import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .graphsage import GraphSAGE
from .attention import SelfAttentionGateFusion
from .vae import VAE


class RegNet(nn.Module):
    """
    Expression → GraphSAGE → Self‑Attention (row‑entropy) → VAE
               ↘————————– latent μ ————————–↙
                     Edge‑classifier
    """
    def __init__(self,
                 input_dim:  int,
                 hidden_dim: int,
                 latent_dim: int,
                 num_layers: int = 2):
        super().__init__()

        # --- encoders ------------------------------------------------------- #
        self.graphsage = GraphSAGE(input_dim, hidden_dim, num_layers)
        self.attention = SelfAttentionGateFusion(hidden_dim, hidden_dim)
        self.vae       = VAE(hidden_dim, latent_dim, hidden_dim)

        # --- edge classifier  (μ_i ⊕ μ_j → p_ij) --------------------------- #
        self.edge_classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    # --------------------------------------------------------------------- #
    #  Encode node features to latent z plus aux outputs
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Returns
        -------
        z, μ, logvar          : VAE outputs         (N, latent_dim)
        att_w                 : attention weights   (N, N)
        gate_vals             : gating coeffs       (N, hidden_dim)
        row_entropy           : mean row entropy    (scalar tensor)
        """
        h0        = self.graphsage(x, edge_index)
        h1, att_w, gate_vals, row_ent = self.attention(h0)
        z, mu, logvar = self.vae(h1)
        return z, mu, logvar, att_w, gate_vals, row_ent

    # --------------------------------------------------------------------- #
    #  Forward for edge prediction
    # --------------------------------------------------------------------- #
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                node_pairs: torch.Tensor):
        """
        Parameters
        ----------
        node_pairs  : (M,2) long tensor with indices of gene pairs to score

        Returns (ordered)
        -------
        edge_preds  : (M,) probabilities
        att_w       : (N,N) attention weights
        row_entropy : scalar ‑ average entropy per attention row
        mu, logvar  : latent stats (N,d)
        """
        z, mu, logvar, att_w, _gate, row_ent = self.encode(x, edge_index)

        pair_feat  = torch.cat([mu[node_pairs[:, 0]],
                                mu[node_pairs[:, 1]]], dim=-1)
        edge_preds = self.edge_classifier(pair_feat).squeeze()

        return edge_preds, att_w, row_ent, mu, logvar

    # --------------------------------------------------------------------- #
    #  Reconstructs input from latent space
    # --------------------------------------------------------------------- #
    def reconstruct(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Reconstruct the input through the full model pipeline
        Parameters
        ----------
        x  : (N, F) input features
        edge_index : (2, E) edge indices
        
        Returns
        -------
        x_recon : (N, F) reconstructed features
        z, μ, logvar : VAE outputs
        att_w : attention weights
        row_entropy : mean row entropy
        """
        h0 = self.graphsage(x, edge_index)
        h1, att_w, gate_vals, row_ent = self.attention(h0)
        x_recon, z, mu, logvar = self.vae.reconstruct(h1)
        return x_recon, z, mu, logvar, att_w, row_ent

    # --------------------------------------------------------------------- #
    #  Convenience: latent mean extractor
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def latent_mu(self, x: torch.Tensor, edge_index: torch.Tensor):
        _, mu, _, _, _, _ = self.encode(x, edge_index)
        return mu


    def edge_variance(self, mu: torch.Tensor, logvar: torch.Tensor,
                    node_pairs: torch.Tensor) -> torch.Tensor:
        """
        Approximate Var[S_ij] = π/8 ∑ w1_k² (σ_i² + σ_j²)_k   (delta‑method)
        Works for *first* linear layer only.
        
        This version adds robustness to handle dimension mismatches that can occur
        when loading pre-trained models with different architectures.
        """
        # Get first layer weights
        w1 = self.edge_classifier[0].weight      # (hidden, 2d_lat)
        
        # Calculate log sigma squared sum for node pairs
        log_sigma2 = logvar[node_pairs[:, 0]] + logvar[node_pairs[:, 1]]  # (M, d_lat)
        sigma2_sum = log_sigma2.exp()
        
        # Determine the dimensions
        latent_dim = mu.shape[1]
        full_latent_dim = 2 * latent_dim  # For concatenated pairs
        
        # Safely get appropriate weights for calculation
        # Use only what we need from w1, up to the available dimensions
        if w1.shape[1] != full_latent_dim:
            # Handle dimension mismatch by using a fixed variance
            print(f"Warning: Dimension mismatch in edge_variance. Using fixed variance. "
                  f"Expected: {full_latent_dim}, Got: {w1.shape[1]}")
            # Return a fixed small variance
            return torch.ones_like(node_pairs[:, 0], dtype=torch.float32) * 0.01
        
        # Pick corresponding columns of w1 (latent → hidden)
        w_sq = (w1**2).sum(0)[:full_latent_dim]  # (2d_lat,)
        
        # Ensure compatible dimensions for multiplication
        if w_sq.shape[0] != sigma2_sum.shape[1] * 2:
            # Truncate or pad w_sq to match dimensions
            if w_sq.shape[0] > sigma2_sum.shape[1] * 2:
                w_sq = w_sq[:sigma2_sum.shape[1] * 2]
            else:
                w_sq = F.pad(w_sq, (0, sigma2_sum.shape[1] * 2 - w_sq.shape[0]))
        
        # Reshape w_sq to match sigma2_sum's dimensions for element-wise multiplication
        w_sq_reshaped = w_sq.view(2, -1)[:, :sigma2_sum.shape[1]].sum(0)
        
        # Final calculation
        var_logit = (math.pi / 8) * (sigma2_sum * w_sq_reshaped).sum(1)
        
        return var_logit  # (M,)
