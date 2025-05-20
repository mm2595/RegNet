"""
RegNet – collection of loss functions
-------------------------------------

• supervised_loss          : BCE on edge labels
• vae_loss                 : standard KL term
• contrastive_loss         : margin‑based (legacy, seldom used)
• masked_recon_loss        : Gaussian NLL on masked genes  (σ configurable)
• info_nce_loss            : cosine InfoNCE / NT‑Xent with hard negatives
• attention_entropy_loss   : row‑entropy regulariser for self‑attention
• discrepancy / EWC terms  : block‑wise weight anchoring
• relational / prototype alignment utilities
"""

import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ #
#  Core losses
# ------------------------------------------------------------------ #
def supervised_loss(preds: torch.Tensor,
                    labels: torch.Tensor,
                    pos_weight: torch.Tensor | None = None):
    return F.binary_cross_entropy(preds, labels, pos_weight=pos_weight)


def vae_loss(mu, logvar, beta: float = 1.0):
    return beta * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))


def vae_reconstruction_loss(model, x, edge_index):
    """
    VAE reconstruction loss: MSE between original encoding and reconstructed encoding
    
    Instead of trying to reconstruct the original input x directly,
    we'll reconstruct the hidden representation from GraphSAGE.
    This ensures dimension compatibility.
    
    Parameters
    ----------
    model : RegNet model with VAE submodule
    x : input features
    edge_index : edge indices
    
    Returns
    -------
    reconstruction_loss : torch.Tensor
        MSE between encoding and reconstructed encoding
    """
    # Get node embeddings via GraphSAGE
    h0 = model.graphsage(x, edge_index)
    
    # Get attention output
    h1, _, _, _ = model.attention(h0)
    
    # Encode and decode through VAE
    _, z, mu, logvar = model.vae.reconstruct(h1)
    
    # Decode back to get reconstruction
    x_recon = model.vae.decoder(z)
    
    # Compute reconstruction loss using h1 as target (same dimensions as x_recon)
    return F.mse_loss(x_recon, h1.detach())


def contrastive_loss(embeddings, pos_pairs, neg_pairs, margin: float = 1.0):
    """
    Legacy margin‑based contrastive loss (Euclidean distances).
    """
    pos_d2 = torch.sum((embeddings[pos_pairs[:, 0]] -
                        embeddings[pos_pairs[:, 1]])**2, dim=1)
    neg_d2 = torch.sum((embeddings[neg_pairs[:, 0]] -
                        embeddings[neg_pairs[:, 1]])**2, dim=1)
    return torch.mean(pos_d2 + F.relu(margin - neg_d2))

# ------------------------------------------------------------------ #
#  Self‑supervised losses
# ------------------------------------------------------------------ #
def masked_recon_loss(x_orig: torch.Tensor,
                      x_recon: torch.Tensor,
                      mask: torch.Tensor,
                      *,
                      sigma_rec: float = 0.1):
    """
    Gaussian negative‑log‑likelihood on *masked* entries only.
    """
    diff2 = ((x_orig - x_recon) * mask)**2
    return diff2.sum() / (2 * sigma_rec**2 * mask.sum().clamp_min(1.))


def info_nce_loss(emb: torch.Tensor,
                  pos_pairs: torch.Tensor,
                  neg_pairs: torch.Tensor | None = None,
                  *,
                  temperature: float = 0.1,
                  neg_push_weight: float = 0.05):
    """
    NT‑Xent / InfoNCE with cosine similarities.

    emb        : (N,d)  latent means (will be ℓ2‑normalised)
    pos_pairs  : (P,2)  LongTensor of positive indices
    neg_pairs  : (Q,2)  optional extra hard negatives
    """
    z   = F.normalize(emb, dim=1)                       # (N,d)
    sim = torch.mm(z, z.t()) / temperature              # (N,N) logits

    # Positive term
    pos_sim = sim[pos_pairs[:, 0], pos_pairs[:, 1]]     # (P,)
    denom   = torch.exp(sim[pos_pairs[:, 0]]).sum(dim=1)  # include self

    # subtract self‑similarity (exp(1/τ) = exp(1/τ))
    denom = denom - torch.exp(torch.ones_like(pos_sim))     # (P,)
    loss  = -torch.log(torch.exp(pos_sim) / denom).mean()

    # Optional hard‑negative additive repulsion
    if neg_pairs is not None and neg_pairs.numel() > 0:
        neg_sim = sim[neg_pairs[:, 0], neg_pairs[:, 1]]
        loss += neg_push_weight * torch.log(torch.exp(neg_sim)).mean()
    return loss


def attention_entropy_loss(row_entropy: torch.Tensor,
                           *,
                           target_entropy: float = 0.0):
    """
    Row‑entropy KL surrogate: encourages sparse attention rows.
    Return is (H_row − target); multiply by λ in training script.
    """
    return row_entropy - target_entropy

# ------------------------------------------------------------------ #
#  Discrepancy / EWC penalties
# ------------------------------------------------------------------ #
def discrepancy_loss(model, pretrained_model, weight: float = 1.0):
    loss = 0.0
    for param, pre_param in zip(model.parameters(),
                                pretrained_model.parameters()):
        loss += torch.sum((param - pre_param.detach())**2)
    return weight * loss


def layerwise_discrepancy_loss(model, pretrained_model, weights: dict):
    """
    Separate λ per block name contained in param path.
    """
    loss = 0.0
    for (name, param), (_, pre_param) in zip(model.named_parameters(),
                                            pretrained_model.named_parameters()):
        if 'attention' in name:
            loss += weights.get('attention', 0.0) * \
                    torch.sum((param - pre_param.detach())**2)
        elif 'vae' in name:
            loss += weights.get('vae', 0.0) * \
                    torch.sum((param - pre_param.detach())**2)
    return loss


def ewc_loss(model,
             pretrained_state_dict: dict,
             fisher_diag: dict,
             lambda_by_block: dict):
    """
    Elastic‑Weight Consolidation quadratic penalty:
    Σ λ_b * F_i * (θ_i − θ*_i)^2
    """
    dev = next(model.parameters()).device
    loss = torch.tensor(0.0, device=dev)

    for name, param in model.named_parameters():
        if name not in pretrained_state_dict or name not in fisher_diag:
            continue
        θ_star = pretrained_state_dict[name].to(dev)
        if param.shape != θ_star.shape:
            continue  # skip mismatched shapes (e.g. input dim changed)

        if   'graphsage' in name:
            lam = lambda_by_block.get('graphsage', 0.0)
        elif 'attention' in name:
            lam = lambda_by_block.get('attention', 0.0)
        elif 'vae' in name:
            lam = lambda_by_block.get('vae', 0.0)
        else:
            lam = lambda_by_block.get('other', 0.0)
        if lam == 0.0:
            continue

        loss += lam * torch.sum(fisher_diag[name] *
                                (param - θ_star)**2)
    return loss

# ------------------------------------------------------------------ #
#  Alignment helpers
# ------------------------------------------------------------------ #
def relational_alignment_loss(current: torch.Tensor,
                              pretrained: torch.Tensor):
    return F.mse_loss(current, pretrained)


def prototype_alignment_loss(emb_pairs: torch.Tensor,
                             prototype: torch.Tensor):
    return F.mse_loss(emb_pairs, prototype)
