import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class VariationalDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=None):
        super(VariationalDecoder, self).__init__()
        if hidden_dim is None:
            hidden_dim = 2 * latent_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=None):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_dim, latent_dim)
        self.decoder = VariationalDecoder(latent_dim, input_dim, hidden_dim)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def reconstruct(self, x):
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar

    def loss_function(self, mu, logvar):
        """KL divergence loss term only"""
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
    
    def reconstruction_loss(self, x, x_recon, reduction='mean'):
        """Mean-squared error reconstruction loss"""
        return F.mse_loss(x_recon, x, reduction=reduction)
