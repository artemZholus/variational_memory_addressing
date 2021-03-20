import torch
from torch import nn
from torch import distributions as dist
from torch.distributions import Normal as N


class ConditionalPrior(nn.Module):
    def __init__(self, encoder=None, latent_dim=50):
        """
        Conditional Prior distribution p(z | m_a) = N(z | μ(m_a), σ(m_a))
        """
        super().__init__()
        self.encoder = encoder

        self.mu = nn.Linear(2 * latent_dim, latent_dim)
        self.sigma = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.Softplus()
        )
    
    def forward(self, m):
        """
        Args:
            m (batch, hidden_dim): condition object
        """
        zm = self.encoder(m)
        mu = self.mu(zm)
        sigma = self.sigma(zm)
        return N(mu, sigma)
