import torch
from torch import nn
from torch.distributions import Bernoulli as B, Normal as N


class GenerativeModel(nn.Module):
    def __init__(self, encoder=None, decoder=None, latent_dim=64,
                 distribution='bernoulli'):
        """
        Generative model (decoder).
        Represents distribution p(x | z, m_a) - usually Bernoulli or Normal
        """
        super().__init__()
        self.m_encoder = encoder
        self.mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim)
        )
        self.decoder = decoder
        self.distribution_kind = distribution

    def distribution(self, param):
        if self.distribution_kind == 'bernoulli':
            return B(logits=param)
        elif self.distribution == 'normal':
            return N(param, param.new_ones(param.shape))
        else:
            raise ValueError('unsupported distribution')

    def forward(self, z, m_a):
        """
        Args:
            z (..., latent_dim): sample from the latent space
                (either from recognition or prior model)
            m_a (..., latent_dim): condition objects
        """
        zm = self.mu(self.m_encoder(m_a))
        z = torch.cat([z, zm], -1)
        return self.distribution(self.decoder(z, skips=self.m_encoder.cache))
