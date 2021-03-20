import torch
from torch import nn
from torch.distributions import Categorical, Normal as N
from ..utils import attention_weights


class AddressRecognition(nn.Module):
    def __init__(self, encoder=None, matching_dim=128, data_dim=784,
                 num_dummies=1):
        """
        Address recognition model (encoder)
        represents distribution
        q(a | M, x)
        """
        super().__init__()
        self.h = encoder(matching_dim)
        self.dummy = nn.Parameter(data=torch.zeros(1, num_dummies, data_dim),
                                  requires_grad=True)

    def add_dummy(self, M):
        dummy = self.dummy.repeat(M.size(0), 1, 1)
        return torch.cat([M, dummy], 1)

    def forward(self, x, M):
        """
        Returns categorical distribution for each memory entity
        using attention-like procedure

        Args:
            x (batch, x_dim): object to address
            M (batch, m_size, x_dim): set of few-shot conditioning objects
        """
        query = self.h(x)
        keys = self.h(M)
        return Categorical(attention_weights(query, keys))


class RecognitionModel(nn.Module):
    def __init__(self, feature_encoder=None, latent_dim=50):
        """
        Recognition model (encoder)
        represents distribution
        q(z | m_a, x) = N(z | μ(x, m_a), σ(x, m_a))
        """
        super().__init__()
        self.feature_encoder = feature_encoder
        self.mu = nn.Linear(4 * latent_dim, latent_dim)
        self.sigma = nn.Sequential(
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Softplus()
        )

    def get_features(self, x):
        return self.feature_encoder(x)

    def forward(self, x, m_a):
        """
        Args:
            x (batch, x_dim):
            m_a (batch, importance_n, x_dim):
        """
        fx = self.get_features(x)
        fx = fx.unsqueeze(1).repeat(1, m_a.shape[1], 1)
        fm = self.get_features(m_a)
        f = torch.cat([fx, fm], 2).contiguous()
        mu = self.mu(f)
        sigma = self.sigma(f)
        return N(mu, sigma)
