import torch
from torch import nn
from torch import optim
from torch import distributions as dist
from torch.nn import functional as F

import numpy as np
from ..utils import ResBlock


class VMAEncoder(nn.Module):
    def __init__(self, latent_dim=288, hidden_size=100, **kwargs):
        super().__init__()        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(),
        )
        # Block 1 output: (48, 13, 13)
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        # block2 output: (64, 7, 7)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim)
        )
        self.latent_dim = latent_dim
        self.cache = []
        
    def forward(self, x):
        self.cache = []
        xs = x.shape[:-1]
        x = x.view(-1, 1, 26, 26)
        x = self.block1(x)
        self.cache.append(x)
        x = self.block2(x)
        self.cache.append(x)
        x = self.block3(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x).view(*xs, self.latent_dim)


class VMADecoder(nn.Module):
    def __init__(self, latent_dim=64, skip=False, hidden_size=100, **kwargs):
        super().__init__()
        self.skip = skip
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64 * 4 * 4)
        )
        # input shape (64, 4, 4)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # input shape (64, 7, 7)
        cond = 64 if skip else 0
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(64 + cond, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        cond = 48 if skip else 0
        # input shape (64, 13, 13)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(64 + cond, 48, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 1, 2, padding=0),
        )
        self.latent_dim = latent_dim

    def forward(self, x, skips=None):
        if self.skip:
            c2, c1 = skips
        xs = x.shape[:-1]
        x = x.view(-1, self.latent_dim)
        x = self.fc(x)
        x = x.view(x.shape[0], 64, 4, 4)
        x = self.block1(x)
        if self.skip:
            x = torch.cat([x, c1], 1)
        x = self.block2(x)
        if self.skip:
            x = torch.cat([x, c2], 1)
        return self.block3(x).view(*xs, -1)
