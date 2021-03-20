import torch
from torch import nn
from torch import optim
from torch import distributions as dist
from torch.nn import functional as F

import numpy as np


class ResBlock(nn.Module):
    def __init__(self, input_shape, out_channels, scale_kernel, block_kernel, stride, downscale=True):
        Conv = nn.Conv2d if downscale else nn.ConvTranspose2d
        in_channels = input_shape[0]
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv(in_channels, out_channels, scale_kernel, stride=stride),
            nn.PReLU(out_channels)
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, block_kernel, 
                                  padding=(block_kernel[0] - 1) // 2)
        self.f = nn.PReLU(out_channels)
        self.scale = Conv(in_channels, out_channels, [1, 1])
        if downscale:
            self.pool = nn.AvgPool2d(scale_kernel, stride=stride)
        else:
            x = torch.zeros(2, *input_shape)
            size = self.conv1(x).shape[2:]
            self.pool = nn.UpsamplingBilinear2d(size)

    def forward(self, x):
        h = self.conv1(x)
        return self.f(self.conv2(h) + h) + self.pool(self.scale(x))
