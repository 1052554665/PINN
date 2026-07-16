"""Squeeze-and-Excitation (SE) Channel Attention Module.

Reference:
    Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018.
"""

import torch
import torch.nn as nn


class SEModule(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Recalibrates channel-wise feature responses by explicitly modelling
    inter-dependencies between channels. The module consists of:
        1. Global average pooling (squeeze)
        2. Two 1×1 convolutions with ReLU (excitation)
        3. Sigmoid gating

    Args:
        channels: number of input/output channels.
        reduction: reduction ratio for the bottleneck (default: 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention.

        Args:
            x: input tensor of shape (B, C, H, W).

        Returns:
            Channel-recalibrated tensor of same shape.
        """
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y
