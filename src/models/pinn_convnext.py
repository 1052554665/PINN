"""PINN-ConvNeXt Classifier.

Lightweight ConvNeXt-Tiny variant with:
- Single-channel input adaptation
- Reduced block count per stage
- Physics-informed output head
- Optional PCNN preprocessing

Reference:
    Liu et al., "A ConvNet for the 2020s," CVPR 2022.
"""

import torch
import torch.nn as nn

from .pcnn import PCNNLayer


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: 7×7 depthwise conv → 1×1 conv → GELU → 1×1 conv."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()  # simplified: no stochastic depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        # (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Module):
    """One ConvNeXt stage: optionally downsample, then blocks."""

    def __init__(self, in_dim: int, out_dim: int, num_blocks: int, downsample: bool):
        super().__init__()
        layers = []
        if downsample:
            layers.append(
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
            )
        else:
            layers.append(
                nn.Conv2d(in_dim, out_dim, kernel_size=1)
            )
        for _ in range(num_blocks):
            layers.append(ConvNeXtBlock(out_dim))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class PINNConvNeXtClassifier(nn.Module):
    """Lightweight ConvNeXt for PINN-based fault diagnosis.

    Architecture:
        Stem: Conv(4×4, stride=4, 96)
        Stage 1: 96→96, no downsample
        Stage 2: 96→192, downsample
        Stage 3: 192→384, downsample
        Stage 4: 384→768, downsample
        Phys_Out: 1×1 Conv
        AdaptiveAvgPool → FC(num_classes)

    Args:
        num_classes: number of fault classes.
        in_channels: input channels.
        use_pcnn: whether to prepend PCNN layer.
        pcnn_steps: number of PCNN iterations.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        use_pcnn: bool = True,
        pcnn_steps: int = 10,
    ):
        super().__init__()

        self.use_pcnn = use_pcnn
        if use_pcnn:
            self.pcnn = PCNNLayer(num_steps=pcnn_steps)

        # Stem: 224 → 56
        self.stem = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)

        # Stages with progressively increasing channel dimensions
        self.stage1 = ConvNeXtStage(96, 96, num_blocks=1, downsample=False)
        self.stage2 = ConvNeXtStage(96, 192, num_blocks=1, downsample=True)
        self.stage3 = ConvNeXtStage(192, 384, num_blocks=1, downsample=True)
        self.stage4 = ConvNeXtStage(384, 768, num_blocks=1, downsample=True)

        # Physics output head
        self.phys_out = nn.Conv2d(768, 1, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: input tensor (B, C, H, W).

        Returns:
            (class_logits, phys_field) tuple.
        """
        if self.use_pcnn:
            x = self.pcnn(x)

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        out_phys = self.phys_out(x)
        pooled = self.avgpool(x).view(x.size(0), -1)
        out_cls = self.classifier(pooled)

        return out_cls, out_phys
