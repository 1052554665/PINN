"""PINN-LeNet Classifier.

Adapted LeNet-5 with:
- Expanded input size for 224×224 images
- ReLU activation for better convergence
- Physics-informed output head
- Optional PCNN preprocessing
"""

import torch
import torch.nn as nn

from .pcnn import PCNNLayer


class PINNLeNetClassifier(nn.Module):
    """LeNet backbone adapted for PINN-based fault diagnosis.

    Architecture:
        Conv1 (5×5, 6) → ReLU → AvgPool(2×2)
        Conv2 (5×5, 16) → ReLU → AvgPool(2×2)
        Phys_Out: 1×1 Conv
        FC(64) → FC(num_classes)

    Args:
        num_classes: number of fault classes.
        in_channels: input channels (1 for grayscale).
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

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Physics output: 1×1 conv on the 16-channel feature map (54×54)
        self.phys_out = nn.Conv2d(16, 1, kernel_size=1)

        # Compute feature dimension after convolutions
        # Input 224 → Conv1(5×5, pad=0): 220 → Pool: 110
        # → Conv2(5×5, pad=0): 106 → Pool: 53
        # 16 * 53 * 53 = 44944
        self.fc = nn.Sequential(
            nn.Linear(16 * 53 * 53, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: input tensor (B, C, H, W).

        Returns:
            (class_logits, phys_field) tuple.
        """
        if self.use_pcnn:
            x = self.pcnn(x)

        feat = self.features(x)
        out_phys = self.phys_out(feat)
        feat_flat = feat.view(feat.size(0), -1)
        out_cls = self.fc(feat_flat)

        return out_cls, out_phys
