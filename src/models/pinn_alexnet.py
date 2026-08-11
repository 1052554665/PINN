"""PINN-AlexNet-SE Classifier.

Improved AlexNet with:
- Batch Normalization after each convolutional layer
- Squeeze-and-Excitation (SE) channel attention modules
- Physics-informed output head (Laplacian regularization)
- Optional PCNN preprocessing layer
"""

import torch
import torch.nn as nn

from .pcnn import PCNNLayer
from .se_block import SEModule


class PINNAlexNetClassifier(nn.Module):
    """AlexNet-SE backbone with physics-informed output.

    Architecture:
        Conv1 (3×3, 64) → BN → ReLU → SE → MaxPool
        Conv2 (3×3, 128) → BN → ReLU → SE → MaxPool
        Conv3 (3×3, 256) → BN → ReLU → SE
        Conv4 (3×3, 256) → BN → ReLU → SE → MaxPool
        Conv5 (3×3, 512) → BN → ReLU → SE
        Conv6 (3×3, 512) → BN → ReLU → SE → MaxPool
        AdaptiveAvgPool → Dropout → FC(1024) → Dropout → FC(num_classes)
        Phys_Out: 1×1 Conv → single-channel physical field

    Args:
        num_classes: number of fault classes.
        in_channels: input channels (1 for grayscale, 3 for RGB).
        use_pcnn: whether to prepend a PCNN preprocessing layer.
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
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEModule(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SEModule(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEModule(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEModule(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEModule(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEModule(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Physics output head: produces a 1-channel physical field
        self.phys_out = nn.Conv2d(512, 1, kernel_size=1)

        # Physical field feedback: pooled phys field is concatenated with
        # feature vector before classification (see manuscript Section 4.4).
        # The +1 accounts for the global-average-pooled phys field.
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7 + 1, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
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
        pooled = self.avgpool(feat)

        # Physical field feedback: global-average-pool phys → scalar per sample
        phys_feedback = out_phys.view(out_phys.size(0), -1).mean(dim=1, keepdim=True)
        fused = torch.cat([pooled.view(pooled.size(0), -1), phys_feedback], dim=1)
        out_cls = self.classifier(fused)

        return out_cls, out_phys
