"""PINN-VGGNet Classifier.

Modified VGG16-BN with:
- Single-channel input adaptation
- Physics-informed output head
- Physical field feedback to classifier
- Optional PCNN preprocessing
"""

import torch
import torch.nn as nn
from torchvision import models

from .pcnn import PCNNLayer


class PINNVGGNetClassifier(nn.Module):
    """VGG16-BN backbone with physics-informed enhancements.

    The first convolutional layer is adapted for single-channel input,
    and a 1×1 physics output layer extracts physical fields. The
    physical field is pooled and concatenated with features before
    entering the classifier.

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

        # VGG16-BN backbone
        vgg16 = models.vgg16_bn(weights=None)
        features = list(vgg16.features)
        # Adapt first conv layer
        features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.features = nn.Sequential(*features)

        # Physics output head (512 channels after VGG features)
        self.phys_out = nn.Conv2d(512, 1, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier with physical field feedback
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 + 1, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
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

        # Pool features
        feat_pooled = self.avgpool(feat)
        phys_pooled = out_phys.view(out_phys.size(0), -1)[:, :1]  # mean of phys field

        feat_flat = feat_pooled.view(feat_pooled.size(0), -1)
        fused = torch.cat([feat_flat, phys_pooled], dim=1)

        out_cls = self.classifier(fused)

        return out_cls, out_phys
