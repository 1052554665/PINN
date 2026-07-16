"""PINN-ResNet Classifier.

Modified ResNet-18 with:
- Single-channel input adaptation
- Physics-informed output head
- Physical field feedback to classifier
- Optional PCNN preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from .pcnn import PCNNLayer


class PINNResNetClassifier(nn.Module):
    """ResNet-18 backbone with physics-informed enhancements.

    The first convolutional layer is adapted for single-channel input,
    and a 1×1 physics output layer extracts physical fields from the
    feature maps. Physical field feedback is concatenated with pooled
    features before classification.

    Args:
        num_classes: number of fault classes.
        in_channels: input channels (1 for grayscale).
        use_pcnn: whether to prepend PCNN layer.
        pcnn_steps: number of PCNN iterations.
        pretrained: whether to use ImageNet pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        use_pcnn: bool = True,
        pcnn_steps: int = 10,
        pretrained: bool = True,
    ):
        super().__init__()

        self.use_pcnn = use_pcnn
        if use_pcnn:
            self.pcnn = PCNNLayer(num_steps=pcnn_steps)

        # Build ResNet-18 backbone
        if pretrained and in_channels == 3:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        base = resnet18(weights=weights)

        # Adapt first conv for non-3-channel input
        if in_channels != 3:
            base.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove final FC and avgpool (handled separately)
        self.features = nn.Sequential(*list(base.children())[:-2])

        # Physics output head
        self.phys_out = nn.Conv2d(512, 1, kernel_size=1)

        # Classifier with physical field feedback
        self.classifier = nn.Linear(512 + 1, num_classes)

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

        # Pool and fuse with physical field
        feat_avg = F.adaptive_avg_pool2d(feat, (1, 1))
        phys_avg = F.adaptive_avg_pool2d(out_phys, (1, 1))
        fused = torch.cat([feat_avg, phys_avg], dim=1).view(x.size(0), -1)

        out_cls = self.classifier(fused)

        return out_cls, out_phys
