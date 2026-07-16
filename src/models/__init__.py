"""Model registry: builds PINN-enhanced classifiers from config."""

from typing import Dict

from .pcnn import PCNNLayer
from .se_block import SEModule
from .pinn_alexnet import PINNAlexNetClassifier
from .pinn_lenet import PINNLeNetClassifier
from .pinn_resnet import PINNResNetClassifier
from .pinn_vggnet import PINNVGGNetClassifier
from .pinn_convnext import PINNConvNeXtClassifier


__all__ = [
    "PCNNLayer",
    "SEModule",
    "PINNAlexNetClassifier",
    "PINNLeNetClassifier",
    "PINNResNetClassifier",
    "PINNVGGNetClassifier",
    "PINNConvNeXtClassifier",
    "build_model",
]


_MODEL_REGISTRY = {
    "alexnet": PINNAlexNetClassifier,
    "alexnet_se": PINNAlexNetClassifier,
    "lenet": PINNLeNetClassifier,
    "resnet": PINNResNetClassifier,
    "resnet18": PINNResNetClassifier,
    "vgg": PINNVGGNetClassifier,
    "vgg16": PINNVGGNetClassifier,
    "convnext": PINNConvNeXtClassifier,
    "convnext_tiny": PINNConvNeXtClassifier,
}


def build_model(config: Dict, use_pcnn: bool = True):
    """Build a PINN model from configuration.

    Args:
        config: full configuration dict.
        use_pcnn: whether to include the PCNN preprocessing layer.

    Returns:
        Instantiated PINN model.
    """
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "alexnet")).lower()
    num_classes = int(model_cfg.get("num_classes", 5))
    in_channels = int(model_cfg.get("in_channels", 1))
    pcnn_steps = int(model_cfg.get("pcnn_steps", 10))

    cls = _MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unsupported model name: {name}. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    model = cls(
        num_classes=num_classes,
        in_channels=in_channels,
        use_pcnn=use_pcnn,
        pcnn_steps=pcnn_steps,
    )
    return model
