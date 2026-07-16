#!/usr/bin/env python3
"""Main training script for PINN-based transformer fault diagnosis.

Supports all PINN backbone architectures:
    - AlexNet-SE (default)
    - ResNet-18
    - LeNet-5
    - VGG16-BN
    - ConvNeXt-Tiny

With optional PCNN preprocessing and Laplacian physics-informed regularization.

Usage:
    # Train with default config
    python scripts/train.py

    # Train with a specific config
    python scripts/train.py --config configs/default.yaml

    # Override model and dataset
    python scripts/train.py \
        --config configs/default.yaml \
        --model.name resnet \
        --dataset.root_dir ./datasets/mel_gadf_pcnn_split \
        --output.root_dir ./experiments/exp_resnet
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, save_yaml
from src.utils.experiment import set_seed, get_device
from src.trainers.workflow import train_and_evaluate


def _parse_overrides(overrides: list) -> dict:
    """Parse key=value overrides into a nested dict.

    Supports dotted keys like 'model.name=resnet'.
    """
    result = {}
    for item in overrides or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to infer type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = float(value)
                if value == int(value):
                    value = int(value)
            except ValueError:
                pass
        d[keys[-1]] = value
    return result


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Train a PINN model for transformer fault diagnosis."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to base configuration YAML.",
    )
    parser.add_argument(
        "--exp_config", type=str, default="",
        help="Path to experiment-specific override YAML.",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in key=value format (e.g., model.name=resnet).",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, args.exp_config)

    # Apply CLI overrides
    if args.overrides:
        overrides = _parse_overrides(args.overrides)
        config = _deep_merge(config, overrides)

    # Setup
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = get_device(str(config.get("device", "auto")))
    print(f"[Device] {device}")

    # Output directory
    output_root = Path(config.get("output", {}).get("root_dir", "./experiments/exp1"))
    run_dir = output_root / config.get("experiment_name", "baseline")

    # Save effective config for reproducibility
    save_yaml(str(run_dir / "config.yaml"), config)

    # Train
    result = train_and_evaluate(config, run_dir, device, seed)

    print(f"\n[Summary] Best val F1: {result['best_val_f1']:.4f}")
    if result["test_metrics"]:
        acc, prec, rec, f1, gmean, bal_acc, kappa = result["test_metrics"]
        print(
            f"[Test] Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} "
            f"F1={f1:.4f} G-Mean={gmean:.4f}"
        )


if __name__ == "__main__":
    main()
