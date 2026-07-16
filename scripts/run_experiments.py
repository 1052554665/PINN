#!/usr/bin/env python3
"""Run a complete suite of PINN experiments for ablation and comparison.

Runs experiments across:
    1. All backbone architectures (AlexNet, ResNet, LeNet, VGG, ConvNeXt)
    2. With and without PCNN preprocessing
    3. With and without physics-informed (Laplacian) regularization
    4. Different feature representations (Mel-only, GAF-only, Mel-GADF fused)

Usage:
    python scripts/run_experiments.py --config configs/default.yaml
    python scripts/run_experiments.py --config configs/default.yaml --quick
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.experiment import set_seed, get_device
from src.trainers.workflow import train_and_evaluate


# Pre-defined experiment configurations
EXPERIMENTS = {
    # ---- Backbone ablation (with PCNN + physics) ----
    "pcnn_pinn_alexnet": {
        "model.name": "alexnet",
        "model.use_pcnn": True,
        "model.lambda_phy": 0.1,
        "output.root_dir": "./experiments/ablation/backbone",
        "experiment_name": "alexnet_pcnn_pinn",
    },
    "pcnn_pinn_resnet": {
        "model.name": "resnet",
        "model.use_pcnn": True,
        "model.lambda_phy": 0.1,
        "output.root_dir": "./experiments/ablation/backbone",
        "experiment_name": "resnet_pcnn_pinn",
    },
    "pcnn_pinn_lenet": {
        "model.name": "lenet",
        "model.use_pcnn": True,
        "model.lambda_phy": 0.1,
        "output.root_dir": "./experiments/ablation/backbone",
        "experiment_name": "lenet_pcnn_pinn",
    },
    "pcnn_pinn_vgg": {
        "model.name": "vgg",
        "model.use_pcnn": True,
        "model.lambda_phy": 0.1,
        "output.root_dir": "./experiments/ablation/backbone",
        "experiment_name": "vgg_pcnn_pinn",
    },
    "pcnn_pinn_convnext": {
        "model.name": "convnext",
        "model.use_pcnn": True,
        "model.lambda_phy": 0.1,
        "output.root_dir": "./experiments/ablation/backbone",
        "experiment_name": "convnext_pcnn_pinn",
    },

    # ---- PCNN ablation (with AlexNet + physics) ----
    "pinn_alexnet_no_pcnn": {
        "model.name": "alexnet",
        "model.use_pcnn": False,
        "model.lambda_phy": 0.1,
        "output.root_dir": "./experiments/ablation/pcnn",
        "experiment_name": "alexnet_no_pcnn",
    },
    "pcnn_pinn_alexnet_ablation": {
        "model.name": "alexnet",
        "model.use_pcnn": True,
        "model.lambda_phy": 0.1,
        "output.root_dir": "./experiments/ablation/pcnn",
        "experiment_name": "alexnet_with_pcnn",
    },

    # ---- Physics ablation (with AlexNet + PCNN) ----
    "pcnn_alexnet_no_physics": {
        "model.name": "alexnet",
        "model.use_pcnn": True,
        "model.lambda_phy": 0.0,
        "output.root_dir": "./experiments/ablation/physics",
        "experiment_name": "alexnet_no_physics",
    },
    "pcnn_alexnet_physics": {
        "model.name": "alexnet",
        "model.use_pcnn": True,
        "model.lambda_phy": 0.1,
        "output.root_dir": "./experiments/ablation/physics",
        "experiment_name": "alexnet_with_physics",
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base."""
    from copy import deepcopy
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_overrides(overrides: list) -> dict:
    """Parse key=value overrides into nested dict."""
    result = {}
    for item in overrides or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        keys = key.split(".")
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
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


def main():
    parser = argparse.ArgumentParser(
        description="Run PINN experiment suite."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Base config YAML.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer epochs for rapid testing.",
    )
    parser.add_argument(
        "--experiments", type=str, nargs="*",
        default=None,
        help="Specific experiment keys to run (default: all).",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Additional config overrides (e.g., dataset.root_dir=...).",
    )
    args = parser.parse_args()

    # Load base config
    config = load_config(args.config)

    # Global overrides
    if args.overrides:
        overrides = _parse_overrides(args.overrides)
        config = _deep_merge(config, overrides)

    if args.quick:
        config.setdefault("train", {})["epochs"] = 5
        print("[Quick Mode] Epochs set to 5")

    # Determine which experiments to run
    if args.experiments:
        selected = {k: EXPERIMENTS[k] for k in args.experiments if k in EXPERIMENTS}
        missing = set(args.experiments) - set(EXPERIMENTS.keys())
        if missing:
            print(f"[WARN] Unknown experiments: {missing}")
    else:
        selected = EXPERIMENTS

    seed = int(config.get("seed", 42))
    device = get_device(str(config.get("device", "auto")))
    print(f"[Device] {device}")
    print(f"[Experiments] {len(selected)} to run\n")

    results = {}
    for exp_key, exp_overrides in selected.items():
        print(f"\n{'='*60}")
        print(f"  Experiment: {exp_key}")
        print(f"{'='*60}")

        exp_config = _deep_merge(config, exp_overrides)

        set_seed(seed)

        output_root = Path(exp_config.get("output", {}).get("root_dir", "./experiments"))
        run_dir = output_root / exp_config.get("experiment_name", exp_key)

        # Save config
        from src.utils.config import save_yaml
        save_yaml(str(run_dir / "config.yaml"), exp_config)

        try:
            result = train_and_evaluate(exp_config, run_dir, device, seed)
            results[exp_key] = {
                "best_val_f1": result["best_val_f1"],
                "test_metrics": result["test_metrics"],
                "n_params": result["n_params"],
            }
        except Exception as e:
            print(f"[ERROR] Experiment {exp_key} failed: {e}")
            results[exp_key] = {"error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("  Experiment Summary")
    print(f"{'='*60}")
    print(f"{'Experiment':<35} {'Val F1':>8} {'Test F1':>8} {'Params':>10}")
    print("-" * 65)
    for exp_key, info in results.items():
        if "error" in info:
            print(f"{exp_key:<35} {'ERROR':>8}")
        else:
            val_f1 = info["best_val_f1"]
            test_metrics = info["test_metrics"]
            test_f1 = test_metrics[3] if test_metrics else float("nan")
            n_params = info["n_params"] / 1e6
            print(
                f"{exp_key:<35} {val_f1:>8.4f} {test_f1:>8.4f} "
                f"{n_params:>8.2f}M"
            )


if __name__ == "__main__":
    main()
