#!/usr/bin/env python3
"""Evaluate a trained PINN model on a test dataset.

Computes comprehensive metrics:
    - Accuracy, Precision, Recall, F1, G-Mean, Balanced Accuracy, Kappa
    - ROC-AUC (macro, one-vs-rest)
    - Confusion matrix
    - t-SNE visualization

Usage:
    python scripts/evaluate.py \
        --checkpoint experiments/exp1/baseline/checkpoints/best.pt \
        --config experiments/exp1/baseline/config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.experiment import get_device
from src.utils.metrics import compute_metrics, compute_roc_auc, count_parameters
from src.utils.plot_confusion import plot_confusion
from src.utils.tsne import extract_features, plot_tsne
from src.utils.train_eval import evaluate
from src.datasets import build_dataloaders
from src.models import build_model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PINN model."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the config YAML used for training.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Directory for evaluation outputs (default: checkpoint dir).",
    )
    parser.add_argument(
        "--no_pcnn", action="store_true",
        help="Disable PCNN even if config says use_pcnn=true.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = get_device(str(config.get("device", "auto")))
    print(f"[Device] {device}")

    # Override PCNN setting if requested
    if args.no_pcnn:
        config.setdefault("model", {})["use_pcnn"] = False

    # Build model and load weights
    use_pcnn = bool(config.get("model", {}).get("use_pcnn", True))
    model = build_model(config, use_pcnn=use_pcnn).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    n_params = count_parameters(model)
    model_name = config.get("model", {}).get("name", "unknown")
    print(f"[Model] {model_name} | Params: {n_params / 1e6:.2f}M")

    # Data
    loaders, class_names = build_dataloaders(config, device=device)

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parents[1] / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    if loaders["test"] is not None:
        lambda_phy = float(config.get("model", {}).get("lambda_phy", 0.1))
        criterion = torch.nn.CrossEntropyLoss()

        test_loss, metrics, y_true, y_pred, y_score = evaluate(
            model, loaders["test"], criterion, device, lambda_phy,
        )
        acc, prec, rec, f1, gmean, bal_acc, kappa = metrics
        auc = compute_roc_auc(y_true, y_score, len(class_names))

        print(f"\n{'='*50}")
        print(f"  Model: {model_name}")
        print(f"  Params: {n_params / 1e6:.2f}M")
        print(f"{'='*50}")
        print(f"  Accuracy:        {acc:.4f}")
        print(f"  Precision (macro): {prec:.4f}")
        print(f"  Recall (macro):    {rec:.4f}")
        print(f"  F1-score (macro):  {f1:.4f}")
        print(f"  G-Mean:            {gmean:.4f}")
        print(f"  Balanced Acc:      {bal_acc:.4f}")
        print(f"  Cohen's Kappa:     {kappa:.4f}")
        print(f"  ROC-AUC (macro):   {auc:.4f}")
        print(f"{'='*50}")

        # Per-class metrics
        from sklearn.metrics import classification_report
        print("\n[Per-Class Report]")
        print(classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
        ))

        # Confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion(
            y_true, y_pred, class_names,
            save_path=str(cm_path),
            title=f"{model_name} Confusion Matrix",
        )
        print(f"\n[Confusion Matrix] {cm_path}")

        # t-SNE
        features, labels = extract_features(model, loaders["test"], device)
        tsne_path = output_dir / "tsne.png"
        plot_tsne(
            features, labels, class_names,
            save_path=str(tsne_path),
            title=f"{model_name} t-SNE",
        )
        print(f"[t-SNE] {tsne_path}")

    else:
        print("[WARN] No test set found in dataset.")


if __name__ == "__main__":
    main()
