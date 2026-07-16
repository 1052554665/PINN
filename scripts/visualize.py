#!/usr/bin/env python3
"""Visualization script for PINN model results.

Generates:
    - Physical field visualizations from PINN models
    - t-SNE embeddings
    - Confusion matrices
    - Training curves
    - Comparative bar charts across experiments

Usage:
    python scripts/visualize.py --results_dir experiments/ablation/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plot_training_curves(log_path: str, save_path: str):
    """Plot training and validation curves from a CSV log."""
    df = pd.read_csv(log_path)
    if df.empty:
        print(f"[WARN] Empty log: {log_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    if "val_loss" in df.columns:
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy / F1 curves
    ax = axes[1]
    ax.plot(df["epoch"], df["train_acc"], label="Train Acc", marker="o")
    if "val_f1" in df.columns:
        ax.plot(df["epoch"], df["val_f1"], label="Val F1", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy / F1 Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Curves] {save_path}")


def plot_comparison(results: dict, save_path: str):
    """Plot bar chart comparing experiments."""
    names = list(results.keys())
    values = list(results.values())

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars = ax.bar(range(len(names)), values, color=plt.cm.Blues(np.linspace(0.3, 0.8, len(names))))

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Comparison] {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PINN experiment results."
    )
    parser.add_argument(
        "--results_dir", type=str, default="./experiments",
        help="Root directory of experiment results.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./experiments/figures",
        help="Output directory for generated figures.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all train logs and plot curves
    for log_file in results_dir.rglob("train_log.csv"):
        exp_name = log_file.parents[1].name + "_" + log_file.parents[2].name
        save_path = output_dir / f"curves_{exp_name}.png"
        plot_training_curves(str(log_file), str(save_path))

    print(f"\n[Done] Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
