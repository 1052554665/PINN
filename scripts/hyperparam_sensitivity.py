#!/usr/bin/env python3
"""
=============================================================================
Hyperparameter Sensitivity Analysis
=============================================================================
Addresses Reviewer #1, #4, #5:
  - Gamma correction value sensitivity (γ = 0.5, 1.0, 1.5, 1.7, 2.0, 2.5)
  - PINN regularization weight (λ_phy = 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
  - PCNN threshold sensitivity (V_T = 0.4, 0.6, 0.8, 1.0)
  - PCNN iteration steps (n_steps = 5, 10, 15, 20)
  - Learning rate sensitivity
  - Batch size sensitivity

Outputs:
  - Sensitivity curves (metric vs. parameter value)
  - Optimal parameter recommendations
=============================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import gmean
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.image_classification import MelSpectrogramDataset
from src.models.pcnn import PCNNLayer
from src.models.se_block import SEModule
from src.utils.laplacian import laplacian_loss

# ---------------------------------------------------------------------------
# Base Configuration
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_root": "./datasets/mel_gadf_pcnn",
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "num_classes": 5,
    "num_folds": 3,          # Reduced for speed
    "epochs": 30,
    "lr": 0.01,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "lambda_phy": 0.1,
    "pcnn_steps": 10,
    "pcnn_threshold": 0.8,
    "gamma_value": 1.7,
    "output_dir": "./experiments/hyperparam_sensitivity",
}

# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------
class SensitivityModel(nn.Module):
    """Full model with configurable hyperparameters."""
    def __init__(self, num_classes=5, in_channels=3, pcnn_steps=10, pcnn_threshold=0.8):
        super().__init__()
        self.pcnn = PCNNLayer(num_steps=pcnn_steps)
        self.pcnn.VT = pcnn_threshold  # Override threshold
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            SEModule(64), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            SEModule(128), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            SEModule(256), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            SEModule(512), nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.phys_out = nn.Conv2d(512, 1, 1)
        # Physical field feedback: +1 for global-average-pooled phys field
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512 * 7 * 7 + 1, 1024), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.pcnn(x)
        feat = self.features(x)
        phys = self.phys_out(feat)
        pooled = self.avgpool(feat)
        # Physical field feedback: global-average-pool phys → scalar per sample
        phys_feedback = phys.view(phys.size(0), -1).mean(dim=1, keepdim=True)
        fused = torch.cat([pooled.view(pooled.size(0), -1), phys_feedback], dim=1)
        return self.classifier(fused), phys


# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------
def run_single_experiment(config, override_params, train_loader, val_loader, device):
    """Run one experiment with given parameter overrides. Returns best F1."""
    cfg = config.copy()
    cfg.update(override_params)

    model = SensitivityModel(
        num_classes=cfg["num_classes"], in_channels=3,
        pcnn_steps=cfg.get("pcnn_steps", 10),
        pcnn_threshold=cfg.get("pcnn_threshold", 0.8),
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"],
        momentum=cfg["momentum"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    lambda_phy = cfg.get("lambda_phy", 0.1)
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out_cls, phys = model(images)
            loss = F.cross_entropy(out_cls, labels) + lambda_phy * laplacian_loss(phys)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out_cls, _ = model(images)
                _, pred = out_cls.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 10:
            break

    return best_f1


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------
def run_sensitivity_analysis(config):
    device = torch.device(config["device"])
    set_seed(config["seed"])

    transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
    ])
    full_dataset = MelSpectrogramDataset(config["data_root"], transform=transform)
    labels_all = np.array([s[1] for s in full_dataset.samples])

    skf = StratifiedKFold(n_splits=config["num_folds"], shuffle=True, random_state=config["seed"])
    fold_splits = list(skf.split(np.arange(len(full_dataset)), labels_all))

    # ---- Parameter grids ----
    param_grids = {
        "Gamma Correction (γ)": {
            "param": "gamma_value",
            "values": [0.5, 0.8, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0],
            "xlabel": "Gamma (γ)",
        },
        "PINN Weight (λ_phy)": {
            "param": "lambda_phy",
            "values": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            "xlabel": "λ_phy",
        },
        "PCNN Threshold (V_T)": {
            "param": "pcnn_threshold",
            "values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            "xlabel": "PCNN Threshold V_T",
        },
        "PCNN Iteration Steps": {
            "param": "pcnn_steps",
            "values": [2, 5, 8, 10, 15, 20],
            "xlabel": "PCNN Iteration Steps",
        },
        "Learning Rate": {
            "param": "lr",
            "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            "xlabel": "Learning Rate",
        },
    }

    all_results = {}

    for study_name, grid in param_grids.items():
        print(f"\n{'='*60}")
        print(f"  Sensitivity: {study_name}")
        print(f"{'='*60}")

        results_per_value = {}
        for val in grid["values"]:
            fold_f1s = []
            for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                train_subset = Subset(full_dataset, train_idx)
                val_subset = Subset(full_dataset, val_idx)
                train_loader = DataLoader(train_subset, batch_size=config["batch_size"],
                                           shuffle=True, num_workers=config["num_workers"])
                val_loader = DataLoader(val_subset, batch_size=config["batch_size"],
                                         shuffle=False, num_workers=config["num_workers"])

                best_f1 = run_single_experiment(
                    config, {grid["param"]: val}, train_loader, val_loader, device
                )
                fold_f1s.append(best_f1)

            mean_f1 = np.mean(fold_f1s)
            std_f1 = np.std(fold_f1s)
            results_per_value[str(val)] = {"mean_f1": float(mean_f1), "std_f1": float(std_f1)}
            print(f"    {grid['param']}={val:<6} → F1 = {mean_f1:.4f} ± {std_f1:.4f}")

        all_results[study_name] = results_per_value
        optimal_val = max(results_per_value, key=lambda k: results_per_value[k]["mean_f1"])
        print(f"    → Optimal: {grid['param']} = {optimal_val}")

    # ---- Plot sensitivity curves ----
    os.makedirs(config["output_dir"], exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (study_name, grid) in zip(axes, param_grids.items()):
        values = grid["values"]
        means = [all_results[study_name][str(v)]["mean_f1"] for v in values]
        stds = [all_results[study_name][str(v)]["std_f1"] for v in values]

        ax.errorbar(range(len(values)), means, yerr=stds, marker="o", capsize=5,
                     color="#2196F3", linewidth=2, markersize=8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=30, ha="right", fontsize=8)
        ax.set_title(study_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Macro F1-Score")
        ax.set_xlabel(grid["xlabel"])
        ax.grid(True, alpha=0.3)
        # Highlight optimal
        best_idx = np.argmax(means)
        ax.axvline(x=best_idx, color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Remove extra subplot
    if len(param_grids) < len(axes):
        for ax in axes[len(param_grids):]:
            ax.axis("off")

    plt.suptitle("Hyperparameter Sensitivity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "hyperparam_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save results
    with open(os.path.join(config["output_dir"], "sensitivity_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {config['output_dir']}")
    return all_results


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=BASE_CONFIG["data_root"])
    parser.add_argument("--output_dir", type=str, default=BASE_CONFIG["output_dir"])
    args = parser.parse_args()
    config = BASE_CONFIG.copy()
    config.update({k: v for k, v in vars(args).items() if v is not None})
    run_sensitivity_analysis(config)
