"""Evaluation metrics for fault diagnosis classification.

Metrics computed:
    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - F1-score (macro)
    - G-Mean (geometric mean of per-class recall)
    - Balanced Accuracy
    - Cohen's Kappa
    - ROC-AUC (macro, one-vs-rest)
"""

import warnings

import numpy as np
from scipy.stats import gmean
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import torch
import torch.nn as nn


def compute_metrics(y_true, y_pred):
    """Compute classification metrics.

    Args:
        y_true: ground-truth labels (1-d array).
        y_pred: predicted labels (1-d array).

    Returns:
        Tuple of (acc, precision, recall, f1, g_mean, bal_acc, kappa).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*y_pred contains classes not in y_true.*"
        )
        warnings.filterwarnings("ignore", message=".*Only one class.*")

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        g_mean = float(gmean(recall_per_class + 1e-6))

        bal_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

    return acc, precision, recall, f1, g_mean, bal_acc, kappa


def compute_roc_auc(
    y_true,
    y_score,
    num_classes: int,
    average: str = "macro",
):
    """Compute multi-class ROC-AUC.

    Args:
        y_true: 1-d array of ground-truth class indices.
        y_score: 2-d array of predicted probabilities [N, num_classes].
        num_classes: total number of classes.
        average: 'macro' (default), 'weighted', or None.

    Returns:
        Scalar AUC value, or list if average=None.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if y_score.ndim != 2 or y_score.shape[1] != num_classes:
        raise ValueError(
            f"y_score shape must be [N, {num_classes}], got {y_score.shape}"
        )

    n_samples = y_true.shape[0]
    y_true_onehot = np.zeros((n_samples, num_classes), dtype=int)
    y_true_onehot[np.arange(n_samples), y_true] = 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            return roc_auc_score(
                y_true_onehot,
                y_score,
                average=average,
                multi_class="ovr",
            )
        except ValueError:
            return float("nan")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: if True, only count requires_grad=True params.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
