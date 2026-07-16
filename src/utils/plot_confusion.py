"""Confusion matrix plotting utility."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str,
    normalize: bool = True,
    figsize=(8, 6),
    title: str = "Confusion Matrix",
):
    """Plot and save a confusion matrix.

    Args:
        y_true: ground-truth labels.
        y_pred: predicted labels.
        class_names: list of class name strings.
        save_path: path to save the figure.
        normalize: if True, normalize rows to sum to 1.
        figsize: figure size tuple.
        title: plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    _, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
