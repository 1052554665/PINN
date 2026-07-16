"""t-SNE feature visualization utility."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple:
    """Extract features and labels from a model's penultimate layer.

    For PINN models returning (logits, phys_field) tuples, the logits
    are used as the feature representation. For standard models, the
    raw output is used.

    Args:
        model: PyTorch model in eval mode.
        loader: DataLoader.
        device: torch device.

    Returns:
        (features, labels) tuple as numpy arrays.
    """
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)

            if isinstance(out, tuple):
                feat = out[0]  # logits as features
            else:
                feat = out

            features.append(feat.cpu().numpy())
            labels.extend(y.tolist())

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    save_path: str,
    perplexity: float = 30.0,
    figsize=(8, 6),
    title: str = "t-SNE Visualization",
    random_state: int = 42,
):
    """Compute t-SNE and save visualization.

    Args:
        features: feature matrix (N, D).
        labels: label array (N,).
        class_names: list of class name strings.
        save_path: path to save the figure.
        perplexity: t-SNE perplexity parameter.
        figsize: figure size tuple.
        title: plot title.
        random_state: random seed for reproducibility.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(features) - 1),
        random_state=random_state,
    )
    embedded = tsne.fit_transform(features)

    _, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)

    # Use a perceptually-uniform colormap
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=[cmap(i)],
            label=class_names[label] if label < len(class_names) else f"Class {label}",
            alpha=0.7,
            s=20,
        )

    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
