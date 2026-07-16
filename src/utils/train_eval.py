"""Training and evaluation loops for PINN models.

Handles:
    - Mixed-precision training (bfloat16/float16 autocast)
    - Multi-output model support (classification + physics)
    - Comprehensive evaluation with metrics
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np

from .metrics import compute_metrics


def _get_autocast_context(device: torch.device):
    """Return autocast context manager for mixed-precision training.

    Prefers bfloat16 when supported (Blackwell/Hopper/Ampere GPUs).
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _extract_logits(output):
    """Extract classification logits from model output.

    Models may return (logits, phys_field) tuples or single tensors.
    """
    if isinstance(output, tuple):
        # First tensor with 2D shape (B, num_classes) is the logits
        candidates = [
            t for t in output
            if isinstance(t, torch.Tensor) and t.ndim == 2
        ]
        if not candidates:
            raise ValueError(
                "Model returned tuple output but no logits tensor found."
            )
        return min(candidates, key=lambda t: t.shape[1])
    return output


def _get_phys_field(output):
    """Extract physical field from model output tuple."""
    if isinstance(output, tuple) and len(output) >= 2:
        return output[1]
    return None


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    lambda_phy: float = 0.1,
):
    """Train the model for one epoch.

    For PINN models that output (logits, phys_field), the loss is
    L_cls + λ_phy · L_laplacian. For standard models, only cross-entropy
    is used.

    Args:
        model: the PyTorch model.
        loader: training DataLoader.
        optimizer: PyTorch optimizer.
        criterion: loss function (may be unused for PINN models).
        device: torch device.
        lambda_phy: weight for physics regularization.

    Returns:
        Tuple of (avg_loss, train_accuracy).
    """
    from .laplacian import total_loss as pinn_loss_fn

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    autocast_ctx = _get_autocast_context(device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with autocast_ctx:
            out = model(x)

            # Check if model returns (logits, phys_field) tuple
            if isinstance(out, tuple) and len(out) == 2:
                pred_cls, phys_field = out
                loss, _, _ = pinn_loss_fn(pred_cls, y, phys_field, lambda_phy)
                logits = pred_cls
            else:
                logits = _extract_logits(out)
                loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total if total > 0 else 0.0
    return running_loss / len(loader), train_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    lambda_phy: float = 0.1,
):
    """Evaluate the model on a dataloader.

    Args:
        model: the PyTorch model.
        loader: evaluation DataLoader.
        criterion: loss function.
        device: torch device.
        lambda_phy: weight for physics regularization.

    Returns:
        Tuple of (avg_loss, metrics_tuple, y_true, y_pred, y_score).
        metrics_tuple = (acc, precision, recall, f1, gmean, bal_acc, kappa).
    """
    from .laplacian import total_loss as pinn_loss_fn

    model.eval()
    losses = []
    y_true, y_pred, y_score = [], [], []
    autocast_ctx = _get_autocast_context(device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with autocast_ctx:
            out = model(x)

            if isinstance(out, tuple) and len(out) == 2:
                pred_cls, phys_field = out
                loss, _, _ = pinn_loss_fn(pred_cls, y, phys_field, lambda_phy)
                logits = pred_cls
            else:
                logits = _extract_logits(out)
                loss = criterion(logits, y)

        losses.append(loss.item())
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_score.extend(probs.cpu().tolist())

    avg_loss = np.mean(losses) if losses else 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred, y_score
