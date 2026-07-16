"""Training workflow for PINN-based fault diagnosis.

Orchestrates the complete training pipeline:
    1. Dataset loading
    2. Model construction
    3. Training loop with physics-informed loss
    4. Evaluation & metrics
    5. Visualization (confusion matrix, t-SNE)
    6. Checkpointing
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.datasets import build_dataloaders
from src.models import build_model
from src.utils.metrics import count_parameters, compute_roc_auc
from src.utils.plot_confusion import plot_confusion
from src.utils.tsne import extract_features, plot_tsne
from src.utils.train_eval import evaluate, train_one_epoch


def _build_optimizer(model: nn.Module, train_cfg: Dict):
    """Build optimizer from config."""
    optimizer_name = str(train_cfg.get("optimizer", "sgd")).lower()
    lr = float(train_cfg.get("lr", 1e-2))
    wd = float(train_cfg.get("weight_decay", 1e-4))

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # Default: SGD with momentum
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=wd,
    )


def _build_scheduler(optimizer, scheduler_cfg: Dict):
    """Build learning rate scheduler from config.

    Returns (scheduler, kind) where kind is 'plateau' or 'epoch'.
    """
    if not scheduler_cfg:
        return None, None

    s_type = str(scheduler_cfg.get("type", "none")).lower()
    if s_type == "none":
        return None, None

    if s_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
        return scheduler, "plateau"

    if s_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 20)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
        return scheduler, "epoch"

    if s_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", 50)),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )
        return scheduler, "epoch"

    raise ValueError(f"Unsupported scheduler type: {s_type}")


def _build_loss(
    train_cfg: Dict,
    dataset,
    num_classes: int,
    device: torch.device,
):
    """Build classification loss with optional class weighting."""
    if not bool(train_cfg.get("class_weighting", False)):
        return nn.CrossEntropyLoss()

    targets = [label for _, label in dataset.samples]
    counter = Counter(targets)
    n_samples = len(targets)
    weights = [
        n_samples / (num_classes * max(counter[i], 1))
        for i in range(num_classes)
    ]
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=class_weights)


def train_and_evaluate(
    config: Dict,
    run_dir: Path,
    device: torch.device,
    seed: Optional[int] = None,
):
    """Complete training and evaluation workflow.

    Args:
        config: full configuration dict.
        run_dir: output directory for this run.
        device: torch device.
        seed: random seed.

    Returns:
        Dict of final metrics.
    """
    # --- Data ---
    loaders, class_names = build_dataloaders(config, device=device, seed=seed)
    num_classes = len(class_names)

    # --- Model ---
    use_pcnn = bool(config.get("model", {}).get("use_pcnn", True))
    model = build_model(config, use_pcnn=use_pcnn).to(device)

    # Performance tuning
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Model stats
    n_params = count_parameters(model, trainable_only=True)
    model_name = str(config.get("model", {}).get("name", "unknown"))
    print(f"[Model] {model_name} | Params: {n_params / 1e6:.2f}M")

    # --- Optimizer, scheduler, loss ---
    train_cfg = config.get("train", {})
    optimizer = _build_optimizer(model, train_cfg)
    scheduler, scheduler_kind = _build_scheduler(
        optimizer, config.get("scheduler", {})
    )
    criterion = _build_loss(
        train_cfg,
        loaders["train"].dataset,
        num_classes,
        device,
    )

    # --- Training config ---
    epochs = int(train_cfg.get("epochs", 20))
    lambda_phy = float(config.get("model", {}).get("lambda_phy", 0.1))

    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 10))
    es_metric = str(es_cfg.get("metric", "val_f1"))

    # --- Output paths ---
    log_path = run_dir / "logs" / "train_log.csv"
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    last_ckpt = run_dir / "checkpoints" / "last.pt"
    cm_path = run_dir / "figures" / "confusion_matrix.png"
    tsne_path = run_dir / "figures" / "tsne.png"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    es_counter = 0
    logs = []

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, lambda_phy
        )

        val_loss = 0.0
        val_f1 = 0.0
        if loaders["val"] is not None:
            val_loss, val_metrics, _, _, val_score = evaluate(
                model, loaders["val"], criterion, device, lambda_phy
            )
            _, _, _, val_f1, _, _, _ = val_metrics

        # Scheduler step
        if scheduler is not None:
            if scheduler_kind == "plateau":
                monitor = val_f1 if es_metric == "val_f1" else -val_loss
                scheduler.step(monitor)
            else:
                scheduler.step()

        # Logging
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_f1": val_f1,
        }
        logs.append(log_entry)
        print(
            f"[Epoch {epoch:3d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        # Early stopping
        if es_enabled:
            is_better = (
                val_loss < best_val_loss
                if es_metric == "val_loss"
                else val_f1 > best_val_f1
            )
            if is_better:
                best_val_f1 = max(val_f1, best_val_f1)
                best_val_loss = min(val_loss, best_val_loss)
                best_epoch = epoch
                es_counter = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                es_counter += 1
                if es_counter >= es_patience:
                    print(
                        f"[Early Stop] No improvement for {es_patience} "
                        f"epochs. Stopping at epoch {epoch}."
                    )
                    break
        else:
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), best_ckpt)

    # Save last checkpoint
    torch.save(model.state_dict(), last_ckpt)

    # --- Final evaluation on test set ---
    # Load best checkpoint
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    test_metrics = None
    if loaders["test"] is not None:
        test_loss, test_metrics, y_true, y_pred, y_score = evaluate(
            model, loaders["test"], criterion, device, lambda_phy
        )
        test_acc, prec, rec, f1, gmean, bal_acc, kappa = test_metrics
        test_auc = compute_roc_auc(y_true, y_score, num_classes)

        print(f"\n[Test] Acc={test_acc:.4f} F1={f1:.4f} AUC={test_auc:.4f}")

        # Confusion matrix
        plot_confusion(
            y_true, y_pred, class_names,
            save_path=str(cm_path),
            title=f"{model_name} Confusion Matrix",
        )

        # t-SNE
        t_features, t_labels = extract_features(model, loaders["test"], device)
        plot_tsne(
            t_features, t_labels, class_names,
            save_path=str(tsne_path),
            title=f"{model_name} t-SNE",
        )

    # --- Save training log ---
    import pandas as pd
    pd.DataFrame(logs).to_csv(log_path, index=False)

    print(f"\n[Done] Best epoch: {best_epoch}, Best val F1: {best_val_f1:.4f}")
    print(f"[Output] {run_dir}")

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "n_params": n_params,
    }
