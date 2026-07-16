"""Experiment utilities: random seed, device selection."""

import random
import os

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed: integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Note: these settings prioritize reproducibility over speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config_device: str = "auto") -> torch.device:
    """Resolve the torch device from a config string.

    Args:
        config_device: one of 'auto', 'cuda', 'cpu', or 'cuda:N'.

    Returns:
        torch.device instance.
    """
    if config_device == "cpu":
        return torch.device("cpu")

    if config_device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(config_device)
        else:
            print(f"[WARN] {config_device} requested but CUDA unavailable. Using CPU.")
            return torch.device("cpu")

    # 'auto' mode
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
