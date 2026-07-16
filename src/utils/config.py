"""Configuration utilities: YAML loading, merging, saving.

Adapted from AW-DPCNN project conventions.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into base dict."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return as dict.

    Args:
        path: absolute or relative path to YAML file.

    Returns:
        Parsed configuration dict.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the YAML root is not a mapping.
    """
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {yaml_path} must be a YAML mapping.")
    return data


def load_config(
    base_config_path: str,
    exp_config_path: str = "",
) -> Dict[str, Any]:
    """Load and optionally merge configuration files.

    Args:
        base_config_path: path to the base YAML config.
        exp_config_path: optional path to an experiment-specific
                         override YAML config.

    Returns:
        Merged configuration dict.
    """
    base = load_yaml(base_config_path)
    if exp_config_path:
        exp = load_yaml(exp_config_path)
        return _deep_update(base, exp)
    return base


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    """Save a dict as a YAML file, creating parent directories as needed.

    Args:
        path: output file path.
        data: dict to serialize.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, sort_keys=False, allow_unicode=False)
