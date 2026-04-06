"""Configuration loading and merging utilities."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Load YAML config with base config inheritance.

    If the config has a '_base_' key, loads and merges the base config first.
    """
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)

    if "_base_" in config:
        base_path = path.parent / config.pop("_base_")
        base_config = load_config(str(base_path))
        config = deep_merge(base_config, config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
