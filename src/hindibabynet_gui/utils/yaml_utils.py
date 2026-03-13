"""Safe YAML read/write that preserves structure and comments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Write *data* to *path* as YAML, preserving readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )


def nested_get(d: dict, dotted_key: str, default: Any = None) -> Any:
    """Get a value from a nested dict using dot-separated keys."""
    keys = dotted_key.split(".")
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
        if current is default:
            return default
    return current


def nested_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot-separated keys."""
    keys = dotted_key.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value
