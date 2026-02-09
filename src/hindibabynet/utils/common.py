from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def make_run_id() -> str:
    """Readable + sortable run id."""
    return time.strftime("%Y%m%d_%H%M%S")
