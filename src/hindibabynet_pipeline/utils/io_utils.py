from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def make_run_id() -> str:
    """Readable + sortable run id."""
    return time.strftime("%Y%m%d_%H%M%S")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, engine="pyarrow")


def write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
