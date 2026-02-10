from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, engine="pyarrow")


def write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
