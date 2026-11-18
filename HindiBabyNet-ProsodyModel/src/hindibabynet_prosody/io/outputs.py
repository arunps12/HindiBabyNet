"""Output writing helpers for tables and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hindibabynet_prosody.utils.paths import ensure_directory


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    frame.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
