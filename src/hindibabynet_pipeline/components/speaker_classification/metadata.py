"""Unified run-metadata helpers for Stage 03 backends."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_info(
    participant_id: str,
    backend: str,
    input_wav: Path,
    output_dir: Path,
    **extra: Any,
) -> dict[str, Any]:
    """Build the base ``run_info`` dict; callers add timing / status later."""
    return {
        "participant_id": participant_id,
        "backend": backend,
        "input_wav": str(input_wav),
        "output_dir": str(output_dir),
        **extra,
    }


def write_run_info(output_dir: Path, info: dict[str, Any]) -> Path:
    """Write ``run_info.json`` into *output_dir* and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path
