"""Path helpers for repository-relative and Windows-friendly workflows."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()
