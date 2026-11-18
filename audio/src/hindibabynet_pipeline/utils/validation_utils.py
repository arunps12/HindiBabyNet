"""Lightweight path / config validation helpers."""
from __future__ import annotations

from pathlib import Path


def assert_file_exists(path: str | Path, label: str = "File") -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def assert_dir_exists(path: str | Path, label: str = "Directory") -> Path:
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"{label} not found: {p}")
    return p


def assert_wav(path: str | Path) -> Path:
    p = assert_file_exists(path, label="WAV")
    if p.suffix.lower() != ".wav":
        raise ValueError(f"Expected .wav file, got: {p.name}")
    return p
