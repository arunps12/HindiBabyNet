"""Path resolution utilities for HindiBabyNet GUI."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root() -> Path:
    """Walk up from CWD or this file to find the project root (contains pyproject.toml)."""
    # Try CWD first
    candidate = Path.cwd()
    for _ in range(10):
        if (candidate / "pyproject.toml").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    # Fallback: relative to this file
    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / "pyproject.toml").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    # Last resort: CWD
    return Path.cwd()


PROJECT_ROOT = find_project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
GUI_DATA_DIR = ARTIFACTS_DIR / "gui_runs"


def resolve_path(p: str | Path) -> Path:
    """Resolve a path; if relative, resolve against PROJECT_ROOT."""
    path = Path(p)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist, return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p
