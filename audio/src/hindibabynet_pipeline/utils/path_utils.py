from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path = __file__) -> Path:
    path = Path(start).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from the given start path.")


def project_path(*parts: str) -> Path:
    return repo_root() .joinpath(*parts)