"""Read the settings of the audio workflow from params.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_params(path: str | Path | None = None) -> dict:
    """Return the settings as a dictionary. Default file: audio/params.yaml."""
    file = Path(path) if path else Path(__file__).resolve().parents[2] / "params.yaml"
    with file.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
