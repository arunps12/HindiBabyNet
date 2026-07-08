"""Configuration helpers for the HindiBabyNet vocal input statistics workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def get_repo_root() -> Path:
    """Return the repository root for the VocalInputStats project."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    repo_root: Path
    config_path: Path
    metadata_csv: Path
    vtc_output_root: Path
    audio_root: Path
    derived_data_dir: Path
    private_data_dir: Path
    figures_dir: Path
    tables_dir: Path
    results_dir: Path
    audio_extensions: tuple[str, ...]
    participant_id_digits: int
    age_month_denominator: float
    ses_source: str
    minimum_recording_hours_warning: float
    manual_mapping_csv: Path | None


def _resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return loaded


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load the repo config and resolve repo-relative paths."""
    repo_root = get_repo_root()
    path = Path(config_path) if config_path is not None else repo_root / "configs" / "config.yaml"
    if not path.is_absolute():
        path = (repo_root / path).resolve()

    payload = _load_yaml(path)
    return ProjectConfig(
        repo_root=repo_root,
        config_path=path,
        metadata_csv=_resolve_path(repo_root, str(payload["metadata_csv"])),
        vtc_output_root=_resolve_path(repo_root, str(payload["vtc_output_root"])),
        audio_root=_resolve_path(repo_root, str(payload["audio_root"])),
        derived_data_dir=_resolve_path(repo_root, str(payload["derived_data_dir"])),
        private_data_dir=_resolve_path(repo_root, str(payload["private_data_dir"])),
        figures_dir=_resolve_path(repo_root, str(payload["figures_dir"])),
        tables_dir=_resolve_path(repo_root, str(payload["tables_dir"])),
        results_dir=_resolve_path(repo_root, str(payload["results_dir"])),
        audio_extensions=tuple(str(value) for value in payload.get("audio_extensions", [])),
        participant_id_digits=int(payload["participant_id_digits"]),
        age_month_denominator=float(payload["age_month_denominator"]),
        ses_source=str(payload["ses_source"]),
        minimum_recording_hours_warning=float(payload["minimum_recording_hours_warning"]),
        manual_mapping_csv=_resolve_path(repo_root, payload.get("manual_mapping_csv")),
    )