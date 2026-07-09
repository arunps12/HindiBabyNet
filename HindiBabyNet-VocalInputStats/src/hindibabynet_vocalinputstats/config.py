"""Configuration helpers for the HindiBabyNet vocal input statistics workflow."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


def get_repo_root() -> Path:
    """Return the repository root for the VocalInputStats project."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AudioLayoutConfig:
    type: str
    participant_folder_name: str
    recursive: bool
    audio_extensions: tuple[str, ...]
    expected_audio_files: str
    prefer_largest_audio_file: bool


@dataclass(frozen=True)
class VtcLayoutConfig:
    type: str
    participant_folder_name: str
    rttm_csv_name: str


@dataclass(frozen=True)
class ProjectConfig:
    repo_root: Path
    config_path: Path
    metadata_path: Path
    metadata_id_column: str
    vtc_output_root: Path
    audio_root: Path
    derived_data_dir: Path
    private_data_dir: Path
    figures_dir: Path
    tables_dir: Path
    results_dir: Path
    audio_layout: AudioLayoutConfig
    vtc_layout: VtcLayoutConfig
    participant_id_digits: int
    age_month_denominator: float
    ses_source: str
    minimum_recording_hours_warning: float
    manual_mapping_csv: Path | None

    @property
    def metadata_csv(self) -> Path:
        return self.metadata_path

    @property
    def audio_extensions(self) -> tuple[str, ...]:
        return self.audio_layout.audio_extensions


def normalize_path(path_value: str | Path | None, repo_root: Path) -> Path | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path_value).strip())))
    if expanded.is_absolute():
        return expanded
    return repo_root / expanded


def _resolve_path(repo_root: Path, value: str | Path | None) -> Path | None:
    return normalize_path(value, repo_root)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return loaded


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load the repo config and resolve repo-relative paths."""
    repo_root = get_repo_root()
    path = normalize_path(config_path, repo_root) if config_path is not None else repo_root / "configs" / "config.yaml"
    if path is None:
        raise ValueError("Config path could not be resolved.")

    payload = _load_yaml(path)
    metadata_value = payload.get("metadata_path", payload.get("metadata_csv"))
    if metadata_value is None:
        raise KeyError("Config must include either 'metadata_path' or 'metadata_csv'.")

    audio_layout_payload = payload.get("audio_layout") or {}
    vtc_layout_payload = payload.get("vtc_layout") or {}
    audio_extensions = audio_layout_payload.get("audio_extensions", payload.get("audio_extensions", []))

    return ProjectConfig(
        repo_root=repo_root,
        config_path=path,
        metadata_path=_resolve_path(repo_root, metadata_value),
        metadata_id_column=str(payload.get("metadata_id_column", "par_id")),
        vtc_output_root=_resolve_path(repo_root, str(payload["vtc_output_root"])),
        audio_root=_resolve_path(repo_root, str(payload["audio_root"])),
        derived_data_dir=_resolve_path(repo_root, str(payload["derived_data_dir"])),
        private_data_dir=_resolve_path(repo_root, str(payload["private_data_dir"])),
        figures_dir=_resolve_path(repo_root, str(payload["figures_dir"])),
        tables_dir=_resolve_path(repo_root, str(payload["tables_dir"])),
        results_dir=_resolve_path(repo_root, str(payload["results_dir"])),
        audio_layout=AudioLayoutConfig(
            type=str(audio_layout_payload.get("type", "participant_folder")),
            participant_folder_name=str(audio_layout_payload.get("participant_folder_name", "{par_id}")),
            recursive=bool(audio_layout_payload.get("recursive", True)),
            audio_extensions=tuple(str(item) for item in audio_extensions),
            expected_audio_files=str(audio_layout_payload.get("expected_audio_files", "auto")),
            prefer_largest_audio_file=bool(audio_layout_payload.get("prefer_largest_audio_file", True)),
        ),
        vtc_layout=VtcLayoutConfig(
            type=str(vtc_layout_payload.get("type", "participant_folder")),
            participant_folder_name=str(vtc_layout_payload.get("participant_folder_name", "{par_id}")),
            rttm_csv_name=str(vtc_layout_payload.get("rttm_csv_name", "rttm.csv")),
        ),
        participant_id_digits=int(payload["participant_id_digits"]),
        age_month_denominator=float(payload["age_month_denominator"]),
        ses_source=str(payload["ses_source"]),
        minimum_recording_hours_warning=float(payload["minimum_recording_hours_warning"]),
        manual_mapping_csv=_resolve_path(repo_root, payload.get("manual_mapping_csv")),
    )