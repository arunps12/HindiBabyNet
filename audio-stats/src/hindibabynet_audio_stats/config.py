"""Settings and data folders of the dataset preparation.

The settings are read from audio-stats/params.yaml. The data folders are read
from config.yaml in the project root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from hbn_paths import Config


# audio-stats/params.yaml
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[2] / "params.yaml"


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
    config_path: Path
    params_path: Path
    metadata_path: Path
    lookup_path: Path
    metadata_id_column: str
    vtc_output_root: Path
    audio_root: Path
    derived_data_dir: Path
    results_dir: Path
    audio_layout: AudioLayoutConfig
    vtc_layout: VtcLayoutConfig
    participant_id_digits: int
    age_month_denominator: float
    ses_source: str
    minimum_recording_hours_warning: float
    manual_mapping_csv: Path | None
    recording_duration_source: str
    session_selection: str
    session_date_format: str

    @property
    def audio_extensions(self) -> tuple[str, ...]:
        return self.audio_layout.audio_extensions


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in settings file: {path}")
    return loaded


def load_config(
    config_path: str | Path | None = None,
    params_path: str | Path | None = None,
) -> ProjectConfig:
    """Read the settings from params.yaml and the folders from the root config.yaml."""
    params_file = Path(params_path) if params_path is not None else DEFAULT_PARAMS_PATH
    params = _load_yaml(params_file)
    folders = Config(config_path)

    audio_layout = params.get("audio_layout") or {}
    vtc_layout = params.get("vtc_layout") or {}
    recording_duration_source = str(params.get("recording_duration_source", "audio_file")).strip().lower()
    manual_mapping_file = str(params.get("manual_mapping_file") or "").strip()

    # Raw recordings are used for the session durations, prepared audio otherwise.
    if recording_duration_source == "raw_audio_sessions":
        audio_root = folders.path("raw_audio")
    else:
        audio_root = folders.path("prepared_audio")

    # The audit files are saved next to the datasets.
    statistical_data_dir = folders.path("statistical_data")

    return ProjectConfig(
        config_path=folders.file,
        params_path=params_file,
        metadata_path=folders.path("metadata", str(params["metadata_file"])),
        lookup_path=folders.path("metadata", str(params["participant_lookup_file"])),
        metadata_id_column=str(params.get("metadata_id_column", "par_id")),
        vtc_output_root=folders.path("vtc"),
        audio_root=audio_root,
        derived_data_dir=statistical_data_dir,
        results_dir=statistical_data_dir,
        audio_layout=AudioLayoutConfig(
            type=str(audio_layout.get("type", "participant_folder")),
            participant_folder_name=str(audio_layout.get("participant_folder_name", "{par_id}")),
            recursive=bool(audio_layout.get("recursive", True)),
            audio_extensions=tuple(str(item) for item in audio_layout.get("audio_extensions", [])),
            expected_audio_files=str(audio_layout.get("expected_audio_files", "auto")),
            prefer_largest_audio_file=bool(audio_layout.get("prefer_largest_audio_file", True)),
        ),
        vtc_layout=VtcLayoutConfig(
            type=str(vtc_layout.get("type", "participant_folder")),
            participant_folder_name=str(vtc_layout.get("participant_folder_name", "{par_id}")),
            rttm_csv_name=str(vtc_layout.get("rttm_csv_name", "rttm.csv")),
        ),
        participant_id_digits=int(params["participant_id_digits"]),
        age_month_denominator=float(params["age_month_denominator"]),
        ses_source=str(params["ses_source"]),
        minimum_recording_hours_warning=float(params["minimum_recording_hours_warning"]),
        manual_mapping_csv=folders.path("metadata", manual_mapping_file) if manual_mapping_file else None,
        recording_duration_source=recording_duration_source,
        session_selection=str(params.get("session_selection", "all")).strip().lower(),
        session_date_format=str(params.get("session_date_format", "%Y%m%d")),
    )