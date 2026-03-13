"""Configuration service: load, edit, validate, and save config.yaml."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hindibabynet_gui.utils import CONFIG_PATH, resolve_path
from hindibabynet_gui.utils.yaml_utils import load_yaml, save_yaml, nested_get, nested_set


# ── Field metadata describing every editable config field ──────────────────

@dataclass(frozen=True)
class FieldMeta:
    key: str            # dot-separated YAML key
    label: str          # human-readable label
    tooltip: str        # help text
    kind: str           # "path" | "int" | "float" | "bool" | "str" | "choice" | "list"
    choices: list[str] = field(default_factory=list)
    group: str = ""     # section heading


FIELD_DEFS: list[FieldMeta] = [
    # ── General ──
    FieldMeta("artifacts_root", "Artifacts Root", "Root directory for pipeline run outputs", "path", group="General"),
    FieldMeta("logs_root", "Logs Root", "Root directory for log files", "path", group="General"),

    # ── Stage 01: Data Ingestion ──
    FieldMeta("data_ingestion.raw_audio_root", "Raw Audio Root",
              "Directory containing participant_id/date/*.WAV tree", "path", group="Stage 01: Data Ingestion"),
    FieldMeta("data_ingestion.allowed_ext", "Allowed Extensions",
              'File extensions to scan, e.g. [".wav", ".WAV"]', "list", group="Stage 01: Data Ingestion"),
    FieldMeta("data_ingestion.recordings_filename", "Recordings Filename",
              "Output parquet filename (recordings.parquet)", "str", group="Stage 01: Data Ingestion"),

    # ── Stage 02: Audio Preparation ──
    FieldMeta("audio_preparation.processed_audio_root", "Processed Audio Root",
              "Directory for pre-processed (mono, 16 kHz, normalised) WAVs", "path",
              group="Stage 02: Audio Preparation"),
    FieldMeta("audio_preparation.target_sr", "Target Sample Rate",
              "Resample to this sample rate (Hz)", "int", group="Stage 02: Audio Preparation"),
    FieldMeta("audio_preparation.to_mono", "Convert to Mono",
              "Mix stereo to mono", "bool", group="Stage 02: Audio Preparation"),
    FieldMeta("audio_preparation.target_peak_dbfs", "Target Peak (dBFS)",
              "Peak-normalise to this level", "float", group="Stage 02: Audio Preparation"),
    FieldMeta("audio_preparation.combine_gap_sec", "Combine Gap (sec)",
              "Silent gap inserted between concatenated WAVs", "float", group="Stage 02: Audio Preparation"),

    # ── Stage 03: Speaker Classification ──
    FieldMeta("speaker_classification.backend", "Classification Backend",
              "Speaker classification engine", "choice", choices=["xgb", "vtc"],
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.model_path", "XGBoost Model Path",
              "Path to serialised XGBoost model (.pkl)", "path",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.output_audio_root", "Output Audio Root",
              "Directory for classified speaker WAVs and TextGrids", "path",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.vad_aggressiveness", "VAD Aggressiveness",
              "WebRTC VAD aggressiveness (0–3, higher = more aggressive)", "int",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.vad_frame_ms", "VAD Frame (ms)",
              "WebRTC frame duration in ms (10, 20, or 30)", "int",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.vad_min_region_ms", "VAD Min Region (ms)",
              "Minimum voiced region duration in ms", "int",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.diarization_model", "Diarization Model",
              "Pyannote model name/path", "str",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.chunk_sec", "Chunk Duration (sec)",
              "Diarization chunk size in seconds", "float",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.overlap_sec", "Chunk Overlap (sec)",
              "Overlap between consecutive chunks in seconds", "float",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.min_speakers", "Min Speakers",
              "Minimum number of speakers for diarization", "int",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.max_speakers", "Max Speakers",
              "Maximum number of speakers for diarization", "int",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.merge_gap_sec", "Merge Gap (sec)",
              "Merge same-speaker segments separated by ≤ this gap", "float",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.min_segment_sec", "Min Segment (sec)",
              "Discard segments shorter than this (after merge)", "float",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.classify_win_sec", "Classify Window (sec)",
              "Feature-extraction window length", "float",
              group="Stage 03: Speaker Classification"),
    FieldMeta("speaker_classification.classify_hop_sec", "Classify Hop (sec)",
              "Feature-extraction hop length", "float",
              group="Stage 03: Speaker Classification"),

    # ── VTC ──
    FieldMeta("vtc.repo_path", "VTC Repo Path",
              "Path to cloned VTC 2.0 repository", "path", group="VTC Backend"),
    FieldMeta("vtc.device", "VTC Device",
              "Compute device for VTC inference", "choice", choices=["cuda", "cpu", "mps"],
              group="VTC Backend"),
    FieldMeta("vtc.output_root", "VTC Output Root",
              "Directory for VTC prediction outputs", "path", group="VTC Backend"),
    FieldMeta("vtc.input_root", "VTC Input Root",
              "Temporary directory for VTC input WAVs", "path", group="VTC Backend"),
    FieldMeta("vtc.keep_inputs", "VTC Keep Inputs",
              "Keep temporary input WAVs after VTC inference", "bool", group="VTC Backend"),
]


class ConfigService:
    """Load, mutate, validate, and persist config.yaml."""

    def __init__(self, config_path: Path | None = None):
        self.path = config_path or CONFIG_PATH
        self._data: dict[str, Any] = {}
        self._backup: dict[str, Any] = {}
        self.reload()

    # ── I/O ───────────────────────────────────────────────
    def reload(self) -> dict[str, Any]:
        self._data = load_yaml(self.path) if self.path.exists() else {}
        self._backup = copy.deepcopy(self._data)
        return self._data

    def save(self) -> None:
        save_yaml(self._data, self.path)
        self._backup = copy.deepcopy(self._data)

    def restore_backup(self) -> None:
        self._data = copy.deepcopy(self._backup)

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    # ── Field access ──────────────────────────────────────
    def get(self, dotted_key: str, default: Any = None) -> Any:
        return nested_get(self._data, dotted_key, default)

    def set(self, dotted_key: str, value: Any) -> None:
        nested_set(self._data, dotted_key, value)

    # ── Validation ────────────────────────────────────────
    def validate_paths(self) -> list[str]:
        """Return list of warning strings for paths that don't exist."""
        warnings: list[str] = []
        path_keys = [
            "data_ingestion.raw_audio_root",
            "speaker_classification.model_path",
        ]
        for key in path_keys:
            val = self.get(key)
            if val and not resolve_path(val).exists():
                warnings.append(f"{key}: path does not exist → {val}")
        return warnings

    @staticmethod
    def field_definitions() -> list[FieldMeta]:
        return FIELD_DEFS
