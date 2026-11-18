"""YAML-backed configuration models for the prosody pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectConfigSection(BaseModel):
    name: str
    random_seed: int = 42


class PathsConfig(BaseModel):
    rttm_root: Path
    audio_root: Path
    participant_lookup: Path
    output_dir: Path
    cache_dir: Path


class RTTMConfig(BaseModel):
    file_pattern: str = "**/rttm.csv"
    participant_column: str = "uid"
    start_column: str = "start_time_s"
    duration_column: str = "duration_s"
    speaker_column: str = "label"


class MappingConfig(BaseModel):
    strict: bool = True
    source_id_column: str = "original_par_id"
    participant_id_column: str = "participant_id"


class SpeakersConfig(BaseModel):
    include: dict[str, str]


class AudioConfig(BaseModel):
    extensions: list[str]
    preferred_extensions: list[str]
    target_sample_rate: int = 16000
    mono: bool = True
    end_tolerance_s: float = 0.05


class PitchSpeakerSettings(BaseModel):
    pitch_floor_hz: float
    pitch_ceiling_hz: float


class AdaptivePitchConfig(BaseModel):
    enabled: bool = True
    level: str = "segment"
    lower_multiplier: float = 0.5
    upper_multiplier: float = 2.0
    minimum_floor_hz: float = 40.0
    maximum_ceiling_hz: float = 1000.0
    fallback_to_first_pass: bool = True


class PraatConfig(BaseModel):
    voicing_threshold: float = 0.45
    octave_cost: float = 0.01
    octave_jump_cost: float = 0.35
    voiced_unvoiced_cost: float = 0.14


class PitchConfig(BaseModel):
    method: str = "praat_ac"
    speaker_settings: dict[str, PitchSpeakerSettings]
    default_speaker_settings: PitchSpeakerSettings | None = None
    time_step_s: float = 0.01
    lower_percentile: float = 5.0
    upper_percentile: float = 95.0
    minimum_voiced_frames: int = 3
    semitone_reference_hz: float = 1.0
    adaptive: AdaptivePitchConfig = Field(default_factory=AdaptivePitchConfig)
    praat: PraatConfig = Field(default_factory=PraatConfig)

    @model_validator(mode="after")
    def validate_pitch(self) -> "PitchConfig":
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("pitch.lower_percentile must be smaller than pitch.upper_percentile")
        if self.minimum_voiced_frames < 1:
            raise ValueError("pitch.minimum_voiced_frames must be at least 1")
        for label, settings in self.speaker_settings.items():
            if settings.pitch_floor_hz <= 0:
                raise ValueError(f"pitch.speaker_settings.{label}.pitch_floor_hz must be positive")
            if settings.pitch_ceiling_hz <= settings.pitch_floor_hz:
                raise ValueError(
                    f"pitch.speaker_settings.{label}.pitch_ceiling_hz must exceed pitch_floor_hz"
                )
        if self.default_speaker_settings is not None:
            if self.default_speaker_settings.pitch_floor_hz <= 0:
                raise ValueError("pitch.default_speaker_settings.pitch_floor_hz must be positive")
            if (
                self.default_speaker_settings.pitch_ceiling_hz
                <= self.default_speaker_settings.pitch_floor_hz
            ):
                raise ValueError(
                    "pitch.default_speaker_settings.pitch_ceiling_hz must exceed pitch_floor_hz"
                )
        if self.adaptive.lower_multiplier <= 0:
            raise ValueError("pitch.adaptive.lower_multiplier must be positive")
        if self.adaptive.upper_multiplier <= 1:
            raise ValueError("pitch.adaptive.upper_multiplier must be greater than 1")
        if self.adaptive.minimum_floor_hz <= 0:
            raise ValueError("pitch.adaptive.minimum_floor_hz must be positive")
        if self.adaptive.maximum_ceiling_hz <= self.adaptive.minimum_floor_hz:
            raise ValueError(
                "pitch.adaptive.maximum_ceiling_hz must exceed pitch.adaptive.minimum_floor_hz"
            )
        if self.adaptive.level not in {"segment", "participant_speaker"}:
            raise ValueError("pitch.adaptive.level must be 'segment' or 'participant_speaker'")
        return self


class SSLConfig(BaseModel):
    wav2vec2_model: str
    hubert_model: str
    wav2vec2_layer: int = -1
    hubert_layer: int = -1
    temporal_pooling: str = "mean"
    segment_pooling: str = "mean"
    minimum_segment_duration_s: float = 0.1
    maximum_segment_duration_s: float | None = None


class PCAConfig(BaseModel):
    n_components: int = 3
    standardize: bool = True
    random_seed: int = 42


class RuntimeConfig(BaseModel):
    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 0
    strict: bool = False
    log_level: str = "INFO"


class OutputConfig(BaseModel):
    final_filename: str = "participant_speaker_acoustic_features.csv"
    save_pca_artifacts: bool = True
    save_qc_reports: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectConfigSection
    paths: PathsConfig
    rttm: RTTMConfig
    mapping: MappingConfig
    speakers: SpeakersConfig
    audio: AudioConfig
    pitch: PitchConfig
    ssl: SSLConfig
    pca: PCAConfig
    runtime: RuntimeConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_included_speakers(self) -> "AppConfig":
        missing = sorted(set(self.speakers.include) - set(self.pitch.speaker_settings))
        if missing:
            raise ValueError(
                "Missing pitch speaker settings for included labels: " + ", ".join(missing)
            )
        return self


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load native YAML; only shared 'prosody' sections rebase filesystem paths."""
    if config_path is None:
        config_path = os.environ.get("HBN_CONFIG_PATH")
        if not config_path:
            raise ValueError("A config path or HBN_CONFIG_PATH is required.")
    resolved_path = Path(config_path).resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        raise TypeError(f"Expected a top-level mapping in {resolved_path}")
    if "prosody" in raw_config:
        section = raw_config["prosody"]
        if not isinstance(section, dict):
            raise TypeError(f"Expected a mapping in section 'prosody' of {resolved_path}")
        raw_config = dict(section)
        project_root = raw_config.pop("project_root", None)
        repo_root = (
            (resolved_path.parent / project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        if isinstance(raw_config.get("paths"), dict):
            raw_config["paths"] = {
                key: (repo_root / value).resolve() if isinstance(value, (str, Path)) else value
                for key, value in raw_config["paths"].items()
            }
    return AppConfig.model_validate(raw_config)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return config.model_dump(mode="python")
