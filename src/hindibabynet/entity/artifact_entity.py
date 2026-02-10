from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    recordings_parquet_path: Path
    n_recordings: int
    n_sessions: int
    n_participants: int

@dataclass(frozen=True)
class AudioPreparationArtifact:
    analysis_wav_path: Path
    manifest_parquet_path: Path
    analysis_meta_json_path: Path
    duration_sec: float
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class SpeakerClassificationArtifact:
    segments_parquet_path: Path
    summary_json_path: Path
    textgrid_path: Path
    main_female_wav_path: Path
    main_male_wav_path: Path
    child_wav_path: Path
    background_wav_path: Path
    n_segments: int
    total_speech_sec: float
    class_durations: dict          # {"adult_male": 123.4, ...}
