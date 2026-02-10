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
