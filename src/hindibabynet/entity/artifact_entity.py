from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    recordings_parquet_path: Path
    n_recordings: int
    n_sessions: int
    n_participants: int
