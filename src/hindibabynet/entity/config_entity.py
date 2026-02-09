from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_audio_root: Path
    allowed_ext: List[str]
    artifacts_dir: Path                # artifacts/runs/<run_id>/data_ingestion
    recordings_parquet_path: Path      # .../recordings.parquet
