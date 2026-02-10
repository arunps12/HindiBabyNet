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

@dataclass(frozen=True)
class AudioPreparationConfig:
    artifacts_dir: Path                  # artifacts/runs/<run_id>/audio_preparation
    processed_audio_root: Path           # scratch
    target_sr: int                       # 16000
    to_mono: bool                        # True
    target_peak_dbfs: float              # -1.0
    combine_gap_sec: float               # 0.0

    # outputs (paths)
    manifest_parquet_path: Path
    analysis_wav_path: Path              # final analysis-ready wav (mono, 16k, normalized)
    analysis_meta_json_path: Path