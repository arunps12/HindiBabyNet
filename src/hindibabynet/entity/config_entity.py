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


@dataclass(frozen=True)
class SpeakerClassificationConfig:
    artifacts_dir: Path                  # artifacts/runs/<run_id>/speaker_classification

    # --- model ---
    model_path: Path                     # models/xgb_egemaps.pkl
    class_names: List[str]               # ["adult_male", "adult_female", "child", "background"]
    egemaps_dim: int                     # 88

    # --- VAD ---
    vad_aggressiveness: int              # 0-3  (webrtcvad)
    vad_frame_ms: int                    # 30
    vad_min_region_ms: int               # 300

    # --- diarization ---
    diarization_model: str               # "pyannote/speaker-diarization-3.1"
    chunk_sec: float                     # 900.0  (chunk length for diarization)
    overlap_sec: float                   # 10.0
    min_speakers: int                    # 2
    max_speakers: int                    # 4

    # --- merge & classify ---
    merge_gap_sec: float                 # 0.7  (merge adjacent same-speaker segments)
    min_segment_sec: float               # 0.2  (discard very short segments)
    classify_win_sec: float              # 1.0  (eGeMAPS window)
    classify_hop_sec: float              # 0.5

    # --- outputs ---
    output_audio_root: Path
    segments_parquet_path: Path          # all classified segments
    summary_json_path: Path              # per-class summary stats
    textgrid_path: Path                  # <participant_id>.TextGrid
    main_female_wav_path: Path
    main_male_wav_path: Path
    child_wav_path: Path
    background_wav_path: Path