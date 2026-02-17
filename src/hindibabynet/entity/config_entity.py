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


# =========================================================================
# Stage 03: Voice Activity Detection
# =========================================================================
@dataclass(frozen=True)
class VADConfig:
    artifacts_dir: Path                  # artifacts/runs/<run_id>/vad

    # --- VAD params ---
    vad_aggressiveness: int              # 0-3  (webrtcvad)
    vad_frame_ms: int                    # 30
    vad_min_region_ms: int               # 300

    # --- outputs ---
    vad_parquet_path: Path               # <pid>_vad.parquet
    summary_json_path: Path              # <pid>_vad_summary.json


# =========================================================================
# Stage 04: Speaker Diarization
# =========================================================================
@dataclass(frozen=True)
class DiarizationConfig:
    artifacts_dir: Path                  # artifacts/runs/<run_id>/diarization

    # --- diarization params ---
    diarization_model: str               # "pyannote/speaker-diarization-3.1"
    chunk_sec: float                     # 900.0  (chunk length for diarization)
    overlap_sec: float                   # 10.0
    min_speakers: int                    # 2
    max_speakers: int                    # 4
    tmp_dir: Path                        # scratch tmp for chunk WAVs

    # --- outputs ---
    diarization_parquet_path: Path       # <pid>_diarization.parquet
    summary_json_path: Path              # <pid>_diarization_summary.json


# =========================================================================
# Stage 05: VAD ∩ Diarization Intersection
# =========================================================================
@dataclass(frozen=True)
class IntersectionConfig:
    artifacts_dir: Path                  # artifacts/runs/<run_id>/intersection

    # --- params ---
    min_segment_sec: float               # 0.2  (discard very short intersected segments)

    # --- inputs (filled at runtime from prior stages) ---
    # (these are passed as arguments, not config fields)

    # --- outputs ---
    speech_segments_parquet_path: Path   # <pid>_speech_segments.parquet
    summary_json_path: Path              # <pid>_intersection_summary.json


# =========================================================================
# Stage 06: Speaker-type Classification + Stream Export
# =========================================================================
@dataclass(frozen=True)
class SpeakerClassificationConfig:
    artifacts_dir: Path                  # artifacts/runs/<run_id>/speaker_classification

    # --- model ---
    model_path: Path                     # models/xgb_egemaps.pkl
    class_names: List[str]               # ["adult_male", "adult_female", "child", "background"]
    egemaps_dim: int                     # 88

    # --- merge & classify ---
    merge_gap_sec: float                 # 0.3  (merge adjacent same-speaker segments)
    min_segment_sec: float               # 0.2  (discard very short segments after merge)
    classify_win_sec: float              # 1.0  (eGeMAPS window)
    classify_hop_sec: float              # 0.5

    # --- secondary diarization (for main-speaker extraction) ---
    diarization_model: str               # "pyannote/speaker-diarization-3.1"
    min_speakers: int                    # 1
    max_speakers: int                    # 3

    # --- outputs ---
    output_audio_root: Path
    classified_segments_parquet_path: Path   # all classified segments
    main_female_parquet_path: Path           # female stream segment info
    main_male_parquet_path: Path             # male stream segment info
    child_parquet_path: Path                 # child stream segment info
    background_parquet_path: Path            # background stream segment info
    summary_json_path: Path                  # per-class summary stats
    textgrid_path: Path                      # <participant_id>.TextGrid
    main_female_wav_path: Path
    main_male_wav_path: Path
    child_wav_path: Path
    background_wav_path: Path