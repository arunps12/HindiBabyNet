"""Command builder: constructs CLI commands for each pipeline mode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _python_m(module: str, *args: str) -> list[str]:
    """Build a `python -m <module> ...` command list."""
    return ["python", "-m", module, *args]


def build_stage_01(run_id: Optional[str] = None) -> list[str]:
    """Stage 01: Data Ingestion."""
    cmd = _python_m("src.hindibabynet.pipeline.stage_01_data_ingestion")
    if run_id:
        cmd += ["--run_id", run_id]
    return cmd


def build_stage_02_from_parquet(
    recordings_parquet: str | Path,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Stage 02: Audio Preparation (batch from parquet)."""
    cmd = _python_m(
        "src.hindibabynet.pipeline.stage_02_audio_preparation_from_parquet",
        "--recordings_parquet", str(recordings_parquet),
    )
    if run_id:
        cmd += ["--run_id", run_id]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def build_stage_02_single_wav(
    wav_path: str | Path,
    recording_id: Optional[str] = None,
) -> list[str]:
    """Stage 02: Audio Preparation (single WAV)."""
    cmd = _python_m(
        "src.hindibabynet.pipeline.stage_02_audio_preparation_single_wav",
        "--wav", str(wav_path),
    )
    if recording_id:
        cmd += ["--recording_id", recording_id]
    return cmd


def build_stage_03(
    wav: Optional[str] = None,
    analysis_dir: Optional[str] = None,
    recordings_parquet: Optional[str] = None,
    run_id: Optional[str] = None,
    backend: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Stage 03: Speaker Classification (flexible dispatch)."""
    cmd = _python_m("src.hindibabynet.pipeline.stage_03_speaker_classification")
    if wav:
        cmd += ["--wav", str(wav)]
    elif analysis_dir:
        cmd += ["--analysis_dir", str(analysis_dir)]
    elif recordings_parquet:
        cmd += ["--recordings_parquet", str(recordings_parquet)]
    if run_id:
        cmd += ["--run_id", run_id]
    if backend:
        cmd += ["--backend", backend]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def build_full_pipeline(
    run_id: str,
    limit: Optional[int] = None,
) -> list[str]:
    """Full pipeline via run_all.sh (all stages)."""
    cmd = ["bash", "scripts/run_all.sh"]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def build_stage_03_vad(
    recordings_parquet: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Stage 03 (VAD only)."""
    cmd = _python_m("src.hindibabynet.pipeline.stage_03_vad")
    if recordings_parquet:
        cmd += ["--recordings_parquet", str(recordings_parquet)]
    if run_id:
        cmd += ["--run_id", run_id]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def build_stage_04_diarization(
    recordings_parquet: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Stage 04: Diarization."""
    cmd = _python_m("src.hindibabynet.pipeline.stage_04_diarization")
    if recordings_parquet:
        cmd += ["--recordings_parquet", str(recordings_parquet)]
    if run_id:
        cmd += ["--run_id", run_id]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def build_stage_05_intersection(
    recordings_parquet: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Stage 05: Intersection."""
    cmd = _python_m("src.hindibabynet.pipeline.stage_05_intersection")
    if recordings_parquet:
        cmd += ["--recordings_parquet", str(recordings_parquet)]
    if run_id:
        cmd += ["--run_id", run_id]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def build_stage_06_classification(
    recordings_parquet: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Stage 06: Speaker Classification (eGeMAPS + XGBoost export)."""
    cmd = _python_m("src.hindibabynet.pipeline.stage_06_speaker_classification")
    if recordings_parquet:
        cmd += ["--recordings_parquet", str(recordings_parquet)]
    if run_id:
        cmd += ["--run_id", run_id]
    if limit is not None and limit > 0:
        cmd += ["--limit", str(limit)]
    return cmd


def build_annotate(
    participant: Optional[str] = None,
    speaker: Optional[str] = None,
    resume: bool = False,
    export_only: bool = False,
    show_all: bool = False,
    show_status: bool = False,
) -> list[str]:
    """ADS/IDS annotation tool."""
    cmd = ["python", "scripts/annotate_ads_ids.py"]
    if show_status:
        cmd += ["--status"]
        return cmd
    if show_all:
        cmd += ["--all"]
        return cmd
    if participant:
        cmd += ["--participant", participant]
    if speaker:
        cmd += ["--speaker", speaker]
    if resume:
        cmd += ["--resume"]
    if export_only:
        cmd += ["--export-only"]
    return cmd
