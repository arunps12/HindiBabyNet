from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from hindibabynet_pipeline.entity.config_entity import DataIngestionConfig
from hindibabynet_pipeline.workflow.data_ingestion import DataIngestion, logger


def _write_wav(path: Path, sr: int = 8000, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.1, 0.1, int(sr * seconds), dtype=np.float32)
    sf.write(path, samples, sr, format="WAV", subtype="PCM_16")


def _make_config(tmp_path: Path, session_selection: str = "earliest") -> DataIngestionConfig:
    return DataIngestionConfig(
        raw_audio_root=tmp_path / "RawAudioData",
        allowed_ext=[".wav", ".WAV"],
        session_selection=session_selection,
        artifacts_dir=tmp_path / "artifacts" / "data_ingestion",
        recordings_parquet_path=tmp_path / "artifacts" / "data_ingestion" / "recordings.parquet",
    )


def test_data_ingestion_selects_earliest_session_only(tmp_path: Path):
    raw_root = tmp_path / "RawAudioData"
    _write_wav(raw_root / "P1" / "20250201" / "audio_1.wav")
    _write_wav(raw_root / "P1" / "20250201" / "audio_2.wav")
    _write_wav(raw_root / "P1" / "20250202" / "audio_3.wav")
    _write_wav(raw_root / "P1" / "20250202" / "audio_4.wav")
    _write_wav(raw_root / "P2" / "20250301" / "audio_5.wav")

    artifact = DataIngestion(_make_config(tmp_path)).initiate_data_ingestion()
    df = pd.read_parquet(artifact.recordings_parquet_path)

    p1_paths = sorted(Path(path).name for path in df.loc[df["participant_id"] == "P1", "path"])
    p1_dates = sorted(df.loc[df["participant_id"] == "P1", "session_date"].unique().tolist())

    assert artifact.n_recordings == 3
    assert p1_dates == ["20250201"]
    assert p1_paths == ["audio_1.wav", "audio_2.wav"]
    assert "audio_3.wav" not in p1_paths
    assert "audio_4.wav" not in p1_paths


def test_data_ingestion_all_session_selection_preserves_all_recordings(tmp_path: Path):
    raw_root = tmp_path / "RawAudioData"
    _write_wav(raw_root / "P1" / "20250201" / "audio_1.wav")
    _write_wav(raw_root / "P1" / "20250202" / "audio_2.wav")

    artifact = DataIngestion(_make_config(tmp_path, session_selection="all")).initiate_data_ingestion()
    df = pd.read_parquet(artifact.recordings_parquet_path)

    assert artifact.n_recordings == 2
    assert sorted(df["session_date"].unique().tolist()) == ["20250201", "20250202"]
    assert sorted(Path(path).name for path in df["path"].tolist()) == ["audio_1.wav", "audio_2.wav"]


def test_data_ingestion_warns_and_preserves_when_no_valid_session_folder_exists(
    tmp_path: Path
):
    raw_root = tmp_path / "RawAudioData"
    _write_wav(raw_root / "P1" / "session_a" / "audio_1.wav")
    _write_wav(raw_root / "P1" / "session_b" / "audio_2.wav")

    warning_stream = io.StringIO()
    warning_handler = logging.StreamHandler(warning_stream)
    warning_handler.setLevel(logging.WARNING)
    logger.addHandler(warning_handler)
    try:
        artifact = DataIngestion(_make_config(tmp_path)).initiate_data_ingestion()
    finally:
        logger.removeHandler(warning_handler)

    df = pd.read_parquet(artifact.recordings_parquet_path)

    assert artifact.n_recordings == 2
    assert sorted(Path(path).name for path in df["path"].tolist()) == ["audio_1.wav", "audio_2.wav"]
    assert "no valid YYYYMMDD session folders found" in warning_stream.getvalue()