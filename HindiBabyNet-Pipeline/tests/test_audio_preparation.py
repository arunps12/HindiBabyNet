from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from hindibabynet_pipeline.workflow.audio_preparation import AudioPreparation
from hindibabynet_pipeline.config.configuration import ConfigurationManager
from hindibabynet_pipeline.entity.config_entity import (
    AudioPreparationConfig,
    DataIngestionConfig,
)
from hindibabynet_pipeline.workflow.data_ingestion import DataIngestion


def _write_wav(path: Path, sr: int = 8000, seconds: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.1, 0.1, int(sr * seconds), dtype=np.float32)
    sf.write(path, samples, sr, format="WAV", subtype="PCM_16")


def _make_audio_prep_config(tmp_path: Path, recording_id: str = "REC001", **overrides) -> AudioPreparationConfig:
    defaults = dict(
        artifacts_dir=tmp_path / "artifacts",
        processed_audio_root=tmp_path / "prepared",
        raw_joined_audio_root=tmp_path / "raw_joined",
        target_sr=16000,
        to_mono=True,
        target_peak_dbfs=-1.0,
        combine_gap_sec=0.0,
        join_multiple_files=True,
        resample=True,
        normalize=True,
        save_raw_joined_audio=True,
        save_prepared_audio=True,
        manifest_parquet_path=tmp_path / "artifacts" / f"{recording_id}_manifest.parquet",
        raw_joined_wav_path=tmp_path / "raw_joined" / recording_id / f"{recording_id}.wav",
        analysis_wav_path=tmp_path / "prepared" / recording_id / f"{recording_id}.wav",
        analysis_meta_json_path=tmp_path / "artifacts" / f"{recording_id}_meta.json",
    )
    defaults.update(overrides)
    return AudioPreparationConfig(**defaults)


def test_audio_preparation_config_parsing(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    params_file = tmp_path / "params.yaml"
    config_file.write_text(
        "\n".join(
            [
                "artifacts_root: artifacts/runs",
                "logs_root: logs",
                "paths:",
                "  raw_audio_root: /tmp/raw",
                "  raw_joined_audio_root: /tmp/raw_joined",
                "  prepared_audio_root: /tmp/prepared",
                "  classification_output_root: /tmp/classification_outputs",
                "  textgrid_output_root: /tmp/textgrids",
                "  manual_annotation_root: /tmp/manual_annotations",
                "  evaluation_output_root: /tmp/evaluation_outputs",
                "audio_preparation:",
                "  save_raw_joined_audio: false",
                "  save_prepared_audio: false",
            ]
        ),
        encoding="utf-8",
    )
    params_file.write_text(
        "\n".join(
            [
                "audio_preparation:",
                "  join_multiple_files: false",
                "  combine_gap_sec: 0.5",
                "  resample: false",
                "  target_sr: 22050",
                "  convert_to_mono: false",
                "  normalize: false",
                "  target_peak_dbfs: -3.0",
            ]
        ),
        encoding="utf-8",
    )

    cfg = ConfigurationManager(config_path=config_file, params_path=params_file)
    ap_cfg = cfg.get_audio_preparation_config(run_id="r1", recording_id="REC001")

    assert ap_cfg.raw_joined_wav_path == Path("/tmp/raw_joined") / "REC001" / "REC001.wav"
    assert ap_cfg.analysis_wav_path == Path("/tmp/prepared") / "REC001" / "REC001.wav"
    assert ap_cfg.join_multiple_files is False
    assert ap_cfg.resample is False
    assert ap_cfg.to_mono is False
    assert ap_cfg.normalize is False
    assert ap_cfg.save_raw_joined_audio is False
    assert ap_cfg.save_prepared_audio is False


def test_audio_preparation_single_wav_without_persisted_output(tmp_path: Path):
    wav_path = tmp_path / "input.wav"
    _write_wav(wav_path)
    cfg = _make_audio_prep_config(
        tmp_path,
        save_prepared_audio=False,
        save_raw_joined_audio=False,
        resample=False,
        to_mono=False,
        normalize=False,
    )

    artifact = AudioPreparation(cfg).run(wav_path=wav_path, recording_id="REC001")

    assert artifact.prepared_audio_saved is False
    assert artifact.analysis_wav_path.exists()
    assert artifact.analysis_wav_path != cfg.analysis_wav_path
    assert not cfg.analysis_wav_path.exists()
    assert artifact.raw_joined_wav_path is None


def test_audio_preparation_rejects_multi_file_dataframe_when_join_disabled(tmp_path: Path):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _write_wav(wav_a)
    _write_wav(wav_b)
    cfg = _make_audio_prep_config(tmp_path, join_multiple_files=False)
    df = pd.DataFrame(
        [
            {"participant_id": "P1", "recording_id": "R1", "path": str(wav_a)},
            {"participant_id": "P1", "recording_id": "R2", "path": str(wav_b)},
        ]
    )

    with pytest.raises(Exception):
        AudioPreparation(cfg).run(recordings_df=df, participant_id="P1", recording_id="P1")


def test_audio_preparation_combines_only_earliest_ingested_session(tmp_path: Path):
    raw_root = tmp_path / "RawAudioData"
    _write_wav(raw_root / "P1" / "20250201" / "audio_1.wav")
    _write_wav(raw_root / "P1" / "20250201" / "audio_2.wav")
    _write_wav(raw_root / "P1" / "20250202" / "audio_3.wav")
    _write_wav(raw_root / "P1" / "20250202" / "audio_4.wav")

    ingestion_cfg = DataIngestionConfig(
        raw_audio_root=raw_root,
        allowed_ext=[".wav", ".WAV"],
        session_selection="earliest",
        artifacts_dir=tmp_path / "artifacts" / "data_ingestion",
        recordings_parquet_path=tmp_path / "artifacts" / "data_ingestion" / "recordings.parquet",
    )
    ingestion_artifact = DataIngestion(ingestion_cfg).initiate_data_ingestion()
    recordings_df = pd.read_parquet(ingestion_artifact.recordings_parquet_path)

    prep_cfg = _make_audio_prep_config(
        tmp_path,
        recording_id="P1",
        resample=False,
        to_mono=False,
        normalize=False,
    )
    artifact = AudioPreparation(prep_cfg).run(
        recordings_df=recordings_df,
        participant_id="P1",
        recording_id="P1",
    )
    manifest_df = pd.read_parquet(artifact.manifest_parquet_path)
    combined_info = sf.info(str(artifact.raw_joined_wav_path))

    assert sorted(Path(path).name for path in manifest_df["source_path"].tolist()) == [
        "audio_1.wav",
        "audio_2.wav",
    ]
    assert combined_info.duration == pytest.approx(0.2, rel=0.05)