from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hindibabynet_pipeline.config.configuration import ConfigurationManager
from hindibabynet_pipeline.components.speaker_classification.dispatcher import get_backend
from hindibabynet_pipeline.components.speaker_classification.output_checks import (
    is_stage03_complete,
    is_vtc_complete,
    is_xgb_complete,
)


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "artifacts_root: artifacts/runs",
                "logs_root: logs",
                "data_ingestion:",
                "  raw_audio_root: /tmp/raw",
                "audio_preparation:",
                "  processed_audio_root: /tmp/processed",
                "speaker_classification:",
                "  backend: vtc",
                "  output_root: /tmp/classification_outputs",
                "  xgb:",
                "    model_path: models/xgb_egemaps.pkl",
                "    diarization_model: pyannote/speaker-diarization-3.1",
                "  vtc:",
                "    repo_path: external_models/VTC",
                "    device: cpu",
                "    keep_inputs: false",
            ]
        ),
        encoding="utf-8",
    )


def test_config_nested_vtc_parsing(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    _write_config(cfg_file)

    cfg = ConfigurationManager(config_path=cfg_file)
    assert cfg.get_speaker_classification_backend() == "vtc"
    assert cfg.get_classification_output_root() == Path("/tmp/classification_outputs")

    vtc_cfg = cfg.get_vtc_config()
    assert vtc_cfg.repo_path == Path("external_models/VTC")
    assert vtc_cfg.device == "cpu"
    assert vtc_cfg.output_root == Path("/tmp/classification_outputs") / "vtc"


def test_dispatcher_returns_vtc_backend(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    _write_config(cfg_file)

    cfg = ConfigurationManager(config_path=cfg_file)
    backend = get_backend(cfg)
    assert backend.name == "vtc"


def test_dispatcher_override_xgb(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    _write_config(cfg_file)

    cfg = ConfigurationManager(config_path=cfg_file)
    backend = get_backend(cfg, override="xgb")
    assert backend.name == "xgb"


def test_output_checks_xgb_and_vtc(tmp_path: Path):
    pid = "PID001"

    xgb_out = tmp_path / "xgb" / pid
    xgb_out.mkdir(parents=True)
    (xgb_out / f"{pid}_main_female.wav").write_text("")
    (xgb_out / f"{pid}_main_male.wav").write_text("")
    (xgb_out / f"{pid}_summary.json").write_text("{}")
    (xgb_out / f"{pid}.TextGrid").write_text("")
    assert is_xgb_complete(pid, xgb_out)
    assert is_stage03_complete(pid, "xgb", xgb_out)

    vtc_out = tmp_path / "vtc" / pid
    (vtc_out / "rttm").mkdir(parents=True)
    (vtc_out / "raw_rttm").mkdir(parents=True)
    (vtc_out / "rttm.csv").write_text("")
    (vtc_out / "raw_rttm.csv").write_text("")
    assert is_vtc_complete(pid, vtc_out)
    assert is_stage03_complete(pid, "vtc", vtc_out)


@patch("hindibabynet_pipeline.components.speaker_classification.vtc_backend.subprocess.run")
def test_vtc_backend_command_and_outputs(mock_run, tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    _write_config(cfg_file)
    cfg = ConfigurationManager(config_path=cfg_file)

    repo = tmp_path / "external_models" / "VTC"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "infer.py").write_text("# stub", encoding="utf-8")

    # Override repo path after config load for isolated temp workspace
    cfg.config["speaker_classification"]["vtc"]["repo_path"] = str(repo)

    wav = tmp_path / "processed" / "PID001" / "PID001.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"\x00" * 32)

    out_dir = tmp_path / "classification_outputs" / "vtc" / "PID001"
    (out_dir / "rttm").mkdir(parents=True)
    (out_dir / "raw_rttm").mkdir(parents=True)
    (out_dir / "rttm.csv").write_text("")
    (out_dir / "raw_rttm.csv").write_text("")

    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

    backend = get_backend(cfg, override="vtc")
    info = backend.run_participant(wav, "PID001", out_dir)

    assert info["backend"] == "vtc"
    assert info["status"] == "success"
    assert (out_dir / "run_info.json").exists()
    assert "scripts/infer.py" in info["command"]


@patch("hindibabynet_pipeline.cli.run_stage_03.get_backend")
def test_stage03_config_driven_defaults(mock_get_backend, tmp_path: Path):
    from hindibabynet_pipeline.cli.run_stage_03 import _discover_participants

    processed = tmp_path / "processed"
    (processed / "P1").mkdir(parents=True)
    (processed / "P1" / "P1.wav").write_text("")

    rows = _discover_participants(processed)
    assert rows == [("P1", processed / "P1" / "P1.wav")]


def test_config_dual_file_loading(tmp_path: Path):
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
                "data_ingestion:",
                "  allowed_ext: ['.wav']",
                "speaker_classification:",
                "  backend: vtc",
                "  vtc:",
                "    repo_path: external_models/VTC",
                "    device: cpu",
                "    keep_inputs: false",
            ]
        ),
        encoding="utf-8",
    )
    params_file.write_text(
        "\n".join(
            [
                "audio_preparation:",
                "  target_sr: 16000",
                "  convert_to_mono: true",
                "  target_peak_dbfs: -1.0",
                "  combine_gap_sec: 0.0",
                "xgb:",
                "  class_names: ['adult_male', 'adult_female', 'child', 'background']",
                "  egemaps_dim: 88",
                "  vad:",
                "    aggressiveness: 2",
                "    frame_ms: 30",
                "    min_region_ms: 300",
                "  diarization:",
                "    model: pyannote/speaker-diarization-3.1",
                "    chunk_sec: 900.0",
                "    overlap_sec: 10.0",
                "    min_speakers: 2",
                "    max_speakers: 4",
                "  segmentation:",
                "    merge_gap_sec: 0.3",
                "    min_segment_sec: 0.2",
                "  classification:",
                "    win_sec: 1.0",
                "    hop_sec: 0.5",
                "vtc:",
                "  batch_size: 32",
            ]
        ),
        encoding="utf-8",
    )

    cfg = ConfigurationManager(config_path=config_file, params_path=params_file)

    assert cfg.get_processed_audio_root() == Path("/tmp/prepared")
    assert cfg.get_raw_joined_audio_root() == Path("/tmp/raw_joined")
    assert cfg.get_textgrid_output_root() == Path("/tmp/textgrids")
    assert cfg.get_vtc_params()["batch_size"] == 32
    assert cfg.get_xgb_params()["vad_frame_ms"] == 30
