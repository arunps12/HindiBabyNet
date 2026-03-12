"""Tests for VTC backend integration."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.hindibabynet.entity.config_entity import VTCConfig
from src.hindibabynet.entity.artifact_entity import VTCInferenceArtifact


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def vtc_config(tmp_path: Path) -> VTCConfig:
    repo = tmp_path / "VTC"
    repo.mkdir()
    (repo / "scripts").mkdir()
    (repo / "scripts" / "infer.py").write_text("# stub")
    return VTCConfig(
        repo_path=repo,
        device="cpu",
        output_root=tmp_path / "vtc_outputs",
        input_root=tmp_path / "vtc_inputs",
        keep_inputs=False,
    )


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    wav = tmp_path / "audio" / "PID001" / "PID001.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"\x00" * 100)  # dummy file
    return wav


# =====================================================================
# Config parsing tests
# =====================================================================

class TestConfigParsing:
    def test_vtc_config_entity(self, vtc_config: VTCConfig):
        assert vtc_config.device == "cpu"
        assert vtc_config.keep_inputs is False
        assert vtc_config.repo_path.name == "VTC"

    def test_vtc_artifact_entity(self, tmp_path: Path):
        art = VTCInferenceArtifact(
            participant_id="PID001",
            output_dir=tmp_path / "out",
            rttm_dir=tmp_path / "out" / "rttm",
            raw_rttm_dir=tmp_path / "out" / "raw_rttm",
            rttm_csv_path=tmp_path / "out" / "rttm.csv",
            raw_rttm_csv_path=tmp_path / "out" / "raw_rttm.csv",
            run_info_json_path=tmp_path / "out" / "vtc_run_info.json",
            runtime_sec=12.3,
            status="success",
        )
        assert art.status == "success"
        assert art.participant_id == "PID001"

    def test_backend_selector_default(self, tmp_path: Path):
        """Default backend is xgb when not specified in config."""
        from src.hindibabynet.config.configuration import ConfigurationManager

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "artifacts_root: artifacts/runs\n"
            "speaker_classification:\n"
            "  model_path: models/xgb_egemaps.pkl\n"
            "  output_audio_root: /tmp/out\n"
        )
        cm = ConfigurationManager(config_path=cfg_file)
        assert cm.get_speaker_classification_backend() == "xgb"

    def test_backend_selector_vtc(self, tmp_path: Path):
        """Backend read from config when set to vtc."""
        from src.hindibabynet.config.configuration import ConfigurationManager

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "artifacts_root: artifacts/runs\n"
            "speaker_classification:\n"
            "  backend: vtc\n"
            "  model_path: models/xgb_egemaps.pkl\n"
            "  output_audio_root: /tmp/out\n"
            "vtc:\n"
            "  repo_path: external_models/VTC\n"
            "  device: cuda\n"
            "  output_root: /tmp/vtc_out\n"
            "  input_root: /tmp/vtc_in\n"
            "  keep_inputs: false\n"
        )
        cm = ConfigurationManager(config_path=cfg_file)
        assert cm.get_speaker_classification_backend() == "vtc"
        vtc = cm.get_vtc_config()
        assert vtc.device == "cuda"
        assert vtc.repo_path == Path("external_models/VTC")


# =====================================================================
# VTC runner tests
# =====================================================================

class TestVTCRunner:
    def test_command_construction(self, vtc_config: VTCConfig):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        runner = VTCInferenceRunner(vtc_config)
        cmd = runner._build_command(Path("/in"), Path("/out"))
        assert cmd == [
            "uv", "run", "scripts/infer.py",
            "--wavs", "/in",
            "--output", "/out",
            "--device", "cpu",
        ]

    def test_repo_validation_missing(self, tmp_path: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        cfg = VTCConfig(
            repo_path=tmp_path / "nonexistent",
            device="cpu",
            output_root=tmp_path / "out",
            input_root=tmp_path / "in",
            keep_inputs=False,
        )
        runner = VTCInferenceRunner(cfg)
        with pytest.raises(FileNotFoundError, match="VTC repo not found"):
            runner._validate_repo()

    def test_repo_validation_no_infer_script(self, tmp_path: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        repo = tmp_path / "VTC"
        repo.mkdir()
        cfg = VTCConfig(
            repo_path=repo,
            device="cpu",
            output_root=tmp_path / "out",
            input_root=tmp_path / "in",
            keep_inputs=False,
        )
        runner = VTCInferenceRunner(cfg)
        with pytest.raises(FileNotFoundError, match="infer script not found"):
            runner._validate_repo()

    def test_input_preparation(self, vtc_config: VTCConfig, sample_wav: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        runner = VTCInferenceRunner(vtc_config)
        input_dir = runner._prepare_input_folder(sample_wav, "PID001")
        assert input_dir == vtc_config.input_root / "PID001"
        assert (input_dir / "PID001.wav").exists()

    def test_output_verification_missing(self, vtc_config: VTCConfig, tmp_path: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        runner = VTCInferenceRunner(vtc_config)
        empty_dir = tmp_path / "empty_out"
        empty_dir.mkdir()
        result = runner._verify_outputs(empty_dir)
        assert all(v is False for v in result.values())

    def test_output_verification_present(self, vtc_config: VTCConfig, tmp_path: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        runner = VTCInferenceRunner(vtc_config)
        out_dir = tmp_path / "full_out"
        out_dir.mkdir()
        (out_dir / "rttm").mkdir()
        (out_dir / "raw_rttm").mkdir()
        (out_dir / "rttm.csv").write_text("")
        (out_dir / "raw_rttm.csv").write_text("")
        result = runner._verify_outputs(out_dir)
        assert all(v is True for v in result.values())

    @patch("src.hindibabynet.components.speaker_classification_vtc.subprocess.run")
    def test_run_success(self, mock_run, vtc_config: VTCConfig, sample_wav: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        # Mock successful subprocess
        mock_run.return_value = MagicMock(
            returncode=0, stdout="OK", stderr=""
        )

        runner = VTCInferenceRunner(vtc_config)

        # Pre-create expected VTC outputs in output folder
        out_dir = vtc_config.output_root / "PID001"
        out_dir.mkdir(parents=True)
        (out_dir / "rttm").mkdir()
        (out_dir / "raw_rttm").mkdir()
        (out_dir / "rttm.csv").write_text("")
        (out_dir / "raw_rttm.csv").write_text("")

        artifact = runner.run(wav_path=sample_wav, participant_id="PID001")

        assert artifact.status == "success"
        assert artifact.participant_id == "PID001"
        assert artifact.rttm_dir == out_dir / "rttm"
        mock_run.assert_called_once()

        # Check run_info.json was written
        assert artifact.run_info_json_path.exists()
        info = json.loads(artifact.run_info_json_path.read_text())
        assert info["backend"] == "vtc"
        assert info["status"] == "success"

    @patch("src.hindibabynet.components.speaker_classification_vtc.subprocess.run")
    def test_run_failure(self, mock_run, vtc_config: VTCConfig, sample_wav: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="CUDA OOM"
        )

        runner = VTCInferenceRunner(vtc_config)
        with pytest.raises(RuntimeError, match="VTC inference failed"):
            runner.run(wav_path=sample_wav, participant_id="PID001")

    def test_run_wav_not_found(self, vtc_config: VTCConfig, tmp_path: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        runner = VTCInferenceRunner(vtc_config)
        with pytest.raises(FileNotFoundError, match="Source WAV not found"):
            runner.run(wav_path=tmp_path / "nope.wav", participant_id="X")

    @patch("src.hindibabynet.components.speaker_classification_vtc.subprocess.run")
    def test_cleanup_input(self, mock_run, vtc_config: VTCConfig, sample_wav: Path):
        from src.hindibabynet.components.speaker_classification_vtc import VTCInferenceRunner

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        runner = VTCInferenceRunner(vtc_config)

        # Pre-create outputs
        out_dir = vtc_config.output_root / "PID001"
        out_dir.mkdir(parents=True)
        (out_dir / "rttm").mkdir()
        (out_dir / "raw_rttm").mkdir()
        (out_dir / "rttm.csv").write_text("")
        (out_dir / "raw_rttm.csv").write_text("")

        runner.run(wav_path=sample_wav, participant_id="PID001")

        # Input should be cleaned up (keep_inputs=False)
        assert not (vtc_config.input_root / "PID001").exists()


# =====================================================================
# Stage 03 dispatch tests
# =====================================================================

class TestStage03Dispatch:
    def test_dispatch_xgb(self, tmp_path: Path):
        """_run_single dispatches to XGB when backend=xgb."""
        from src.hindibabynet.pipeline.stage_03_speaker_classification import (
            _run_single_xgb,
        )

        # Just verify the function exists and is callable
        assert callable(_run_single_xgb)

    def test_dispatch_vtc(self, tmp_path: Path):
        """_run_single dispatches to VTC when backend=vtc."""
        from src.hindibabynet.pipeline.stage_03_speaker_classification import (
            _run_single_vtc,
        )

        assert callable(_run_single_vtc)

    def test_valid_backends_constant(self):
        from src.hindibabynet.pipeline.stage_03_speaker_classification import (
            _VALID_BACKENDS,
        )

        assert "xgb" in _VALID_BACKENDS
        assert "vtc" in _VALID_BACKENDS
