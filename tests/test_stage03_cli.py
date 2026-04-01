from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_pipeline_stage03_wrapper_exports_valid_backends():
    from src.hindibabynet.pipeline.stage_03_speaker_classification import _VALID_BACKENDS

    assert "xgb" in _VALID_BACKENDS
    assert "vtc" in _VALID_BACKENDS


@patch("src.hindibabynet.cli.run_stage_03.ConfigurationManager")
@patch("src.hindibabynet.cli.run_stage_03.get_backend")
def test_stage03_main_config_driven_batch(mock_get_backend, mock_cfg, tmp_path: Path):
    from src.hindibabynet.cli.run_stage_03 import main

    processed = tmp_path / "processed"
    (processed / "P1").mkdir(parents=True)
    (processed / "P1" / "P1.wav").write_text("")

    cfg_obj = MagicMock()
    cfg_obj.make_run_id.return_value = "r1"
    cfg_obj.get_logs_root.return_value = tmp_path / "logs"
    cfg_obj.get_processed_audio_root.return_value = processed
    cfg_obj.get_classification_output_root.return_value = tmp_path / "outputs"
    mock_cfg.return_value = cfg_obj

    backend = MagicMock()
    backend.name = "vtc"
    backend.is_complete.return_value = False
    mock_get_backend.return_value = backend

    with patch("sys.argv", ["run_stage_03"]):
        main()

    backend.run_participant.assert_called_once()
