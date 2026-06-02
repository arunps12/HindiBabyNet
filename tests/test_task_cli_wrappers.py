from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


@patch("hindibabynet_pipeline.workflow.audio_preparation.subprocess.run")
def test_prepare_audio_batch_wrapper_builds_stage02_command(mock_run):
    from hindibabynet_pipeline.workflow.audio_preparation import run_prepare_audio

    mock_run.return_value = MagicMock(returncode=0)

    run_prepare_audio(recordings_parquet=Path("artifacts/runs/r1/data_ingestion/recordings.parquet"), limit=2, run_id="r1")

    cmd = mock_run.call_args.args[0]
    assert "hindibabynet_pipeline.pipeline.stage_02_audio_preparation_from_parquet" in cmd
    assert "--limit" in cmd
    assert "--run_id" in cmd


@patch("hindibabynet_pipeline.workflow.xgb_classification.subprocess.run")
def test_classify_xgb_wrapper_sets_backend(mock_run):
    from hindibabynet_pipeline.workflow.xgb_classification import run_classify_xgb

    mock_run.return_value = MagicMock(returncode=0)

    run_classify_xgb(recordings_parquet="recordings.parquet", limit=3)

    cmd = mock_run.call_args.args[0]
    assert cmd[:4] == [cmd[0], "-m", "hindibabynet_pipeline.cli.run_stage_03", "--backend"]
    assert "xgb" in cmd


@patch("hindibabynet_pipeline.workflow.vtc_classification.subprocess.run")
def test_classify_vtc_wrapper_sets_backend(mock_run):
    from hindibabynet_pipeline.workflow.vtc_classification import run_classify_vtc

    mock_run.return_value = MagicMock(returncode=0)

    run_classify_vtc(analysis_dir="prepared", limit=1)

    cmd = mock_run.call_args.args[0]
    assert "vtc" in cmd
    assert "--analysis_dir" in cmd