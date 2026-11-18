from __future__ import annotations

from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pandas as pd
import pytest

from hindibabynet_pipeline.workflow.classification import run_classify


@pytest.fixture(params=[("vtc", "shared"), ("vtc", "wrapper"),
                        ("xgb", "shared"), ("xgb", "wrapper")])
def workflow(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
             tmp_path: Path) -> SimpleNamespace:
    """Use only synthetic files, mocked configuration, and mocked inference."""
    from hindibabynet_pipeline.components import speaker_classification
    from hindibabynet_pipeline.config import configuration
    from hindibabynet_pipeline.exception import exception
    from hindibabynet_pipeline.logging import logger as logging_module

    backend_name, entrypoint = request.param
    monkeypatch.chdir(tmp_path)
    processed = Path("processed")  # Deliberately relative: do not normalize config roots.
    processed.mkdir()
    cfg = Mock()
    cfg.make_run_id.return_value = "classification-run"
    cfg.get_logs_root.return_value = Path("logs")
    cfg.get_classification_output_root.return_value = Path("outputs")
    cfg.get_processed_audio_root.return_value = processed
    classifier = Mock()
    classifier.name = backend_name
    classifier.is_complete.return_value = False
    logger = Mock()
    factory = Mock(return_value=cfg)
    dispatch = Mock(return_value=classifier)
    get_logger = Mock(return_value=logger)
    add_handler = Mock()
    traceback = Mock(return_value="formatted traceback")
    monkeypatch.setattr(configuration, "ConfigurationManager", factory)
    monkeypatch.setattr(speaker_classification, "get_backend", dispatch)
    monkeypatch.setattr(logging_module, "get_logger", get_logger)
    monkeypatch.setattr(logging_module, "add_file_handler", add_handler)
    monkeypatch.setattr(exception, "format_traceback", traceback)

    events = Mock()
    for name, mock in [("get_logger", get_logger), ("ConfigurationManager", factory),
                       ("cfg", cfg), ("get_backend", dispatch),
                       ("add_file_handler", add_handler), ("classifier", classifier)]:
        events.attach_mock(mock, name)

    if entrypoint == "shared":
        run = partial(run_classify, backend=backend_name)
    else:
        module = import_module(
            f"hindibabynet_pipeline.workflow.{backend_name}_classification"
        )
        run = getattr(module, f"run_classify_{backend_name}")
    return SimpleNamespace(
        run=run, backend_name=backend_name, classifier=classifier, cfg=cfg,
        logger=logger, dispatch=dispatch, traceback=traceback, events=events,
        processed=processed, output_root=Path("outputs") / backend_name,
    )


def _wav(root: Path, pid: str) -> Path:
    path = root / pid / f"{pid}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic WAV placeholder")
    return path


@pytest.mark.parametrize("participant_id", [None, "", "explicit-id"])
def test_single_file_precedence_paths_and_setup(workflow: SimpleNamespace,
                                               participant_id: str | None) -> None:
    # No existence precheck, no manifest read, no analysis scan, and no limit slicing.
    wav = Path("input") / "missing.wav"
    pid = participant_id or "missing"
    workflow.classifier.name = "backend-output"
    out = Path("outputs") / "backend-output" / pid

    assert workflow.run(wav=wav, participant_id=participant_id, limit=0,
                        recordings_parquet="missing.parquet",
                        analysis_dir="missing-directory") is None

    assert workflow.events.mock_calls == [
        call.get_logger(
            f"hindibabynet_pipeline.workflow.{workflow.backend_name}_classification"
        ),
        call.ConfigurationManager(),
        call.cfg.make_run_id(),
        call.get_backend(workflow.cfg, override=workflow.backend_name),
        call.cfg.get_logs_root(),
        call.add_file_handler(
            workflow.logger,
            Path("logs") / "classification-run"
            / f"{workflow.backend_name}_classification_classification-run.log",
        ),
        call.cfg.get_classification_output_root(),
        call.classifier.is_complete(pid, out),
        call.classifier.run_participant(wav.resolve(), pid, out),
    ]
    workflow.logger.info.assert_not_called()
    workflow.logger.error.assert_not_called()
    # The workflow must not create output/provenance artifacts of its own.
    assert not Path("outputs").exists()


def test_single_file_skips_complete(workflow: SimpleNamespace) -> None:
    workflow.classifier.is_complete.return_value = True
    workflow.run(wav="missing.wav")
    workflow.classifier.is_complete.assert_called_once_with(
        "missing", workflow.output_root / "missing"
    )
    workflow.classifier.run_participant.assert_not_called()
    workflow.logger.info.assert_not_called()


@pytest.mark.parametrize("operation", ["is_complete", "run_participant"])
def test_single_file_errors_propagate(workflow: SimpleNamespace, operation: str) -> None:
    error = RuntimeError("single-file failure")
    getattr(workflow.classifier, operation).side_effect = error
    with pytest.raises(RuntimeError) as raised:
        workflow.run(wav="missing.wav")
    assert raised.value is error
    workflow.logger.error.assert_not_called()
    workflow.logger.info.assert_not_called()
    workflow.traceback.assert_not_called()


@pytest.mark.parametrize("limit", [None, 0, 2, -1, 99])
def test_manifest_selection_order_and_limit(workflow: SimpleNamespace,
                                           monkeypatch: pytest.MonkeyPatch,
                                           limit: int | None) -> None:
    for pid in ["C", "B", "A", "7", "unlisted"]:
        _wav(workflow.processed, pid)
    manifest = Path("relative") / "recordings.parquet"
    read = Mock(return_value=pd.DataFrame({
        "participant_id": ["C", None, "A", "B", "A", "0-missing", 7],
        "wav_path": ["ignored.wav"] * 7,
    }))
    monkeypatch.setattr(pd, "read_parquet", read)

    workflow.run(recordings_parquet=manifest, analysis_dir="nonexistent",
                 participant_id="ignored-in-batch", limit=limit)

    selected = ["7", "A", "B", "C"]
    if limit is not None:
        selected = selected[:limit]
    read.assert_called_once_with(str(manifest))
    workflow.cfg.get_processed_audio_root.assert_called_once_with()
    assert workflow.classifier.mock_calls == [
        entry
        for pid in selected
        for entry in [
            call.is_complete(pid, workflow.output_root / pid),
            call.run_participant(workflow.processed / pid / f"{pid}.wav",
                                 pid, workflow.output_root / pid),
        ]
    ]
    workflow.logger.info.assert_called_once_with(
        f"{workflow.backend_name.upper()} classification finished "
        f"| ok={len(selected)} skip=0 fail=0"
    )


@pytest.mark.parametrize("source", ["configured", "explicit", "empty-string"])
@pytest.mark.parametrize("limit", [None, 0, 2, -1])
def test_directory_selection_order_paths_and_limit(workflow: SimpleNamespace,
                                                  source: str,
                                                  limit: int | None) -> None:
    root = Path("analysis") if source == "explicit" else workflow.processed
    for pid in ["C", "B", "A"]:
        _wav(root, pid)
    (root / "0-missing").mkdir()
    (root / "0-missing" / "wrong.wav").write_bytes(b"")
    (root / "1-not-a-directory").write_bytes(b"")
    (root / "2-directory-wav" / "2-directory-wav.wav").mkdir(parents=True)
    analysis_dir = "analysis" if source == "explicit" else (
        "" if source == "empty-string" else None
    )

    workflow.run(analysis_dir=analysis_dir, limit=limit)

    selected = ["A", "B", "C"]
    if limit is not None:
        selected = selected[:limit]
    if source == "explicit":
        root = root.resolve()
        workflow.cfg.get_processed_audio_root.assert_not_called()
    else:
        workflow.cfg.get_processed_audio_root.assert_called_once_with()
    assert workflow.classifier.run_participant.call_args_list == [
        call(root / pid / f"{pid}.wav", pid, workflow.output_root / pid)
        for pid in selected
    ]
    assert workflow.classifier.is_complete.call_args_list == [
        call(pid, workflow.output_root / pid) for pid in selected
    ]
    workflow.logger.info.assert_called_once_with(
        f"{workflow.backend_name.upper()} classification finished "
        f"| ok={len(selected)} skip=0 fail=0"
    )


@pytest.mark.parametrize("source", ["manifest", "directory"])
def test_batch_skips_complete_logs_failure_and_continues(
    workflow: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    for pid in ["D", "C", "B", "A"]:
        _wav(workflow.processed, pid)
    monkeypatch.setattr(pd, "read_parquet", Mock(return_value=pd.DataFrame({
        "participant_id": ["D", "C", "B", "A"],
    })))
    workflow.classifier.is_complete.side_effect = [True, False, False]
    error = RuntimeError("inference failed")
    workflow.classifier.run_participant.side_effect = [error, {"status": "success"}]

    # Limit counts selected participants, including the complete and failed entries.
    manifest = "synthetic.parquet" if source == "manifest" else None
    assert workflow.run(recordings_parquet=manifest, limit=3) is None

    assert workflow.classifier.mock_calls == [
        call.is_complete("A", workflow.output_root / "A"),
        call.is_complete("B", workflow.output_root / "B"),
        call.run_participant(workflow.processed / "B" / "B.wav",
                             "B", workflow.output_root / "B"),
        call.is_complete("C", workflow.output_root / "C"),
        call.run_participant(workflow.processed / "C" / "C.wav",
                             "C", workflow.output_root / "C"),
    ]
    workflow.traceback.assert_called_once_with(error)
    workflow.logger.error.assert_called_once_with("FAIL | B\nformatted traceback")
    workflow.logger.info.assert_called_once_with(
        f"{workflow.backend_name.upper()} classification finished | ok=1 skip=1 fail=1"
    )
    assert not Path("outputs").exists()


@pytest.mark.parametrize("source", ["manifest", "directory"])
def test_empty_batch_reports_zero_counts(workflow: SimpleNamespace,
                                        monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    monkeypatch.setattr(pd, "read_parquet", Mock(return_value=pd.DataFrame({
        "participant_id": [None, None],
    })))
    manifest = "synthetic.parquet" if source == "manifest" else None
    assert workflow.run(recordings_parquet=manifest) is None
    workflow.classifier.is_complete.assert_not_called()
    workflow.classifier.run_participant.assert_not_called()
    workflow.logger.error.assert_not_called()
    workflow.logger.info.assert_called_once_with(
        f"{workflow.backend_name.upper()} classification finished | ok=0 skip=0 fail=0"
    )


def test_batch_completion_error_aborts_outside_handler(workflow: SimpleNamespace) -> None:
    for pid in ["B", "A"]:
        _wav(workflow.processed, pid)
    error = OSError("completion check failed")
    workflow.classifier.is_complete.side_effect = error

    with pytest.raises(OSError) as raised:
        workflow.run()

    assert raised.value is error
    workflow.classifier.is_complete.assert_called_once_with("A", workflow.output_root / "A")
    workflow.classifier.run_participant.assert_not_called()
    workflow.logger.error.assert_not_called()
    workflow.logger.info.assert_not_called()
    workflow.traceback.assert_not_called()


@pytest.mark.parametrize("source", ["manifest", "directory"])
def test_selection_errors_propagate(workflow: SimpleNamespace,
                                   monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    read = Mock(side_effect=FileNotFoundError("missing manifest"))
    monkeypatch.setattr(pd, "read_parquet", read)
    kwargs = ({"recordings_parquet": "missing.parquet"} if source == "manifest"
              else {"analysis_dir": "missing-directory"})
    with pytest.raises(FileNotFoundError):
        workflow.run(**kwargs)
    workflow.classifier.is_complete.assert_not_called()
    workflow.classifier.run_participant.assert_not_called()
    workflow.logger.error.assert_not_called()
    workflow.logger.info.assert_not_called()


def test_dispatch_failure_precedes_log_attachment(workflow: SimpleNamespace) -> None:
    error = RuntimeError("backend unavailable")
    workflow.dispatch.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        workflow.run(wav="missing.wav")
    assert raised.value is error
    workflow.cfg.get_logs_root.assert_not_called()
    workflow.cfg.get_classification_output_root.assert_not_called()
    workflow.classifier.is_complete.assert_not_called()


@pytest.mark.parametrize("backend_name", ["vtc", "xgb", "unsupported"])
@pytest.mark.parametrize("explicit", [False, True])
def test_pipeline_keeps_wrapper_imports_and_stage_order(monkeypatch: pytest.MonkeyPatch,
                                                       backend_name: str,
                                                       explicit: bool) -> None:
    from hindibabynet_pipeline.workflow import pipeline_runner

    cfg = Mock()
    cfg.make_run_id.return_value = "pipeline-run"
    cfg.get_logs_root.return_value = Path("logs")
    cfg.get_speaker_classification_backend.return_value = backend_name
    monkeypatch.setattr(pipeline_runner, "ConfigurationManager", Mock(return_value=cfg))
    monkeypatch.setattr(pipeline_runner, "add_file_handler", Mock())
    stages = Mock()
    stages.ingest.return_value = SimpleNamespace(recordings_parquet_path=Path("manifest.parquet"))
    monkeypatch.setattr(pipeline_runner, "run_data_ingestion", stages.ingest)
    monkeypatch.setattr(pipeline_runner, "run_prepare_audio", stages.prepare)
    monkeypatch.setattr(pipeline_runner, "run_classify_vtc", stages.vtc)
    monkeypatch.setattr(pipeline_runner, "run_classify_xgb", stages.xgb)

    selected = backend_name if explicit else None
    if backend_name == "unsupported":
        with pytest.raises(ValueError, match="^Unsupported backend: unsupported$"):
            pipeline_runner.run_pipeline(limit=2, backend=selected)
    else:
        assert pipeline_runner.run_pipeline(limit=2, backend=selected) is None

    expected = [
        call.ingest(run_id="pipeline-run"),
        call.prepare(recordings_parquet=Path("manifest.parquet"), limit=2,
                     run_id="pipeline-run"),
    ]
    if backend_name != "unsupported":
        expected.append(getattr(call, backend_name)(
            recordings_parquet=Path("manifest.parquet"), limit=2
        ))
    assert stages.mock_calls == expected
    if explicit:
        cfg.get_speaker_classification_backend.assert_not_called()
    else:
        cfg.get_speaker_classification_backend.assert_called_once_with()