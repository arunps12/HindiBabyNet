from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from hindibabynet_pipeline.logging import logger as logger_module
from hindibabynet_pipeline.logging.logger import add_file_handler, get_logger


@pytest.fixture
def loggers(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[logging.Logger]]:
    monkeypatch.delenv("HBN_LOG_FILE", raising=False)
    monkeypatch.setattr(logger_module, "_SHARED_FILE_HANDLERS", {})
    instances = [logging.getLogger(f"test_logging.{uuid4().hex}") for _ in range(2)]
    try:
        yield instances
    finally:
        handlers = {handler for logger in instances for handler in logger.handlers}
        for logger in instances:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        for handler in handlers:
            handler.close()


def test_console_is_simple_and_not_duplicated(
    loggers: list[logging.Logger], capsys: pytest.CaptureFixture[str]
) -> None:
    logger = loggers[0]
    assert get_logger(logger.name) is logger
    assert get_logger(logger.name) is logger

    logger.info("Preparing audio")

    assert len(logger.handlers) == 1
    assert not logger.propagate
    assert capsys.readouterr().err == "[INFO] Preparing audio\n"


def test_file_handler_does_not_prevent_console_handler(
    loggers: list[logging.Logger], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logger = loggers[0]
    destination = tmp_path / "stage.log"
    add_file_handler(logger, destination)
    get_logger(logger.name)
    get_logger(logger.name)

    logger.warning("QC warning")

    assert len(logger.handlers) == 2
    assert capsys.readouterr().err == "[WARNING] QC warning\n"
    assert destination.read_text(encoding="utf-8").count("QC warning") == 1


@pytest.mark.parametrize("override", [None, ""])
def test_default_preserves_separate_append_only_audit_files(
    loggers: list[logging.Logger], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, override: str | None,
) -> None:
    if override is not None:
        monkeypatch.setenv("HBN_LOG_FILE", override)
    logger = get_logger(loggers[0].name)
    first = tmp_path / "first.log"
    second = tmp_path / "nested" / "second.log"
    first.write_text("previous run\n", encoding="utf-8")
    add_file_handler(logger, first)
    add_file_handler(logger, second)

    logger.error("QC failed")

    contents = first.read_text(encoding="utf-8")
    assert contents.startswith("previous run\n")
    assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", contents)
    assert f"ERROR - {logger.name} - QC failed" in contents
    assert "QC failed" in second.read_text(encoding="utf-8")


def test_relative_and_absolute_file_requests_do_not_duplicate(
    loggers: list[logging.Logger], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    logger = get_logger(loggers[0].name)
    add_file_handler(logger, "stage.log")
    add_file_handler(logger, tmp_path / "stage.log")
    add_file_handler(logger, Path("nested") / ".." / "stage.log")

    logger.info("one record")

    assert len([h for h in logger.handlers if isinstance(h, logging.FileHandler)]) == 1
    assert (tmp_path / "stage.log").read_text(encoding="utf-8").count("one record") == 1


def test_override_shares_one_handler_and_preserves_audit_records(
    loggers: list[logging.Logger], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "runs" / "combined.log"
    destination.parent.mkdir()
    destination.write_text("previous run\n", encoding="utf-8")
    monkeypatch.setenv("HBN_LOG_FILE", str(destination))
    first, second = [get_logger(logger.name) for logger in loggers]
    stage_root = tmp_path / "unused_stage_logs"
    add_file_handler(first, stage_root / "prepare.log")
    add_file_handler(first, stage_root / "classify.log")
    add_file_handler(second, stage_root / "evaluate.log")
    add_file_handler(second, stage_root / "evaluate.log")

    first.info("Prepared audio")
    try:
        raise ValueError("synthetic QC failure")
    except ValueError:
        second.exception("Classification failed")

    first_files = [h for h in first.handlers if isinstance(h, logging.FileHandler)]
    second_files = [h for h in second.handlers if isinstance(h, logging.FileHandler)]
    assert len(first.handlers) == len(second.handlers) == 2
    assert len(first_files) == len(second_files) == 1
    assert first_files[0] is second_files[0]
    assert not stage_root.exists()
    contents = destination.read_text(encoding="utf-8")
    assert contents.startswith("previous run\n")
    assert contents.count(f"INFO - {first.name} - Prepared audio") == 1
    assert contents.count(f"ERROR - {second.name} - Classification failed") == 1
    assert "Traceback (most recent call last):" in contents
    assert "ValueError: synthetic QC failure" in contents
    console = capsys.readouterr().err
    assert console.count("[INFO] Prepared audio") == 1
    assert console.count("[ERROR] Classification failed") == 1


def test_relative_override_is_resolved_and_creates_only_shared_destination(
    loggers: list[logging.Logger], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HBN_LOG_FILE", "runs/combined.log")
    logger = get_logger(loggers[0].name)
    add_file_handler(logger, tmp_path / "unused" / "stage.log")
    monkeypatch.setenv("HBN_LOG_FILE", str(tmp_path / "runs" / "combined.log"))
    add_file_handler(logger, "another-stage.log")

    logger.info("shared record")

    assert len([h for h in logger.handlers if isinstance(h, logging.FileHandler)]) == 1
    assert (tmp_path / "runs" / "combined.log").read_text(encoding="utf-8").count("shared record") == 1
    assert not (tmp_path / "unused").exists()
    assert not (tmp_path / "another-stage.log").exists()


def test_shared_handler_keeps_lowest_requested_level(
    loggers: list[logging.Logger], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "combined.log"
    monkeypatch.setenv("HBN_LOG_FILE", str(destination))
    first, second = [get_logger(logger.name, logging.DEBUG) for logger in loggers]
    add_file_handler(first, "first.log", logging.WARNING)
    add_file_handler(second, "second.log", logging.DEBUG)
    add_file_handler(first, "first.log", logging.ERROR)

    first.debug("debug audit record")
    second.info("info audit record")

    contents = destination.read_text(encoding="utf-8")
    assert contents.count("debug audit record") == 1
    assert contents.count("info audit record") == 1