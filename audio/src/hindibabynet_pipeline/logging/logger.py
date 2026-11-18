# Logging helpers for hindibabynet_pipeline.
from __future__ import annotations

import logging
import os
from pathlib import Path


_CONSOLE_FORMAT = "[%(levelname)s] %(message)s"
_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_SHARED_FILE_HANDLERS: dict[Path, logging.FileHandler] = {}


def get_logger(name: str = "hindibabynet_pipeline", level: int = logging.INFO) -> logging.Logger:
    """
    Get a console logger that doesn't add duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent double logging if called multiple times
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        logger.addHandler(sh)

    logger.propagate = False
    return logger


def add_file_handler(logger: logging.Logger, log_file: str | Path, level: int = logging.INFO) -> None:
    """
    Append audit records to the requested stage file by default.

    A nonempty HBN_LOG_FILE redirects all requests to one shared file handler.
    Repeated requests for the same resolved destination do not add duplicates.
    With a shared destination, the lowest requested handler level is retained;
    each logger still applies its own level.
    """
    shared_file = os.environ.get("HBN_LOG_FILE")
    log_file = Path(shared_file or log_file).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate file handlers for the same path
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file:
            if shared_file:
                h.setLevel(min(h.level, level))
            return

    if shared_file and log_file in _SHARED_FILE_HANDLERS:
        fh = _SHARED_FILE_HANDLERS[log_file]
        fh.setLevel(min(fh.level, level))
        logger.addHandler(fh)
        return

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(fh)
    if shared_file:
        _SHARED_FILE_HANDLERS[log_file] = fh
