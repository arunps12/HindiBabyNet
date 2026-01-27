# src/hindibabynet/logging/logger.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "hindibabynet", level: int = logging.INFO) -> logging.Logger:
    """
    Get a console logger that doesn't add duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent double logging if called multiple times
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(sh)

    logger.propagate = False
    return logger


def add_file_handler(logger: logging.Logger, log_file: str | Path, level: int = logging.INFO) -> None:
    """
    Add a file handler to an existing logger .
    Safe to call multiple times (won't duplicate).
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate file handlers for the same path
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_file:
            return

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(fh)
