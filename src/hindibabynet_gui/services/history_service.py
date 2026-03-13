"""History service: persist and query GUI-triggered run history."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hindibabynet_gui.utils import GUI_DATA_DIR, ensure_dir

HISTORY_FILE = GUI_DATA_DIR / "gui_run_history.json"


def _load_history() -> list[dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(records: list[dict[str, Any]]) -> None:
    ensure_dir(GUI_DATA_DIR)
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2, default=str)


def add_record(
    command: str,
    mode: str,
    backend: str = "xgb",
    input_paths: Optional[list[str]] = None,
    participant_limit: Optional[int] = None,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create a new run history record (status=running). Returns the record."""
    record: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "mode": mode,
        "backend": backend,
        "input_paths": input_paths or [],
        "participant_limit": participant_limit,
        "run_id": run_id,
        "status": "running",
        "exit_code": None,
        "log_file": None,
    }
    records = _load_history()
    records.append(record)
    _save_history(records)
    return record


def update_last(status: str, exit_code: Optional[int] = None) -> None:
    """Update the last record with final status."""
    records = _load_history()
    if records:
        records[-1]["status"] = status
        records[-1]["exit_code"] = exit_code
        records[-1]["finished_at"] = datetime.now().isoformat()
        _save_history(records)


def get_history(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent *limit* run records, newest first."""
    records = _load_history()
    return list(reversed(records[-limit:]))
