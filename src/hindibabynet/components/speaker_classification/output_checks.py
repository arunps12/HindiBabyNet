"""Completion checks for Stage 03 outputs."""
from __future__ import annotations

from pathlib import Path


def is_xgb_complete(participant_id: str, output_dir: Path) -> bool:
    """XGB is complete when the four speaker WAVs, TextGrid, and summary exist."""
    pid = participant_id
    return (
        (output_dir / f"{pid}_main_female.wav").exists()
        and (output_dir / f"{pid}_main_male.wav").exists()
        and (output_dir / f"{pid}_summary.json").exists()
        and (output_dir / f"{pid}.TextGrid").exists()
    )


def is_vtc_complete(participant_id: str, output_dir: Path) -> bool:
    """VTC is complete when RTTM dirs and CSV files exist."""
    return (
        (output_dir / "rttm").is_dir()
        and (output_dir / "raw_rttm").is_dir()
        and (output_dir / "rttm.csv").is_file()
        and (output_dir / "raw_rttm.csv").is_file()
    )


def is_stage03_complete(
    participant_id: str,
    backend: str,
    output_dir: Path,
) -> bool:
    """
    Unified completion check.

    Parameters
    ----------
    participant_id : str
    backend : str
        ``'xgb'`` or ``'vtc'``.
    output_dir : Path
        The backend+participant output directory,
        e.g. ``<output_root>/xgb/<pid>`` or ``<output_root>/vtc/<pid>``.
    """
    if backend == "vtc":
        return is_vtc_complete(participant_id, output_dir)
    if backend == "xgb":
        return is_xgb_complete(participant_id, output_dir)
    return False
