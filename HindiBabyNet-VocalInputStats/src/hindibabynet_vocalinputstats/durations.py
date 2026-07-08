"""Recording duration helpers based on full audio files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio_duration_seconds(path: str | Path) -> float:
    """Read full recording duration in seconds from an audio file."""
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(f"Invalid audio samplerate for {path}")
    return float(info.frames) / float(info.samplerate)


def seconds_to_hours(seconds: float | None) -> float:
    """Convert seconds to hours, preserving missing values."""
    if seconds is None or np.isnan(seconds):
        return np.nan
    return float(seconds) / 3600.0