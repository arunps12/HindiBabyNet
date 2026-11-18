"""Core runtime schemas used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParticipantMapping:
    source_participant_id: str
    participant_id: str


@dataclass(frozen=True)
class NormalizedSegment:
    source_participant_id: str
    participant_id: str | None
    speaker_label: str
    speaker: str | None
    start_time_s: float
    duration_s: float
    end_time_s: float
    rttm_path: Path
