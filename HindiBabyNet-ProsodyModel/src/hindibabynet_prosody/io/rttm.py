"""RTTM CSV discovery and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hindibabynet_prosody.config import AppConfig
from hindibabynet_prosody.schemas import NormalizedSegment


@dataclass(frozen=True)
class RTTMFileSummary:
    path: Path
    source_participant_id: str
    n_rows: int


def discover_rttm_files(config: AppConfig) -> list[Path]:
    return sorted(config.paths.rttm_root.glob(config.rttm.file_pattern))


def load_rttm_dataframe(config: AppConfig, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        config.rttm.participant_column,
        config.rttm.start_column,
        config.rttm.duration_column,
        config.rttm.speaker_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"RTTM CSV missing columns at {path}: {sorted(missing)}")
    return frame


def summarize_rttm_file(config: AppConfig, path: Path) -> RTTMFileSummary:
    frame = load_rttm_dataframe(config, path)
    source_values = frame[config.rttm.participant_column].dropna().astype(str).str.strip().unique()
    source_participant_id = path.parent.name
    if len(source_values) == 1:
        source_participant_id = str(source_values[0])
    return RTTMFileSummary(path=path, source_participant_id=source_participant_id, n_rows=len(frame))


def normalize_segments(
    config: AppConfig,
    path: Path,
    source_to_participant: dict[str, str],
) -> tuple[list[NormalizedSegment], list[dict[str, object]]]:
    frame = load_rttm_dataframe(config, path)
    segments: list[NormalizedSegment] = []
    issues: list[dict[str, object]] = []

    participant_column = config.rttm.participant_column
    start_column = config.rttm.start_column
    duration_column = config.rttm.duration_column
    speaker_column = config.rttm.speaker_column

    for row in frame.itertuples(index=False):
        source_participant_id = str(getattr(row, participant_column)).strip()
        speaker_label = str(getattr(row, speaker_column)).strip()
        try:
            start_time_s = float(getattr(row, start_column))
            duration_s = float(getattr(row, duration_column))
        except (TypeError, ValueError) as exc:
            issues.append(
                {
                    "rttm_path": str(path),
                    "source_participant_id": source_participant_id,
                    "speaker": speaker_label,
                    "failure_reason": f"invalid_numeric_fields: {exc}",
                }
            )
            continue

        end_time_s = start_time_s + duration_s
        if start_time_s < 0 or duration_s <= 0:
            issues.append(
                {
                    "rttm_path": str(path),
                    "source_participant_id": source_participant_id,
                    "speaker": speaker_label,
                    "start_time_s": start_time_s,
                    "duration_s": duration_s,
                    "failure_reason": "invalid_segment_timing",
                }
            )
            continue

        participant_id = source_to_participant.get(source_participant_id)
        mapped_speaker = config.speakers.include.get(speaker_label)
        segments.append(
            NormalizedSegment(
                source_participant_id=source_participant_id,
                participant_id=participant_id,
                speaker_label=speaker_label,
                speaker=mapped_speaker,
                start_time_s=start_time_s,
                duration_s=duration_s,
                end_time_s=end_time_s,
                rttm_path=path,
            )
        )

    return segments, issues
