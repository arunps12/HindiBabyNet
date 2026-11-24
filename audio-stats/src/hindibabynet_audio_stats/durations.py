"""Recording duration helpers based on full audio files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioDurationResult:
    duration_sec: float
    matched_audio: bool
    warning_message: str | None = None
    selected_folder: str | None = None
    selected_date: str | None = None
    audio_file_count: int = 0
    ignored_later_folders: tuple[str, ...] = ()
    audit_reason: str | None = None


def _matches_extension(path: Path, audio_extensions: Sequence[str]) -> bool:
    normalized = {extension.lower() for extension in audio_extensions}
    return path.suffix.lower() in normalized


def _participant_folder_name(template: str, participant_source_id: str) -> str:
    return template.format(par_id=participant_source_id)


def _parse_session_name(session_name: str, session_date_format: str) -> datetime | None:
    try:
        return datetime.strptime(session_name, session_date_format)
    except ValueError:
        return None


def read_audio_duration_seconds(path: str | Path) -> float:
    """Read full recording duration in seconds from an audio file."""
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(f"Invalid audio samplerate for {path}")
    return float(info.frames) / float(info.samplerate)


def read_total_audio_duration_seconds(paths: Sequence[Path]) -> float:
    return float(sum(read_audio_duration_seconds(path) for path in paths))


def resolve_recording_duration(
    audio_root: str | Path,
    participant_source_id: str,
    audio_extensions: Sequence[str],
    *,
    participant_folder_name: str = "{par_id}",
    recursive: bool = True,
    prefer_largest_audio_file: bool = True,
    recording_duration_source: str = "audio_file",
    session_selection: str = "all",
    session_date_format: str = "%Y%m%d",
) -> AudioDurationResult:
    participant_root = Path(audio_root) / _participant_folder_name(participant_folder_name, participant_source_id)

    if recording_duration_source == "audio_file":
        candidates: list[Path] = []
        if participant_root.is_file() and _matches_extension(participant_root, audio_extensions):
            candidates = [participant_root]
        elif participant_root.is_dir():
            iterator = participant_root.rglob("*") if recursive else participant_root.glob("*")
            candidates = sorted(
                (candidate for candidate in iterator if candidate.is_file() and _matches_extension(candidate, audio_extensions)),
                key=lambda candidate: candidate.as_posix(),
            )
        else:
            parent = Path(audio_root)
            for extension in audio_extensions:
                direct_candidate = parent / f"{participant_source_id}{extension}"
                if direct_candidate.exists() and direct_candidate.is_file():
                    candidates.append(direct_candidate)
            candidates = sorted(candidates, key=lambda candidate: candidate.as_posix())

        if not candidates:
            return AudioDurationResult(duration_sec=np.nan, matched_audio=False, audit_reason="No matching audio file found.")
        if len(candidates) == 1:
            return AudioDurationResult(
                duration_sec=read_audio_duration_seconds(candidates[0]),
                matched_audio=True,
                selected_folder=candidates[0].parent.as_posix(),
                audio_file_count=1,
                audit_reason="Single audio file matched."
            )
        if prefer_largest_audio_file:
            selected = max(candidates, key=lambda candidate: (candidate.stat().st_size, candidate.as_posix()))
            return AudioDurationResult(
                duration_sec=read_audio_duration_seconds(selected),
                matched_audio=True,
                warning_message=f"Multiple audio files were found ({len(candidates)} matches); selected the largest file.",
                selected_folder=selected.parent.as_posix(),
                audio_file_count=1,
                ignored_later_folders=tuple(candidate.parent.as_posix() for candidate in candidates if candidate != selected),
                audit_reason="Selected the largest matching audio file."
            )
        return AudioDurationResult(
            duration_sec=read_audio_duration_seconds(candidates[0]),
            matched_audio=True,
            warning_message=f"Multiple audio files were found ({len(candidates)} matches); selected the first sorted file.",
            selected_folder=candidates[0].parent.as_posix(),
            audio_file_count=1,
            ignored_later_folders=tuple(candidate.parent.as_posix() for candidate in candidates[1:]),
            audit_reason="Selected the first sorted matching audio file."
        )

    if recording_duration_source != "raw_audio_sessions":
        raise ValueError(
            "recording_duration_source must be one of: audio_file, raw_audio_sessions"
        )
    if session_selection not in {"earliest", "all"}:
        raise ValueError("session_selection must be one of: earliest, all")

    if not participant_root.exists() or not participant_root.is_dir():
        return AudioDurationResult(duration_sec=np.nan, matched_audio=False, audit_reason="Participant audio folder is missing.")

    child_dirs = sorted((path for path in participant_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    valid_session_dirs = [
        session_dir
        for session_dir in child_dirs
        if _parse_session_name(session_dir.name, session_date_format) is not None
    ]

    if valid_session_dirs:
        selected_dirs = [valid_session_dirs[0]] if session_selection == "earliest" else valid_session_dirs
        warning_message = None
        ignored_later_folders = tuple(path.name for path in valid_session_dirs[1:]) if session_selection == "earliest" else ()
        selected_date = selected_dirs[0].name if session_selection == "earliest" else None
        audit_reason = "Selected the earliest valid session-date folder."
    else:
        selected_dirs = [participant_root]
        warning_message = (
            "No valid session-date folders were found; summed all participant audio files instead."
        )
        ignored_later_folders = ()
        selected_date = None
        audit_reason = warning_message

    audio_paths: list[Path] = []
    for selected_dir in selected_dirs:
        iterator = selected_dir.rglob("*") if recursive else selected_dir.glob("*")
        audio_paths.extend(
            candidate
            for candidate in iterator
            if candidate.is_file() and _matches_extension(candidate, audio_extensions)
        )

    unique_audio_paths = sorted(set(audio_paths), key=lambda candidate: candidate.as_posix())
    if not unique_audio_paths:
        return AudioDurationResult(
            duration_sec=np.nan,
            matched_audio=False,
            warning_message=warning_message,
            selected_folder=selected_dirs[0].as_posix() if selected_dirs else None,
            selected_date=selected_date,
            ignored_later_folders=ignored_later_folders,
            audit_reason="No audio files found in the selected folder(s)."
        )
    return AudioDurationResult(
        duration_sec=read_total_audio_duration_seconds(unique_audio_paths),
        matched_audio=True,
        warning_message=warning_message,
        selected_folder=selected_dirs[0].as_posix() if selected_dirs else None,
        selected_date=selected_date,
        audio_file_count=len(unique_audio_paths),
        ignored_later_folders=ignored_later_folders,
        audit_reason=audit_reason,
    )


def seconds_to_hours(seconds: float | None) -> float:
    """Convert seconds to hours, preserving missing values."""
    if seconds is None or np.isnan(seconds):
        return np.nan
    return float(seconds) / 3600.0