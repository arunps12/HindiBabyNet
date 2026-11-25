"""Audio file resolution against source participant identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from hindibabynet_prosody.config import AppConfig


@dataclass(frozen=True)
class AudioResolution:
    source_participant_id: str
    path: Path | None
    duration_s: float | None
    issue: str | None


def resolve_audio_for_source(config: AppConfig, source_participant_id: str) -> AudioResolution:
    participant_dir = config.paths.audio_root / source_participant_id
    candidates: list[Path] = []
    for extension in config.audio.extensions:
        direct = participant_dir / f"{source_participant_id}{extension}"
        if direct.exists():
            candidates.append(direct)
    if not candidates and participant_dir.exists():
        for extension in config.audio.extensions:
            candidates.extend(sorted(participant_dir.glob(f"*{extension}")))
    if not candidates and config.paths.audio_root.is_dir():
        # The prepared file is saved directly in the audio folder: <audio_root>/<id>.wav
        extensions = {extension.lower() for extension in config.audio.extensions}
        candidates = sorted(
            path
            for path in config.paths.audio_root.iterdir()
            if path.is_file()
            and path.stem == source_participant_id
            and path.suffix.lower() in extensions
        )

    if not candidates:
        return AudioResolution(source_participant_id, None, None, "audio_file_missing")

    preferred = [path for path in candidates if path.suffix.lower() in config.audio.preferred_extensions]
    selected_pool = preferred or candidates
    unique_paths = list(dict.fromkeys(selected_pool))
    if len(unique_paths) > 1:
        return AudioResolution(source_participant_id, None, None, "ambiguous_audio_files")

    selected = unique_paths[0]
    try:
        info = sf.info(str(selected))
    except Exception as exc:  # pragma: no cover - depends on local audio backend/runtime
        return AudioResolution(
            source_participant_id=source_participant_id,
            path=selected,
            duration_s=None,
            issue=f"audio_unreadable: {type(exc).__name__}",
        )
    return AudioResolution(
        source_participant_id=source_participant_id,
        path=selected,
        duration_s=float(info.duration),
        issue=None,
    )
