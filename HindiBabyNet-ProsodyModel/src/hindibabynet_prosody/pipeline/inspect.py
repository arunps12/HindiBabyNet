"""Small-scope inspection helpers for early pipeline verification."""

from __future__ import annotations

from dataclasses import dataclass

from hindibabynet_prosody.config import AppConfig
from hindibabynet_prosody.io.audio import resolve_audio_for_source
from hindibabynet_prosody.io.participant_lookup import load_participant_lookup
from hindibabynet_prosody.io.rttm import discover_rttm_files, normalize_segments


@dataclass(frozen=True)
class InspectReport:
    source_participant_id: str
    participant_id: str | None
    audio_path: str | None
    n_segments_total: int
    n_segments_included_speakers: int
    speaker_counts: dict[str, int]


def run_inspect(config: AppConfig, participant: str, max_segments: int) -> InspectReport:
    lookup = load_participant_lookup(config)
    target_rttm = None
    for rttm_path in discover_rttm_files(config):
        if rttm_path.parent.name == participant:
            target_rttm = rttm_path
            break
    if target_rttm is None:
        raise FileNotFoundError(f"No RTTM file found for participant {participant}")

    segments, _issues = normalize_segments(config, target_rttm, lookup.mapping)
    trimmed_segments = segments[:max_segments]
    source_participant_id = participant
    if trimmed_segments:
        source_participant_id = trimmed_segments[0].source_participant_id
    audio = resolve_audio_for_source(config, source_participant_id)

    speaker_counts: dict[str, int] = {}
    included = 0
    participant_id = trimmed_segments[0].participant_id if trimmed_segments else lookup.mapping.get(participant)
    for segment in trimmed_segments:
        speaker_counts[segment.speaker_label] = speaker_counts.get(segment.speaker_label, 0) + 1
        if segment.speaker is not None:
            included += 1

    return InspectReport(
        source_participant_id=source_participant_id,
        participant_id=participant_id,
        audio_path=str(audio.path) if audio.path is not None else None,
        n_segments_total=len(trimmed_segments),
        n_segments_included_speakers=included,
        speaker_counts=speaker_counts,
    )
