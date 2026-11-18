"""Validation workflow for configuration, RTTM files, mapping, and audio resolution."""

from __future__ import annotations

from dataclasses import dataclass

from hindibabynet_prosody.config import AppConfig
from hindibabynet_prosody.io.audio import AudioResolution, resolve_audio_for_source
from hindibabynet_prosody.io.participant_lookup import ParticipantLookupResult, load_participant_lookup
from hindibabynet_prosody.io.rttm import discover_rttm_files, normalize_segments, summarize_rttm_file


@dataclass(frozen=True)
class ValidationReport:
    lookup: ParticipantLookupResult
    rttm_files: list[str]
    audio_resolutions: list[AudioResolution]
    unmapped_sources: list[str]
    skipped_segment_issues: list[dict[str, object]]
    duplicate_participant_speaker_keys: list[str]


def run_validation(config: AppConfig) -> ValidationReport:
    lookup = load_participant_lookup(config)
    rttm_files = discover_rttm_files(config)
    if not rttm_files:
        raise FileNotFoundError(
            f"No RTTM files found under {config.paths.rttm_root} using {config.rttm.file_pattern}"
        )

    unmapped_sources: set[str] = set()
    skipped_segment_issues: list[dict[str, object]] = []
    duplicate_participant_speaker_keys: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_source_ids: set[str] = set()

    for rttm_path in rttm_files:
        summarize_rttm_file(config, rttm_path)
        segments, issues = normalize_segments(config, rttm_path, lookup.mapping)
        skipped_segment_issues.extend(issues)
        for segment in segments:
            seen_source_ids.add(segment.source_participant_id)
            if segment.participant_id is None:
                unmapped_sources.add(segment.source_participant_id)
            if segment.speaker is None:
                skipped_segment_issues.append(
                    {
                        "rttm_path": str(segment.rttm_path),
                        "source_participant_id": segment.source_participant_id,
                        "speaker": segment.speaker_label,
                        "start_time_s": segment.start_time_s,
                        "duration_s": segment.duration_s,
                        "failure_reason": "unsupported_speaker_label",
                    }
                )
                continue
            if segment.participant_id is not None:
                key = (segment.participant_id, segment.speaker)
                if key in seen_keys:
                    duplicate_participant_speaker_keys.append(f"{segment.participant_id}:{segment.speaker}")
                seen_keys.add(key)

    audio_resolutions = [resolve_audio_for_source(config, source_id) for source_id in sorted(seen_source_ids)]
    if config.mapping.strict and unmapped_sources:
        raise ValueError(
            "Unmapped source participant IDs found in RTTM files: "
            + ", ".join(sorted(unmapped_sources))
        )

    return ValidationReport(
        lookup=lookup,
        rttm_files=[str(path) for path in rttm_files],
        audio_resolutions=audio_resolutions,
        unmapped_sources=sorted(unmapped_sources),
        skipped_segment_issues=skipped_segment_issues,
        duplicate_participant_speaker_keys=sorted(set(duplicate_participant_speaker_keys)),
    )
