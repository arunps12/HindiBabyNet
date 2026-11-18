"""Speaker-specific and adaptive two-pass pitch extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hindibabynet_prosody.config import PitchConfig, PitchSpeakerSettings

try:
    import parselmouth
except ImportError:  # pragma: no cover
    parselmouth = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PitchStatistics:
    mean_f0_hz: float
    median_f0_hz: float
    sd_f0_hz: float
    p05_f0_hz: float
    p95_f0_hz: float
    robust_pitch_range_hz: float
    mean_f0_st: float
    median_f0_st: float
    sd_f0_st: float
    p05_f0_st: float
    p95_f0_st: float
    robust_pitch_range_st: float


@dataclass(frozen=True)
class SegmentPitchResult:
    statistics: PitchStatistics | None
    first_pass_floor_hz: float | None
    first_pass_ceiling_hz: float | None
    first_pass_voiced_frames: int
    first_pass_median_f0_hz: float | None
    adaptive_floor_hz: float | None
    adaptive_ceiling_hz: float | None
    second_pass_attempted: bool
    second_pass_voiced_frames: int
    final_pass_used: str
    final_voiced_frames: int
    voiced_proportion: float
    pitch_extraction_success: bool
    fallback_used: bool
    failure_reason: str | None
    warning_reason: str | None


@dataclass(frozen=True)
class FirstPassPitchResult:
    speaker_settings: PitchSpeakerSettings | None
    voiced_values_hz: np.ndarray
    voiced_frames: int
    median_f0_hz: float | None
    success: bool
    failure_reason: str | None


def select_speaker_settings(
    pitch_config: PitchConfig,
    speaker_label: str,
    default_range: PitchSpeakerSettings | None = None,
) -> PitchSpeakerSettings:
    settings = pitch_config.speaker_settings.get(speaker_label)
    if settings is None:
        if default_range is not None:
            return default_range
        raise ValueError(f"Unsupported speaker label for pitch extraction: {speaker_label}")
    return settings


def hz_to_semitones(values_hz: np.ndarray, reference_hz: float) -> np.ndarray:
    return 12.0 * np.log2(values_hz / reference_hz)


def compute_adaptive_range(
    speaker_settings: PitchSpeakerSettings,
    median_f0_hz: float,
    pitch_config: PitchConfig,
) -> tuple[float, float]:
    adaptive = pitch_config.adaptive
    adaptive_floor_hz = max(
        speaker_settings.pitch_floor_hz,
        median_f0_hz * adaptive.lower_multiplier,
        adaptive.minimum_floor_hz,
    )
    adaptive_ceiling_hz = min(
        speaker_settings.pitch_ceiling_hz,
        median_f0_hz * adaptive.upper_multiplier,
        adaptive.maximum_ceiling_hz,
    )
    return float(adaptive_floor_hz), float(adaptive_ceiling_hz)


def summarize_pitch_values(values_hz: np.ndarray, reference_hz: float, lower: float, upper: float) -> PitchStatistics:
    semitones = hz_to_semitones(values_hz, reference_hz)
    p05_hz = float(np.percentile(values_hz, lower))
    p95_hz = float(np.percentile(values_hz, upper))
    p05_st = float(np.percentile(semitones, lower))
    p95_st = float(np.percentile(semitones, upper))
    return PitchStatistics(
        mean_f0_hz=float(np.mean(values_hz)),
        median_f0_hz=float(np.median(values_hz)),
        sd_f0_hz=float(np.std(values_hz, ddof=0)),
        p05_f0_hz=p05_hz,
        p95_f0_hz=p95_hz,
        robust_pitch_range_hz=float(p95_hz - p05_hz),
        mean_f0_st=float(np.mean(semitones)),
        median_f0_st=float(np.median(semitones)),
        sd_f0_st=float(np.std(semitones, ddof=0)),
        p05_f0_st=p05_st,
        p95_f0_st=p95_st,
        robust_pitch_range_st=float(p95_st - p05_st),
    )


def _extract_voiced_values(sound: Any, pitch_config: PitchConfig, floor_hz: float, ceiling_hz: float) -> np.ndarray:
    if parselmouth is None:
        raise RuntimeError("praat-parselmouth is required for pitch extraction")
    pitch = sound.to_pitch_ac(
        time_step=pitch_config.time_step_s,
        pitch_floor=floor_hz,
        pitch_ceiling=ceiling_hz,
        voicing_threshold=pitch_config.praat.voicing_threshold,
        octave_cost=pitch_config.praat.octave_cost,
        octave_jump_cost=pitch_config.praat.octave_jump_cost,
        voiced_unvoiced_cost=pitch_config.praat.voiced_unvoiced_cost,
    )
    values = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    return values[np.isfinite(values) & (values > 0.0)]


def extract_first_pass_pitch(
    sound: Any,
    speaker_label: str,
    pitch_config: PitchConfig,
    *,
    default_range: PitchSpeakerSettings | None = None,
) -> FirstPassPitchResult:
    try:
        speaker_settings = select_speaker_settings(pitch_config, speaker_label, default_range)
    except ValueError as exc:
        return FirstPassPitchResult(
            speaker_settings=None,
            voiced_values_hz=np.asarray([], dtype=np.float64),
            voiced_frames=0,
            median_f0_hz=None,
            success=False,
            failure_reason=str(exc),
        )

    voiced_values_hz = _extract_voiced_values(
        sound,
        pitch_config,
        speaker_settings.pitch_floor_hz,
        speaker_settings.pitch_ceiling_hz,
    )
    voiced_frames = int(voiced_values_hz.size)
    median_f0_hz = float(np.median(voiced_values_hz)) if voiced_frames else None
    success = voiced_frames >= pitch_config.minimum_voiced_frames
    failure_reason = None if success else "insufficient_first_pass_voiced_frames"
    return FirstPassPitchResult(
        speaker_settings=speaker_settings,
        voiced_values_hz=voiced_values_hz,
        voiced_frames=voiced_frames,
        median_f0_hz=median_f0_hz,
        success=success,
        failure_reason=failure_reason,
    )


def summarize_participant_speaker_adaptive_range(
    first_pass_results: list[FirstPassPitchResult],
    pitch_config: PitchConfig,
) -> tuple[float, float] | None:
    valid_results = [result for result in first_pass_results if result.success and result.median_f0_hz is not None and result.speaker_settings is not None]
    if not valid_results:
        return None
    group_median_f0_hz = float(np.median(np.asarray([result.median_f0_hz for result in valid_results], dtype=np.float64)))
    reference_settings = valid_results[0].speaker_settings
    adaptive_floor_hz, adaptive_ceiling_hz = compute_adaptive_range(
        reference_settings,
        group_median_f0_hz,
        pitch_config,
    )
    if adaptive_floor_hz >= adaptive_ceiling_hz:
        return None
    return adaptive_floor_hz, adaptive_ceiling_hz


def build_segment_pitch_result(
    *,
    first_pass_result: FirstPassPitchResult,
    pitch_config: PitchConfig,
    final_values_hz: np.ndarray | None,
    adaptive_floor_hz: float | None,
    adaptive_ceiling_hz: float | None,
    second_pass_attempted: bool,
    second_pass_voiced_frames: int,
    final_pass_used: str,
    fallback_used: bool,
    failure_reason: str | None,
    warning_reason: str | None,
) -> SegmentPitchResult:
    frame_total = max(first_pass_result.voiced_frames, 1)
    if final_values_hz is None:
        return SegmentPitchResult(
            statistics=None,
            first_pass_floor_hz=(first_pass_result.speaker_settings.pitch_floor_hz if first_pass_result.speaker_settings is not None else None),
            first_pass_ceiling_hz=(first_pass_result.speaker_settings.pitch_ceiling_hz if first_pass_result.speaker_settings is not None else None),
            first_pass_voiced_frames=first_pass_result.voiced_frames,
            first_pass_median_f0_hz=first_pass_result.median_f0_hz,
            adaptive_floor_hz=adaptive_floor_hz,
            adaptive_ceiling_hz=adaptive_ceiling_hz,
            second_pass_attempted=second_pass_attempted,
            second_pass_voiced_frames=second_pass_voiced_frames,
            final_pass_used=final_pass_used,
            final_voiced_frames=0,
            voiced_proportion=0.0,
            pitch_extraction_success=False,
            fallback_used=fallback_used,
            failure_reason=failure_reason,
            warning_reason=warning_reason,
        )

    statistics = summarize_pitch_values(
        final_values_hz,
        pitch_config.semitone_reference_hz,
        pitch_config.lower_percentile,
        pitch_config.upper_percentile,
    )
    final_voiced_frames = int(final_values_hz.size)
    return SegmentPitchResult(
        statistics=statistics,
        first_pass_floor_hz=(first_pass_result.speaker_settings.pitch_floor_hz if first_pass_result.speaker_settings is not None else None),
        first_pass_ceiling_hz=(first_pass_result.speaker_settings.pitch_ceiling_hz if first_pass_result.speaker_settings is not None else None),
        first_pass_voiced_frames=first_pass_result.voiced_frames,
        first_pass_median_f0_hz=first_pass_result.median_f0_hz,
        adaptive_floor_hz=adaptive_floor_hz,
        adaptive_ceiling_hz=adaptive_ceiling_hz,
        second_pass_attempted=second_pass_attempted,
        second_pass_voiced_frames=second_pass_voiced_frames,
        final_pass_used=final_pass_used,
        final_voiced_frames=final_voiced_frames,
        voiced_proportion=float(final_voiced_frames / frame_total),
        pitch_extraction_success=True,
        fallback_used=fallback_used,
        failure_reason=None,
        warning_reason=warning_reason,
    )


def extract_segment_pitch_with_shared_adaptive_range(
    sound: Any,
    speaker_label: str,
    pitch_config: PitchConfig,
    adaptive_range_hz: tuple[float, float] | None,
    *,
    default_range: PitchSpeakerSettings | None = None,
) -> SegmentPitchResult:
    first_pass_result = extract_first_pass_pitch(
        sound,
        speaker_label,
        pitch_config,
        default_range=default_range,
    )
    if first_pass_result.speaker_settings is None:
        return build_segment_pitch_result(
            first_pass_result=first_pass_result,
            pitch_config=pitch_config,
            final_values_hz=None,
            adaptive_floor_hz=None,
            adaptive_ceiling_hz=None,
            second_pass_attempted=False,
            second_pass_voiced_frames=0,
            final_pass_used="failed",
            fallback_used=False,
            failure_reason=first_pass_result.failure_reason,
            warning_reason=None,
        )

    if not pitch_config.adaptive.enabled or adaptive_range_hz is None:
        if first_pass_result.success:
            return build_segment_pitch_result(
                first_pass_result=first_pass_result,
                pitch_config=pitch_config,
                final_values_hz=first_pass_result.voiced_values_hz,
                adaptive_floor_hz=None,
                adaptive_ceiling_hz=None,
                second_pass_attempted=False,
                second_pass_voiced_frames=0,
                final_pass_used="first_pass_fallback",
                fallback_used=True,
                failure_reason=None,
                warning_reason=(None if not pitch_config.adaptive.enabled else "shared_adaptive_range_unavailable"),
            )
        return build_segment_pitch_result(
            first_pass_result=first_pass_result,
            pitch_config=pitch_config,
            final_values_hz=None,
            adaptive_floor_hz=None,
            adaptive_ceiling_hz=None,
            second_pass_attempted=False,
            second_pass_voiced_frames=0,
            final_pass_used="failed",
            fallback_used=False,
            failure_reason=first_pass_result.failure_reason,
            warning_reason=None,
        )

    adaptive_floor_hz, adaptive_ceiling_hz = adaptive_range_hz
    if adaptive_floor_hz >= adaptive_ceiling_hz:
        if first_pass_result.success and pitch_config.adaptive.fallback_to_first_pass:
            return build_segment_pitch_result(
                first_pass_result=first_pass_result,
                pitch_config=pitch_config,
                final_values_hz=first_pass_result.voiced_values_hz,
                adaptive_floor_hz=adaptive_floor_hz,
                adaptive_ceiling_hz=adaptive_ceiling_hz,
                second_pass_attempted=False,
                second_pass_voiced_frames=0,
                final_pass_used="first_pass_fallback",
                fallback_used=True,
                failure_reason=None,
                warning_reason="invalid_adaptive_range",
            )
        return build_segment_pitch_result(
            first_pass_result=first_pass_result,
            pitch_config=pitch_config,
            final_values_hz=None,
            adaptive_floor_hz=adaptive_floor_hz,
            adaptive_ceiling_hz=adaptive_ceiling_hz,
            second_pass_attempted=False,
            second_pass_voiced_frames=0,
            final_pass_used="failed",
            fallback_used=False,
            failure_reason="invalid_adaptive_range",
            warning_reason=None,
        )

    second_values = _extract_voiced_values(
        sound,
        pitch_config,
        adaptive_floor_hz,
        adaptive_ceiling_hz,
    )
    second_pass_voiced_frames = int(second_values.size)
    if second_pass_voiced_frames >= pitch_config.minimum_voiced_frames:
        return build_segment_pitch_result(
            first_pass_result=first_pass_result,
            pitch_config=pitch_config,
            final_values_hz=second_values,
            adaptive_floor_hz=adaptive_floor_hz,
            adaptive_ceiling_hz=adaptive_ceiling_hz,
            second_pass_attempted=True,
            second_pass_voiced_frames=second_pass_voiced_frames,
            final_pass_used="second_pass",
            fallback_used=False,
            failure_reason=None,
            warning_reason=None,
        )
    if first_pass_result.success and pitch_config.adaptive.fallback_to_first_pass:
        return build_segment_pitch_result(
            first_pass_result=first_pass_result,
            pitch_config=pitch_config,
            final_values_hz=first_pass_result.voiced_values_hz,
            adaptive_floor_hz=adaptive_floor_hz,
            adaptive_ceiling_hz=adaptive_ceiling_hz,
            second_pass_attempted=True,
            second_pass_voiced_frames=second_pass_voiced_frames,
            final_pass_used="first_pass_fallback",
            fallback_used=True,
            failure_reason=None,
            warning_reason="second_pass_failed_fallback_used",
        )
    return build_segment_pitch_result(
        first_pass_result=first_pass_result,
        pitch_config=pitch_config,
        final_values_hz=None,
        adaptive_floor_hz=adaptive_floor_hz,
        adaptive_ceiling_hz=adaptive_ceiling_hz,
        second_pass_attempted=True,
        second_pass_voiced_frames=second_pass_voiced_frames,
        final_pass_used="failed",
        fallback_used=False,
        failure_reason=(first_pass_result.failure_reason if not first_pass_result.success else "insufficient_second_pass_voiced_frames"),
        warning_reason=None,
    )


def extract_segment_pitch(
    sound: Any,
    speaker_label: str,
    pitch_config: PitchConfig,
    *,
    default_range: PitchSpeakerSettings | None = None,
) -> SegmentPitchResult:
    first_pass_result = extract_first_pass_pitch(
        sound,
        speaker_label,
        pitch_config,
        default_range=default_range,
    )
    return extract_segment_pitch_with_shared_adaptive_range(
        sound,
        speaker_label,
        pitch_config,
        (
            compute_adaptive_range(first_pass_result.speaker_settings, first_pass_result.median_f0_hz, pitch_config)
            if pitch_config.adaptive.enabled and first_pass_result.success and first_pass_result.median_f0_hz is not None and first_pass_result.speaker_settings is not None
            else None
        ),
        default_range=default_range,
    )
