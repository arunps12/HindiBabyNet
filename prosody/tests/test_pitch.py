from __future__ import annotations

import numpy as np
import parselmouth
import pytest

from hindibabynet_prosody.features.pitch import (
    compute_adaptive_range,
    extract_segment_pitch,
    hz_to_semitones,
    select_speaker_settings,
    summarize_pitch_values,
)


@pytest.fixture
def pitch_config(make_config):
    return make_config().pitch


def _sine_sound(frequency_hz: float, duration_s: float = 1.0, sample_rate: int = 16000) -> parselmouth.Sound:
    timeline = np.arange(int(duration_s * sample_rate), dtype=np.float64) / sample_rate
    waveform = 0.2 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    return parselmouth.Sound(waveform, sample_rate)


def test_select_speaker_settings_for_supported_labels(pitch_config) -> None:
    assert select_speaker_settings(pitch_config, "FEM").pitch_floor_hz == 75.0
    assert select_speaker_settings(pitch_config, "MAL").pitch_ceiling_hz == 400.0
    assert select_speaker_settings(pitch_config, "OCH").pitch_floor_hz == 100.0


def test_select_speaker_settings_rejects_unsupported_label(pitch_config) -> None:
    with pytest.raises(ValueError, match="Unsupported speaker label"):
        select_speaker_settings(pitch_config, "KCHI")


def test_compute_adaptive_range_clips_to_speaker_limits(pitch_config) -> None:
    settings = select_speaker_settings(pitch_config, "MAL")
    floor_hz, ceiling_hz = compute_adaptive_range(settings, 300.0, pitch_config)
    assert floor_hz == 150.0
    assert ceiling_hz == 400.0


def test_semitone_conversion_matches_known_ratio() -> None:
    values = np.asarray([2.0, 4.0], dtype=np.float64)
    converted = hz_to_semitones(values, 1.0)
    assert converted[1] - converted[0] == pytest.approx(12.0)


def test_robust_percentile_range_uses_percentiles() -> None:
    values = np.asarray([100.0, 100.0, 100.0, 200.0, 1000.0], dtype=np.float64)
    stats = summarize_pitch_values(values, 1.0, 5.0, 95.0)
    assert stats.robust_pitch_range_hz < 900.0


def test_extract_segment_pitch_two_pass_sine_wave(pitch_config) -> None:
    result = extract_segment_pitch(_sine_sound(220.0), "FEM", pitch_config)
    assert result.pitch_extraction_success is True
    assert result.statistics is not None
    assert result.statistics.median_f0_hz == pytest.approx(220.0, rel=1e-2)
    assert result.second_pass_attempted is True


def test_extract_segment_pitch_falls_back_to_first_pass_when_adaptive_invalid(pitch_config) -> None:
    pitch_config.adaptive.lower_multiplier = 4.0
    pitch_config.adaptive.upper_multiplier = 0.6
    sound = _sine_sound(220.0)
    result = extract_segment_pitch(sound, "FEM", pitch_config)
    assert result.pitch_extraction_success is True
    assert result.final_pass_used == "first_pass_fallback"
    assert result.fallback_used is True


def test_extract_segment_pitch_fails_for_unvoiced_audio(pitch_config) -> None:
    silent = parselmouth.Sound(np.zeros(16000, dtype=np.float64), 16000)
    result = extract_segment_pitch(silent, "FEM", pitch_config)
    assert result.pitch_extraction_success is False
    assert result.failure_reason == "insufficient_first_pass_voiced_frames"
