import wave
from pathlib import Path

import numpy as np
import pandas as pd

from hindibabynet_vocalinputstats.durations import (
    read_audio_duration_seconds,
    resolve_recording_duration,
    seconds_to_hours,
)
from hindibabynet_vocalinputstats.vtc_summary import normalize_vtc_label, summarize_vtc_dataframe


def _write_test_wav(path: Path, *, seconds: float, samplerate: int = 16000) -> None:
    frame_count = int(seconds * samplerate)
    samples = np.zeros(frame_count, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(samplerate)
        handle.writeframes(samples.tobytes())


def test_read_audio_duration_seconds_uses_full_audio_file(tmp_path: Path) -> None:
    path = tmp_path / "participant.wav"
    _write_test_wav(path, seconds=2.5)

    duration_sec = read_audio_duration_seconds(path)

    assert duration_sec == 2.5
    assert seconds_to_hours(duration_sec) == 2.5 / 3600.0


def test_resolve_recording_duration_uses_earliest_session_sum(tmp_path: Path) -> None:
    participant_root = tmp_path / "P001"
    _write_test_wav(participant_root / "20250201" / "a.wav", seconds=1.0)
    _write_test_wav(participant_root / "20250201" / "b.wav", seconds=2.0)
    _write_test_wav(participant_root / "20250202" / "c.wav", seconds=5.0)

    result = resolve_recording_duration(
        tmp_path,
        "P001",
        [".wav", ".WAV"],
        participant_folder_name="{par_id}",
        recording_duration_source="raw_audio_sessions",
        session_selection="earliest",
    )

    assert result.matched_audio is True
    assert result.warning_message is None
    assert result.duration_sec == 3.0
    assert result.selected_date == "20250201"
    assert result.audio_file_count == 2
    assert result.ignored_later_folders == ("20250202",)


def test_resolve_recording_duration_falls_back_when_no_valid_session_dirs(tmp_path: Path) -> None:
    participant_root = tmp_path / "P001"
    _write_test_wav(participant_root / "session_a" / "a.wav", seconds=1.5)
    _write_test_wav(participant_root / "session_b" / "b.wav", seconds=2.5)

    result = resolve_recording_duration(
        tmp_path,
        "P001",
        [".wav", ".WAV"],
        participant_folder_name="{par_id}",
        recording_duration_source="raw_audio_sessions",
        session_selection="earliest",
    )

    assert result.matched_audio is True
    assert result.duration_sec == 4.0
    assert result.warning_message is not None
    assert result.audio_file_count == 2


def test_normalize_vtc_label_maps_expected_values() -> None:
    assert normalize_vtc_label("FEM") == "adult_female"
    assert normalize_vtc_label("mal") == "adult_male"
    assert normalize_vtc_label("KCHI") == "key_child"
    assert normalize_vtc_label("OCH") == "other_child"
    assert normalize_vtc_label("noise") is None


def test_summarize_vtc_dataframe_aggregates_count_and_duration() -> None:
    dataframe = pd.DataFrame(
        {
            "uid": ["A", "A", "A", "A"],
            "start_time_s": [0.0, 0.5, 1.0, 1.5],
            "duration_s": [0.5, 0.25, 0.75, 1.0],
            "label": ["FEM", "FEM", "MAL", "OTHER"],
        }
    )

    summary, warnings = summarize_vtc_dataframe(dataframe, participant_id="P001", original_par_id="A001")

    assert summary.to_dict("records") == [
        {
            "participant_id": "P001",
            "original_par_id": "A001",
            "speaker": "adult_female",
            "count": 2,
            "duration_sec": 0.75,
        },
        {
            "participant_id": "P001",
            "original_par_id": "A001",
            "speaker": "adult_male",
            "count": 1,
            "duration_sec": 0.75,
        },
    ]
    assert warnings == [
        {
            "participant_id": "P001",
            "original_par_id": "A001",
            "issue_type": "unknown_vtc_label",
            "message": "Unexpected VTC label: OTHER",
        }
    ]