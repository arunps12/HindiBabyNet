from pathlib import Path
import wave

import numpy as np
import pandas as pd

from hindibabynet_audio_stats.cli import prepare_dataset


def _write_test_wav(path: Path, *, seconds: float, samplerate: int = 16000) -> None:
    frame_count = int(seconds * samplerate)
    samples = np.zeros(frame_count, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(samplerate)
        handle.writeframes(samples.tobytes())


def test_build_master_dataset_creates_public_outputs(make_config) -> None:
    config = make_config(params={"metadata_file": "metadata.csv", "recording_duration_source": "audio_file"})
    metadata_path = config.metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "par_id": ["BBB", "AAA"],
            "REC_date": ["2024-01-02", "2024-01-01"],
            "birthdate": ["2023-01-02", "2024-01-10"],
            "child_sex": ["F", "M"],
            "mother_education": ["college", "school"],
            "father_education": ["college", "school"],
            "Location": ["Delhi", "Jaipur"],
        }
    ).to_csv(metadata_path, index=False)

    vtc_root = config.vtc_output_root
    (vtc_root / "BBB").mkdir(parents=True)
    pd.DataFrame(
        {
            "uid": ["BBB", "BBB"],
            "start_time_s": [0.0, 1.0],
            "duration_s": [1.0, 2.0],
            "label": ["FEM", "KCHI"],
        }
    ).to_csv(vtc_root / "BBB" / "rttm.csv", index=False)

    _write_test_wav(config.audio_root / "BBB" / "joined.wav", seconds=7200.0)

    final_master, input_long, input_output_long = prepare_dataset(config)

    assert final_master["participant_id"].tolist() == ["P001", "P002"]
    assert pd.isna(final_master.loc[0, "recording_duration_hours"])
    assert final_master.loc[1, "recording_duration_hours"] == 2.0
    assert final_master.loc[1, "adult_female_count_hour"] == 0.5
    assert final_master.loc[1, "adult_female_count_min_hour"] == final_master.loc[1, "adult_female_count_hour"] / 60.0
    assert final_master.loc[1, "key_child_duration_hour"] == 1.0
    assert final_master.loc[1, "key_child_duration_min_hour"] == final_master.loc[1, "key_child_duration_hour"] / 60.0
    assert "SES_num" in final_master.columns
    assert "age_days_sq" in final_master.columns
    assert "age_days_sq_z" in final_master.columns
    assert final_master.loc[1, "age_days_sq"] == final_master.loc[1, "age_days"] ** 2
    assert "age_z2" in final_master.columns
    assert final_master.loc[1, "age_z2"] == final_master.loc[1, "age_z"] ** 2
    assert "original_par_id" not in final_master.columns

    validation = pd.read_csv(config.results_dir / "validation_report.csv")
    assert set(validation["issue_type"]) == {"missing_audio", "missing_vtc", "negative_age"}
    age_validation = pd.read_csv(config.results_dir / "age_validation.csv")
    assert age_validation["value"].all()
    assert "original_par_id" not in validation.columns
    private_lookup = pd.read_csv(config.lookup_path)
    assert list(private_lookup.columns) == ["original_par_id", "participant_id"]

    # One run saves the participant table, both long tables and the build report.
    saved_master = pd.read_csv(config.derived_data_dir / "final_master.csv")
    assert saved_master["participant_id"].tolist() == ["P001", "P002"]
    assert len(input_long) == 6
    assert len(input_output_long) == 6
    assert len(pd.read_csv(config.derived_data_dir / "input_long.csv")) == 6
    assert len(pd.read_csv(config.derived_data_dir / "input_output_long.csv")) == 6
    assert "original_par_id" not in input_long.columns
    report = (config.results_dir / "dataset_build_report.txt").read_text(encoding="utf-8")
    for line in [
        "metadata_participants: 2",
        "matched_vtc_output: 1",
        "matched_audio_duration: 1",
        "missing_vtc: 1",
        "missing_audio: 1",
        "final_master_rows: 2",
        "input_long_rows: 6",
        "input_output_long_rows: 6",
    ]:
        assert line in report


def test_build_master_dataset_uses_earliest_session_duration_for_rates(make_config) -> None:
    config = make_config(
        params={
            "metadata_file": "metadata.csv",
            "recording_duration_source": "raw_audio_sessions",
            "session_selection": "earliest",
            "session_date_format": "%Y%m%d",
        }
    )
    metadata_path = config.metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "par_id": ["AAA"],
            "REC_date": ["2024-01-01"],
            "birthdate": ["2023-01-01"],
            "child_sex": ["F"],
            "mother_education": ["college"],
            "father_education": ["college"],
            "Location": ["Delhi"],
        }
    ).to_csv(metadata_path, index=False)

    vtc_root = config.vtc_output_root
    (vtc_root / "AAA").mkdir(parents=True)
    pd.DataFrame(
        {
            "uid": ["AAA", "AAA"],
            "start_time_s": [0.0, 5.0],
            "duration_s": [60.0, 120.0],
            "label": ["FEM", "KCHI"],
        }
    ).to_csv(vtc_root / "AAA" / "rttm.csv", index=False)

    audio_root = config.audio_root
    _write_test_wav(audio_root / "AAA" / "20250201" / "part_1.wav", seconds=3600.0)
    _write_test_wav(audio_root / "AAA" / "20250201" / "part_2.wav", seconds=1800.0)
    _write_test_wav(audio_root / "AAA" / "20250202" / "part_3.wav", seconds=7200.0)

    final_master, _, _ = prepare_dataset(config)

    assert final_master.loc[0, "recording_duration_hours"] == 1.5
    assert final_master.loc[0, "adult_female_count_hour"] == 1.0 / 1.5
    assert final_master.loc[0, "adult_female_count_min_hour"] == (1.0 / 1.5) / 60.0
    assert final_master.loc[0, "key_child_duration_hour"] == 120.0 / 1.5
    assert final_master.loc[0, "key_child_duration_min_hour"] == (120.0 / 1.5) / 60.0
    duration_audit = pd.read_csv(config.results_dir / "recording_duration_audit.csv")
    assert duration_audit.loc[0, "selected_date"] == 20250201
    assert duration_audit.loc[0, "audio_file_count"] == 2
    assert str(duration_audit.loc[0, "ignored_later_folders"]) == "20250202"