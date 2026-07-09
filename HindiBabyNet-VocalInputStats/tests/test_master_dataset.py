from pathlib import Path
import wave

import numpy as np
import pandas as pd

from hindibabynet_vocalinputstats.build_master_dataset import run_build_master


def _write_test_wav(path: Path, *, seconds: float, samplerate: int = 16000) -> None:
    frame_count = int(seconds * samplerate)
    samples = np.zeros(frame_count, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(samplerate)
        handle.writeframes(samples.tobytes())


def test_build_master_dataset_creates_public_outputs(tmp_path: Path) -> None:
    metadata_path = tmp_path / "data" / "raw" / "metadata.csv"
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

    vtc_root = tmp_path / "vtc"
    (vtc_root / "BBB").mkdir(parents=True)
    pd.DataFrame(
        {
            "uid": ["BBB", "BBB"],
            "start_time_s": [0.0, 1.0],
            "duration_s": [1.0, 2.0],
            "label": ["FEM", "KCHI"],
        }
    ).to_csv(vtc_root / "BBB" / "rttm.csv", index=False)

    audio_root = tmp_path / "audio"
    _write_test_wav(audio_root / "BBB" / "joined.wav", seconds=7200.0)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"metadata_path: {metadata_path.as_posix()}",
                f"vtc_output_root: {vtc_root.as_posix()}",
                f"audio_root: {audio_root.as_posix()}",
                f"derived_data_dir: {(tmp_path / 'data' / 'derived').as_posix()}",
                f"private_data_dir: {(tmp_path / 'data' / 'private').as_posix()}",
                f"figures_dir: {(tmp_path / 'figures').as_posix()}",
                f"tables_dir: {(tmp_path / 'tables').as_posix()}",
                f"results_dir: {(tmp_path / 'results').as_posix()}",
                "metadata_id_column: par_id",
                "audio_layout:",
                "  type: participant_folder",
                "  participant_folder_name: '{par_id}'",
                "  recursive: true",
                "  audio_extensions: ['.wav', '.WAV']",
                "  expected_audio_files: auto",
                "  prefer_largest_audio_file: true",
                "vtc_layout:",
                "  type: participant_folder",
                "  participant_folder_name: '{par_id}'",
                "  rttm_csv_name: rttm.csv",
                "participant_id_digits: 3",
                "age_month_denominator: 30.44",
                "ses_source: mother_education",
                "minimum_recording_hours_warning: 1.0",
                "manual_mapping_csv:",
            ]
        ),
        encoding="utf-8",
    )

    final_master = run_build_master(config_path)

    assert final_master["participant_id"].tolist() == ["P001", "P002"]
    assert pd.isna(final_master.loc[0, "recording_duration_hours"])
    assert final_master.loc[1, "recording_duration_hours"] == 2.0
    assert final_master.loc[1, "adult_female_count_hour"] == 0.5
    assert final_master.loc[1, "key_child_duration_hour"] == 1.0
    assert "original_par_id" not in final_master.columns

    validation = pd.read_csv(tmp_path / "results" / "validation_report.csv")
    assert set(validation["issue_type"]) == {"missing_audio", "missing_vtc", "negative_age"}
    assert "original_par_id" not in validation.columns
    private_lookup = pd.read_csv(tmp_path / "data" / "private" / "participant_lookup.csv")
    assert list(private_lookup.columns) == ["original_par_id", "participant_id"]