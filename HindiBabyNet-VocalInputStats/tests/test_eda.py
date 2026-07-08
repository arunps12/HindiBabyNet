from pathlib import Path

import pandas as pd

from hindibabynet_vocalinputstats.eda import run_eda


def test_generate_eda_creates_required_tables(tmp_path: Path) -> None:
    derived_dir = tmp_path / "data" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "participant_id": ["P001", "P002"],
            "REC_date": ["2024-01-01", "2024-01-02"],
            "birthdate": ["2023-01-01", "2023-01-02"],
            "child_sex": ["F", "M"],
            "mother_education": ["college", "school"],
            "father_education": ["college", "school"],
            "SES": ["college", "school"],
            "Location": ["Delhi", "Jaipur"],
            "age_days": [365.0, 366.0],
            "age_months": [11.99, 12.02],
            "age_z": [-1.0, 1.0],
            "recording_duration_sec": [3600.0, 7200.0],
            "recording_duration_hours": [1.0, 2.0],
            "adult_female_count": [1.0, 2.0],
            "adult_male_count": [3.0, 4.0],
            "other_child_count": [5.0, 6.0],
            "key_child_count": [7.0, 8.0],
            "adult_female_duration_sec": [10.0, 20.0],
            "adult_male_duration_sec": [30.0, 40.0],
            "other_child_duration_sec": [50.0, 60.0],
            "key_child_duration_sec": [70.0, 80.0],
            "adult_female_count_hour": [1.0, 1.0],
            "adult_male_count_hour": [3.0, 2.0],
            "other_child_count_hour": [5.0, 3.0],
            "key_child_count_hour": [7.0, 4.0],
            "adult_female_duration_hour": [10.0, 10.0],
            "adult_male_duration_hour": [30.0, 20.0],
            "other_child_duration_hour": [50.0, 30.0],
            "key_child_duration_hour": [70.0, 40.0],
        }
    ).to_csv(derived_dir / "final_master.csv", index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"metadata_csv: {(tmp_path / 'data' / 'raw' / 'metadata.csv').as_posix()}",
                f"vtc_output_root: {(tmp_path / 'vtc').as_posix()}",
                f"audio_root: {(tmp_path / 'audio').as_posix()}",
                f"derived_data_dir: {derived_dir.as_posix()}",
                f"private_data_dir: {(tmp_path / 'data' / 'private').as_posix()}",
                f"figures_dir: {(tmp_path / 'figures').as_posix()}",
                f"tables_dir: {(tmp_path / 'tables').as_posix()}",
                f"results_dir: {(tmp_path / 'results').as_posix()}",
                "audio_extensions: ['.wav', '.WAV']",
                "participant_id_digits: 3",
                "age_month_denominator: 30.44",
                "ses_source: mother_education",
                "minimum_recording_hours_warning: 1.0",
                "manual_mapping_csv:",
            ]
        ),
        encoding="utf-8",
    )

    run_eda(config_path)

    output_files = sorted(path.name for path in (tmp_path / "tables").glob("*.csv"))
    assert output_files == [
        "age_summary.csv",
        "education_distribution.csv",
        "location_distribution.csv",
        "missing_values_summary.csv",
        "participant_summary.csv",
        "recording_duration_summary.csv",
        "sex_distribution.csv",
        "speaker_count_summary.csv",
        "speaker_duration_summary.csv",
    ]