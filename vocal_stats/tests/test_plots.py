from pathlib import Path

import pandas as pd

from hindibabynet_vocalinputstats.plots import EXPECTED_PLOT_BASENAMES, run_plots


def test_generate_plots_creates_png_and_pdf_outputs(tmp_path: Path) -> None:
    derived_dir = tmp_path / "data" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    master = pd.DataFrame(
        {
            "participant_id": ["P001", "P002", "P003"],
            "REC_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "birthdate": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "child_sex": ["F", "M", "F"],
            "mother_education": ["college", "school", "college"],
            "father_education": ["college", "school", "school"],
            "SES": ["college", "school", "college"],
            "Location": ["Delhi", "Jaipur", "Delhi"],
            "age_days": [365.0, 390.0, 420.0],
            "age_months": [11.99, 12.81, 13.80],
            "age_z": [-1.0, 0.0, 1.0],
            "recording_duration_sec": [3600.0, 5400.0, 7200.0],
            "recording_duration_hours": [1.0, 1.5, 2.0],
            "adult_female_count": [1.0, 2.0, 3.0],
            "adult_male_count": [2.0, 3.0, 4.0],
            "other_child_count": [3.0, 4.0, 5.0],
            "key_child_count": [4.0, 5.0, 6.0],
            "adult_female_duration_sec": [10.0, 20.0, 30.0],
            "adult_male_duration_sec": [20.0, 30.0, 40.0],
            "other_child_duration_sec": [30.0, 40.0, 50.0],
            "key_child_duration_sec": [40.0, 50.0, 60.0],
            "adult_female_count_hour": [1.0, 1.33, 1.5],
            "adult_male_count_hour": [2.0, 2.0, 2.0],
            "other_child_count_hour": [3.0, 2.67, 2.5],
            "key_child_count_hour": [4.0, 3.33, 3.0],
            "adult_female_duration_hour": [10.0, 13.33, 15.0],
            "adult_male_duration_hour": [20.0, 20.0, 20.0],
            "other_child_duration_hour": [30.0, 26.67, 25.0],
            "key_child_duration_hour": [40.0, 33.33, 30.0],
        }
    )
    input_long = pd.DataFrame(
        {
            "participant_id": ["P001", "P001", "P001", "P002", "P002", "P002", "P003", "P003", "P003"],
            "speaker": ["adult_female", "adult_male", "other_child"] * 3,
            "input_count_hour": [1.0, 2.0, 3.0, 1.33, 2.0, 2.67, 1.5, 2.0, 2.5],
            "input_duration_hour": [10.0, 20.0, 30.0, 13.33, 20.0, 26.67, 15.0, 20.0, 25.0],
            "age_days": [365.0] * 3 + [390.0] * 3 + [420.0] * 3,
            "age_months": [11.99] * 3 + [12.81] * 3 + [13.80] * 3,
            "age_z": [-1.0] * 3 + [0.0] * 3 + [1.0] * 3,
            "child_sex": ["F"] * 3 + ["M"] * 3 + ["F"] * 3,
            "SES": ["college"] * 3 + ["school"] * 3 + ["college"] * 3,
            "mother_education": ["college"] * 3 + ["school"] * 3 + ["college"] * 3,
            "father_education": ["college"] * 3 + ["school"] * 3 + ["school"] * 3,
            "Location": ["Delhi"] * 3 + ["Jaipur"] * 3 + ["Delhi"] * 3,
            "recording_duration_hours": [1.0] * 3 + [1.5] * 3 + [2.0] * 3,
        }
    )
    master.to_csv(derived_dir / "final_master.csv", index=False)
    input_long.to_csv(derived_dir / "input_long.csv", index=False)

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

    run_plots(config_path)

    for basename in EXPECTED_PLOT_BASENAMES:
        assert (tmp_path / "figures" / f"{basename}.png").exists()
        assert (tmp_path / "figures" / f"{basename}.pdf").exists()