from pathlib import Path

import numpy as np
import pandas as pd

from hindibabynet_audio_stats.create_long_format import build_long_tables, build_wide_table


def test_create_long_format_builds_three_rows_per_participant(tmp_path: Path) -> None:
    master = pd.DataFrame(
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
            "age_days_sq": [133225.0, 133956.0],
            "age_days_sq_z": [-0.7071067811865475, 0.7071067811865475],
            "age_months": [11.99, 12.02],
            "age_z": [-1.0, 1.0],
            "age_z2": [1.0, 1.0],
            "SES_num": [None, None],
            "recording_duration_sec": [3600.0, 3600.0],
            "recording_duration_hours": [1.0, 1.0],
            "adult_female_count": [1.0, 2.0],
            "adult_male_count": [3.0, 4.0],
            "other_child_count": [5.0, 6.0],
            "key_child_count": [7.0, 8.0],
            "adult_female_duration_sec": [10.0, 20.0],
            "adult_male_duration_sec": [30.0, 40.0],
            "other_child_duration_sec": [50.0, 60.0],
            "key_child_duration_sec": [70.0, 80.0],
            "adult_female_count_hour": [1.0, 2.0],
            "adult_male_count_hour": [3.0, 4.0],
            "other_child_count_hour": [5.0, 6.0],
            "key_child_count_hour": [7.0, 8.0],
            "adult_female_count_min_hour": [1.0 / 60.0, 2.0 / 60.0],
            "adult_male_count_min_hour": [3.0 / 60.0, 4.0 / 60.0],
            "other_child_count_min_hour": [5.0 / 60.0, 6.0 / 60.0],
            "key_child_count_min_hour": [7.0 / 60.0, 8.0 / 60.0],
            "adult_female_duration_hour": [10.0, 20.0],
            "adult_male_duration_hour": [30.0, 40.0],
            "other_child_duration_hour": [50.0, 60.0],
            "key_child_duration_hour": [70.0, 80.0],
            "adult_female_duration_min_hour": [10.0 / 60.0, 20.0 / 60.0],
            "adult_male_duration_min_hour": [30.0 / 60.0, 40.0 / 60.0],
            "other_child_duration_min_hour": [50.0 / 60.0, 60.0 / 60.0],
            "key_child_duration_min_hour": [70.0 / 60.0, 80.0 / 60.0],
        }
    )
    # The long tables are built from the participant table as it is saved in the CSV file.
    master.to_csv(tmp_path / "final_master.csv", index=False)

    input_long, input_output_long = build_long_tables(pd.read_csv(tmp_path / "final_master.csv"))

    assert len(input_long) == 6
    assert len(input_output_long) == 6
    assert input_long["speaker"].tolist() == [
        "adult_female",
        "adult_male",
        "other_child",
        "adult_female",
        "adult_male",
        "other_child",
    ]
    assert input_output_long["key_child_count_hour"].tolist() == [7.0, 7.0, 7.0, 8.0, 8.0, 8.0]
    assert np.allclose(input_long["input_count_min_hour"], [1.0 / 60.0, 3.0 / 60.0, 5.0 / 60.0, 2.0 / 60.0, 4.0 / 60.0, 6.0 / 60.0])
    assert np.allclose(input_long["input_duration_min_hour"], [10.0 / 60.0, 30.0 / 60.0, 50.0 / 60.0, 20.0 / 60.0, 40.0 / 60.0, 60.0 / 60.0])
    assert np.allclose(input_output_long["key_child_count_min_hour"], [7.0 / 60.0, 7.0 / 60.0, 7.0 / 60.0, 8.0 / 60.0, 8.0 / 60.0, 8.0 / 60.0])
    assert np.allclose(input_output_long["key_child_duration_min_hour"], [70.0 / 60.0, 70.0 / 60.0, 70.0 / 60.0, 80.0 / 60.0, 80.0 / 60.0, 80.0 / 60.0])
    assert "age_days_sq" in input_long.columns
    assert "age_days_sq_z" in input_long.columns
    assert "SES_num" in input_long.columns
    assert "age_z2" in input_long.columns
    assert (input_long["age_z2"] == 1.0).all()
    assert "original_par_id" not in input_long.columns

    # The wide table has one row per participant and one column for each input speaker.
    saved = pd.read_csv(tmp_path / "final_master.csv")
    wide = build_wide_table(saved.iloc[::-1])
    assert wide["participant_id"].tolist() == ["P001", "P002"]
    assert "speaker" not in wide.columns and "original_par_id" not in wide.columns
    rates = [
        f"{speaker}_{rate}"
        for speaker in ("adult_female", "adult_male", "other_child", "key_child")
        for rate in ("count_hour", "count_min_hour", "duration_hour", "duration_min_hour")
    ]
    assert list(wide.columns) == list(input_long.columns[:1]) + list(input_long.columns[2:15]) + rates
    for column in wide.columns:
        pd.testing.assert_series_equal(wide[column], saved[column])
    assert wide["adult_male_count_hour"].tolist() == [3.0, 4.0]
    assert wide["key_child_duration_hour"].tolist() == [70.0, 80.0]