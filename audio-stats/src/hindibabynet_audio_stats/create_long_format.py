"""Create long-format vocal input datasets from the master dataset."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from hindibabynet_audio_stats.config import ProjectConfig


INPUT_SPEAKERS = ["adult_female", "adult_male", "other_child"]
COMMON_COLUMNS = [
    "participant_id",
    "age_days",
    "age_days_sq",
    "age_days_sq_z",
    "age_months",
    "age_z",
    "age_z2",
    "child_sex",
    "SES",
    "SES_num",
    "mother_education",
    "father_education",
    "Location",
    "recording_duration_hours",
]


def _build_input_long(master: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for speaker in INPUT_SPEAKERS:
        frame = master[COMMON_COLUMNS].copy()
        frame.insert(1, "speaker", speaker)
        frame["input_count_hour"] = master[f"{speaker}_count_hour"]
        frame["input_count_min_hour"] = master[f"{speaker}_count_min_hour"]
        frame["input_duration_hour"] = master[f"{speaker}_duration_hour"]
        frame["input_duration_min_hour"] = master[f"{speaker}_duration_min_hour"]
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(["participant_id", "speaker"], kind="stable").reset_index(drop=True)


def _build_input_output_long(master: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for speaker in INPUT_SPEAKERS:
        frame = master[COMMON_COLUMNS].copy()
        frame.insert(1, "speaker", speaker)
        frame["input_count_hour"] = master[f"{speaker}_count_hour"]
        frame["input_count_min_hour"] = master[f"{speaker}_count_min_hour"]
        frame["input_duration_hour"] = master[f"{speaker}_duration_hour"]
        frame["input_duration_min_hour"] = master[f"{speaker}_duration_min_hour"]
        frame["key_child_count_hour"] = master["key_child_count_hour"]
        frame["key_child_count_min_hour"] = master["key_child_count_min_hour"]
        frame["key_child_duration_hour"] = master["key_child_duration_hour"]
        frame["key_child_duration_min_hour"] = master["key_child_duration_min_hour"]
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(["participant_id", "speaker"], kind="stable").reset_index(drop=True)


def dataset_report_lines(
    config: ProjectConfig,
    master: pd.DataFrame,
    input_long: pd.DataFrame,
    input_output_long: pd.DataFrame,
    validation_records: list[dict[str, object]],
) -> list[str]:
    """Lines of dataset_build_report.txt: numbers of participants, matches and rows."""
    issue_types = [record["issue_type"] for record in validation_records]

    total = len(master)
    missing_vtc = issue_types.count("missing_vtc")
    missing_audio = issue_types.count("missing_audio") + issue_types.count("unreadable_audio")
    return [
        f"build_datetime_utc: {datetime.now(timezone.utc).isoformat()}",
        f"metadata_participants: {total}",
        f"matched_vtc_output: {total - missing_vtc}",
        f"matched_audio_duration: {total - missing_audio}",
        f"missing_vtc: {missing_vtc}",
        f"missing_audio: {missing_audio}",
        f"missing_age: {int(master['age_days'].isna().sum())}",
        f"negative_or_invalid_age: {int(((master['age_days'] < 0) & master['age_days'].notna()).sum())}",
        f"final_master_rows: {total}",
        f"input_long_rows: {len(input_long)}",
        f"input_output_long_rows: {len(input_output_long)}",
        f"denominator_note: Recording duration source was {config.recording_duration_source} with session_selection={config.session_selection}.",
    ]


def build_long_tables(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build input_long and input_output_long from the participant table."""
    missing_age_columns = [column for column in ("age_days", "age_days_sq", "age_days_sq_z", "age_z", "age_z2") if column not in master.columns]
    if missing_age_columns:
        raise ValueError(f"final_master.csv is missing required age columns: {', '.join(missing_age_columns)}")
    return _build_input_long(master), _build_input_output_long(master)