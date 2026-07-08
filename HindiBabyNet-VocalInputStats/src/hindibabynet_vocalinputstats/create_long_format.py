"""Create long-format vocal input datasets from the master dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hindibabynet_vocalinputstats.config import ProjectConfig, load_config
from hindibabynet_vocalinputstats.io import read_csv, write_csv, write_dataset_build_report


INPUT_SPEAKERS = ["adult_female", "adult_male", "other_child"]
COMMON_COLUMNS = [
    "participant_id",
    "age_days",
    "age_months",
    "age_z",
    "child_sex",
    "SES",
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
        frame["input_duration_hour"] = master[f"{speaker}_duration_hour"]
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(["participant_id", "speaker"], kind="stable").reset_index(drop=True)


def _build_input_output_long(master: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for speaker in INPUT_SPEAKERS:
        frame = master[COMMON_COLUMNS].copy()
        frame.insert(1, "speaker", speaker)
        frame["input_count_hour"] = master[f"{speaker}_count_hour"]
        frame["input_duration_hour"] = master[f"{speaker}_duration_hour"]
        frame["key_child_count_hour"] = master["key_child_count_hour"]
        frame["key_child_duration_hour"] = master["key_child_duration_hour"]
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(["participant_id", "speaker"], kind="stable").reset_index(drop=True)


def _refresh_dataset_build_report(config: ProjectConfig, master: pd.DataFrame, input_long: pd.DataFrame, input_output_long: pd.DataFrame) -> None:
    validation_path = config.results_dir / "validation_report.csv"
    issues = read_csv(validation_path) if validation_path.exists() else pd.DataFrame(columns=["issue_type"])

    def _issue_count(issue_type: str) -> int:
        if issues.empty or "issue_type" not in issues.columns:
            return 0
        return int((issues["issue_type"] == issue_type).sum())

    total = len(master)
    missing_vtc = _issue_count("missing_vtc")
    missing_audio = _issue_count("missing_audio") + _issue_count("unreadable_audio")
    lines = [
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
        "denominator_note: Full recording duration was used as the denominator for count/hour and duration/hour.",
    ]
    write_dataset_build_report(lines, config.results_dir / "dataset_build_report.txt")


def create_long_format(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    master_path = config.derived_data_dir / "final_master.csv"
    master = read_csv(master_path)
    input_long = _build_input_long(master)
    input_output_long = _build_input_output_long(master)
    write_csv(input_long, config.derived_data_dir / "input_long.csv")
    write_csv(input_output_long, config.derived_data_dir / "input_output_long.csv")
    _refresh_dataset_build_report(config, master, input_long, input_output_long)
    return input_long, input_output_long


def run_create_long(config_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    return create_long_format(config)