"""Create EDA summary tables for the vocal input statistics datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hindibabynet_vocalinputstats.config import ProjectConfig, load_config
from hindibabynet_vocalinputstats.io import read_csv, write_csv


COUNT_COLUMNS = ["adult_female_count_hour", "adult_male_count_hour", "other_child_count_hour", "key_child_count_hour"]
DURATION_COLUMNS = [
    "adult_female_duration_hour",
    "adult_male_duration_hour",
    "other_child_duration_hour",
    "key_child_duration_hour",
]


def _numeric_summary(series: pd.Series, variable: str) -> dict[str, object]:
    clean = pd.to_numeric(series, errors="coerce")
    return {
        "variable": variable,
        "non_missing": int(clean.notna().sum()),
        "mean": clean.mean(),
        "std": clean.std(ddof=0),
        "min": clean.min(),
        "median": clean.median(),
        "max": clean.max(),
    }


def build_eda_tables(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["participant_summary.csv"] = master[
        ["participant_id", "age_days", "age_months", "age_z", "child_sex", "SES", "Location", "recording_duration_hours"]
    ].copy()
    tables["missing_values_summary.csv"] = pd.DataFrame(
        {
            "column": master.columns,
            "missing_count": [int(master[column].isna().sum()) for column in master.columns],
            "missing_percent": [float(master[column].isna().mean() * 100.0) for column in master.columns],
        }
    )
    tables["recording_duration_summary.csv"] = pd.DataFrame(
        [_numeric_summary(master["recording_duration_hours"], "recording_duration_hours")]
    )
    tables["speaker_count_summary.csv"] = pd.DataFrame([_numeric_summary(master[column], column) for column in COUNT_COLUMNS])
    tables["speaker_duration_summary.csv"] = pd.DataFrame([_numeric_summary(master[column], column) for column in DURATION_COLUMNS])
    tables["age_summary.csv"] = pd.DataFrame(
        [
            _numeric_summary(master["age_days"], "age_days"),
            _numeric_summary(master["age_months"], "age_months"),
            _numeric_summary(master["age_z"], "age_z"),
        ]
    )
    tables["sex_distribution.csv"] = (
        master["child_sex"].fillna("missing").value_counts(dropna=False).rename_axis("child_sex").reset_index(name="count")
    )
    tables["location_distribution.csv"] = (
        master["Location"].fillna("missing").value_counts(dropna=False).rename_axis("Location").reset_index(name="count")
    )
    education_rows = []
    for education_type in ["mother_education", "father_education"]:
        counts = master[education_type].fillna("missing").value_counts(dropna=False)
        for value, count in counts.items():
            education_rows.append({"education_type": education_type, "education_value": value, "count": int(count)})
    tables["education_distribution.csv"] = pd.DataFrame(education_rows).sort_values(
        ["education_type", "education_value"],
        kind="stable",
    )
    return tables


def generate_eda(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    master = read_csv(config.derived_data_dir / "final_master.csv")
    tables = build_eda_tables(master)
    for filename, dataframe in tables.items():
        write_csv(dataframe, config.tables_dir / filename)
    return tables


def run_eda(config_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    return generate_eda(config)