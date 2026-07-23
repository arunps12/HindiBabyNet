from __future__ import annotations

import math

import pandas as pd

from .cleaning import parse_completed_months, parse_date_value, parse_timestamp
from .config import ProjectConfig


def add_age_columns(participant_linkage: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    data = participant_linkage.copy()

    data["reported_age_months"] = data["reported_age_months"].map(parse_completed_months)
    data["birthdate_parsed"] = data["birthdate"].map(parse_date_value)
    data["cdi_created_timestamp"] = data["cdi_created"].map(parse_timestamp)
    data["cdi_completion_date"] = data["cdi_created_timestamp"].map(lambda value: value.date() if value is not None else None)

    data["age_days"] = pd.NA
    valid_age_rows = data["birthdate_parsed"].notna() & data["cdi_completion_date"].notna()
    data.loc[valid_age_rows, "age_days"] = (
        pd.to_datetime(data.loc[valid_age_rows, "cdi_completion_date"]) - pd.to_datetime(data.loc[valid_age_rows, "birthdate_parsed"])
    ).dt.days
    data["calculated_age_months_exact"] = pd.to_numeric(data["age_days"], errors="coerce") / config.analysis.age_month_divisor
    data["age_month"] = data["calculated_age_months_exact"].map(
        lambda value: math.floor(value) if pd.notna(value) else pd.NA
    )
    data["age_difference_months"] = pd.to_numeric(data["reported_age_months"], errors="coerce") - data["calculated_age_months_exact"]
    data["age_discrepancy_flag"] = data["age_difference_months"].abs().ge(config.analysis.age_discrepancy_flag_months)

    data["age_is_valid"] = data["calculated_age_months_exact"].between(
        config.age.cdi1_min_month,
        config.age.cdi2_max_month + 0.999999,
        inclusive="both",
    )

    def _expected_form(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "unknown"
        if config.age.cdi1_min_month <= value < config.age.cdi2_min_month:
            return "CDI-I"
        if config.age.cdi2_min_month <= value <= config.age.cdi2_max_month + 0.999999:
            return "CDI-II"
        return "out_of_range"

    data["expected_form"] = data["calculated_age_months_exact"].map(_expected_form)
    data["age_form_match"] = data["submitted_form"].eq(data["expected_form"])
    return data