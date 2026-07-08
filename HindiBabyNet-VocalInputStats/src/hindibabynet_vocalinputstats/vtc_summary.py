"""VTC label normalization and aggregation helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


VTC_LABEL_MAP = {
    "FEM": "adult_female",
    "MAL": "adult_male",
    "KCHI": "key_child",
    "OCH": "other_child",
}


def normalize_vtc_label(label: object) -> str | None:
    raw_label = str(label).strip().upper()
    return VTC_LABEL_MAP.get(raw_label)


def summarize_vtc_dataframe(
    dataframe: pd.DataFrame,
    *,
    participant_id: str,
    original_par_id: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Aggregate VTC segments to count and duration by speaker class."""
    records: list[dict[str, object]] = []
    warning_records: list[dict[str, object]] = []

    for row in dataframe.itertuples(index=False):
        normalized_label = normalize_vtc_label(getattr(row, "label"))
        if normalized_label is None:
            warning_records.append(
                {
                    "participant_id": participant_id,
                    "original_par_id": original_par_id,
                    "issue_type": "unknown_vtc_label",
                    "message": f"Unexpected VTC label: {getattr(row, 'label')}",
                }
            )
            continue
        records.append(
            {
                "participant_id": participant_id,
                "original_par_id": original_par_id,
                "speaker": normalized_label,
                "duration_sec": float(getattr(row, "duration_s")),
            }
        )

    if not records:
        empty = pd.DataFrame(columns=["participant_id", "original_par_id", "speaker", "count", "duration_sec"])
        return empty, warning_records

    segments = pd.DataFrame.from_records(records)
    summary = (
        segments.groupby(["participant_id", "original_par_id", "speaker"], as_index=False, sort=True)
        .agg(count=("speaker", "size"), duration_sec=("duration_sec", "sum"))
        .sort_values(["participant_id", "speaker"], kind="stable")
        .reset_index(drop=True)
    )
    return summary, warning_records


def expected_speakers() -> Iterable[str]:
    return VTC_LABEL_MAP.values()