"""Participant anonymization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hindibabynet_vocalinputstats.io import ensure_directory, write_csv


LOOKUP_COLUMNS = ["original_par_id", "participant_id"]


def create_participant_lookup(metadata: pd.DataFrame, participant_id_digits: int) -> pd.DataFrame:
    """Create a stable anonymized lookup ordered by sorted original participant IDs."""
    source_ids = metadata["par_id"].astype(str).str.strip()
    unique_ids = sorted(source_ids.dropna().unique().tolist())
    rows = []
    for index, original_par_id in enumerate(unique_ids, start=1):
        participant_id = f"P{index:0{participant_id_digits}d}"
        rows.append({"original_par_id": original_par_id, "participant_id": participant_id})
    return pd.DataFrame(rows, columns=LOOKUP_COLUMNS)


def attach_participant_ids(metadata: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Attach participant IDs to metadata without dropping rows."""
    merged = metadata.copy()
    merged["par_id"] = merged["par_id"].astype(str).str.strip()
    return merged.merge(lookup, how="left", left_on="par_id", right_on="original_par_id", sort=False)


def save_participant_lookup(lookup: pd.DataFrame, path: str | Path) -> Path:
    """Save the private original-to-anonymized lookup file."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    return write_csv(lookup.loc[:, LOOKUP_COLUMNS], output_path)