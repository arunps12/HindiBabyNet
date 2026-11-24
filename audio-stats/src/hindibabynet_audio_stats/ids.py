"""Participant anonymization helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hindibabynet_audio_stats.io import ensure_directory, write_csv


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
    source_column = "par_id" if "par_id" in merged.columns else "original_par_id"
    merged[source_column] = merged[source_column].astype(str).str.strip()
    return merged.merge(lookup, how="left", left_on=source_column, right_on="original_par_id", sort=False)


def save_participant_lookup(lookup: pd.DataFrame, path: str | Path) -> Path:
    """Save the private original-to-anonymized lookup file."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    return write_csv(lookup.loc[:, LOOKUP_COLUMNS], output_path)


def save_or_reuse_participant_lookup(lookup: pd.DataFrame, path: str | Path) -> bool:
    """Save the lookup file the first time. An existing file is never replaced.

    Returns True when the file was written and False when the existing file
    has the same content and is reused. A different existing file is an error.
    """
    output_path = Path(path)
    if not output_path.exists():
        save_participant_lookup(lookup, output_path)
        return True

    existing = pd.read_csv(output_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    new_rows = lookup.loc[:, LOOKUP_COLUMNS].astype(str).values.tolist()
    if list(existing.columns) != LOOKUP_COLUMNS or existing.values.tolist() != new_rows:
        raise FileExistsError(
            f"{output_path} already exists and is different from the participant ID list "
            "built from the metadata. The file is not replaced. Check that the metadata file "
            "has the same participants as before, or move the old file if a new list is wanted."
        )
    return False