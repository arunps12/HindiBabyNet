"""Participant lookup loading and validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hindibabynet_prosody.config import AppConfig


@dataclass(frozen=True)
class ParticipantLookupResult:
    dataframe: pd.DataFrame
    mapping: dict[str, str]


def load_participant_lookup(config: AppConfig) -> ParticipantLookupResult:
    frame = pd.read_csv(config.paths.participant_lookup)
    source_column = config.mapping.source_id_column
    participant_column = config.mapping.participant_id_column
    missing = {source_column, participant_column} - set(frame.columns)
    if missing:
        raise ValueError(
            "participant lookup is missing required columns: " + ", ".join(sorted(missing))
        )

    normalized = frame[[source_column, participant_column]].copy()
    normalized[source_column] = normalized[source_column].astype(str).str.strip()
    normalized[participant_column] = normalized[participant_column].astype(str).str.strip()
    normalized = normalized.loc[
        normalized[source_column].ne("") & normalized[participant_column].ne("")
    ].reset_index(drop=True)

    duplicated_source = normalized.duplicated(subset=[source_column], keep=False)
    if duplicated_source.any():
        duplicates = normalized.loc[duplicated_source].sort_values(source_column)
        conflict_counts = duplicates.groupby(source_column)[participant_column].nunique()
        conflicts = conflict_counts.loc[conflict_counts.gt(1)]
        if not conflicts.empty:
            raise ValueError(
                "participant lookup contains conflicting source mappings for: "
                + ", ".join(conflicts.index.astype(str).tolist())
            )
        raise ValueError(
            "participant lookup contains duplicate source mappings for: "
            + ", ".join(conflict_counts.index.astype(str).tolist())
        )

    duplicated_target = normalized.duplicated(subset=[participant_column], keep=False)
    if duplicated_target.any():
        duplicates = normalized.loc[duplicated_target].sort_values(participant_column)
        raise ValueError(
            "participant lookup contains non-unique participant_id values: "
            + ", ".join(duplicates[participant_column].unique().tolist())
        )

    mapping = dict(zip(normalized[source_column], normalized[participant_column], strict=True))
    return ParticipantLookupResult(dataframe=normalized, mapping=mapping)
