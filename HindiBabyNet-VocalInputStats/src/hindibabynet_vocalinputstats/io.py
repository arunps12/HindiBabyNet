"""I/O helpers for deterministic dataset building and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


EXPECTED_VTC_FILENAME = "rttm.csv"


@dataclass(frozen=True)
class ParticipantMatch:
    original_par_id: str
    source_id: str


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_csv(path: str | Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_csv(dataframe: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    dataframe.to_csv(output_path, index=False)
    return output_path


def resolve_participant_match(original_par_id: object) -> ParticipantMatch:
    source_id = str(original_par_id).strip()
    return ParticipantMatch(original_par_id=source_id, source_id=source_id)


def find_audio_file(audio_root: str | Path, participant_source_id: str, audio_extensions: Sequence[str]) -> Path | None:
    participant_root = Path(audio_root) / participant_source_id
    candidates: list[Path] = []
    if participant_root.is_file() and participant_root.suffix in set(audio_extensions):
        return participant_root
    if participant_root.is_dir():
        for extension in audio_extensions:
            candidates.extend(sorted(participant_root.rglob(f"*{extension}")))
    else:
        parent = Path(audio_root)
        for extension in audio_extensions:
            direct_candidate = parent / f"{participant_source_id}{extension}"
            if direct_candidate.exists():
                candidates.append(direct_candidate)
    return candidates[0] if candidates else None


def find_vtc_csv(vtc_output_root: str | Path, participant_source_id: str) -> Path | None:
    vtc_path = Path(vtc_output_root) / participant_source_id / EXPECTED_VTC_FILENAME
    return vtc_path if vtc_path.exists() else None


def load_vtc_csv(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    missing = {"uid", "start_time_s", "duration_s", "label"}.difference(dataframe.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"VTC CSV is missing expected columns: {missing_text}")
    return dataframe


def write_validation_report(records: Sequence[dict[str, object]], path: str | Path) -> Path:
    columns = ["participant_id", "issue_type", "message"]
    dataframe = pd.DataFrame(records, columns=columns)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=columns, kind="stable", na_position="last")
    return write_csv(dataframe, path)


def write_dataset_build_report(lines: Iterable[str], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    normalized = [line.rstrip() for line in lines]
    output_path.write_text("\n".join(normalized) + "\n", encoding="utf-8")
    return output_path