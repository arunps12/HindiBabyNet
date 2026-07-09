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


@dataclass(frozen=True)
class AudioDiscoveryResult:
    selected_path: Path | None
    warning_message: str | None = None


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_table(path: str | Path, **kwargs: object) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(table_path, **kwargs)
    raise ValueError(f"Unsupported tabular input format: {table_path.suffix or '<no suffix>'}")


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


def _participant_folder_name(template: str, participant_source_id: str) -> str:
    return template.format(par_id=participant_source_id)


def _matches_extension(path: Path, audio_extensions: Sequence[str]) -> bool:
    normalized = {extension.lower() for extension in audio_extensions}
    return path.suffix.lower() in normalized


def find_audio_file(
    audio_root: str | Path,
    participant_source_id: str,
    audio_extensions: Sequence[str],
    *,
    recursive: bool = True,
    participant_folder_name: str = "{par_id}",
    prefer_largest_audio_file: bool = True,
) -> AudioDiscoveryResult:
    participant_root = Path(audio_root) / _participant_folder_name(participant_folder_name, participant_source_id)
    candidates: list[Path] = []

    if participant_root.is_file() and _matches_extension(participant_root, audio_extensions):
        candidates = [participant_root]
    elif participant_root.is_dir():
        iterator = participant_root.rglob("*") if recursive else participant_root.glob("*")
        candidates = sorted(
            (candidate for candidate in iterator if candidate.is_file() and _matches_extension(candidate, audio_extensions)),
            key=lambda candidate: candidate.as_posix(),
        )
    else:
        parent = Path(audio_root)
        for extension in audio_extensions:
            direct_candidate = parent / f"{participant_source_id}{extension}"
            if direct_candidate.exists() and direct_candidate.is_file():
                candidates.append(direct_candidate)
        candidates = sorted(candidates, key=lambda candidate: candidate.as_posix())

    if not candidates:
        return AudioDiscoveryResult(selected_path=None)
    if len(candidates) == 1:
        return AudioDiscoveryResult(selected_path=candidates[0])
    if prefer_largest_audio_file:
        selected = max(candidates, key=lambda candidate: (candidate.stat().st_size, candidate.as_posix()))
        return AudioDiscoveryResult(
            selected_path=selected,
            warning_message=f"Multiple audio files were found ({len(candidates)} matches); selected the largest file.",
        )
    return AudioDiscoveryResult(
        selected_path=candidates[0],
        warning_message=f"Multiple audio files were found ({len(candidates)} matches); selected the first sorted file.",
    )


def find_vtc_csv(
    vtc_output_root: str | Path,
    participant_source_id: str,
    *,
    participant_folder_name: str = "{par_id}",
    rttm_csv_name: str = EXPECTED_VTC_FILENAME,
) -> Path | None:
    participant_root = Path(vtc_output_root) / _participant_folder_name(participant_folder_name, participant_source_id)
    vtc_path = participant_root / rttm_csv_name
    if vtc_path.exists():
        return vtc_path
    if not participant_root.exists() or not participant_root.is_dir():
        return None
    matches = sorted(
        (candidate for candidate in participant_root.rglob(rttm_csv_name) if candidate.is_file()),
        key=lambda candidate: candidate.as_posix(),
    )
    return matches[0] if matches else None


def _infer_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        resolved = lower_map.get(candidate.lower())
        if resolved is not None:
            return resolved
    return None


def load_vtc_csv(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    start_column = _infer_column(dataframe.columns, ["start_time_s", "start_sec", "start", "onset"])
    duration_column = _infer_column(dataframe.columns, ["duration_s", "duration_sec", "duration"])
    end_column = _infer_column(dataframe.columns, ["end_time_s", "end_sec", "end"])
    label_column = _infer_column(dataframe.columns, ["label", "speaker", "speaker_type"])
    uid_column = _infer_column(dataframe.columns, ["uid", "recording_id", "participant", "par_id"])

    found_columns = ", ".join(str(column) for column in dataframe.columns)
    if label_column is None:
        raise ValueError(f"VTC CSV is missing an inferable label column. Found columns: {found_columns}")
    if duration_column is None and (start_column is None or end_column is None):
        raise ValueError(
            "VTC CSV is missing inferable timing columns. Expected one of duration_s, duration_sec, duration, "
            "or start_time_s/start_sec/start/onset plus end_time_s/end_sec/end. "
            f"Found columns: {found_columns}"
        )

    normalized = pd.DataFrame(index=dataframe.index)
    normalized["uid"] = dataframe[uid_column] if uid_column is not None else Path(path).stem
    normalized["start_time_s"] = pd.to_numeric(dataframe[start_column], errors="coerce") if start_column is not None else pd.NA
    if duration_column is not None:
        normalized["duration_s"] = pd.to_numeric(dataframe[duration_column], errors="coerce")
    else:
        normalized["duration_s"] = pd.to_numeric(dataframe[end_column], errors="coerce") - pd.to_numeric(
            dataframe[start_column],
            errors="coerce",
        )
    normalized["label"] = dataframe[label_column].astype(str).str.strip()
    return normalized


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