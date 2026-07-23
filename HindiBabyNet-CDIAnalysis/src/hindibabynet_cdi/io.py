from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cleaning import deduplicate_names, normalize_column_name, normalize_text
from .config import ProjectConfig


METADATA_COLUMNS = {"$submission_id", "$created", "$answer_time_ms", "$forwarded_to_form"}
ROLE_ORDER = ("consent", "eligibility", "background", "cdi_8_18", "cdi_19_36", "contact")


@dataclass(frozen=True)
class RawFileRecord:
    path: Path
    source_group: str


@dataclass(frozen=True)
class FormDetectionResult:
    path: Path
    source_group: str
    detected_role: str | None
    matched_form_ids: tuple[str, ...]
    filename_form_ids: tuple[str, ...]
    matched_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    score: int
    is_ambiguous: bool


@dataclass(frozen=True)
class RawFileComparisonResult:
    source_stem: str
    source_group: str
    preferred_path: Path
    alternate_path: Path | None
    preferred_suffix: str
    alternate_suffix: str | None
    same_row_count: bool | None
    same_column_count: bool | None
    same_normalized_columns: bool | None
    normalized_columns_only_in_preferred: tuple[str, ...]
    normalized_columns_only_in_alternate: tuple[str, ...]


@dataclass
class LoadedForm:
    role: str
    path: Path
    source_group: str
    data: pd.DataFrame
    detection: FormDetectionResult


def _file_priority(path: Path) -> tuple[int, str]:
    suffix_priority = {".xlsx": 0, ".txt": 1}
    return (suffix_priority.get(path.suffix.lower(), 99), path.name)


def discover_raw_files(config: ProjectConfig) -> list[RawFileRecord]:
    records: list[RawFileRecord] = []
    for directory, source_group in (
        (config.paths.raw_data, "raw"),
        (config.paths.personal_info, "personal_info"),
    ):
        if not directory.exists():
            continue
        for extension in ("*.txt", "*.xlsx"):
            for path in sorted(directory.glob(extension)):
                if path.name.startswith("~$"):
                    continue
                records.append(RawFileRecord(path=path, source_group=source_group))
    return records


def _group_raw_file_variants(records: Iterable[RawFileRecord]) -> dict[tuple[str, str], list[RawFileRecord]]:
    grouped: dict[tuple[str, str], list[RawFileRecord]] = {}
    for record in records:
        grouped.setdefault((record.source_group, record.path.stem), []).append(record)
    return grouped


def _extract_form_ids_from_name(path: Path) -> tuple[str, ...]:
    return tuple(part for part in path.stem.split("-") if part.isdigit())


def _candidate_role_scores(columns: Iterable[str], config: ProjectConfig) -> dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]]:
    normalized_columns = {normalize_column_name(column) for column in columns}
    scores: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {}
    for role, defining_columns in config.form_detection_rules.items():
        matched = tuple(column for column in defining_columns if normalize_column_name(column) in normalized_columns)
        missing = tuple(column for column in defining_columns if normalize_column_name(column) not in normalized_columns)
        scores[role] = (len(matched), matched, missing)
    return scores


def read_raw_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".txt":
        data = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    elif path.suffix.lower() == ".xlsx":
        data = pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    data.columns = deduplicate_names([normalize_column_name(column) for column in data.columns])
    return data


def compare_raw_file_variants(config: ProjectConfig) -> pd.DataFrame:
    comparisons: list[dict[str, object]] = []
    grouped = _group_raw_file_variants(discover_raw_files(config))

    for (source_group, source_stem), variant_records in sorted(grouped.items()):
        ordered_records = sorted(variant_records, key=lambda item: _file_priority(item.path))
        preferred_record = ordered_records[0]
        alternate_record = ordered_records[1] if len(ordered_records) > 1 else None
        preferred_frame = read_raw_file(preferred_record.path)
        alternate_frame = read_raw_file(alternate_record.path) if alternate_record is not None else None

        preferred_columns = tuple(preferred_frame.columns)
        alternate_columns = tuple(alternate_frame.columns) if alternate_frame is not None else ()
        preferred_set = set(preferred_columns)
        alternate_set = set(alternate_columns)

        comparisons.append(
            {
                "source_group": source_group,
                "source_stem": source_stem,
                "preferred_file": preferred_record.path.name,
                "preferred_suffix": preferred_record.path.suffix.lower(),
                "alternate_file": alternate_record.path.name if alternate_record is not None else "",
                "alternate_suffix": alternate_record.path.suffix.lower() if alternate_record is not None else "",
                "preferred_rows": len(preferred_frame),
                "alternate_rows": len(alternate_frame) if alternate_frame is not None else pd.NA,
                "same_row_count": len(preferred_frame) == len(alternate_frame) if alternate_frame is not None else pd.NA,
                "preferred_columns": len(preferred_columns),
                "alternate_columns": len(alternate_columns) if alternate_frame is not None else pd.NA,
                "same_column_count": len(preferred_columns) == len(alternate_columns) if alternate_frame is not None else pd.NA,
                "same_normalized_columns": preferred_columns == alternate_columns if alternate_frame is not None else pd.NA,
                "normalized_columns_only_in_preferred": " | ".join(sorted(preferred_set - alternate_set)),
                "normalized_columns_only_in_alternate": " | ".join(sorted(alternate_set - preferred_set)),
            }
        )

    return pd.DataFrame(comparisons).sort_values(["source_group", "source_stem"]).reset_index(drop=True)


def _matched_roles_from_filename(filename_form_ids: tuple[str, ...], config: ProjectConfig) -> list[str]:
    matched_roles: list[str] = []
    form_id_map = {
        "consent": config.form_ids.consent,
        "eligibility": config.form_ids.eligibility,
        "background": config.form_ids.background,
        "cdi_8_18": config.form_ids.cdi_8_18,
        "cdi_19_36": config.form_ids.cdi_19_36,
        "contact": config.form_ids.contact,
    }
    for role, form_id in form_id_map.items():
        if form_id and form_id in filename_form_ids:
            matched_roles.append(role)
    return matched_roles


def detect_form_role(path: Path, data: pd.DataFrame, config: ProjectConfig, source_group: str) -> FormDetectionResult:
    filename_form_ids = _extract_form_ids_from_name(path)
    matched_form_ids = _matched_roles_from_filename(filename_form_ids, config)
    candidate_scores = _candidate_role_scores(data.columns, config)

    scored_roles: list[tuple[int, str, tuple[str, ...], tuple[str, ...]]] = []
    for role in ROLE_ORDER:
        matched_count, matched_columns, missing_columns = candidate_scores[role]
        role_score = matched_count + (100 if role in matched_form_ids else 0)
        scored_roles.append((role_score, role, matched_columns, missing_columns))

    scored_roles.sort(key=lambda item: (item[0], ROLE_ORDER.index(item[1])), reverse=True)
    top_score = scored_roles[0][0]
    top_roles = [item for item in scored_roles if item[0] == top_score and top_score > 0]
    chosen = top_roles[0] if top_roles else None

    if chosen is None:
        return FormDetectionResult(
            path=path,
            source_group=source_group,
            detected_role=None,
            matched_form_ids=tuple(matched_form_ids),
            filename_form_ids=filename_form_ids,
            matched_columns=(),
            missing_columns=(),
            score=0,
            is_ambiguous=False,
        )

    return FormDetectionResult(
        path=path,
        source_group=source_group,
        detected_role=chosen[1],
        matched_form_ids=tuple(matched_form_ids),
        filename_form_ids=filename_form_ids,
        matched_columns=tuple(chosen[2]),
        missing_columns=tuple(chosen[3]),
        score=chosen[0],
        is_ambiguous=len(top_roles) > 1,
    )


def load_detected_forms(config: ProjectConfig) -> list[LoadedForm]:
    candidates: dict[str, list[LoadedForm]] = {}
    for record in discover_raw_files(config):
        data = read_raw_file(record.path)
        detection = detect_form_role(record.path, data, config, record.source_group)
        if detection.detected_role is None:
            continue
        candidates.setdefault(detection.detected_role, []).append(
            LoadedForm(
                role=detection.detected_role,
                path=record.path,
                source_group=record.source_group,
                data=data,
                detection=detection,
            )
        )
    selected: list[LoadedForm] = []
    for role in ROLE_ORDER:
        role_candidates = candidates.get(role, [])
        if not role_candidates:
            continue
        selected.append(sorted(role_candidates, key=lambda item: _file_priority(item.path))[0])
    return selected


def build_form_detection_report(config: ProjectConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped_variants = _group_raw_file_variants(discover_raw_files(config))
    for record in discover_raw_files(config):
        data = read_raw_file(record.path)
        detection = detect_form_role(record.path, data, config, record.source_group)
        preferred_record = sorted(
            grouped_variants[(record.source_group, record.path.stem)],
            key=lambda item: _file_priority(item.path),
        )[0]
        rows.append(
            {
                "file_name": record.path.name,
                "source_group": record.source_group,
                "source_stem": record.path.stem,
                "is_preferred_variant": record.path == preferred_record.path,
                "detected_role": detection.detected_role or "unknown",
                "matched_form_ids": ", ".join(detection.matched_form_ids),
                "filename_form_ids": ", ".join(detection.filename_form_ids),
                "matched_columns": " | ".join(detection.matched_columns),
                "missing_columns": " | ".join(detection.missing_columns),
                "detection_score": detection.score,
                "is_ambiguous": detection.is_ambiguous,
                "n_rows": len(data),
                "n_columns": len(data.columns),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_group", "file_name"]).reset_index(drop=True)


def summarize_response_vocabularies(forms: Iterable[LoadedForm]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for loaded_form in forms:
        value_columns = [column for column in loaded_form.data.columns if column not in METADATA_COLUMNS]
        observed_values = {
            normalize_text(value)
            for column in value_columns[:25]
            for value in loaded_form.data[column].dropna().unique().tolist()
            if normalize_text(value)
        }
        rows.append(
            {
                "role": loaded_form.role,
                "file_name": loaded_form.path.name,
                "sample_values": " | ".join(sorted(observed_values)[:10]),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "file_name"]).reset_index(drop=True)