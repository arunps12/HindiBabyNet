"""Find and read the questionnaire exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cleaning import deduplicate_names, normalize_column_name
from .config import FORM_ROLES, ProjectConfig


@dataclass(frozen=True)
class LoadedForm:
    """The export of one form. not_used: other files of the same form."""

    role: str
    path: Path
    data: pd.DataFrame
    not_used: tuple[Path, ...] = ()


def _file_priority(path: Path) -> tuple[int, str]:
    """An .xlsx file is used before a .txt file, then the first file name."""
    suffix_priority = {".xlsx": 0, ".txt": 1}
    return (suffix_priority.get(path.suffix.lower(), 99), path.name)


def discover_raw_files(folder: Path) -> list[Path]:
    """All .txt and .xlsx files of the folder, the preferred files first."""
    files = [
        path
        for extension in ("*.txt", "*.xlsx")
        for path in folder.glob(extension)
        if not path.name.startswith("~$")
    ]
    return sorted(files, key=_file_priority)


def detect_form_role(path: Path, columns: Iterable[str], config: ProjectConfig) -> str | None:
    """Form of an export, from the form ID in the file name and the column names."""
    ids_in_name = {part for part in path.stem.split("-") if part.isdigit()}
    normalized_columns = {normalize_column_name(column) for column in columns}
    best_role, best_score = None, 0
    for role in FORM_ROLES:
        score = sum(
            normalize_column_name(column) in normalized_columns
            for column in config.form_detection_rules[role]
        )
        if config.form_ids[role] and config.form_ids[role] in ids_in_name:
            score += 100
        if score > 0 and score >= best_score:
            best_role, best_score = role, score
    return best_role


def read_raw_file(path: Path) -> pd.DataFrame:
    """Read one export as text. Column names are cleaned and made unique."""
    if path.suffix.lower() == ".txt":
        data = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    elif path.suffix.lower() == ".xlsx":
        data = pd.read_excel(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    data.columns = deduplicate_names([normalize_column_name(column) for column in data.columns])
    return data


def load_forms(config: ProjectConfig) -> dict[str, LoadedForm]:
    """Find the export of each form and read it once.

    A file with the form ID in its name is used first. A file without a form ID
    is read only when a form is still missing; its form is found from the columns.
    """
    named: dict[str, list[Path]] = {}
    other_files: list[Path] = []
    for path in discover_raw_files(config.raw_dir):
        role = detect_form_role(path, (), config)
        if role is None:
            other_files.append(path)
        else:
            named.setdefault(role, []).append(path)

    forms = {
        role: LoadedForm(role, paths[0], read_raw_file(paths[0]), tuple(paths[1:]))
        for role, paths in named.items()
    }
    for path in other_files:
        if len(forms) == len(FORM_ROLES):
            break
        data = read_raw_file(path)
        role = detect_form_role(path, data.columns, config)
        if role is not None and role not in forms:
            forms[role] = LoadedForm(role, path, data)

    missing = [f"{role} (form ID {config.form_ids[role]})" for role in FORM_ROLES if role not in forms]
    if missing:
        raise FileNotFoundError(
            f"No export file found in {config.raw_dir} for: {', '.join(missing)}. "
            "The export files are .xlsx or .txt files with the form ID in the file name."
        )
    return {role: forms[role] for role in FORM_ROLES}
