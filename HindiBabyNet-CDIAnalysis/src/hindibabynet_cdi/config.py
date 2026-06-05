"""Configuration helpers for the Hindi CDI analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def get_repo_root() -> Path:
	"""Return the repository root for the CDI analysis project."""
	return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
	raw_data: Path
	interim_data: Path
	processed_data: Path
	metadata: Path
	outputs: Path


@dataclass(frozen=True)
class RawFiles:
	consent: str
	eligibility: str
	background: str
	cdi_8_18: str
	cdi_19_36: str


@dataclass(frozen=True)
class FormIds:
	consent_form_id: int
	eligibility_form_id: int
	background_form_id: int
	cdi_8_18_form_id: int
	cdi_19_36_form_id: int
	contact_form_id: int


@dataclass(frozen=True)
class AnalysisSettings:
	age_month_divisor: float
	younger_questionnaire: str
	older_questionnaire: str
	younger_age_bins: tuple[str, ...]
	older_age_bins: tuple[str, ...]


@dataclass(frozen=True)
class ProjectConfig:
	repo_root: Path
	config_path: Path
	paths: ProjectPaths
	raw_files: RawFiles
	forms: FormIds
	analysis: AnalysisSettings


def _resolve_path(repo_root: Path, value: str) -> Path:
	path = Path(value)
	if path.is_absolute():
		return path
	return (repo_root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		loaded = yaml.safe_load(handle) or {}
	if not isinstance(loaded, dict):
		raise ValueError(f"Expected mapping at top level of config file: {path}")
	return loaded


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
	"""Load the project config and resolve repo-relative paths."""
	repo_root = get_repo_root()
	path = Path(config_path) if config_path is not None else repo_root / "configs" / "config.yaml"
	if not path.is_absolute():
		path = (repo_root / path).resolve()

	payload = _load_yaml(path)
	path_settings = payload.get("paths", {})
	raw_file_settings = payload.get("raw_files", {})
	form_settings = payload.get("forms", {})
	analysis_settings = payload.get("analysis", {})

	paths = ProjectPaths(
		raw_data=_resolve_path(repo_root, str(path_settings["raw_data"])),
		interim_data=_resolve_path(repo_root, str(path_settings["interim_data"])),
		processed_data=_resolve_path(repo_root, str(path_settings["processed_data"])),
		metadata=_resolve_path(repo_root, str(path_settings["metadata"])),
		outputs=_resolve_path(repo_root, str(path_settings["outputs"])),
	)
	raw_files = RawFiles(
		consent=str(raw_file_settings["consent"]),
		eligibility=str(raw_file_settings["eligibility"]),
		background=str(raw_file_settings["background"]),
		cdi_8_18=str(raw_file_settings["cdi_8_18"]),
		cdi_19_36=str(raw_file_settings["cdi_19_36"]),
	)
	forms = FormIds(
		consent_form_id=int(form_settings["consent_form_id"]),
		eligibility_form_id=int(form_settings["eligibility_form_id"]),
		background_form_id=int(form_settings["background_form_id"]),
		cdi_8_18_form_id=int(form_settings["cdi_8_18_form_id"]),
		cdi_19_36_form_id=int(form_settings["cdi_19_36_form_id"]),
		contact_form_id=int(form_settings["contact_form_id"]),
	)
	analysis = AnalysisSettings(
		age_month_divisor=float(analysis_settings["age_month_divisor"]),
		younger_questionnaire=str(analysis_settings["younger_questionnaire"]),
		older_questionnaire=str(analysis_settings["older_questionnaire"]),
		younger_age_bins=tuple(analysis_settings.get("younger_age_bins", [])),
		older_age_bins=tuple(analysis_settings.get("older_age_bins", [])),
	)
	return ProjectConfig(
		repo_root=repo_root,
		config_path=path,
		paths=paths,
		raw_files=raw_files,
		forms=forms,
		analysis=analysis,
	)