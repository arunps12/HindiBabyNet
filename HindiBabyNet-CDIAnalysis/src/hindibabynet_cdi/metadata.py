"""Metadata helpers for raw-driven CDI word mapping generation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hindibabynet_cdi.cleaning import normalize_column_name, normalize_text
from hindibabynet_cdi.config import ProjectConfig, load_config
from hindibabynet_cdi.io import get_cdi_item_columns
from hindibabynet_cdi.word_mapping import build_word_mapping_table, load_reference_word_mapping, normalize_word_for_match

EXPECTED_RAW_WORDS_8_18 = 409
EXPECTED_RAW_WORDS_19_36 = 645
EXPECTED_SHARED_WORDS = 383


@dataclass(frozen=True)
class MappingDiagnostics:
	total_rows: int
	shared_words: int
	only_8_18: int
	only_19_36: int
	missing_word_english: int
	missing_category: int
	duplicate_raw_columns: int
	raw_columns_8_18: int
	raw_columns_19_36: int


def _get_config(config: ProjectConfig | None = None) -> ProjectConfig:
	return config if config is not None else load_config()


def load_word_mapping(path: str | Path | None = None, *, config: ProjectConfig | None = None) -> pd.DataFrame:
	resolved_path = Path(path) if path is not None else _get_config(config).paths.metadata / "word_mapping.csv"
	if not resolved_path.exists():
		return pd.DataFrame(columns=["item_id", "word", "word_clean", "word_english", "category", "category_english", "questionnaire", "in_8_18", "in_19_36", "raw_column_8_18", "raw_column_19_36", "is_duplicate_raw_column", "notes", "normalized_word", "questionnaire_list"])
	dataframe = pd.read_csv(resolved_path, dtype=str, encoding="utf-8-sig").fillna("")
	dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]
	if "word_clean" not in dataframe.columns:
		dataframe["word_clean"] = dataframe.get("word", "").map(normalize_text)
	if "questionnaire" not in dataframe.columns:
		dataframe["questionnaire"] = dataframe.apply(
			lambda row: ";".join(questionnaire for questionnaire, present in (("8_18", normalize_text(row.get("raw_column_8_18", "")) != ""), ("19_36", normalize_text(row.get("raw_column_19_36", "")) != "")) if present),
			axis=1,
		)
	if "in_8_18" not in dataframe.columns:
		dataframe["in_8_18"] = dataframe["questionnaire"].map(lambda value: "8_18" in _split_questionnaires(value))
	if "in_19_36" not in dataframe.columns:
		dataframe["in_19_36"] = dataframe["questionnaire"].map(lambda value: "19_36" in _split_questionnaires(value))
	if "raw_column_8_18" not in dataframe.columns:
		dataframe["raw_column_8_18"] = ""
	if "raw_column_19_36" not in dataframe.columns:
		dataframe["raw_column_19_36"] = ""
	if "item_id" not in dataframe.columns:
		dataframe.insert(0, "item_id", [f"legacy_item_{index + 1:04d}" for index in range(len(dataframe))])
	if "is_duplicate_raw_column" not in dataframe.columns:
		dataframe["is_duplicate_raw_column"] = False
	if "notes" not in dataframe.columns:
		dataframe["notes"] = ""
	dataframe["normalized_word"] = dataframe["word_clean"].map(normalize_word_for_match)
	dataframe["questionnaire_list"] = dataframe["questionnaire"].map(_split_questionnaires)
	return dataframe


def _split_questionnaires(value: str) -> list[str]:
	normalized = normalize_text(value)
	if normalized == "":
		return []
	return [part.strip() for part in normalized.split(";") if part.strip()]


def explode_word_mapping_by_questionnaire(dataframe: pd.DataFrame) -> pd.DataFrame:
	prepared = dataframe.copy()
	if "questionnaire_list" not in prepared.columns:
		prepared["questionnaire_list"] = prepared.get("questionnaire", "").map(_split_questionnaires)
	exploded = prepared.explode("questionnaire_list")
	return exploded.rename(columns={"questionnaire_list": "questionnaire_name"}).reset_index(drop=True)


def summarize_word_mapping(dataframe: pd.DataFrame, *, raw_columns_8_18: int = 0, raw_columns_19_36: int = 0) -> MappingDiagnostics:
	if not dataframe.empty and "in_8_18" not in dataframe.columns:
		dataframe = dataframe.copy()
		dataframe["in_8_18"] = dataframe.get("questionnaire", "").map(lambda value: "8_18" in _split_questionnaires(value))
		dataframe["in_19_36"] = dataframe.get("questionnaire", "").map(lambda value: "19_36" in _split_questionnaires(value))
	shared_words = int((dataframe["in_8_18"].astype(str).str.lower() == "true").astype(int).mul((dataframe["in_19_36"].astype(str).str.lower() == "true").astype(int)).sum()) if not dataframe.empty else 0
	only_8_18 = int(((dataframe["in_8_18"].astype(str).str.lower() == "true") & (dataframe["in_19_36"].astype(str).str.lower() != "true")).sum()) if not dataframe.empty else 0
	only_19_36 = int(((dataframe["in_19_36"].astype(str).str.lower() == "true") & (dataframe["in_8_18"].astype(str).str.lower() != "true")).sum()) if not dataframe.empty else 0
	missing_word_english = int((dataframe.get("word_english", "") == "").sum()) if not dataframe.empty else 0
	missing_category = int(((dataframe.get("category", "") == "") | (dataframe.get("category_english", "") == "")).sum()) if not dataframe.empty else 0
	duplicate_raw_columns = int((dataframe.get("is_duplicate_raw_column", False).astype(str).str.lower() == "true").sum()) if not dataframe.empty else 0
	return MappingDiagnostics(
		total_rows=int(len(dataframe)),
		shared_words=shared_words,
		only_8_18=only_8_18,
		only_19_36=only_19_36,
		missing_word_english=missing_word_english,
		missing_category=missing_category,
		duplicate_raw_columns=duplicate_raw_columns,
		raw_columns_8_18=raw_columns_8_18,
		raw_columns_19_36=raw_columns_19_36,
	)


def match_cdi_columns_to_metadata(cdi_columns: list[str], word_mapping: pd.DataFrame, *, questionnaire: str | None = None) -> pd.DataFrame:
	column_mapping_rows: list[dict[str, object]] = []
	for _, row in word_mapping.iterrows():
		if questionnaire in (None, "8_18") and normalize_text(row.get("raw_column_8_18", "")) != "":
			column_mapping_rows.append(
				{
					"column_name": row["raw_column_8_18"],
					"word": row.get("word", ""),
					"word_english": row.get("word_english", ""),
					"category": row.get("category", ""),
					"category_english": row.get("category_english", ""),
					"questionnaire": "8_18",
					"item_id": row.get("item_id", ""),
				}
			)
		if questionnaire in (None, "19_36") and normalize_text(row.get("raw_column_19_36", "")) != "":
			column_mapping_rows.append(
				{
					"column_name": row["raw_column_19_36"],
					"word": row.get("word", ""),
					"word_english": row.get("word_english", ""),
					"category": row.get("category", ""),
					"category_english": row.get("category_english", ""),
					"questionnaire": "19_36",
					"item_id": row.get("item_id", ""),
				}
			)
	column_mapping = pd.DataFrame(column_mapping_rows)
	if questionnaire is not None and not column_mapping.empty:
		column_mapping = column_mapping[column_mapping["questionnaire"] == questionnaire].copy()
	lookup = column_mapping.set_index("column_name") if not column_mapping.empty else pd.DataFrame()
	normalized_lookup = None
	if word_mapping is not None and not word_mapping.empty:
		fallback_mapping = word_mapping.copy()
		if questionnaire is not None:
			fallback_mapping = explode_word_mapping_by_questionnaire(fallback_mapping)
			fallback_mapping = fallback_mapping[fallback_mapping["questionnaire_name"] == questionnaire].copy()
		if "normalized_word" not in fallback_mapping.columns:
			fallback_mapping["normalized_word"] = fallback_mapping.get("word_clean", fallback_mapping.get("word", "")).map(normalize_word_for_match)
		fallback_mapping = fallback_mapping.drop_duplicates(subset=["normalized_word"], keep="first")
		normalized_lookup = fallback_mapping.set_index("normalized_word")
	rows = []
	for column in cdi_columns:
		normalized_column = normalize_word_for_match(column)
		matched = (not column_mapping.empty and column in lookup.index) or (normalized_lookup is not None and normalized_column in normalized_lookup.index)
		row: dict[str, object] = {
			"column_name": column,
			"normalized_column": normalized_column,
			"matched": matched,
		}
		if not column_mapping.empty and column in lookup.index:
			metadata_row = lookup.loc[column]
			if isinstance(metadata_row, pd.DataFrame):
				metadata_row = metadata_row.iloc[0]
			row.update(metadata_row.to_dict())
		elif normalized_lookup is not None and normalized_column in normalized_lookup.index:
			metadata_row = normalized_lookup.loc[normalized_column]
			if isinstance(metadata_row, pd.DataFrame):
				metadata_row = metadata_row.iloc[0]
			row.update(metadata_row.to_dict())
		rows.append(row)
	return pd.DataFrame(rows)


def generate_metadata_outputs(forms: dict[str, pd.DataFrame], *, config: ProjectConfig | None = None) -> dict[str, pd.DataFrame]:
	project_config = config or load_config()
	raw_columns_8_18 = get_cdi_item_columns(forms["cdi_8_18"])
	raw_columns_19_36 = get_cdi_item_columns(forms["cdi_19_36"])
	reference_mapping = load_reference_word_mapping(project_config.paths.metadata / "word_mapping.csv")
	word_mapping, column_mapping, missing_mapping = build_word_mapping_table(
		raw_columns_8_18=raw_columns_8_18,
		raw_columns_19_36=raw_columns_19_36,
		reference_mapping=reference_mapping,
	)
	quality = summarize_word_mapping(word_mapping, raw_columns_8_18=len(raw_columns_8_18), raw_columns_19_36=len(raw_columns_19_36))
	column_mapping = column_mapping.merge(
		word_mapping[["item_id", "word_english", "category", "category_english", "is_duplicate_raw_column", "notes"]],
		on="item_id",
		how="left",
	)
	category_counts = (
		word_mapping.assign(category_display=word_mapping["category_english"].where(word_mapping["category_english"] != "", word_mapping["category"]))
		.groupby("category_display", dropna=False)["item_id"]
		.nunique()
	)
	warnings: list[tuple[str, object]] = []
	if quality.raw_columns_8_18 != EXPECTED_RAW_WORDS_8_18:
		warnings.append(("warning_raw_columns_8_18_expected_409", quality.raw_columns_8_18))
	if quality.raw_columns_19_36 != EXPECTED_RAW_WORDS_19_36:
		warnings.append(("warning_raw_columns_19_36_expected_645", quality.raw_columns_19_36))
	if quality.shared_words != EXPECTED_SHARED_WORDS:
		warnings.append(("warning_shared_words_expected_383", quality.shared_words))
	quality_report = pd.DataFrame(
		{
			"metric": [
				"raw_columns_8_18",
				"raw_columns_19_36",
				"mapped_words",
				"missing_words",
				"duplicate_raw_columns",
				"shared_words",
				"only_8_18",
				"only_19_36",
				"missing_word_english",
				"missing_category",
				*[f"category_count:{category}" for category in category_counts.index.tolist()],
				*[warning[0] for warning in warnings],
			],
			"value": [
				quality.raw_columns_8_18,
				quality.raw_columns_19_36,
				quality.total_rows,
				len(missing_mapping),
				quality.duplicate_raw_columns,
				quality.shared_words,
				quality.only_8_18,
				quality.only_19_36,
				quality.missing_word_english,
				quality.missing_category,
				*category_counts.tolist(),
				*[warning[1] for warning in warnings],
			],
		}
	)
	education_mapping = pd.DataFrame(
		{
			"code": [1, 2, 3, 4, 5, 6, 99],
			"label": ["primary_school", "high_school", "some_college", "bachelor", "master", "other_qualifications", "missing"],
		}
	)
	language_mapping = pd.DataFrame(
		{
			"raw_value": [
				"अंग्रेज़ी",
				"हिंदी",
				"हिंदी + अन्य",
				"हिंदी &#43; अन्य",
				"अन्य कोई भाषा",
				"अन्य भारतीय भाषा",
				"लागू नहीं",
				"शहर",
				"कस्बा",
				"गाँव",
				"मेट्रो शहर/राजधानी",
				"भारत",
				"उत्तर नहीं देना चाहता/चाहती",
			],
			"english": [
				"English",
				"Hindi",
				"Hindi + other",
				"Hindi + other",
				"Other language",
				"Other Indian language",
				"Not applicable",
				"City",
				"Town",
				"Village",
				"Metro/capital city",
				"India",
				"Prefer not to answer",
			],
		}
	)
	sex_mapping = pd.DataFrame(
		{
			"raw_value": ["male", "female", "unknown", "other", ""],
			"code": [1, 2, 99, 99, 99],
		}
	)
	response_mapping = pd.DataFrame(
		{
			"questionnaire": ["8_18", "8_18", "8_18", "19_36", "19_36"],
			"response_raw": ["", "केवल समझता/समझती है", "समझता/समझती है और कहता/कहती है", "", "कहता/कहती है"],
			"score_code": [0, 1, 2, 0, 1],
			"comprehension": [0, 1, 1, pd.NA, pd.NA],
			"production": [0, 0, 1, 0, 1],
		}
	)
	return {
		"word_mapping": word_mapping,
		"column_mapping": column_mapping,
		"education_mapping": education_mapping,
		"language_mapping": language_mapping,
		"sex_mapping": sex_mapping,
		"response_mapping": response_mapping,
		"missing_word_mapping_to_fill": missing_mapping,
		"word_mapping_quality_report": quality_report,
	}


def write_metadata_outputs(outputs: dict[str, pd.DataFrame], *, config: ProjectConfig | None = None) -> dict[str, Path]:
	project_config = config or load_config()
	project_config.paths.metadata.mkdir(parents=True, exist_ok=True)
	(project_config.paths.outputs / "reports").mkdir(parents=True, exist_ok=True)
	paths: dict[str, Path] = {}
	for name, dataframe in outputs.items():
		path = project_config.paths.metadata / f"{name}.csv"
		dataframe.to_csv(path, index=False)
		paths[name] = path
	report_path = project_config.paths.outputs / "reports" / "metadata_quality_report.md"
	quality = outputs["word_mapping_quality_report"]
	lines = ["# Metadata Quality Report", ""]
	for _, row in quality.iterrows():
		lines.append(f"- {row['metric']}: {row['value']}")
	missing_rows = outputs.get("missing_word_mapping_to_fill", pd.DataFrame())
	if not missing_rows.empty:
		lines.extend(["", "## Missing Word Mappings", ""])
		for _, row in missing_rows.iterrows():
			lines.append(f"- {row['questionnaire']} | {row['raw_column_name']} | {row['suggested_action']}")
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	paths["metadata_quality_report"] = report_path
	return paths