"""Word mapping generation and duplicate raw-column handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from hindibabynet_cdi.cleaning import normalize_column_name, normalize_text

DUPLICATE_SUFFIX_RE = re.compile(r"\s*\.\d+$")
PUNCTUATION_SPACING_RE = re.compile(r"\s*([/()])\s*")


def normalize_word_for_match(value: object) -> str:
	normalized = normalize_column_name(value)
	normalized = DUPLICATE_SUFFIX_RE.sub("", normalized)
	normalized = PUNCTUATION_SPACING_RE.sub(r"\1", normalized)
	return normalized.casefold()


def clean_word_label(value: object) -> str:
	normalized = normalize_text(value)
	return DUPLICATE_SUFFIX_RE.sub("", normalized)


@dataclass(frozen=True)
class RawWordOccurrence:
	questionnaire: str
	raw_column_name: str
	word_clean: str
	is_duplicate_raw_column: bool


def build_occurrences(questionnaire: str, raw_columns: list[str]) -> list[RawWordOccurrence]:
	normalized_counts: dict[str, int] = {}
	for column in raw_columns:
		word_clean = clean_word_label(column)
		normalized_counts[word_clean] = normalized_counts.get(word_clean, 0) + 1
	return [
		RawWordOccurrence(
			questionnaire=questionnaire,
			raw_column_name=column,
			word_clean=clean_word_label(column),
			is_duplicate_raw_column=normalized_counts.get(clean_word_label(column), 0) > 1,
		)
		for column in raw_columns
	]


def _ensure_reference_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
	prepared = dataframe.copy()
	if "word_clean" not in prepared.columns:
		prepared["word_clean"] = prepared.get("word", "").map(clean_word_label)
	if "questionnaire" not in prepared.columns:
		prepared["questionnaire"] = ""
	return prepared.fillna("")


def load_reference_word_mapping(path: str | Path | None) -> pd.DataFrame:
	if path is None:
		return pd.DataFrame(columns=["word", "word_clean", "word_english", "category", "category_english", "questionnaire"])
	resolved_path = Path(path)
	if not resolved_path.exists():
		return pd.DataFrame(columns=["word", "word_clean", "word_english", "category", "category_english", "questionnaire"])
	dataframe = pd.read_csv(resolved_path, dtype=str, encoding="utf-8-sig").fillna("")
	dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]
	prepared = _ensure_reference_columns(dataframe)
	if "raw_column_8_18" in prepared.columns or "raw_column_19_36" in prepared.columns:
		prepared["questionnaire"] = prepared.apply(
			lambda row: ";".join(
				questionnaire
				for questionnaire, present in (("8_18", normalize_text(row.get("raw_column_8_18", "")) != ""), ("19_36", normalize_text(row.get("raw_column_19_36", "")) != ""))
				if present
			),
			axis=1,
		)
	return prepared[["word", "word_clean", "word_english", "category", "category_english", "questionnaire"]].copy()


def _questionnaire_membership(value: str) -> set[str]:
	normalized = normalize_text(value)
	if normalized == "":
		return set()
	return {part.strip() for part in normalized.split(";") if part.strip()}


def build_word_mapping_table(
	*,
	raw_columns_8_18: list[str],
	raw_columns_19_36: list[str],
	reference_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	occurrences_8_18 = build_occurrences("8_18", raw_columns_8_18)
	occurrences_19_36 = build_occurrences("19_36", raw_columns_19_36)
	remaining: dict[str, list[RawWordOccurrence]] = {
		"8_18": list(occurrences_8_18),
		"19_36": list(occurrences_19_36),
	}
	rows: list[dict[str, object]] = []
	column_rows: list[dict[str, object]] = []
	missing_rows: list[dict[str, object]] = []

	prepared_reference = _ensure_reference_columns(reference_mapping)
	prepared_reference = prepared_reference.reset_index(drop=True)

	def pop_match(questionnaire: str, word_clean: str) -> RawWordOccurrence | None:
		for index, occurrence in enumerate(remaining[questionnaire]):
			if normalize_word_for_match(occurrence.word_clean) == normalize_word_for_match(word_clean):
				return remaining[questionnaire].pop(index)
		return None

	for _, reference_row in prepared_reference.iterrows():
		questionnaires = _questionnaire_membership(str(reference_row.get("questionnaire", "")))
		if not questionnaires:
			questionnaires = {"8_18", "19_36"}
		match_8_18 = pop_match("8_18", str(reference_row.get("word_clean", ""))) if "8_18" in questionnaires else None
		match_19_36 = pop_match("19_36", str(reference_row.get("word_clean", ""))) if "19_36" in questionnaires else None
		if match_8_18 is None and match_19_36 is None:
			continue
		rows.append(
			{
				"word": normalize_text(reference_row.get("word", "")) or (match_8_18.word_clean if match_8_18 is not None else match_19_36.word_clean),
				"word_clean": normalize_text(reference_row.get("word_clean", "")) or (match_8_18.word_clean if match_8_18 is not None else match_19_36.word_clean),
				"word_english": normalize_text(reference_row.get("word_english", "")),
				"category": normalize_text(reference_row.get("category", "")),
				"category_english": normalize_text(reference_row.get("category_english", "")),
				"questionnaire": ";".join(questionnaire for questionnaire, occurrence in (("8_18", match_8_18), ("19_36", match_19_36)) if occurrence is not None),
				"raw_column_8_18": "" if match_8_18 is None else match_8_18.raw_column_name,
				"raw_column_19_36": "" if match_19_36 is None else match_19_36.raw_column_name,
				"is_duplicate_raw_column": bool((match_8_18 is not None and match_8_18.is_duplicate_raw_column) or (match_19_36 is not None and match_19_36.is_duplicate_raw_column)),
				"notes": "",
			}
		)

	for questionnaire, occurrences in remaining.items():
		for occurrence in occurrences:
			rows.append(
				{
					"word": occurrence.word_clean,
					"word_clean": occurrence.word_clean,
					"word_english": "",
					"category": "",
					"category_english": "",
					"questionnaire": questionnaire,
					"raw_column_8_18": occurrence.raw_column_name if questionnaire == "8_18" else "",
					"raw_column_19_36": occurrence.raw_column_name if questionnaire == "19_36" else "",
					"is_duplicate_raw_column": occurrence.is_duplicate_raw_column,
					"notes": "manual_review_required",
				}
			)
			missing_rows.append(
				{
					"word": occurrence.word_clean,
					"word_clean": occurrence.word_clean,
					"questionnaire": questionnaire,
					"raw_column_name": occurrence.raw_column_name,
					"suggested_action": "add_translation_and_category",
				}
			)

	word_mapping = pd.DataFrame(rows)
	if word_mapping.empty:
		word_mapping = pd.DataFrame(columns=["item_id", "word", "word_clean", "word_english", "category", "category_english", "questionnaire", "in_8_18", "in_19_36", "raw_column_8_18", "raw_column_19_36", "is_duplicate_raw_column", "notes"])
	else:
		word_mapping = word_mapping.reset_index(drop=True)
		word_mapping.insert(0, "item_id", [f"item_{index + 1:04d}" for index in range(len(word_mapping))])
		word_mapping["in_8_18"] = word_mapping["raw_column_8_18"].map(lambda value: normalize_text(value) != "")
		word_mapping["in_19_36"] = word_mapping["raw_column_19_36"].map(lambda value: normalize_text(value) != "")

	for _, row in word_mapping.iterrows():
		if normalize_text(row.get("raw_column_8_18", "")) != "":
			column_rows.append({
				"item_id": row["item_id"],
				"questionnaire": "8_18",
				"raw_column_name": row["raw_column_8_18"],
				"word": row["word"],
				"word_clean": row["word_clean"],
			})
		if normalize_text(row.get("raw_column_19_36", "")) != "":
			column_rows.append({
				"item_id": row["item_id"],
				"questionnaire": "19_36",
				"raw_column_name": row["raw_column_19_36"],
				"word": row["word"],
				"word_clean": row["word_clean"],
			})

	column_mapping = pd.DataFrame(column_rows)
	missing_mapping = pd.DataFrame(missing_rows)
	return word_mapping, column_mapping, missing_mapping