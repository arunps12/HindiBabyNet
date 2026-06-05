"""Participant metadata, scored CDI outputs, and aggregate summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd

from hindibabynet_cdi.cleaning import (
	SAYS_WORD,
	UNDERSTANDS_AND_SAYS,
	UNDERSTANDS_ONLY,
	assess_age_quality,
	code_education,
	code_sex,
	normalize_response,
	normalize_text,
	parse_age_months,
	parse_datetime_safe,
	translate_language_value,
	translate_residence_value,
)
from hindibabynet_cdi.config import ProjectConfig, load_config
from hindibabynet_cdi.linking import (
	BIRTHDATE_COLUMN,
	CHILD_AGE_GROUP_COLUMN,
	CHILD_AGE_MONTHS_COLUMN,
	build_participant_linkage,
	load_pipeline_forms,
)
from hindibabynet_cdi.metadata import load_word_mapping

SECOND_LANGUAGE_COLUMN = "यदि लागू हो, तो आपके बच्चे की दूसरी भाषा क्या है?"
SECOND_LANGUAGE_PCT_COLUMN = "आपका बच्चा कितने प्रतिशत समय दूसरी भाषा सुनता है?"
THIRD_LANGUAGE_COLUMN = "यदि लागू हो, तो आपके बच्चे की तीसरी भाषा क्या है?"
THIRD_LANGUAGE_PCT_COLUMN = "आपका बच्चा कितने प्रतिशत समय तीसरी भाषा सुनता है?"
MOTHER_EDUCATION_COLUMN = "माता की वर्तमान शिक्षा स्तर:"
FATHER_EDUCATION_COLUMN = "पिता की वर्तमान शिक्षा स्तर:"
MOTHER_GREW_UP_COLUMN = "माँ कहाँ पली-बढ़ी हैं?"
FATHER_GREW_UP_COLUMN = "पिता कहाँ पले-बढ़े हैं?"
RESIDENCE_COLUMN = "आप कहाँ रहते हैं?"
MOTHER_TONGUE_COLUMN = "माँ की मातृभाषा क्या है?"
MOTHER_OTHER_LANGUAGE_COLUMN = "यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया माता की अन्य भाषा बताएं।"
MOTHER_OTHER_LANGUAGE_ALT_COLUMN = "यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया माता की अन्य भाषा बताएं।"
FATHER_TONGUE_COLUMN = "पिता की मातृभाषा क्या है?"
FATHER_OTHER_LANGUAGE_COLUMN = "यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया पिता की अन्य भाषा बताएं।"
FATHER_OTHER_LANGUAGE_ALT_COLUMN = "यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया पिता की अन्य भाषा बताएं।"
MOTHER_CONTACT_PCT_COLUMN = "बच्चा माँ के संपर्क में कितने प्रतिशत समय रहता है?"
FATHER_CONTACT_PCT_COLUMN = "बच्चा पिता के संपर्क में कितने प्रतिशत समय रहता है?"
CHILD_SEX_COLUMN = "बच्चे का लिंग"
OTHER_EDUCATION_COLUMN = "other_education"

PARTICIPANT_METADATA_COLUMNS = [
	"participant_id",
	"questionnaire",
	"age_days",
	"age_months",
	"age_bin",
	"age_months_from_raw",
	"age_discrepancy_flag",
	"age_quality_flag",
	"child_age_months_raw",
	"child_age_group_raw",
	"sex_raw",
	"sex_code",
	"maternal_education_raw",
	"maternal_education_code",
	"father_education_raw",
	"father_education_code",
	"other_education_raw",
	"ses_maternal_education_code",
	"second_language_raw",
	"second_language_english",
	"second_language_percent",
	"third_language_raw",
	"third_language_english",
	"third_language_percent",
	"mother_tongue_mother_raw",
	"mother_tongue_mother_english",
	"mother_other_language_raw",
	"mother_tongue_father_raw",
	"mother_tongue_father_english",
	"father_other_language_raw",
	"mother_contact_percent_bin_raw",
	"father_contact_percent_bin_raw",
	"mother_grew_up_raw",
	"father_grew_up_raw",
	"residence_raw",
	"residence_english",
	"background_submission_id",
	"cdi_submission_id",
	"cdi_form_id",
	"included_analysis",
	"exclusion_reason",
]

WIDE_METADATA_COLUMNS = [
	"participant_id",
	"questionnaire",
	"age_days",
	"age_months",
	"age_bin",
	"sex_raw",
	"sex_code",
	"maternal_education_raw",
	"maternal_education_code",
	"father_education_raw",
	"father_education_code",
	"ses_maternal_education_code",
]

LONG_METADATA_COLUMNS = [
	"participant_id",
	"questionnaire",
	"age_days",
	"age_months",
	"age_bin",
	"sex_raw",
	"sex_code",
	"maternal_education_raw",
	"maternal_education_code",
	"father_education_raw",
	"father_education_code",
	"ses_maternal_education_code",
]

WORD_LEVEL_COLUMNS = [
	*LONG_METADATA_COLUMNS,
	"item_id",
	"word",
	"word_clean",
	"word_english",
	"category",
	"category_english",
	"raw_column_name",
	"response_raw",
	"score_code",
	"comprehension",
	"production",
]


def score_response(questionnaire: str, response: object) -> dict[str, int | None]:
	normalized = normalize_response(response)
	if questionnaire == "8_18":
		if normalized == UNDERSTANDS_ONLY:
			return {"comprehension_score": 1, "production_score": 0}
		if normalized in {UNDERSTANDS_AND_SAYS, SAYS_WORD}:
			return {"comprehension_score": 1, "production_score": 1}
		return {"comprehension_score": 0, "production_score": 0}
	if questionnaire == "19_36":
		if normalized == SAYS_WORD:
			return {"comprehension_score": None, "production_score": 1}
		return {"comprehension_score": None, "production_score": 0}
	raise ValueError(f"Unsupported questionnaire: {questionnaire}")


def _score_code(questionnaire: str, response: object) -> int:
	normalized = normalize_response(response)
	if questionnaire == "8_18":
		if normalized == UNDERSTANDS_ONLY:
			return 1
		if normalized in {UNDERSTANDS_AND_SAYS, SAYS_WORD}:
			return 2
		return 0
	if questionnaire == "19_36":
		return 1 if normalized == SAYS_WORD else 0
	raise ValueError(f"Unsupported questionnaire: {questionnaire}")


def _parse_percent(value: object) -> float | None:
	normalized = normalize_text(value).replace("%", "")
	if normalized == "":
		return None
	try:
		return float(normalized)
	except ValueError:
		return None


def _age_bin(age_months: float | None, questionnaire: str, config: ProjectConfig) -> str:
	if age_months is None:
		return ""
	bins = config.analysis.younger_age_bins if questionnaire == config.analysis.younger_questionnaire else config.analysis.older_age_bins
	for value in bins:
		try:
			start_text, end_text = value.split("_", 1)
			start_value = float(start_text)
			end_value = float(end_text)
		except ValueError:
			continue
		if start_value <= age_months <= end_value + 0.999:
			return value
	return ""


def _set_submission_lookup(dataframe: pd.DataFrame) -> pd.DataFrame:
	prepared = dataframe.copy()
	prepared["_submission_id_norm"] = prepared.get("$submission_id", "").map(normalize_text)
	return prepared.set_index("_submission_id_norm", drop=False)


def _mapping_for_questionnaire(mapping: pd.DataFrame, questionnaire: str) -> pd.DataFrame:
	column_name = "raw_column_8_18" if questionnaire == "8_18" else "raw_column_19_36"
	filtered = mapping[mapping[column_name].map(lambda value: normalize_text(value) != "")].copy()
	filtered["raw_column_name"] = filtered[column_name]
	return filtered.reset_index(drop=True)


def _get_value(row: pd.Series | None, primary_column: str, alternate_column: str | None = None) -> str:
	if row is None:
		return ""
	value = normalize_text(row.get(primary_column, ""))
	if value or alternate_column is None:
		return value
	return normalize_text(row.get(alternate_column, ""))


def _append_age_quality_flag(age_quality_flag: str, questionnaire_mismatch: bool) -> str:
	flags = [flag.strip() for flag in normalize_text(age_quality_flag).split(";") if flag.strip()]
	if questionnaire_mismatch and "questionnaire_age_range_mismatch" not in flags:
		flags.append("questionnaire_age_range_mismatch")
	return "; ".join(flags) if flags else "ok"


def build_participant_metadata(
	*,
	forms: dict[str, pd.DataFrame] | None = None,
	linkage: pd.DataFrame | None = None,
	config: ProjectConfig | None = None,
) -> pd.DataFrame:
	project_config = config or load_config()
	loaded_forms = forms or load_pipeline_forms(project_config)
	participant_linkage = linkage if linkage is not None else build_participant_linkage(loaded_forms, config=project_config)
	background_lookup = _set_submission_lookup(loaded_forms["background"])
	cdi_lookup = {
		"8_18": _set_submission_lookup(loaded_forms["cdi_8_18"]),
		"19_36": _set_submission_lookup(loaded_forms["cdi_19_36"]),
	}
	rows: list[dict[str, Any]] = []
	for _, participant in participant_linkage.iterrows():
		questionnaire = normalize_text(participant.get("questionnaire", ""))
		background_id = normalize_text(participant.get("background_submission_id", ""))
		cdi_id = normalize_text(participant.get("cdi_submission_id", ""))
		background_row = background_lookup.loc[background_id] if background_id and background_id in background_lookup.index else None
		questionnaire_lookup = cdi_lookup.get(questionnaire)
		cdi_row = questionnaire_lookup.loc[cdi_id] if questionnaire_lookup is not None and cdi_id and cdi_id in questionnaire_lookup.index else None
		birthdate = None if background_row is None else parse_datetime_safe(background_row.get(BIRTHDATE_COLUMN, ""))
		background_created = None if background_row is None else parse_datetime_safe(background_row.get("$created", ""))
		assessment_date = None if cdi_row is None else parse_datetime_safe(cdi_row.get("$created", ""))
		raw_age_months = None if background_row is None else parse_age_months(background_row.get(CHILD_AGE_MONTHS_COLUMN, ""))
		age_quality = assess_age_quality(
			birthdate=birthdate,
			reference_datetime=assessment_date or background_created,
			raw_age_months=raw_age_months,
			age_month_divisor=project_config.analysis.age_month_divisor,
		)
		questionnaire_age_range_mismatch = bool(participant.get("questionnaire_age_range_mismatch", False))
		age_months_for_bin = age_quality["age_months"]
		if raw_age_months is not None and (age_months_for_bin is None or bool(age_quality["age_discrepancy_flag"])):
			age_months_for_bin = raw_age_months
		maternal_education_code = 99 if background_row is None else code_education(background_row.get(MOTHER_EDUCATION_COLUMN, ""))
		father_education_code = 99 if background_row is None else code_education(background_row.get(FATHER_EDUCATION_COLUMN, ""))
		rows.append(
			{
				"participant_id": participant["participant_id"],
				"questionnaire": questionnaire,
				"age_days": age_quality["age_days"],
				"age_months": age_quality["age_months"],
				"age_bin": _age_bin(age_months_for_bin, questionnaire, project_config) if questionnaire else "",
				"age_months_from_raw": age_quality["age_months_from_raw"],
				"age_discrepancy_flag": age_quality["age_discrepancy_flag"],
				"age_quality_flag": _append_age_quality_flag(str(age_quality["age_quality_flag"]), questionnaire_age_range_mismatch),
				"child_age_months_raw": "" if background_row is None else normalize_text(background_row.get(CHILD_AGE_MONTHS_COLUMN, "")),
				"child_age_group_raw": "" if background_row is None else normalize_text(background_row.get(CHILD_AGE_GROUP_COLUMN, "")),
				"sex_raw": _get_value(background_row, CHILD_SEX_COLUMN),
				"sex_code": 99 if background_row is None else code_sex(background_row.get(CHILD_SEX_COLUMN, "")),
				"maternal_education_raw": _get_value(background_row, MOTHER_EDUCATION_COLUMN),
				"maternal_education_code": maternal_education_code,
				"father_education_raw": _get_value(background_row, FATHER_EDUCATION_COLUMN),
				"father_education_code": father_education_code,
				"other_education_raw": _get_value(background_row, OTHER_EDUCATION_COLUMN),
				"ses_maternal_education_code": maternal_education_code,
				"second_language_raw": _get_value(background_row, SECOND_LANGUAGE_COLUMN),
				"second_language_english": "" if background_row is None else translate_language_value(background_row.get(SECOND_LANGUAGE_COLUMN, "")),
				"second_language_percent": None if background_row is None else _parse_percent(background_row.get(SECOND_LANGUAGE_PCT_COLUMN, "")),
				"third_language_raw": _get_value(background_row, THIRD_LANGUAGE_COLUMN),
				"third_language_english": "" if background_row is None else translate_language_value(background_row.get(THIRD_LANGUAGE_COLUMN, "")),
				"third_language_percent": None if background_row is None else _parse_percent(background_row.get(THIRD_LANGUAGE_PCT_COLUMN, "")),
				"mother_tongue_mother_raw": _get_value(background_row, MOTHER_TONGUE_COLUMN),
				"mother_tongue_mother_english": "" if background_row is None else translate_language_value(background_row.get(MOTHER_TONGUE_COLUMN, "")),
				"mother_other_language_raw": _get_value(background_row, MOTHER_OTHER_LANGUAGE_COLUMN, MOTHER_OTHER_LANGUAGE_ALT_COLUMN),
				"mother_tongue_father_raw": _get_value(background_row, FATHER_TONGUE_COLUMN),
				"mother_tongue_father_english": "" if background_row is None else translate_language_value(background_row.get(FATHER_TONGUE_COLUMN, "")),
				"father_other_language_raw": _get_value(background_row, FATHER_OTHER_LANGUAGE_COLUMN, FATHER_OTHER_LANGUAGE_ALT_COLUMN),
				"mother_contact_percent_bin_raw": _get_value(background_row, MOTHER_CONTACT_PCT_COLUMN),
				"father_contact_percent_bin_raw": _get_value(background_row, FATHER_CONTACT_PCT_COLUMN),
				"mother_grew_up_raw": _get_value(background_row, MOTHER_GREW_UP_COLUMN),
				"father_grew_up_raw": _get_value(background_row, FATHER_GREW_UP_COLUMN),
				"residence_raw": _get_value(background_row, RESIDENCE_COLUMN),
				"residence_english": "" if background_row is None else translate_residence_value(background_row.get(RESIDENCE_COLUMN, "")),
				"background_submission_id": background_id,
				"cdi_submission_id": cdi_id,
				"cdi_form_id": participant.get("cdi_form_id", ""),
				"included_analysis": bool(participant.get("included_analysis", False)),
				"exclusion_reason": participant.get("exclusion_reason", ""),
			}
		)
	return pd.DataFrame(rows, columns=PARTICIPANT_METADATA_COLUMNS)


def build_participant_info(tracking: pd.DataFrame, forms: dict[str, pd.DataFrame], *, config: ProjectConfig | None = None) -> pd.DataFrame:
	return build_participant_metadata(forms=forms, linkage=tracking, config=config)


def build_scored_wide(
	questionnaire: str,
	*,
	forms: dict[str, pd.DataFrame],
	participant_metadata: pd.DataFrame,
	mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	mapping_q = _mapping_for_questionnaire(mapping, questionnaire)
	cdi_lookup = _set_submission_lookup(forms["cdi_8_18" if questionnaire == "8_18" else "cdi_19_36"])
	participants = participant_metadata[(participant_metadata["included_analysis"] == True) & (participant_metadata["questionnaire"] == questionnaire)].copy()
	denominator = int(len(mapping_q))
	raw_rows: list[dict[str, Any]] = []
	safe_rows: list[dict[str, Any]] = []
	for _, participant in participants.iterrows():
		cdi_id = normalize_text(participant.get("cdi_submission_id", ""))
		if cdi_id == "" or cdi_id not in cdi_lookup.index:
			continue
		cdi_row = cdi_lookup.loc[cdi_id]
		base = {column: participant.get(column, "") for column in WIDE_METADATA_COLUMNS}
		raw_scored = dict(base)
		safe_scored = dict(base)
		for _, item in mapping_q.iterrows():
			response = cdi_row.get(item["raw_column_name"], "")
			score_code = _score_code(questionnaire, response)
			raw_scored[item["raw_column_name"]] = score_code
			safe_scored[item["item_id"]] = score_code
		raw_rows.append(raw_scored)
		safe_rows.append(safe_scored)
	return pd.DataFrame(raw_rows), pd.DataFrame(safe_rows)


def build_word_level_long(
	*,
	forms: dict[str, pd.DataFrame] | None = None,
	tracking: pd.DataFrame | None = None,
	mapping: pd.DataFrame | None = None,
	participant_metadata: pd.DataFrame | None = None,
	config: ProjectConfig | None = None,
	questionnaire: str | None = None,
) -> pd.DataFrame:
	project_config = config or load_config()
	loaded_forms = forms or load_pipeline_forms(project_config)
	word_mapping = mapping if mapping is not None else load_word_mapping(config=project_config)
	linkage = tracking if tracking is not None else build_participant_linkage(loaded_forms, config=project_config)
	metadata_df = participant_metadata if participant_metadata is not None else build_participant_metadata(forms=loaded_forms, linkage=linkage, config=project_config)
	questionnaires = [questionnaire] if questionnaire is not None else ["8_18", "19_36"]
	rows: list[dict[str, Any]] = []
	for questionnaire_name in questionnaires:
		mapping_q = _mapping_for_questionnaire(word_mapping, questionnaire_name)
		cdi_lookup = _set_submission_lookup(loaded_forms["cdi_8_18" if questionnaire_name == "8_18" else "cdi_19_36"])
		participants = metadata_df[(metadata_df["included_analysis"] == True) & (metadata_df["questionnaire"] == questionnaire_name)].copy()
		for _, participant in participants.iterrows():
			cdi_id = normalize_text(participant.get("cdi_submission_id", ""))
			if cdi_id == "" or cdi_id not in cdi_lookup.index:
				continue
			cdi_row = cdi_lookup.loc[cdi_id]
			for _, item in mapping_q.iterrows():
				response = cdi_row.get(item["raw_column_name"], "")
				scores = score_response(questionnaire_name, response)
				row = {column: participant.get(column, None) for column in LONG_METADATA_COLUMNS}
				row.update(
					{
						"item_id": item["item_id"],
						"word": item.get("word", ""),
						"word_clean": item.get("word_clean", ""),
						"word_english": item.get("word_english", ""),
						"category": item.get("category", ""),
						"category_english": item.get("category_english", ""),
						"raw_column_name": item["raw_column_name"],
						"response_raw": normalize_text(response),
						"score_code": _score_code(questionnaire_name, response),
						"comprehension": scores["comprehension_score"],
						"production": scores["production_score"],
					}
				)
				rows.append(
					row
				)
	return pd.DataFrame(rows, columns=WORD_LEVEL_COLUMNS)


def aggregate_scores(
	word_level_long: pd.DataFrame,
	participant_info: pd.DataFrame,
	*,
	mapping: pd.DataFrame | None = None,
	config: ProjectConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	project_config = config or load_config()
	word_mapping = mapping if mapping is not None else load_word_mapping(config=project_config)
	participant_info_included = participant_info[participant_info.get("included_analysis", False) == True].copy() if "included_analysis" in participant_info.columns else participant_info.copy()
	questionnaire_denominators = {
		"8_18": int((word_mapping["in_8_18"].astype(str).str.lower() == "true").sum()),
		"19_36": int((word_mapping["in_19_36"].astype(str).str.lower() == "true").sum()),
	}
	category_denominators_rows: list[dict[str, Any]] = []
	for questionnaire_name in ["8_18", "19_36"]:
		mapping_q = _mapping_for_questionnaire(word_mapping, questionnaire_name)
		for (category, category_english), frame in mapping_q.groupby(["category", "category_english"], dropna=False):
			category_denominators_rows.append(
				{
					"questionnaire": questionnaire_name,
					"category": category,
					"category_english": category_english,
					"category_denominator": int(frame["item_id"].nunique()),
				}
			)
	category_denominators = pd.DataFrame(category_denominators_rows)
	participant_scores = (
		word_level_long.groupby(["participant_id", "questionnaire"], dropna=False)
		.agg(
			production_total=("production", "sum"),
			comprehension_total=("comprehension", "sum"),
		)
		.reset_index()
	)
	participant_scores["n_words_inventory"] = participant_scores["questionnaire"].map(questionnaire_denominators).fillna(0).astype(int)
	participant_scores["production_proportion"] = participant_scores.apply(lambda row: row["production_total"] / row["n_words_inventory"] if row["n_words_inventory"] else 0.0, axis=1)
	participant_scores["comprehension_proportion"] = participant_scores.apply(lambda row: row["comprehension_total"] / row["n_words_inventory"] if row["questionnaire"] == "8_18" and row["n_words_inventory"] else pd.NA, axis=1)
	participant_scores["comprehension_production_gap"] = participant_scores.apply(lambda row: row["comprehension_total"] - row["production_total"] if row["questionnaire"] == "8_18" else pd.NA, axis=1)
	participant_scores = participant_info_included.merge(participant_scores, on=["participant_id", "questionnaire"], how="inner")
	participant_scores = participant_scores[
		[
			"participant_id",
			"questionnaire",
			"age_months",
			"age_bin",
			"sex_raw",
			"sex_code",
			"maternal_education_code",
			"father_education_code",
			"ses_maternal_education_code",
			"comprehension_total",
			"production_total",
			"comprehension_proportion",
			"production_proportion",
			"comprehension_production_gap",
			"n_words_inventory",
		]
	]
	category_scores = (
		word_level_long.groupby(["participant_id", "questionnaire", "category", "category_english"], dropna=False)
		.agg(production_score=("production", "sum"), comprehension_score=("comprehension", "sum"))
		.reset_index()
		.merge(category_denominators, on=["questionnaire", "category", "category_english"], how="left")
	)
	category_scores["n_words_category"] = category_scores["category_denominator"].fillna(0).astype(int)
	category_scores["production_proportion"] = category_scores.apply(lambda row: row["production_score"] / row["n_words_category"] if row["n_words_category"] else 0.0, axis=1)
	category_scores["comprehension_proportion"] = category_scores.apply(lambda row: row["comprehension_score"] / row["n_words_category"] if row["questionnaire"] == "8_18" and row["n_words_category"] else pd.NA, axis=1)
	category_scores = participant_info_included[
		[
			"participant_id",
			"questionnaire",
			"age_months",
			"age_bin",
			"sex_raw",
			"sex_code",
			"maternal_education_code",
			"father_education_code",
			"ses_maternal_education_code",
		]
	].merge(category_scores, on=["participant_id", "questionnaire"], how="inner")
	category_scores = category_scores[
		[
			"participant_id",
			"questionnaire",
			"age_months",
			"age_bin",
			"sex_raw",
			"sex_code",
			"maternal_education_code",
			"father_education_code",
			"ses_maternal_education_code",
			"category",
			"category_english",
			"n_words_category",
			"comprehension_score",
			"production_score",
			"comprehension_proportion",
			"production_proportion",
		]
	]
	master_dataset = participant_info_included.merge(participant_scores, on=["participant_id", "questionnaire"], how="inner")
	return participant_scores, category_scores, master_dataset



def _percentile_value(series: pd.Series, quantile: float) -> float | None:
	clean = pd.to_numeric(series, errors="coerce").dropna()
	if clean.empty:
		return None
	return float(clean.quantile(quantile))


def build_wordbank_tables(
	word_level_long: pd.DataFrame,
	*,
	participant_scores: pd.DataFrame | None = None,
	mapping: pd.DataFrame | None = None,
	config: ProjectConfig | None = None,
) -> dict[str, pd.DataFrame]:
	project_config = config or load_config()
	word_mapping = mapping if mapping is not None else load_word_mapping(config=project_config)
	participant_scores_df = participant_scores if participant_scores is not None else pd.DataFrame()
	wordbank_word_by_age = (
		word_level_long.groupby(["questionnaire", "age_bin", "word", "word_clean", "word_english", "category", "category_english"], dropna=False)
		.agg(
			n_children=("participant_id", "nunique"),
			production_rate=("production", "mean"),
			comprehension_rate=("comprehension", "mean"),
		)
		.reset_index()
	)
	wordbank_category_by_age = (
		word_level_long.groupby(["questionnaire", "age_bin", "category", "category_english"], dropna=False)
		.agg(
			n_children=("participant_id", "nunique"),
			mean_production=("production", "mean"),
			mean_comprehension=("comprehension", "mean"),
		)
		.reset_index()
	)
	word_frequency_overall = (
		word_level_long.groupby(["questionnaire", "item_id", "word", "word_clean", "word_english", "category", "category_english"], dropna=False)
		.agg(
			n_children=("participant_id", "nunique"),
			production_rate=("production", "mean"),
			comprehension_rate=("comprehension", "mean"),
		)
		.reset_index()
	)
	word_frequency_by_age = (
		word_level_long.groupby(["questionnaire", "age_bin", "item_id", "word", "word_clean", "word_english", "category", "category_english"], dropna=False)
		.agg(
			n_children=("participant_id", "nunique"),
			production_rate=("production", "mean"),
			comprehension_rate=("comprehension", "mean"),
		)
		.reset_index()
	)
	category_frequency_by_age = wordbank_category_by_age.copy()
	if participant_scores_df.empty:
		wordbank_age_summary = pd.DataFrame(columns=["questionnaire", "age_bin", "n_children", "mean_age_months", "mean_production", "mean_comprehension"])
		wordbank_percentile_curves = pd.DataFrame(columns=["questionnaire", "age_bin", "p10_production", "p25_production", "p50_production", "p75_production", "p90_production", "p10_comprehension", "p25_comprehension", "p50_comprehension", "p75_comprehension", "p90_comprehension"])
	else:
		wordbank_age_summary = (
			participant_scores_df.groupby(["questionnaire", "age_bin"], dropna=False)
			.agg(
				n_children=("participant_id", "nunique"),
				mean_age_months=("age_months", "mean"),
				mean_production=("production_proportion", "mean"),
				mean_comprehension=("comprehension_proportion", "mean"),
			)
			.reset_index()
		)
		percentile_rows: list[dict[str, object]] = []
		for (questionnaire, age_bin), frame in participant_scores_df.groupby(["questionnaire", "age_bin"], dropna=False):
			percentile_rows.append(
				{
					"questionnaire": questionnaire,
					"age_bin": age_bin,
					"p10_production": _percentile_value(frame["production_total"], 0.10),
					"p25_production": _percentile_value(frame["production_total"], 0.25),
					"p50_production": _percentile_value(frame["production_total"], 0.50),
					"p75_production": _percentile_value(frame["production_total"], 0.75),
					"p90_production": _percentile_value(frame["production_total"], 0.90),
					"p10_comprehension": _percentile_value(frame["comprehension_total"], 0.10) if questionnaire == "8_18" else None,
					"p25_comprehension": _percentile_value(frame["comprehension_total"], 0.25) if questionnaire == "8_18" else None,
					"p50_comprehension": _percentile_value(frame["comprehension_total"], 0.50) if questionnaire == "8_18" else None,
					"p75_comprehension": _percentile_value(frame["comprehension_total"], 0.75) if questionnaire == "8_18" else None,
					"p90_comprehension": _percentile_value(frame["comprehension_total"], 0.90) if questionnaire == "8_18" else None,
				}
			)
		wordbank_percentile_curves = pd.DataFrame(percentile_rows)
	shared_item_ids = set(
		word_mapping[
			(word_mapping["in_8_18"].astype(str).str.lower() == "true")
			& (word_mapping["in_19_36"].astype(str).str.lower() == "true")
		]["item_id"].tolist()
	)
	shared_long = word_level_long[word_level_long["item_id"].isin(shared_item_ids)].copy()
	shared_word_production_by_age = (
		shared_long.groupby(["age_bin", "questionnaire"], dropna=False)
		.agg(
			n_children=("participant_id", "nunique"),
			n_shared_words=("item_id", "nunique"),
			mean_shared_word_production=("production", "mean"),
			production_rate=("production", "mean"),
		)
		.reset_index()
	)
	return {
		"wordbank_age_summary": wordbank_age_summary,
		"wordbank_word_by_age": wordbank_word_by_age,
		"wordbank_category_by_age": wordbank_category_by_age,
		"wordbank_percentile_curves": wordbank_percentile_curves,
		"word_frequency_overall": word_frequency_overall,
		"word_frequency_by_age": word_frequency_by_age,
		"category_frequency_by_age": category_frequency_by_age,
		"shared_word_production_by_age": shared_word_production_by_age,
		"wordbank_item_summary": word_frequency_overall,
		"wordbank_age_bin_summary": wordbank_age_summary,
	}


def build_scoring_outputs(*, config: ProjectConfig | None = None) -> dict[str, pd.DataFrame]:
	project_config = config or load_config()
	forms = load_pipeline_forms(project_config)
	linkage = build_participant_linkage(forms, config=project_config)
	mapping = load_word_mapping(config=project_config)
	participant_metadata = build_participant_metadata(forms=forms, linkage=linkage, config=project_config)
	wide_8_18, wide_8_18_safe = build_scored_wide("8_18", forms=forms, participant_metadata=participant_metadata, mapping=mapping)
	wide_19_36, wide_19_36_safe = build_scored_wide("19_36", forms=forms, participant_metadata=participant_metadata, mapping=mapping)
	long_8_18 = build_word_level_long(forms=forms, tracking=linkage, mapping=mapping, participant_metadata=participant_metadata, config=project_config, questionnaire="8_18")
	long_19_36 = build_word_level_long(forms=forms, tracking=linkage, mapping=mapping, participant_metadata=participant_metadata, config=project_config, questionnaire="19_36")
	combined_long = pd.concat([long_8_18, long_19_36], ignore_index=True)
	participant_scores, category_scores, master_dataset = aggregate_scores(combined_long, participant_metadata, mapping=mapping, config=project_config)
	wordbank_tables = build_wordbank_tables(combined_long, participant_scores=participant_scores, mapping=mapping, config=project_config)
	return {
		"participant_linkage": linkage,
		"participant_metadata": participant_metadata,
		"cdi_8_18_scored_wide": wide_8_18,
		"cdi_8_18_scored_wide_safe": wide_8_18_safe,
		"cdi_8_18_scored_wide_safe_columns": wide_8_18_safe,
		"cdi_19_36_scored_wide": wide_19_36,
		"cdi_19_36_scored_wide_safe": wide_19_36_safe,
		"cdi_19_36_scored_wide_safe_columns": wide_19_36_safe,
		"cdi_8_18_word_level_long": long_8_18,
		"cdi_19_36_word_level_long": long_19_36,
		"cdi_combined_word_level_long": combined_long,
		"cdi_8_18_participant_scores": participant_scores[participant_scores["questionnaire"] == "8_18"].reset_index(drop=True),
		"cdi_19_36_participant_scores": participant_scores[participant_scores["questionnaire"] == "19_36"].reset_index(drop=True),
		"cdi_combined_participant_scores": participant_scores,
		"cdi_participant_scores": participant_scores,
		"cdi_category_scores": category_scores,
		"cdi_master_dataset": master_dataset,
		**wordbank_tables,
	}