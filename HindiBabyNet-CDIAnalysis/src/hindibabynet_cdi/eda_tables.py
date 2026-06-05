"""EDA summary tables and markdown report helpers for Hindi CDI outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _subset_questionnaire(dataframe: pd.DataFrame, questionnaire: str | None) -> pd.DataFrame:
	if questionnaire is None or dataframe.empty or "questionnaire" not in dataframe.columns:
		return dataframe.copy()
	return dataframe[dataframe["questionnaire"] == questionnaire].copy()


def _count_values(dataframe: pd.DataFrame, column: str, *, value_name: str | None = None) -> pd.DataFrame:
	if dataframe.empty or column not in dataframe.columns:
		return pd.DataFrame(columns=[value_name or column, "n"])
	counts = dataframe[column].fillna("<NA>").astype(str).value_counts(dropna=False).reset_index()
	counts.columns = [value_name or column, "n"]
	return counts


def _numeric_summary(series: pd.Series, prefix: str) -> pd.DataFrame:
	clean = pd.to_numeric(series, errors="coerce").dropna()
	if clean.empty:
		return pd.DataFrame({"metric": [f"{prefix}_n"], "value": [0]})
	return pd.DataFrame(
		{
			"metric": [f"{prefix}_n", f"{prefix}_mean", f"{prefix}_median", f"{prefix}_min", f"{prefix}_max"],
			"value": [int(clean.shape[0]), round(float(clean.mean()), 3), round(float(clean.median()), 3), round(float(clean.min()), 3), round(float(clean.max()), 3)],
		}
	)


def _explode_semicolon_counts(series: pd.Series) -> pd.DataFrame:
	if series.empty:
		return pd.DataFrame(columns=["value", "n"])
	exploded = (
		series.fillna("")
		.astype(str)
		.str.split(";")
		.explode()
		.str.strip()
	)
	exploded = exploded[exploded != ""]
	if exploded.empty:
		return pd.DataFrame(columns=["value", "n"])
	counts = exploded.value_counts().reset_index()
	counts.columns = ["value", "n"]
	return counts


def build_eda_tables(
	*,
	participant_linkage: pd.DataFrame,
	participant_metadata: pd.DataFrame,
	participant_scores: pd.DataFrame,
	category_scores: pd.DataFrame,
	word_level_long: pd.DataFrame,
	questionnaire: str | None = None,
) -> dict[str, pd.DataFrame]:
	linkage = _subset_questionnaire(participant_linkage, questionnaire)
	metadata = _subset_questionnaire(participant_metadata, questionnaire)
	scores = _subset_questionnaire(participant_scores, questionnaire)
	categories = _subset_questionnaire(category_scores, questionnaire)
	word_level = _subset_questionnaire(word_level_long, questionnaire)
	included_metadata = metadata[metadata.get("included_analysis", False) == True].copy() if not metadata.empty else metadata.copy()
	summary = pd.DataFrame(
		{
			"metric": ["raw_participants", "included_participants", "excluded_participants"],
			"value": [
				int(len(linkage)),
				int(linkage.get("included_analysis", pd.Series(dtype=bool)).fillna(False).sum()) if not linkage.empty else 0,
				int((~linkage.get("included_analysis", pd.Series(dtype=bool)).fillna(False)).sum()) if not linkage.empty else 0,
			],
		}
	)
	exclusion_reasons = _explode_semicolon_counts(linkage.loc[linkage.get("included_analysis", pd.Series(dtype=bool)) == False, "exclusion_reason"]) if not linkage.empty and "exclusion_reason" in linkage.columns else pd.DataFrame(columns=["value", "n"])
	age_summary = _numeric_summary(included_metadata.get("age_months", pd.Series(dtype=float)), "age_months")
	age_bin_counts = _count_values(included_metadata, "age_bin")
	sex_distribution = _count_values(included_metadata, "sex_raw")
	maternal_education_distribution = _count_values(included_metadata, "maternal_education_raw")
	father_education_distribution = _count_values(included_metadata, "father_education_raw")
	language_exposure_summary = pd.DataFrame(
		{
			"metric": ["second_language_percent_mean", "third_language_percent_mean"],
			"value": [
				round(float(pd.to_numeric(included_metadata.get("second_language_percent", pd.Series(dtype=float)), errors="coerce").dropna().mean()), 3) if not included_metadata.empty and pd.to_numeric(included_metadata.get("second_language_percent", pd.Series(dtype=float)), errors="coerce").dropna().shape[0] else None,
				round(float(pd.to_numeric(included_metadata.get("third_language_percent", pd.Series(dtype=float)), errors="coerce").dropna().mean()), 3) if not included_metadata.empty and pd.to_numeric(included_metadata.get("third_language_percent", pd.Series(dtype=float)), errors="coerce").dropna().shape[0] else None,
			],
		}
	)
	total_vocabulary_summary = pd.concat(
		[
			_numeric_summary(scores.get("production_total", pd.Series(dtype=float)), "production_total"),
			_numeric_summary(scores.get("comprehension_total", pd.Series(dtype=float)), "comprehension_total"),
		],
		ignore_index=True,
	)
	category_summary = pd.DataFrame(columns=["category", "category_english", "mean_production", "mean_comprehension"])
	if not categories.empty:
		category_summary = (
			categories.groupby(["category", "category_english"], dropna=False)
			.agg(mean_production=("production_proportion", "mean"), mean_comprehension=("comprehension_proportion", "mean"))
			.reset_index()
			.sort_values(by=["mean_production", "mean_comprehension"], ascending=[False, False])
		)
	top_produced_words = pd.DataFrame(columns=["word", "word_english", "category", "category_english", "production_rate"])
	if not word_level.empty:
		top_produced_words = (
			word_level.groupby(["word", "word_english", "category", "category_english"], dropna=False)
			.agg(production_rate=("production", "mean"))
			.reset_index()
			.sort_values(by="production_rate", ascending=False)
			.head(20)
		)
	top_comprehended_words = pd.DataFrame(columns=["word", "word_english", "category", "category_english", "comprehension_rate"])
	if not word_level.empty and questionnaire != "19_36":
		top_comprehended_words = (
			word_level.groupby(["word", "word_english", "category", "category_english"], dropna=False)
			.agg(comprehension_rate=("comprehension", "mean"))
			.reset_index()
			.sort_values(by="comprehension_rate", ascending=False)
			.head(20)
		)
	metadata_warnings = pd.concat(
		[
			_count_values(included_metadata[included_metadata.get("age_quality_flag", "ok") != "ok"], "age_quality_flag", value_name="warning"),
			_count_values(linkage[linkage.get("linkage_quality_flag", "ok") != "ok"], "linkage_quality_flag", value_name="warning") if not linkage.empty and "linkage_quality_flag" in linkage.columns else pd.DataFrame(columns=["warning", "n"]),
		],
		ignore_index=True,
	)
	return {
		"summary": summary,
		"exclusion_reasons": exclusion_reasons,
		"age_summary": age_summary,
		"age_bin_counts": age_bin_counts,
		"sex_distribution": sex_distribution,
		"maternal_education_distribution": maternal_education_distribution,
		"father_education_distribution": father_education_distribution,
		"language_exposure_summary": language_exposure_summary,
		"total_vocabulary_summary": total_vocabulary_summary,
		"category_summary": category_summary,
		"top_produced_words": top_produced_words,
		"top_comprehended_words": top_comprehended_words,
		"metadata_warnings": metadata_warnings,
	}


def build_eda_markdown_report(title: str, tables: dict[str, pd.DataFrame]) -> str:
	lines = [f"# {title}", ""]
	section_order = [
		("Summary", "summary"),
		("Exclusion Reasons", "exclusion_reasons"),
		("Age Summary", "age_summary"),
		("Age Bin Counts", "age_bin_counts"),
		("Sex Distribution", "sex_distribution"),
		("Maternal Education Distribution", "maternal_education_distribution"),
		("Father Education Distribution", "father_education_distribution"),
		("Language Exposure Summary", "language_exposure_summary"),
		("Total Vocabulary Score Summary", "total_vocabulary_summary"),
		("Category Score Summary", "category_summary"),
		("Top Produced Words", "top_produced_words"),
		("Top Comprehended Words", "top_comprehended_words"),
		("Metadata Warnings", "metadata_warnings"),
	]
	for heading, key in section_order:
		frame = tables.get(key, pd.DataFrame())
		lines.extend([f"## {heading}", ""])
		if frame.empty:
			lines.append("- None")
			lines.append("")
			continue
		for _, row in frame.iterrows():
			items = [f"{column}={row[column]}" for column in frame.columns]
			lines.append(f"- {', '.join(items)}")
		lines.append("")
	return "\n".join(lines).strip() + "\n"