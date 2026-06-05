"""Run questionnaire-specific and combined EDA reports and figures for the Hindi CDI pipeline."""

from __future__ import annotations

import pandas as pd

from hindibabynet_cdi.eda_tables import build_eda_markdown_report, build_eda_tables
from hindibabynet_cdi.config import load_config
from hindibabynet_cdi.plotting import save_bar_chart, save_heatmap, save_histogram, save_horizontal_bar_chart, save_line_plot
from hindibabynet_cdi.scoring import build_scoring_outputs


def _subset(dataframe: pd.DataFrame, questionnaire: str) -> pd.DataFrame:
	if dataframe.empty or "questionnaire" not in dataframe.columns:
		return dataframe.copy()
	return dataframe[dataframe["questionnaire"] == questionnaire].copy()


def _age_means(dataframe: pd.DataFrame, value_column: str) -> pd.DataFrame:
	if dataframe.empty:
		return pd.DataFrame(columns=["age_bin", value_column])
	return dataframe.groupby("age_bin", dropna=False)[value_column].mean().reset_index()


def _top_words(word_level: pd.DataFrame, score_column: str) -> pd.DataFrame:
	if word_level.empty:
		return pd.DataFrame(columns=["word", "word_english", score_column])
	return (
		word_level.groupby(["word", "word_english"], dropna=False)[score_column]
		.mean()
		.reset_index()
		.sort_values(by=score_column, ascending=False)
		.head(20)
	)


def _word_curves(word_level: pd.DataFrame, score_column: str) -> pd.DataFrame:
	if word_level.empty:
		return pd.DataFrame(columns=["age_bin", "word_english", score_column])
	top_words = (
		word_level.groupby("word_english", dropna=False)[score_column]
		.mean()
		.sort_values(ascending=False)
		.head(10)
		.index.tolist()
	)
	return (
		word_level[word_level["word_english"].isin(top_words)]
		.groupby(["age_bin", "word_english"], dropna=False)[score_column]
		.mean()
		.reset_index()
	)


def _write_report(report_dir, filename: str, title: str, *, outputs: dict[str, pd.DataFrame], questionnaire: str | None) -> None:
	tables = build_eda_tables(
		participant_linkage=outputs["participant_linkage"],
		participant_metadata=outputs["participant_metadata"],
		participant_scores=outputs["cdi_combined_participant_scores"],
		category_scores=outputs["cdi_category_scores"],
		word_level_long=outputs["cdi_combined_word_level_long"],
		questionnaire=questionnaire,
	)
	(report_dir / filename).write_text(build_eda_markdown_report(title, tables), encoding="utf-8")


def _generate_questionnaire_figures(outputs: dict[str, pd.DataFrame], figure_dir, questionnaire: str) -> None:
	suffix = questionnaire
	participant_metadata = _subset(outputs["participant_metadata"], questionnaire)
	participant_scores = _subset(outputs["cdi_combined_participant_scores"], questionnaire)
	category_scores = _subset(outputs["cdi_category_scores"], questionnaire)
	word_level = _subset(outputs["cdi_combined_word_level_long"], questionnaire)
	percentiles = _subset(outputs["wordbank_percentile_curves"], questionnaire)
	category_by_age = _subset(outputs["category_frequency_by_age"], questionnaire)

	save_histogram(participant_metadata.get("age_months", pd.Series(dtype=float)), title=f"Age Distribution ({questionnaire})", xlabel="Age in months", output_dir=figure_dir, filename=f"age_histogram_{suffix}.png")
	save_line_plot(_age_means(participant_scores, "production_total"), x="age_bin", y="production_total", title=f"Production Total by Age ({questionnaire})", output_dir=figure_dir, filename=f"production_total_by_age_{suffix}.png", xlabel="Age bin", ylabel="Production total")
	if questionnaire == "8_18":
		save_line_plot(_age_means(participant_scores, "comprehension_total"), x="age_bin", y="comprehension_total", title="Comprehension Total by Age (8-18)", output_dir=figure_dir, filename="comprehension_total_by_age_8_18.png", xlabel="Age bin", ylabel="Comprehension total")
		save_line_plot(_age_means(participant_scores, "comprehension_production_gap"), x="age_bin", y="comprehension_production_gap", title="Comprehension-Production Gap (8-18)", output_dir=figure_dir, filename="comprehension_production_gap_8_18.png", xlabel="Age bin", ylabel="Gap")
		save_line_plot(percentiles, x="age_bin", y="p50_production", title="Production Percentile Curves (8-18)", output_dir=figure_dir, filename="production_percentile_curves_8_18.png", xlabel="Age bin", ylabel="Median production")
		save_line_plot(percentiles, x="age_bin", y="p50_comprehension", title="Comprehension Percentile Curves (8-18)", output_dir=figure_dir, filename="comprehension_percentile_curves_8_18.png", xlabel="Age bin", ylabel="Median comprehension")
		save_bar_chart(category_scores.groupby(["category_english"], dropna=False)["production_proportion"].mean().reset_index(), x="category_english", y="production_proportion", title="Category Production (8-18)", output_dir=figure_dir, filename="category_production_8_18.png")
		save_bar_chart(category_scores.groupby(["category_english"], dropna=False)["comprehension_proportion"].mean().reset_index(), x="category_english", y="comprehension_proportion", title="Category Comprehension (8-18)", output_dir=figure_dir, filename="category_comprehension_8_18.png")
		save_heatmap(category_by_age, index="category_english", columns="age_bin", values="mean_production", title="Category Production Heatmap (8-18)", output_dir=figure_dir, filename="category_age_heatmap_production_8_18.png")
		save_heatmap(category_by_age, index="category_english", columns="age_bin", values="mean_comprehension", title="Category Comprehension Heatmap (8-18)", output_dir=figure_dir, filename="category_age_heatmap_comprehension_8_18.png")
		save_horizontal_bar_chart(_top_words(word_level, "production"), x="production", y="word_english", title="Top 20 Produced Words (8-18)", output_dir=figure_dir, filename="top20_produced_words_8_18.png")
		save_horizontal_bar_chart(_top_words(word_level, "comprehension"), x="comprehension", y="word_english", title="Top 20 Comprehended Words (8-18)", output_dir=figure_dir, filename="top20_comprehended_words_8_18.png")
		save_line_plot(_word_curves(word_level, "production"), x="age_bin", y="production", hue="word_english", title="Word Acquisition Curves (8-18)", output_dir=figure_dir, filename="word_acquisition_curves_8_18.png", xlabel="Age bin", ylabel="Production rate")
	else:
		save_line_plot(_age_means(participant_scores, "production_proportion"), x="age_bin", y="production_proportion", title="Production Proportion by Age (19-36)", output_dir=figure_dir, filename="production_proportion_by_age_19_36.png", xlabel="Age bin", ylabel="Production proportion")
		save_line_plot(percentiles, x="age_bin", y="p50_production", title="Production Percentile Curves (19-36)", output_dir=figure_dir, filename="production_percentile_curves_19_36.png", xlabel="Age bin", ylabel="Median production")
		save_bar_chart(category_scores.groupby(["category_english"], dropna=False)["production_proportion"].mean().reset_index(), x="category_english", y="production_proportion", title="Category Production (19-36)", output_dir=figure_dir, filename="category_production_19_36.png")
		save_heatmap(category_by_age, index="category_english", columns="age_bin", values="mean_production", title="Category Production Heatmap (19-36)", output_dir=figure_dir, filename="category_age_heatmap_production_19_36.png")
		save_horizontal_bar_chart(_top_words(word_level, "production"), x="production", y="word_english", title="Top 20 Produced Words (19-36)", output_dir=figure_dir, filename="top20_produced_words_19_36.png")
		save_line_plot(_word_curves(word_level, "production"), x="age_bin", y="production", hue="word_english", title="Word Acquisition Curves (19-36)", output_dir=figure_dir, filename="word_acquisition_curves_19_36.png", xlabel="Age bin", ylabel="Production rate")


def main() -> None:
	config = load_config()
	outputs = build_scoring_outputs(config=config)
	figure_dir = config.paths.outputs / "figures"
	report_dir = config.paths.outputs / "reports"
	figure_dir.mkdir(parents=True, exist_ok=True)
	report_dir.mkdir(parents=True, exist_ok=True)

	_write_report(report_dir, "eda_8_18_summary.md", "Hindi CDI EDA Summary: 8-18", outputs=outputs, questionnaire="8_18")
	_write_report(report_dir, "eda_19_36_summary.md", "Hindi CDI EDA Summary: 19-36", outputs=outputs, questionnaire="19_36")
	_write_report(report_dir, "eda_combined_summary.md", "Hindi CDI EDA Summary: Combined", outputs=outputs, questionnaire=None)
	_generate_questionnaire_figures(outputs, figure_dir, "8_18")
	_generate_questionnaire_figures(outputs, figure_dir, "19_36")
	print(f"Wrote {report_dir / 'eda_8_18_summary.md'}")
	print(f"Wrote {report_dir / 'eda_19_36_summary.md'}")
	print(f"Wrote {report_dir / 'eda_combined_summary.md'}")


if __name__ == "__main__":
    main()