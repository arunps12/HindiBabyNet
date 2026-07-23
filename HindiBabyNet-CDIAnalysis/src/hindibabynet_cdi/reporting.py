from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ProjectConfig
from .norms import (
    DECILES,
    QUARTILES,
    SELECTED_QUANTILES,
    SmoothCurveResult,
    category_proportion_long,
    empirical_norm_table,
    fit_group_mean_curves,
    fit_item_probability_curves,
    fit_mean_curve,
    fit_quantile_curves,
    grouped_empirical_table,
)
from .plotting import (
    plot_category_trajectories,
    plot_group_trajectories,
    plot_item_trajectories,
    plot_overall_trajectory,
    plot_quantile_curves,
    plot_sample_size,
    plot_sex_specific_quantiles,
    plot_score_distribution,
)
from .qc import build_age_counts
from .scoring import ScoringOutputs


@dataclass(frozen=True)
class ReportingOutputs:
    tables: dict[str, pd.DataFrame]
    model_metadata: pd.DataFrame
    figure_paths: dict[str, Path]


FULL_TABLE_QUANTILES = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)


def _measure_age_months(form: str) -> list[int]:
    return list(range(8, 19)) if form == "CDI-I" else list(range(19, 37))


def _max_score(participant_analysis: pd.DataFrame, measure_column: str) -> int:
    return int(pd.to_numeric(participant_analysis[measure_column], errors="coerce").max())


def _selected_item_predictions(item_long: pd.DataFrame, response_column: str, age_months: list[int], n_items: int = 6) -> SmoothCurveResult:
    included = item_long.loc[item_long["included_final"] & item_long["cdi_is_complete_enough"]].copy()
    item_summary = (
        included.groupby(["item_id", "word", "word_english"], dropna=False)[response_column]
        .mean()
        .reset_index(name="mean_response")
    )
    item_summary["distance_from_mid"] = (item_summary["mean_response"] - 0.5).abs()
    selected_ids = item_summary.sort_values(["distance_from_mid", "word_english"]).head(n_items)["item_id"]
    selected = included.loc[included["item_id"].isin(selected_ids)].copy()
    return fit_item_probability_curves(selected, response_column, age_months)


def _augment_group_predictions(predictions: pd.DataFrame, participant_analysis: pd.DataFrame, group_column: str) -> pd.DataFrame:
    counts = (
        participant_analysis.loc[participant_analysis["included_final"]]
        .groupby(group_column, dropna=False)
        .size()
        .reset_index(name="n_obs")
        .rename(columns={group_column: "group"})
    )
    return predictions.merge(counts, on="group", how="left")


def build_reporting_outputs(config: ProjectConfig, scoring_outputs: ScoringOutputs) -> ReportingOutputs:
    tables: dict[str, pd.DataFrame] = {}
    figure_paths: dict[str, Path] = {}
    metadata_frames: list[pd.DataFrame] = []

    report_specs = [
        (
            "cdi1_comprehension",
            scoring_outputs.participant_analysis_cdi1,
            scoring_outputs.cdi1_items_long,
            "comprehension_total",
            "understand",
            "CDI-I comprehension",
            config.outputs.cdi1_tables_dir,
            config.outputs.cdi1_comprehension_figures_dir,
            "CDI-I",
        ),
        (
            "cdi1_production",
            scoring_outputs.participant_analysis_cdi1,
            scoring_outputs.cdi1_items_long,
            "production_total",
            "produce",
            "CDI-I production",
            config.outputs.cdi1_tables_dir,
            config.outputs.cdi1_production_figures_dir,
            "CDI-I",
        ),
        (
            "cdi2_production",
            scoring_outputs.participant_analysis_cdi2,
            scoring_outputs.cdi2_items_long,
            "production_total",
            "produce",
            "CDI-II production",
            config.outputs.cdi2_tables_dir,
            config.outputs.cdi2_production_figures_dir,
            "CDI-II",
        ),
    ]

    for prefix, participant_analysis, item_long, measure_column, item_response_column, label, table_dir, figure_dir, form in report_specs:
        age_months = _measure_age_months(form)
        maximum_score = _max_score(participant_analysis.loc[participant_analysis["included_final"]], measure_column)

        empirical = empirical_norm_table(participant_analysis, measure_column, age_months, quantiles=FULL_TABLE_QUANTILES)
        tables[f"{prefix}_norms"] = empirical
        tables[f"{prefix}_sex"] = grouped_empirical_table(participant_analysis, measure_column, "sex", age_months, SELECTED_QUANTILES)
        tables[f"{prefix}_education"] = grouped_empirical_table(participant_analysis, measure_column, "mother_education", age_months, (0.25, 0.50, 0.75))

        overall_curve = fit_mean_curve(participant_analysis, measure_column, age_months, maximum_score, f"{prefix}_overall")
        quartile_curves = fit_quantile_curves(participant_analysis, measure_column, age_months, QUARTILES, maximum_score, prefix)
        decile_curves = fit_quantile_curves(participant_analysis, measure_column, age_months, DECILES, maximum_score, prefix)
        selected_curves = fit_quantile_curves(participant_analysis, measure_column, age_months, SELECTED_QUANTILES, maximum_score, prefix)
        sex_quantile_frames: list[pd.DataFrame] = []
        sex_meta: list[pd.DataFrame] = []
        for sex in sorted(participant_analysis.loc[participant_analysis["included_final"], "sex"].dropna().unique().tolist()):
            sex_result = fit_quantile_curves(
                participant_analysis.loc[participant_analysis["sex"].eq(sex)].assign(included_final=True),
                measure_column,
                age_months,
                SELECTED_QUANTILES,
                maximum_score,
                f"{prefix}_{sex}",
            )
            sex_pred = sex_result.predictions.copy()
            sex_pred["group"] = sex
            sex_quantile_frames.append(sex_pred)
            sex_meta.append(sex_result.metadata.assign(group=sex))
        sex_quantiles = pd.concat(sex_quantile_frames, ignore_index=True) if sex_quantile_frames else pd.DataFrame()

        education_curves = fit_group_mean_curves(participant_analysis, measure_column, "mother_education", age_months, maximum_score)
        education_predictions = _augment_group_predictions(education_curves.predictions, participant_analysis, "mother_education")

        item_predictions = _selected_item_predictions(item_long, item_response_column, age_months)
        category_long = category_proportion_long(item_long, item_response_column)
        category_curves = fit_group_mean_curves(
            category_long.rename(columns={"category_english": "group", "category_proportion": measure_column}).assign(included_final=True),
            measure_column,
            "group",
            age_months,
            1,
            min_group_n=10,
        )

        metadata_frames.extend(
            [
                overall_curve.metadata.assign(output=prefix),
                quartile_curves.metadata.assign(output=prefix),
                decile_curves.metadata.assign(output=prefix),
                selected_curves.metadata.assign(output=prefix),
                education_curves.metadata.assign(output=prefix),
                item_predictions.metadata.assign(output=prefix),
                category_curves.metadata.assign(output=prefix),
            ]
        )
        metadata_frames.extend(sex_meta)

        plot_overall_trajectory(participant_analysis, overall_curve.predictions, measure_column, f"{label}: overall trajectory", figure_dir / f"{prefix}_overall.png")
        plot_quantile_curves(participant_analysis, quartile_curves.predictions, measure_column, f"{label}: quartiles", figure_dir / f"{prefix}_quartiles.png")
        plot_quantile_curves(participant_analysis, decile_curves.predictions, measure_column, f"{label}: deciles", figure_dir / f"{prefix}_deciles.png")
        plot_quantile_curves(participant_analysis, selected_curves.predictions, measure_column, f"{label}: selected quantiles", figure_dir / f"{prefix}_selected_quantiles.png")
        if not sex_quantiles.empty:
            plot_sex_specific_quantiles(
                participant_analysis,
                sex_quantiles,
                measure_column,
                f"{label}: sex-specific selected quantiles",
                figure_dir / f"{prefix}_sex_selected_quantiles.png",
            )
        plot_group_trajectories(
            education_predictions,
            f"{label}: mother's education trajectories",
            "Vocabulary score",
            figure_dir.parent / "categories" / f"{prefix}_education.png",
        )

        age_counts = build_age_counts(participant_analysis)
        plot_sample_size(age_counts.loc[age_counts["form"].eq(form)], f"{label}: sample size by age", figure_dir / f"{prefix}_sample_size.png")
        plot_score_distribution(participant_analysis, measure_column, f"{label}: score distribution by age", figure_dir / f"{prefix}_score_distribution.png")

        plot_item_trajectories(item_predictions.predictions, f"{label}: representative item trajectories", figure_dir.parent / "items" / f"{prefix}_items.png")
        plot_category_trajectories(category_curves.predictions, f"{label}: category trajectories", figure_dir.parent / "categories" / f"{prefix}_categories.png")

        if item_response_column == "understand":
            figure_paths[f"{prefix}_items"] = figure_dir.parent / "items" / f"{prefix}_items.png"
        else:
            figure_paths[f"{prefix}_items"] = figure_dir.parent / "items" / f"{prefix}_items.png"

        figure_paths[f"{prefix}_overall"] = figure_dir / f"{prefix}_overall.png"
        figure_paths[f"{prefix}_quartiles"] = figure_dir / f"{prefix}_quartiles.png"
        figure_paths[f"{prefix}_deciles"] = figure_dir / f"{prefix}_deciles.png"
        figure_paths[f"{prefix}_selected_quantiles"] = figure_dir / f"{prefix}_selected_quantiles.png"
        figure_paths[f"{prefix}_sex_selected_quantiles"] = figure_dir / f"{prefix}_sex_selected_quantiles.png"
        figure_paths[f"{prefix}_sample_size"] = figure_dir / f"{prefix}_sample_size.png"
        figure_paths[f"{prefix}_score_distribution"] = figure_dir / f"{prefix}_score_distribution.png"
        figure_paths[f"{prefix}_education"] = figure_dir.parent / "categories" / f"{prefix}_education.png"
        figure_paths[f"{prefix}_categories"] = figure_dir.parent / "categories" / f"{prefix}_categories.png"

        tables[f"{prefix}_overall_curve"] = overall_curve.predictions
        tables[f"{prefix}_quartile_curves"] = quartile_curves.predictions
        tables[f"{prefix}_decile_curves"] = decile_curves.predictions
        tables[f"{prefix}_selected_quantile_curves"] = selected_curves.predictions
        tables[f"{prefix}_sex_quantiles"] = sex_quantiles
        tables[f"{prefix}_education_curves"] = education_predictions
        tables[f"{prefix}_item_trajectories"] = item_predictions.predictions
        tables[f"{prefix}_category_trajectories"] = category_curves.predictions

    model_metadata = pd.concat(metadata_frames, ignore_index=True, sort=False) if metadata_frames else pd.DataFrame()
    return ReportingOutputs(tables=tables, model_metadata=model_metadata, figure_paths=figure_paths)