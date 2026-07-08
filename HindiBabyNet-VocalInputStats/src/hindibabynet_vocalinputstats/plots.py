"""Create publication-style plots for the vocal input statistics workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hindibabynet_vocalinputstats.config import ProjectConfig, load_config
from hindibabynet_vocalinputstats.io import ensure_directory, read_csv


INPUT_SPEAKERS = ["adult_female", "adult_male", "other_child"]
RATE_COLUMNS = [
    "adult_female_count_hour",
    "adult_male_count_hour",
    "other_child_count_hour",
    "key_child_count_hour",
    "adult_female_duration_hour",
    "adult_male_duration_hour",
    "other_child_duration_hour",
    "key_child_duration_hour",
]
EXPECTED_PLOT_BASENAMES = [
    "age_distribution_histogram",
    "recording_duration_histogram",
    "child_sex_distribution",
    "location_distribution",
    "mother_education_distribution",
    "father_education_distribution",
    "input_count_hour_by_speaker_boxplot",
    "input_duration_hour_by_speaker_boxplot",
    "age_vs_adult_female_count_hour",
    "age_vs_adult_male_count_hour",
    "age_vs_other_child_count_hour",
    "age_vs_key_child_count_hour",
    "key_child_vs_adult_female_count_hour",
    "key_child_vs_adult_male_count_hour",
    "key_child_vs_other_child_count_hour",
    "hourly_variables_correlation_heatmap",
    "input_composition_per_participant_stacked_bar",
    "mean_input_count_hour_by_speaker",
    "mean_input_duration_hour_by_speaker",
]


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")


def _save_figure(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    ensure_directory(output_dir)
    fig.tight_layout()
    fig.savefig(output_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _empty_figure(title: str, xlabel: str = "", ylabel: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _save_histogram(data: pd.Series, output_dir: Path, basename: str, title: str, xlabel: str) -> None:
    clean = pd.to_numeric(data, errors="coerce").dropna()
    if clean.empty:
        _save_figure(_empty_figure(title, xlabel, "Count"), output_dir, basename)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(clean, bins=min(20, max(5, clean.nunique())), color="#355c7d", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    _save_figure(fig, output_dir, basename)


def _save_barplot(series: pd.Series, output_dir: Path, basename: str, title: str, xlabel: str) -> None:
    counts = series.fillna("missing").value_counts(dropna=False)
    if counts.empty:
        _save_figure(_empty_figure(title, xlabel, "Count"), output_dir, basename)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#6c8ead", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, output_dir, basename)


def _save_boxplot(long_df: pd.DataFrame, y: str, output_dir: Path, basename: str, title: str, ylabel: str) -> None:
    clean = long_df.dropna(subset=[y]).copy()
    if clean.empty:
        _save_figure(_empty_figure(title, "Speaker", ylabel), output_dir, basename)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=clean,
        x="speaker",
        y=y,
        hue="speaker",
        order=INPUT_SPEAKERS,
        hue_order=INPUT_SPEAKERS,
        palette="Blues",
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Speaker")
    ax.set_ylabel(ylabel)
    _save_figure(fig, output_dir, basename)


def _save_scatter(master: pd.DataFrame, x: str, y: str, output_dir: Path, basename: str, title: str, xlabel: str, ylabel: str) -> None:
    clean = master[[x, y]].dropna().copy()
    if clean.empty:
        _save_figure(_empty_figure(title, xlabel, ylabel), output_dir, basename)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.regplot(data=clean, x=x, y=y, scatter_kws={"s": 35, "alpha": 0.8}, line_kws={"color": "#c06c84"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_figure(fig, output_dir, basename)


def _save_correlation_heatmap(master: pd.DataFrame, output_dir: Path) -> None:
    clean = master[RATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if clean.dropna(how="all").empty:
        _save_figure(_empty_figure("Correlation heatmap", "Variable", "Variable"), output_dir, "hourly_variables_correlation_heatmap")
        return
    corr = clean.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap for count/hour and duration/hour variables")
    _save_figure(fig, output_dir, "hourly_variables_correlation_heatmap")


def _save_stacked_bar(input_long: pd.DataFrame, output_dir: Path) -> None:
    pivot = input_long.pivot(index="participant_id", columns="speaker", values="input_count_hour")
    pivot = pivot.reindex(columns=INPUT_SPEAKERS)
    if pivot.dropna(how="all").empty:
        _save_figure(_empty_figure("Input composition per participant", "Participant", "Input count/hour"), output_dir, "input_composition_per_participant_stacked_bar")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.fillna(0.0).plot(kind="bar", stacked=True, ax=ax, color=["#355c7d", "#6c8ead", "#c06c84"])
    ax.set_title("Input composition per participant")
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Input count/hour")
    ax.legend(title="Speaker")
    ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, output_dir, "input_composition_per_participant_stacked_bar")


def _save_mean_ci(input_long: pd.DataFrame, value_column: str, output_dir: Path, basename: str, title: str, ylabel: str) -> None:
    clean = input_long[["speaker", value_column]].dropna().copy()
    if clean.empty:
        _save_figure(_empty_figure(title, "Speaker", ylabel), output_dir, basename)
        return
    summary = (
        clean.groupby("speaker", sort=False)[value_column]
        .agg(["mean", "count", "std"])
        .reindex(INPUT_SPEAKERS)
        .reset_index()
    )
    summary["sem"] = summary["std"] / summary["count"].pow(0.5)
    summary["ci95"] = 1.96 * summary["sem"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=summary,
        x="speaker",
        y="mean",
        hue="speaker",
        order=INPUT_SPEAKERS,
        hue_order=INPUT_SPEAKERS,
        palette="Blues",
        legend=False,
        ax=ax,
    )
    ax.errorbar(x=range(len(summary)), y=summary["mean"], yerr=summary["ci95"], fmt="none", ecolor="#222222", capsize=4)
    ax.set_title(title)
    ax.set_xlabel("Speaker")
    ax.set_ylabel(ylabel)
    _save_figure(fig, output_dir, basename)


def generate_plots(config: ProjectConfig) -> list[str]:
    _set_style()
    master = read_csv(config.derived_data_dir / "final_master.csv")
    input_long = read_csv(config.derived_data_dir / "input_long.csv")
    output_dir = config.figures_dir

    _save_histogram(master["age_months"], output_dir, "age_distribution_histogram", "Age distribution", "Age (months)")
    _save_histogram(
        master["recording_duration_hours"],
        output_dir,
        "recording_duration_histogram",
        "Recording duration distribution",
        "Recording duration (hours)",
    )
    _save_barplot(master["child_sex"], output_dir, "child_sex_distribution", "Child sex distribution", "Child sex")
    _save_barplot(master["Location"], output_dir, "location_distribution", "Location distribution", "Location")
    _save_barplot(master["mother_education"], output_dir, "mother_education_distribution", "Mother education distribution", "Mother education")
    _save_barplot(master["father_education"], output_dir, "father_education_distribution", "Father education distribution", "Father education")
    _save_boxplot(input_long, "input_count_hour", output_dir, "input_count_hour_by_speaker_boxplot", "Input count/hour by speaker", "Input count/hour")
    _save_boxplot(input_long, "input_duration_hour", output_dir, "input_duration_hour_by_speaker_boxplot", "Input duration/hour by speaker", "Input duration/hour")
    _save_scatter(master, "age_months", "adult_female_count_hour", output_dir, "age_vs_adult_female_count_hour", "Age vs adult female count/hour", "Age (months)", "Adult female count/hour")
    _save_scatter(master, "age_months", "adult_male_count_hour", output_dir, "age_vs_adult_male_count_hour", "Age vs adult male count/hour", "Age (months)", "Adult male count/hour")
    _save_scatter(master, "age_months", "other_child_count_hour", output_dir, "age_vs_other_child_count_hour", "Age vs other child count/hour", "Age (months)", "Other child count/hour")
    _save_scatter(master, "age_months", "key_child_count_hour", output_dir, "age_vs_key_child_count_hour", "Age vs key child count/hour", "Age (months)", "Key child count/hour")
    _save_scatter(master, "adult_female_count_hour", "key_child_count_hour", output_dir, "key_child_vs_adult_female_count_hour", "Key child vs adult female count/hour", "Adult female count/hour", "Key child count/hour")
    _save_scatter(master, "adult_male_count_hour", "key_child_count_hour", output_dir, "key_child_vs_adult_male_count_hour", "Key child vs adult male count/hour", "Adult male count/hour", "Key child count/hour")
    _save_scatter(master, "other_child_count_hour", "key_child_count_hour", output_dir, "key_child_vs_other_child_count_hour", "Key child vs other child count/hour", "Other child count/hour", "Key child count/hour")
    _save_correlation_heatmap(master, output_dir)
    _save_stacked_bar(input_long, output_dir)
    _save_mean_ci(input_long, "input_count_hour", output_dir, "mean_input_count_hour_by_speaker", "Mean input count/hour by speaker", "Mean input count/hour")
    _save_mean_ci(input_long, "input_duration_hour", output_dir, "mean_input_duration_hour_by_speaker", "Mean input duration/hour by speaker", "Mean input duration/hour")
    return EXPECTED_PLOT_BASENAMES


def run_plots(config_path: str | Path | None = None) -> list[str]:
    config = load_config(config_path)
    return generate_plots(config)