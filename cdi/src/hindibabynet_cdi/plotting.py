from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_overall_trajectory(
    participant_analysis: pd.DataFrame,
    smooth_curve: pd.DataFrame,
    measure_column: str,
    title: str,
    path: Path,
) -> None:
    included = participant_analysis.loc[participant_analysis["included_final"]].copy()
    plt.figure(figsize=(8, 5))
    plt.scatter(included["age_months_exact"], included[measure_column], alpha=0.35, color="#8a8a8a", s=24)
    plt.plot(smooth_curve["age_month"], smooth_curve["predicted"], color="#1b4d3e", linewidth=2.5)
    plt.xlabel("Age in months")
    plt.ylabel("Vocabulary score")
    plt.title(title)
    _save_figure(path)


def plot_quantile_curves(
    participant_analysis: pd.DataFrame,
    quantile_predictions: pd.DataFrame,
    measure_column: str,
    title: str,
    path: Path,
) -> None:
    included = participant_analysis.loc[participant_analysis["included_final"]].copy()
    plt.figure(figsize=(8, 5))
    plt.scatter(included["age_months_exact"], included[measure_column], alpha=0.20, color="#b5b5b5", s=18)
    for curve_type, frame in quantile_predictions.groupby("curve_type", dropna=False):
        label = curve_type.split("_p")[-1]
        plt.plot(frame["age_month"], frame["predicted"], linewidth=2, label=f"P{label}")
    plt.xlabel("Age in months")
    plt.ylabel("Vocabulary score")
    plt.title(title)
    plt.legend(title="Quantile", ncol=3, fontsize=8)
    _save_figure(path)


def plot_sex_specific_quantiles(
    participant_analysis: pd.DataFrame,
    group_predictions: pd.DataFrame,
    measure_column: str,
    title: str,
    path: Path,
) -> None:
    included = participant_analysis.loc[participant_analysis["included_final"]].copy()
    sexes = [sex for sex in ["female", "male"] if sex in set(included["sex"].dropna())]
    fig, axes = plt.subplots(1, max(1, len(sexes)), figsize=(6 * max(1, len(sexes)), 5), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for axis, sex in zip(axes, sexes):
        sex_frame = included.loc[included["sex"].eq(sex)]
        axis.scatter(sex_frame["age_months_exact"], sex_frame[measure_column], alpha=0.2, color="#b5b5b5", s=18)
        for curve_type, frame in group_predictions.loc[group_predictions["group"].eq(sex)].groupby("curve_type", dropna=False):
            label = curve_type.split("_p")[-1]
            axis.plot(frame["age_month"], frame["predicted"], linewidth=2, label=f"P{label}")
        axis.set_title(sex.title())
        axis.set_xlabel("Age in months")
        axis.set_ylabel("Vocabulary score")
        axis.legend(fontsize=8)
    fig.suptitle(title)
    _save_figure(path)


def plot_group_trajectories(group_predictions: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for group, frame in group_predictions.groupby("group", dropna=False):
        label = f"{group} (n={int(frame['n_obs'].iloc[0])})" if "n_obs" in frame.columns else str(group)
        plt.plot(frame["age_month"], frame["predicted"], linewidth=2, label=label)
    plt.xlabel("Age in months")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    _save_figure(path)


def plot_sample_size(age_counts: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(age_counts["age_month"], age_counts["n"], color="#315c83")
    plt.xlabel("Age month")
    plt.ylabel("Included participant count")
    plt.title(title)
    _save_figure(path)


def plot_score_distribution(participant_analysis: pd.DataFrame, measure_column: str, title: str, path: Path) -> None:
    included = participant_analysis.loc[participant_analysis["included_final"]].copy()
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=included, x="age_month", y=measure_column, color="#d4c5a9")
    sns.stripplot(data=included, x="age_month", y=measure_column, color="#7b3f00", alpha=0.4, size=3)
    plt.xlabel("Age month")
    plt.ylabel("Vocabulary score")
    plt.title(title)
    _save_figure(path)


def plot_item_trajectories(item_predictions: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for _, frame in item_predictions.groupby(["item_id", "word", "word_english"], dropna=False):
        label = str(frame["word_english"].iloc[0])
        linestyle = "--" if bool(frame["insufficient_data"].iloc[0]) else "-"
        plt.plot(frame["age_month"], frame["predicted"], linewidth=2, linestyle=linestyle, label=label)
    plt.xlabel("Age in months")
    plt.ylabel("Estimated proportion")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    _save_figure(path)


def plot_category_trajectories(category_predictions: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for _, frame in category_predictions.groupby(["group"], dropna=False):
        plt.plot(frame["age_month"], frame["predicted"], linewidth=2, label=str(frame["group"].iloc[0]))
    plt.xlabel("Age in months")
    plt.ylabel("Category proportion")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    _save_figure(path)