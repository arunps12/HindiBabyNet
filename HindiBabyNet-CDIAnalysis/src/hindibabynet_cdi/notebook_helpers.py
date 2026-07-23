from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

from .norms import fit_item_probability_curves


def load_processed_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def item_dropdown_options(item_long: pd.DataFrame) -> list[tuple[str, int]]:
    unique_items = item_long[["item_id", "word", "word_english"]].drop_duplicates().sort_values("word_english")
    return [
        (f"{row.word_english} / {row.word}", int(row.item_id))
        for row in unique_items.itertuples(index=False)
    ]


def representative_items(item_long: pd.DataFrame, response_column: str, n_items: int = 6) -> pd.DataFrame:
    eligible = item_long.loc[item_long["included_final"] & item_long["cdi_is_complete_enough"]].copy()
    summary = (
        eligible.groupby(["item_id", "word", "word_english"], dropna=False)[response_column]
        .mean()
        .reset_index(name="mean_response")
    )
    summary["distance_from_mid"] = (summary["mean_response"] - 0.5).abs()
    return summary.sort_values(["distance_from_mid", "word_english"]).head(n_items)


def item_sample_size_by_age(item_long: pd.DataFrame, item_id: int) -> pd.DataFrame:
    frame = item_long.loc[item_long["item_id"].eq(item_id) & item_long["included_final"]].copy()
    return frame.groupby("age_month", dropna=False).size().reset_index(name="n")


def render_item_widget(item_long: pd.DataFrame, response_column: str, age_months: list[int], title_prefix: str):
    options = item_dropdown_options(item_long)
    dropdown = widgets.Dropdown(options=options, description="Item:", layout=widgets.Layout(width="70%"))
    output = widgets.Output()

    def _render(item_id: int) -> None:
        with output:
            output.clear_output(wait=True)
            selected = item_long.loc[item_long["item_id"].eq(item_id)].copy()
            if selected.empty:
                display(Markdown("No observations available for this item."))
                return
            curves = fit_item_probability_curves(selected, response_column, age_months)
            sample_sizes = item_sample_size_by_age(item_long, item_id)
            word = selected["word"].iloc[0]
            word_english = selected["word_english"].iloc[0]
            insufficient = bool(curves.predictions["insufficient_data"].iloc[0])

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(curves.predictions["age_month"], curves.predictions["predicted"], linewidth=2.5)
            axes[0].set_ylim(0, 1)
            axes[0].set_xlabel("Age month")
            axes[0].set_ylabel("Estimated proportion")
            axes[0].set_title(f"{title_prefix}: {word_english} / {word}")
            axes[1].bar(sample_sizes["age_month"], sample_sizes["n"], color="#315c83")
            axes[1].set_xlabel("Age month")
            axes[1].set_ylabel("Sample size")
            axes[1].set_title("Sample size by age")
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            if insufficient:
                display(Markdown("Insufficient observations for a stable smooth model; showing empirical proportions."))
            display(sample_sizes)

    def _on_change(change: dict) -> None:
        if change.get("name") == "value" and change.get("new") is not None:
            _render(int(change["new"]))

    dropdown.observe(_on_change)
    if options:
        _render(int(options[0][1]))
    return widgets.VBox([dropdown, output])