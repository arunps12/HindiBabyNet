"""Command hbn-cdi: process the Hindi CDI questionnaires and save four CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from hbn_paths import new_output

from .columns import CDI1_FORM, CDI2_FORM
from .config import ProjectConfig, load_config
from .eligibility import EXCLUSION_REASONS, evaluate_participant_criteria
from .io import load_forms
from .item_dictionary import build_item_list
from .linking import build_participant_linkage
from .scoring import score_form


# The four output files (each name with ".csv").
OUTPUT_NAMES = ("cdi1_items", "cdi1_summary", "cdi2_items", "cdi2_summary")

FORM_NAMES = {
    "consent": "consent form",
    "eligibility": "eligibility form",
    "background": "background form",
    "cdi_8_18": "CDI-1",
    "cdi_19_36": "CDI-2",
}


def process_cdi(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    """Read the forms, apply the inclusion rules and score the questionnaires.

    Returns the four tables. No file is written.
    """
    print(f"Reading the questionnaire exports in {config.raw_dir}")
    forms = load_forms(config)
    for role, form in forms.items():
        print(f"  {FORM_NAMES[role]}: {form.path.name}")
        if form.not_used:
            other_files = ", ".join(path.name for path in form.not_used)
            print(f"    other files of this form (not used): {other_files}")
    if config.education.father_education_column not in forms["background"].data.columns:
        print("  Note: the background form has no column for the education of the father. It stays empty.")

    linkage = build_participant_linkage(config, forms)
    criteria = evaluate_participant_criteria(linkage, config)
    item_list = build_item_list(config, forms)

    age = config.age
    limit = f"{config.analysis.strict_hindi_threshold:g}"
    tables: dict[str, pd.DataFrame] = {}
    for prefix, title, form_label in (
        ("cdi1", f"CDI-1 ({age.cdi1_min_month} to {age.cdi1_max_month} months)", CDI1_FORM),
        ("cdi2", f"CDI-2 ({age.cdi2_min_month} to {age.cdi2_max_month} months)", CDI2_FORM),
    ):
        items, summary, n_not_recognised = score_form(form_label, criteria, forms, item_list)
        tables[f"{prefix}_items"] = items
        tables[f"{prefix}_summary"] = summary

        submitted = criteria.loc[criteria["submitted_form"].eq(form_label)]
        reasons = submitted.loc[~submitted["included_final"], "primary_exclusion_reason"].value_counts()
        print(title)
        print(f"  submissions found: {len(submitted)}")
        print(f"  not included: {int(reasons.sum())}")
        for reason, text in EXCLUSION_REASONS.items():
            if reason in reasons.index:
                print(f"    {text.format(limit=limit)}: {int(reasons[reason])}")
        print(f"  included: {len(summary)}")
        repeated = summary.loc[summary["n_submissions_for_child"] > 1, "child_id"].nunique()
        duplicates = int(summary["exact_duplicate_of"].ne("").sum())
        print(f"  children with more than one included questionnaire: {repeated}")
        print(f"  exact duplicates of an earlier questionnaire: {duplicates}")
        if n_not_recognised:
            print(f"  answers that were not recognised (no score): {n_not_recognised}")
    return tables


def write_outputs(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> list[Path]:
    """Save the four tables as CSV files. Existing files are never replaced."""
    paths = {name: new_output(Path(output_dir) / f"{name}.csv") for name in OUTPUT_NAMES}
    for name, path in paths.items():
        tables[name].to_csv(path, index=False, encoding="utf-8-sig")
    return list(paths.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hbn-cdi",
        description="Process the Hindi CDI questionnaires and save four CSV files.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Default: config.yaml in the project root.",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        # Stop before any processing when an output file already exists.
        for name in OUTPUT_NAMES:
            new_output(config.output_dir / f"{name}.csv")
        tables = process_cdi(config)
        saved = write_outputs(tables, config.output_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Stopped: {error}")
    print(f"Saved in {config.output_dir}:")
    for path in saved:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
