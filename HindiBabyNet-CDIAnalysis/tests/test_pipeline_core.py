from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

from hindibabynet_cdi import (
    build_eligibility_outputs,
    build_master_item_dictionary,
    build_reporting_outputs,
    build_scoring_outputs,
    load_config,
)
from hindibabynet_cdi.cleaning import parse_completed_months
from hindibabynet_cdi.io import compare_raw_file_variants, load_detected_forms
from hindibabynet_cdi.linking import build_participant_linkage


def test_parse_completed_months_handles_digits_and_year_labels() -> None:
    assert parse_completed_months("९") == 9
    assert parse_completed_months("1 year") == 12
    assert parse_completed_months("12 months") == 12


def test_detected_forms_use_xlsx_variants() -> None:
    config = load_config()
    forms = load_detected_forms(config)

    assert {form.role for form in forms} == {"consent", "eligibility", "background", "cdi_8_18", "cdi_19_36", "contact"}
    assert all(form.path.suffix.lower() == ".xlsx" for form in forms)


def test_variant_comparison_confirms_matching_txt_xlsx_shapes() -> None:
    config = load_config()
    comparison = compare_raw_file_variants(config)

    assert not comparison.empty
    assert comparison["same_row_count"].dropna().all()
    assert comparison["same_column_count"].dropna().all()


def test_participant_linkage_covers_reference_chain() -> None:
    config = load_config()
    outputs = build_participant_linkage(config)
    linkage = outputs.participant_linkage

    assert len(linkage) == 306
    assert linkage["participant_id"].str.fullmatch(r"P\d{6}").all()
    assert linkage["has_background_link"].all()
    assert linkage["has_eligibility_link"].all()
    assert linkage["has_consent_link"].all()
    assert outputs.merge_validation.loc[outputs.merge_validation["metric"].eq("duplicate_cdi_submission_ids"), "value"].iloc[0] == 0


def test_eligibility_outputs_respect_strict_hindi_threshold_and_age_flags() -> None:
    config = load_config()
    outputs = build_eligibility_outputs(config)
    criteria = outputs.participant_criteria

    assert (criteria.loc[criteria["hindi_percentage"].eq(75), "included_final"] == False).all()
    assert criteria["age_form_match"].notna().any()
    assert {"CDI-I", "CDI-II"} <= set(criteria["submitted_form"])


def test_master_dictionary_reports_validation_issues_explicitly() -> None:
    config = load_config()
    outputs = build_master_item_dictionary(config)

    assert not outputs.master_dictionary.empty
    assert {"item_id", "word", "cdi1", "cdi2", "cdi1_order", "cdi2_order", "active"} <= set(outputs.master_dictionary.columns)
    assert not outputs.validation_report.empty
    assert "normalization_only_match" in set(outputs.validation_report["issue_type"])


def test_scoring_outputs_keep_cdi2_comprehension_missing() -> None:
    config = load_config()
    outputs = build_scoring_outputs(config)

    assert not outputs.participant_analysis_cdi1.empty
    assert not outputs.participant_analysis_cdi2.empty
    assert outputs.participant_analysis_cdi2["comprehension_total"].isna().all()
    assert outputs.cdi2_items_long["understand"].isna().all()
    assert outputs.unknown_responses.empty


def test_reporting_outputs_include_norms_and_figure_paths() -> None:
    config = load_config()
    scoring_outputs = build_scoring_outputs(config)
    reporting_outputs = build_reporting_outputs(config, scoring_outputs)

    assert "cdi1_comprehension_norms" in reporting_outputs.tables
    assert "cdi1_production_norms" in reporting_outputs.tables
    assert "cdi2_production_norms" in reporting_outputs.tables
    assert "cdi1_comprehension_overall" in reporting_outputs.figure_paths
    assert "cdi2_production_sex_selected_quantiles" in reporting_outputs.figure_paths
    assert not reporting_outputs.model_metadata.empty


def test_notebooks_have_metadata_ids_and_execute() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "notebooks" / "01_CDI1_EDA_and_Norms.ipynb",
        repo_root / "notebooks" / "02_CDI2_EDA_and_Norms.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook = nbformat.read(notebook_path, as_version=4)
        assert notebook.cells
        assert all("id" in cell.get("metadata", {}) for cell in notebook.cells)

        client = NotebookClient(notebook, timeout=1200, kernel_name="python3")
        client.execute(cwd=str(repo_root))