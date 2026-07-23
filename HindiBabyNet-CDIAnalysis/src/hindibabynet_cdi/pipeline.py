from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ProjectConfig, load_config
from .eligibility import build_eligibility_outputs
from .item_dictionary import build_master_item_dictionary
from .linking import build_participant_linkage
from .qc import build_age_counts, build_sample_characteristics
from .reporting import build_reporting_outputs
from .scoring import build_scoring_outputs


def _ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_pipeline(config: ProjectConfig | None = None) -> dict[str, Path]:
    config = config or load_config()
    _ensure_directories(
        [
            config.paths.processed_data,
            config.outputs.qc_dir,
            config.outputs.cdi1_tables_dir,
            config.outputs.cdi2_tables_dir,
            config.outputs.combined_tables_dir,
        ]
    )

    linkage_outputs = build_participant_linkage(config)
    eligibility_outputs = build_eligibility_outputs(config)
    dictionary_outputs = build_master_item_dictionary(config)
    scoring_outputs = build_scoring_outputs(config)
    reporting_outputs = build_reporting_outputs(config, scoring_outputs)

    participant_analysis_all = scoring_outputs.participant_analysis_all.copy()
    participant_analysis_cdi1 = scoring_outputs.participant_analysis_cdi1.copy()
    participant_analysis_cdi2 = scoring_outputs.participant_analysis_cdi2.copy()
    cdi_all_items_long = pd.concat(
        [scoring_outputs.cdi1_items_long, scoring_outputs.cdi2_items_long],
        ignore_index=True,
        sort=False,
    )

    sample_characteristics = build_sample_characteristics(participant_analysis_all)
    age_counts = build_age_counts(participant_analysis_all)

    output_paths = {
        "word_mapping_master": config.paths.processed_data / "word_mapping_master.csv",
        "participant_linkage": config.paths.processed_data / "participant_linkage.csv",
        "participant_crosswalk": config.paths.processed_data / "participant_id_crosswalk.csv",
        "participant_analysis_all": config.paths.processed_data / "participant_analysis_all.csv",
        "participant_analysis_cdi1": config.paths.processed_data / "participant_analysis_cdi1.csv",
        "participant_analysis_cdi2": config.paths.processed_data / "participant_analysis_cdi2.csv",
        "cdi1_items_long": config.paths.processed_data / "cdi1_items_long.csv",
        "cdi2_items_long": config.paths.processed_data / "cdi2_items_long.csv",
        "cdi_all_items_long": config.paths.processed_data / "cdi_all_items_long.csv",
        "included_participant_contacts": config.paths.processed_data / "included_participant_contacts.csv",
        "item_mapping_validation": config.paths.processed_data / "item_mapping_validation.csv",
        "unknown_responses": config.outputs.qc_dir / "unknown_responses.csv",
        "completeness_report": config.outputs.qc_dir / "completeness_report.csv",
        "contact_warnings": config.outputs.qc_dir / "contact_warnings.csv",
        "merge_validation": config.outputs.qc_dir / "merge_validation.csv",
        "form_detection": config.outputs.qc_dir / "form_detection_report.csv",
        "file_variant_comparison": config.outputs.qc_dir / "file_variant_comparison.csv",
        "participant_flow": config.outputs.qc_dir / "participant_flow.csv",
        "exclusions": config.outputs.qc_dir / "exclusions.csv",
        "sample_characteristics": config.outputs.qc_dir / "sample_characteristics.csv",
        "age_counts": config.outputs.qc_dir / "age_counts.csv",
        "model_metadata": config.outputs.qc_dir / "model_metadata.csv",
    }

    from .io import build_form_detection_report, compare_raw_file_variants

    _write_csv(dictionary_outputs.master_dictionary, output_paths["word_mapping_master"])
    _write_csv(linkage_outputs.participant_linkage, output_paths["participant_linkage"])
    _write_csv(linkage_outputs.participant_crosswalk, output_paths["participant_crosswalk"])
    _write_csv(participant_analysis_all, output_paths["participant_analysis_all"])
    _write_csv(participant_analysis_cdi1, output_paths["participant_analysis_cdi1"])
    _write_csv(participant_analysis_cdi2, output_paths["participant_analysis_cdi2"])
    _write_csv(scoring_outputs.cdi1_items_long, output_paths["cdi1_items_long"])
    _write_csv(scoring_outputs.cdi2_items_long, output_paths["cdi2_items_long"])
    _write_csv(cdi_all_items_long, output_paths["cdi_all_items_long"])
    _write_csv(scoring_outputs.included_participant_contacts, output_paths["included_participant_contacts"])
    _write_csv(dictionary_outputs.validation_report, output_paths["item_mapping_validation"])
    _write_csv(scoring_outputs.unknown_responses, output_paths["unknown_responses"])
    _write_csv(scoring_outputs.completeness_report, output_paths["completeness_report"])
    _write_csv(linkage_outputs.contact_warnings, output_paths["contact_warnings"])
    _write_csv(linkage_outputs.merge_validation, output_paths["merge_validation"])
    _write_csv(build_form_detection_report(config), output_paths["form_detection"])
    _write_csv(compare_raw_file_variants(config), output_paths["file_variant_comparison"])
    _write_csv(eligibility_outputs.participant_flow, output_paths["participant_flow"])
    _write_csv(eligibility_outputs.exclusions, output_paths["exclusions"])
    _write_csv(sample_characteristics, output_paths["sample_characteristics"])
    _write_csv(age_counts, output_paths["age_counts"])
    _write_csv(reporting_outputs.model_metadata, output_paths["model_metadata"])

    for table_name, frame in reporting_outputs.tables.items():
        if table_name.startswith("cdi1_"):
            target_dir = config.outputs.cdi1_tables_dir
        elif table_name.startswith("cdi2_"):
            target_dir = config.outputs.cdi2_tables_dir
        else:
            target_dir = config.outputs.combined_tables_dir
        table_path = target_dir / f"{table_name}.csv"
        _write_csv(frame, table_path)
        output_paths[table_name] = table_path

    output_paths.update(reporting_outputs.figure_paths)

    return output_paths