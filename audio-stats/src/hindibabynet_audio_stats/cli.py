"""Command hbn-audio-stats: prepare the dataset for the audio statistics."""

from __future__ import annotations

import argparse
from io import StringIO

import pandas as pd
from hbn_paths import new_output

from hindibabynet_audio_stats.build_master_dataset import build_master_dataset
from hindibabynet_audio_stats.config import ProjectConfig, load_config
from hindibabynet_audio_stats.create_long_format import build_long_tables, dataset_report_lines
from hindibabynet_audio_stats.io import write_csv, write_dataset_build_report, write_validation_report


DATASET_FILES = ["final_master.csv", "input_long.csv", "input_output_long.csv"]
AUDIT_FILES = [
    "recording_duration_audit.csv",
    "age_validation.csv",
    "validation_report.csv",
    "dataset_build_report.txt",
]


def check_outputs_are_new(config: ProjectConfig) -> None:
    """Stop when an output file already exists. Output files are never replaced."""
    targets = [config.derived_data_dir / name for name in DATASET_FILES]
    targets += [config.results_dir / name for name in AUDIT_FILES]
    existing = [path for path in targets if path.exists()]
    if existing:
        files = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            f"These output files already exist and are not replaced:\n{files}\n"
            "Move or rename the old files, or write another folder for "
            f"paths.statistical_data in {config.config_path}."
        )


def prepare_dataset(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the participant table and the long tables, then save every file once."""
    check_outputs_are_new(config)

    print("Step 1 of 3: participant table (metadata, recording durations, VTC results)")
    master, duration_audit, age_validation, validation_records = build_master_dataset(config)
    print(f"  {len(master)} participants")

    print("Step 2 of 3: long tables (one row per participant and speaker)")
    # The long tables use the values exactly as they are saved in final_master.csv.
    saved_master = pd.read_csv(StringIO(master.to_csv(index=False)))
    input_long, input_output_long = build_long_tables(saved_master)
    print(f"  {len(input_long)} rows")

    print("Step 3 of 3: saving the files")
    data_dir = config.derived_data_dir
    audit_dir = config.results_dir
    write_csv(master, new_output(data_dir / "final_master.csv"))
    write_csv(input_long, new_output(data_dir / "input_long.csv"))
    write_csv(input_output_long, new_output(data_dir / "input_output_long.csv"))
    write_csv(duration_audit, new_output(audit_dir / "recording_duration_audit.csv"))
    write_csv(age_validation, new_output(audit_dir / "age_validation.csv"))
    write_validation_report(validation_records, new_output(audit_dir / "validation_report.csv"))
    report_lines = dataset_report_lines(config, saved_master, input_long, input_output_long, validation_records)
    write_dataset_build_report(report_lines, new_output(audit_dir / "dataset_build_report.txt"))
    print(f"  Datasets saved in: {data_dir}")
    print(f"  Audit files saved in: {audit_dir}")

    print(
        f"Finished: {len(master)} participants, {len(input_long)} rows in each long table, "
        f"{len(validation_records)} notes in validation_report.csv."
    )
    return master, input_long, input_output_long


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hbn-audio-stats",
        description="Prepare the dataset for the audio statistics.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Default: config.yaml in the project root.",
    )
    args = parser.parse_args()
    try:
        prepare_dataset(load_config(args.config))
    except (FileExistsError, FileNotFoundError) as error:
        raise SystemExit(f"Stopped: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())