"""Command hbn-audio-stats: prepare the dataset for the audio statistics."""

from __future__ import annotations

import argparse
from io import StringIO

import pandas as pd
from hbn_paths import new_output

from hindibabynet_audio_stats.build_master_dataset import build_master_dataset
from hindibabynet_audio_stats.config import ProjectConfig, load_config
from hindibabynet_audio_stats.create_long_format import build_long_tables, build_wide_table, dataset_report_lines
from hindibabynet_audio_stats.io import write_csv, write_dataset_build_report, write_validation_report


WIDE_FILE = "input_output_wide.csv"
DATASET_FILES = ["final_master.csv", "input_long.csv", "input_output_long.csv", WIDE_FILE]
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


def build_wide_table_from_csv(master_csv) -> pd.DataFrame:
    """Wide table from final_master.csv (a path or the CSV text). The numbers are read
    exactly as they are written, so the wide table has the same numbers as final_master.csv."""
    return build_wide_table(pd.read_csv(master_csv, float_precision="round_trip"))


def add_wide_table(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add input_output_wide.csv to an existing dataset. It is built from the saved
    final_master.csv. No other file is read from the recordings or changed."""
    data_dir = config.derived_data_dir
    wide = build_wide_table_from_csv(data_dir / "final_master.csv")
    write_csv(wide, new_output(data_dir / WIDE_FILE))
    print(f"The dataset already exists in: {data_dir}")
    print(f"Only {WIDE_FILE} was missing. It was made from final_master.csv ({len(wide)} participants).")
    print("The other files were not changed.")
    return tuple(pd.read_csv(data_dir / name) for name in DATASET_FILES if name != WIDE_FILE)


def prepare_dataset(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the participant table, the long tables and the wide table, then save every file once."""
    data_dir = config.derived_data_dir
    older_files = [data_dir / name for name in DATASET_FILES if name != WIDE_FILE]
    if all(path.exists() for path in older_files) and not (data_dir / WIDE_FILE).exists():
        return add_wide_table(config)
    check_outputs_are_new(config)

    print("Step 1 of 3: participant table (metadata, recording durations, VTC results)")
    master, duration_audit, age_validation, validation_records = build_master_dataset(config)
    print(f"  {len(master)} participants")

    print("Step 2 of 3: long tables (one row per participant and speaker) and wide table (one row per participant)")
    # The long and wide tables use the values as they are saved in final_master.csv.
    master_csv = master.to_csv(index=False)
    saved_master = pd.read_csv(StringIO(master_csv))
    input_long, input_output_long = build_long_tables(saved_master)
    input_output_wide = build_wide_table_from_csv(StringIO(master_csv))
    print(f"  {len(input_long)} rows in each long table, {len(input_output_wide)} rows in the wide table")

    print("Step 3 of 3: saving the files")
    audit_dir = config.results_dir
    write_csv(master, new_output(data_dir / "final_master.csv"))
    write_csv(input_long, new_output(data_dir / "input_long.csv"))
    write_csv(input_output_long, new_output(data_dir / "input_output_long.csv"))
    write_csv(input_output_wide, new_output(data_dir / WIDE_FILE))
    write_csv(duration_audit, new_output(audit_dir / "recording_duration_audit.csv"))
    write_csv(age_validation, new_output(audit_dir / "age_validation.csv"))
    write_validation_report(validation_records, new_output(audit_dir / "validation_report.csv"))
    report_lines = dataset_report_lines(
        config, saved_master, input_long, input_output_long, input_output_wide, validation_records
    )
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