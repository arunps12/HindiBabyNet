"""Validate that the expected raw Excel exports are present and structurally usable."""

from __future__ import annotations

from pathlib import Path

from hindibabynet_cdi.config import load_config
from hindibabynet_cdi.io import (
    FORM_COLUMN_REQUIREMENTS,
    get_expected_raw_files,
    read_form_export,
    validate_form_columns,
    validate_raw_file_inventory,
)


def _write_markdown_report(report_path: Path, inventory_rows: list[dict[str, object]], column_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Raw File Check",
        "",
        "## Expected files",
        "",
        "| form_key | form_id | expected_filename | exists | filename_matches_expectation |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in inventory_rows:
        lines.append(
            f"| {row['form_key']} | {row['form_id']} | {row['expected_filename']} | {row['exists']} | {row['filename_matches_expectation']} |"
        )
    lines.extend(
        [
            "",
            "## Required columns",
            "",
            "| form_key | missing_columns |",
            "| --- | --- |",
        ]
    )
    for row in column_rows:
        missing = ", ".join(row["missing_columns"])
        lines.append(f"| {row['form_key']} | {missing or 'None'} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    config = load_config()
    inventory = validate_raw_file_inventory(config)
    inventory_rows = inventory.to_dict(orient="records")

    column_rows: list[dict[str, object]] = []
    for form_key, path in get_expected_raw_files(config).items():
        missing_columns = list(FORM_COLUMN_REQUIREMENTS[form_key])
        if path.exists():
            dataframe = read_form_export(path)
            missing_columns = validate_form_columns(form_key, dataframe)
        column_rows.append({"form_key": form_key, "missing_columns": missing_columns})

    reports_dir = config.paths.outputs / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "raw_file_check.md"
    _write_markdown_report(report_path, inventory_rows, column_rows)

    failures = [row for row in inventory_rows if not row["exists"] or not row["filename_matches_expectation"]]
    failures.extend(row for row in column_rows if row["missing_columns"])

    print(f"Wrote {report_path}")
    if failures:
        print("Raw file check failed; see report for details.")
        return 1
    print("Raw file check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())