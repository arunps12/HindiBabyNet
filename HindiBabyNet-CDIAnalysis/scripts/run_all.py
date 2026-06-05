"""Run the full CDI pipeline script sequence from raw checks through EDA."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_SEQUENCE = [
	"check_raw_files.py",
	"create_metadata_files.py",
	"create_participant_linkage.py",
	"create_participant_metadata.py",
	"score_cdi_8_18.py",
	"score_cdi_19_36.py",
	"create_word_level_long.py",
	"compute_participant_scores.py",
	"compute_category_scores.py",
	"create_wordbank_tables.py",
	"run_eda.py",
]

LEGACY_OUTPUTS = [
	Path("data/processed/cdi_8_18_scored_wide_safe.csv"),
	Path("data/processed/cdi_19_36_scored_wide_safe.csv"),
	Path("data/processed/cdi_participant_scores.csv"),
	Path("data/processed/cdi_master_dataset.csv"),
	Path("data/processed/vocabulary_scores.csv"),
	Path("data/processed/participant_info.csv"),
	Path("data/processed/master_dataset.csv"),
	Path("data/processed/category_scores.csv"),
	Path("outputs/tables/wordbank_item_summary.csv"),
	Path("outputs/tables/wordbank_age_bin_summary.csv"),
	Path("outputs/tables/questionnaire_summary.csv"),
	Path("outputs/tables/category_summary.csv"),
	Path("outputs/reports/eda_report.md"),
]


def _run_script(script_name: str) -> None:
	script_path = Path(__file__).with_name(script_name)
	result = subprocess.run([sys.executable, str(script_path)], check=False)
	if result.returncode != 0:
		raise SystemExit(result.returncode)


def _remove_legacy_outputs() -> None:
	for path in LEGACY_OUTPUTS:
		if path.exists():
			path.unlink()


def main() -> int:
	_remove_legacy_outputs()
	for script_name in SCRIPT_SEQUENCE:
		_run_script(script_name)
	print("Pipeline outputs written")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())