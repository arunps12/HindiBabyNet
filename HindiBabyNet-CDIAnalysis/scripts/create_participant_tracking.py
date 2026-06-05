"""Compatibility wrapper for building participant linkage/tracking outputs."""

from __future__ import annotations

from pathlib import Path

from hindibabynet_cdi.linking import (
	build_participant_linkage,
	load_pipeline_forms,
	summarize_participant_linkage,
	write_data_linking_report,
)


def main() -> int:
	forms = load_pipeline_forms()
	linkage = build_participant_linkage(forms)
	linkage_summary = summarize_participant_linkage(linkage)
	Path("data/interim").mkdir(parents=True, exist_ok=True)
	Path("outputs/tables").mkdir(parents=True, exist_ok=True)
	Path("outputs/reports").mkdir(parents=True, exist_ok=True)
	linkage.to_csv("data/interim/participant_linkage.csv", index=False)
	linkage_summary.to_csv("outputs/tables/participant_linkage_summary.csv", index=False)
	write_data_linking_report(linkage, linkage_summary, Path("outputs/reports/data_linking_report.md"))
	print("Wrote data/interim/participant_linkage.csv")
	print("Wrote outputs/tables/participant_linkage_summary.csv")
	print("Wrote outputs/reports/data_linking_report.md")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())