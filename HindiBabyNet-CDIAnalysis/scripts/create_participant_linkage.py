"""Create participant linkage outputs for the Hindi CDI pipeline."""

from __future__ import annotations

from hindibabynet_cdi.config import load_config
from hindibabynet_cdi.linking import build_participant_linkage, summarize_participant_linkage, write_data_linking_report


def main() -> int:
	config = load_config()
	linkage = build_participant_linkage(config=config)
	summary = summarize_participant_linkage(linkage)

	config.paths.interim_data.mkdir(parents=True, exist_ok=True)
	(config.paths.outputs / "tables").mkdir(parents=True, exist_ok=True)
	(config.paths.outputs / "reports").mkdir(parents=True, exist_ok=True)

	linkage_path = config.paths.interim_data / "participant_linkage.csv"
	summary_path = config.paths.outputs / "tables" / "participant_linkage_summary.csv"
	report_path = config.paths.outputs / "reports" / "data_linking_report.md"

	linkage.to_csv(linkage_path, index=False)
	summary.to_csv(summary_path, index=False)
	write_data_linking_report(linkage, summary, report_path)

	print(f"Wrote {linkage_path}")
	print(f"Wrote {summary_path}")
	print(f"Wrote {report_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())