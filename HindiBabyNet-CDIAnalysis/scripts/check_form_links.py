"""Generate a simple linkage summary for the Hindi CDI pipeline."""

from hindibabynet_cdi.config import load_config
from hindibabynet_cdi.linking import build_participant_tracking, summarize_form_links, write_form_link_report


def main() -> None:
    config = load_config()
    tracking = build_participant_tracking(config=config)
    summary = summarize_form_links(tracking)

    output_tables = config.paths.outputs / "tables"
    output_reports = config.paths.outputs / "reports"
    output_tables.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)

    summary_path = output_tables / "form_link_summary.csv"
    tracking_path = output_tables / "form_link_tracking_preview.csv"
    report_path = output_reports / "form_link_check.md"

    summary.to_csv(summary_path, index=False)
    tracking.to_csv(tracking_path, index=False)
    write_form_link_report(tracking, summary, report_path)
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()