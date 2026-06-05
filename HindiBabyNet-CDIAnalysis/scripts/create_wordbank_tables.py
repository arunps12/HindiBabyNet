"""Create Wordbank-style summary tables from the scored long dataset."""

from __future__ import annotations

from pathlib import Path

from hindibabynet_cdi.scoring import build_scoring_outputs


def main() -> int:
	outputs = build_scoring_outputs()
	Path("outputs/tables").mkdir(parents=True, exist_ok=True)
	for name in [
		"wordbank_age_summary",
		"wordbank_word_by_age",
		"wordbank_category_by_age",
		"wordbank_percentile_curves",
		"word_frequency_overall",
		"word_frequency_by_age",
		"category_frequency_by_age",
		"shared_word_production_by_age",
	]:
		path = Path("outputs/tables") / f"{name}.csv"
		outputs[name].to_csv(path, index=False)
		print(f"Wrote {path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())