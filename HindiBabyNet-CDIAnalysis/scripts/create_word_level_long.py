"""Create questionnaire-specific and combined word-level long files."""

from __future__ import annotations

from pathlib import Path

from hindibabynet_cdi.scoring import build_scoring_outputs


def main() -> int:
	outputs = build_scoring_outputs()
	Path("data/processed").mkdir(parents=True, exist_ok=True)
	outputs["cdi_8_18_word_level_long"].to_csv("data/processed/cdi_8_18_word_level_long.csv", index=False)
	outputs["cdi_19_36_word_level_long"].to_csv("data/processed/cdi_19_36_word_level_long.csv", index=False)
	outputs["cdi_combined_word_level_long"].to_csv("data/processed/cdi_combined_word_level_long.csv", index=False)
	print("Wrote data/processed/cdi_8_18_word_level_long.csv")
	print("Wrote data/processed/cdi_19_36_word_level_long.csv")
	print("Wrote data/processed/cdi_combined_word_level_long.csv")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())