"""Compute participant-by-category CDI scores."""

from __future__ import annotations

from pathlib import Path

from hindibabynet_cdi.scoring import build_scoring_outputs


def main() -> int:
	outputs = build_scoring_outputs()
	Path("data/processed").mkdir(parents=True, exist_ok=True)
	outputs["cdi_category_scores"].to_csv("data/processed/cdi_category_scores.csv", index=False)
	print("Wrote data/processed/cdi_category_scores.csv")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())