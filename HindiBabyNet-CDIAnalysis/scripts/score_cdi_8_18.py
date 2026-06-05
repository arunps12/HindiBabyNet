"""Create the 8-18 CDI wide scored outputs."""

from __future__ import annotations

from pathlib import Path

from hindibabynet_cdi.scoring import build_scoring_outputs


def main() -> int:
	outputs = build_scoring_outputs()
	Path("data/processed").mkdir(parents=True, exist_ok=True)
	outputs["cdi_8_18_scored_wide"].to_csv("data/processed/cdi_8_18_scored_wide.csv", index=False)
	outputs["cdi_8_18_scored_wide_safe_columns"].to_csv("data/processed/cdi_8_18_scored_wide_safe_columns.csv", index=False)
	print("Wrote data/processed/cdi_8_18_scored_wide.csv")
	print("Wrote data/processed/cdi_8_18_scored_wide_safe_columns.csv")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())