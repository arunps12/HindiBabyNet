"""Create participant metadata from linkage and background/CDI forms."""

from __future__ import annotations

from pathlib import Path

from hindibabynet_cdi.linking import build_participant_linkage, load_pipeline_forms
from hindibabynet_cdi.scoring import build_participant_metadata


def main() -> int:
	forms = load_pipeline_forms()
	linkage = build_participant_linkage(forms)
	participant_metadata = build_participant_metadata(forms=forms, linkage=linkage)
	Path("data/processed").mkdir(parents=True, exist_ok=True)
	participant_metadata.to_csv("data/processed/participant_metadata.csv", index=False)
	print("Wrote data/processed/participant_metadata.csv")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())