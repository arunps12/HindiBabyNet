"""Create metadata and word mapping files from the raw CDI Excel exports."""

from __future__ import annotations

from hindibabynet_cdi.linking import load_pipeline_forms
from hindibabynet_cdi.metadata import generate_metadata_outputs, write_metadata_outputs


def main() -> int:
	forms = load_pipeline_forms()
	outputs = generate_metadata_outputs(forms)
	paths = write_metadata_outputs(outputs)
	for path in paths.values():
		print(f"Wrote {path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())