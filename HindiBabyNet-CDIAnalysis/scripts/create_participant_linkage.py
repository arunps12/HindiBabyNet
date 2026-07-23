from __future__ import annotations

from hindibabynet_cdi.config import load_config
from hindibabynet_cdi.linking import build_participant_linkage


def main() -> None:
    config = load_config()
    outputs = build_participant_linkage(config)

    protected_dir = config.paths.processed_data / "protected"
    qc_dir = config.paths.outputs / "tables" / "qc"
    protected_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    linkage_path = protected_dir / "participant_linkage.csv"
    qc_path = qc_dir / "participant_linkage_qc.csv"

    outputs.participant_linkage.to_csv(linkage_path, index=False, encoding="utf-8-sig")
    outputs.linkage_qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {linkage_path}")
    print(f"Wrote {qc_path}")
    print(outputs.linkage_qc.to_string(index=False))


if __name__ == "__main__":
    main()