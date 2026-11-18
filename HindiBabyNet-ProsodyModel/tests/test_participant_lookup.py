from __future__ import annotations

from pathlib import Path

from hindibabynet_prosody.config import load_config
from hindibabynet_prosody.io.participant_lookup import load_participant_lookup


def test_load_participant_lookup_reads_expected_columns(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "config.example.yaml")
    lookup_path = tmp_path / "participant_lookup.csv"
    lookup_path.write_text("original_par_id,participant_id\nTEST000001,P001\n", encoding="utf-8")
    config.paths.participant_lookup = lookup_path

    lookup = load_participant_lookup(config)

    assert lookup.mapping["TEST000001"] == "P001"
    assert lookup.dataframe.columns.tolist() == ["original_par_id", "participant_id"]
