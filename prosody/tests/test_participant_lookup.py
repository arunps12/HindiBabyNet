from __future__ import annotations

from pathlib import Path

from hindibabynet_prosody.io.participant_lookup import load_participant_lookup


def test_load_participant_lookup_reads_expected_columns(make_config, tmp_path: Path) -> None:
    config = make_config()
    lookup_path = tmp_path / "participant_lookup.csv"
    lookup_path.write_text("original_par_id,participant_id\nTEST000001,P001\n", encoding="utf-8")
    config.paths.participant_lookup = lookup_path

    lookup = load_participant_lookup(config)

    assert lookup.mapping["TEST000001"] == "P001"
    assert lookup.dataframe.columns.tolist() == ["original_par_id", "participant_id"]
