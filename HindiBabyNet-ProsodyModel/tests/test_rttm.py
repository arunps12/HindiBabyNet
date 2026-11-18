from __future__ import annotations

from pathlib import Path

from hindibabynet_prosody.config import load_config
from hindibabynet_prosody.io.rttm import normalize_segments


def test_normalize_segments_maps_supported_speakers(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "config.example.yaml")
    rttm_path = tmp_path / "rttm.csv"
    rttm_path.write_text(
        "uid,start_time_s,duration_s,label\n"
        "TEST000001,0.0,1.0,KCHI\n"
        "TEST000001,1.0,1.0,MAL\n"
        "TEST000001,2.0,1.0,FEM\n", encoding="utf-8",
    )

    segments, issues = normalize_segments(config, rttm_path, {"TEST000001": "P001"})

    assert issues == []
    assert segments[0].source_participant_id == "TEST000001"
    assert segments[0].participant_id == "P001"
    assert segments[0].speaker_label == "KCHI"
    assert segments[0].speaker is None
    assert segments[2].speaker_label == "FEM"
    assert segments[2].speaker == "adult_female"
