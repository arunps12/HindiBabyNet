"""Tests of the names of the prepared recordings and of the tools that use them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from hindibabynet_audio import evaluation, textgrid
from hindibabynet_audio.recordings import prepared_recordings, recording_id, select

RTTM_CSV = (
    "uid,start_time_s,duration_s,label\n"
    "REC_0001,0.0,0.5,FEM\n"
    "REC_0001,1.0,0.5,KCHI\n"
)
ANNOTATION = {
    "n_segments_per_participant": 50,
    "random_seed": 42,
    "labels": {"adult_female": 1, "adult_male": 2, "key_child": 3, "other_child": 4,
               "unclear": 5, "noise": 6},
}


def write_file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_tone(path: Path, seconds: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(seconds * 16000)) / 16000
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), 16000,
             format="WAV", subtype="PCM_16")
    return path


def test_recording_id():
    assert recording_id("P001.wav") == "P001"
    assert recording_id("REC_0001.WAV") == "REC_0001"
    assert recording_id(Path("P001") / "20250323" / "REC_0001.WAV") == "P001/20250323/REC_0001"
    assert recording_id(Path("P001") / "REC_0001.WAV") == "P001/REC_0001"
    # One folder per recording: the folder name is the recording.
    assert recording_id(Path("P001") / "P001.wav") == "P001"


def test_prepared_recordings_lists_every_wav_file(tmp_path):
    root = tmp_path / "prepared-audio"
    assert prepared_recordings(root) == []

    files = {
        "P2": write_file(root / "P2.wav"),
        "P1/20250201/REC_0001": write_file(root / "P1" / "20250201" / "REC_0001.WAV"),
        "P3": write_file(root / "P3" / "P3.wav"),
        "REC_0002": write_file(root / "REC_0002.WAV"),
    }
    # Records, unfinished files and other files are not recordings.
    write_file(root / "P2_preparation.json", "{}")
    write_file(root / "P4.wav.part")
    write_file(root / "notes.txt")

    assert prepared_recordings(root) == sorted(files.items())


def test_two_prepared_files_with_the_same_name_are_an_error(tmp_path):
    root = tmp_path / "prepared-audio"
    write_file(root / "P1.wav")
    write_file(root / "P1" / "P1.wav")

    with pytest.raises(ValueError, match="same name P1"):
        prepared_recordings(root)


def test_select_by_name_and_limit():
    items = [("P1/a", 1), ("P1/b", 2), ("P10", 3), ("P2", 4)]

    assert select(items, None, None) == items
    assert select(items, "P1", None) == [("P1/a", 1), ("P1/b", 2)]  # not P10
    assert select(items, "P1/b", None) == [("P1/b", 2)]
    assert select(items, None, 3) == items[:3]
    assert select(items, "P9", None) == []


def test_textgrid_for_results_in_sub_folders(make_config, params, capsys):
    pytest.importorskip("praatio")
    config = make_config()
    write_file(config.path("vtc", "P1", "20250201", "REC_0001", "rttm.csv"), RTTM_CSV)
    write_file(config.path("vtc", "REC_0002", "rttm.csv"), RTTM_CSV)
    segments = config.path("xgb", "P1", "REC_0003", "REC_0003_segments.csv")
    segments.parent.mkdir(parents=True)
    pd.DataFrame(
        {"start_sec": [0.0], "end_sec": [0.5], "predicted_class": ["adult_male"]}
    ).to_csv(segments, index=False)
    # The table of the merged segments is not taken as the results of a recording.
    (segments.parent / "REC_0003_merged_segments.csv").write_text("merged_segment\n", encoding="utf-8")

    assert textgrid.find_results(config, "vtc") == [
        ("P1/20250201/REC_0001", config.path("vtc", "P1", "20250201", "REC_0001", "rttm.csv")),
        ("REC_0002", config.path("vtc", "REC_0002", "rttm.csv")),
    ]
    assert textgrid.find_results(config, "xgb") == [("P1/REC_0003", segments)]

    assert textgrid.run(config, params, "vtc", participant="P1") == 0
    assert textgrid.run(config, params, "xgb") == 0

    assert config.path("textgrids", "vtc", "P1", "20250201", "REC_0001.TextGrid").is_file()
    assert not config.path("textgrids", "vtc", "REC_0002.TextGrid").exists()
    assert config.path("textgrids", "xgb", "P1", "REC_0003.TextGrid").is_file()


@pytest.mark.parametrize("place", ["REC_0001.WAV", "REC_0001/REC_0001.wav"])
def test_annotation_finds_the_prepared_file(make_config, monkeypatch, place):
    config = make_config()
    write_file(config.path("vtc", "REC_0001", "rttm.csv"), RTTM_CSV)
    wav = write_tone(config.path("prepared_audio", *place.split("/")))
    played = []
    replies = iter(["1", "k"])
    monkeypatch.setattr(evaluation, "play_wav", lambda path: played.append(Path(path)))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    assert evaluation.run_annotation(config, {"annotation": ANNOTATION}, "vtc", "REC_0001") == 0

    table = pd.read_csv(config.path("annotations", "REC_0001", evaluation.ANNOTATION_FILE))
    assert list(table["manual_label"]) == ["adult_female", "key_child"]
    assert len(played) == 2
    assert sorted(path.name for path in wav.parent.iterdir()) == [wav.name]


def test_annotation_and_evaluation_of_a_recording_in_a_sub_folder(make_config, monkeypatch):
    config = make_config()
    name = "P1/20250201/REC_0001"
    write_file(config.path("vtc", "P1", "20250201", "REC_0001", "rttm.csv"), RTTM_CSV)
    write_tone(config.path("prepared_audio", "P1", "20250201", "REC_0001.WAV"))
    replies = iter(["1", "k"])
    monkeypatch.setattr(evaluation, "play_wav", lambda path: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    settings = {"annotation": ANNOTATION}

    assert evaluation.run_annotation(config, settings, "vtc", name) == 0
    annotations = config.path("annotations", "P1", "20250201", "REC_0001",
                              evaluation.ANNOTATION_FILE)
    assert annotations.is_file()

    # One recording, and all recordings together.
    assert evaluation.run_evaluation(config, settings, "vtc", participant=name) == 0
    assert evaluation.run_evaluation(config, settings, "vtc") == 0
    results = config.results("classification-evaluation", "vtc")
    assert (results / "P1" / "20250201" / "REC_0001" / "metrics.json").is_file()
    assert (results / "summary" / "metrics.json").is_file()
