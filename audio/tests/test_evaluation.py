from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from hbn_paths import Config
from hindibabynet_audio import evaluation
from hindibabynet_audio.evaluation import (
    ANNOTATION_FILE,
    compute_accuracy,
    compute_classwise_metrics,
    compute_confusion_matrix,
    generate_evaluation_report,
    label_shortcuts,
    sample_segments,
)

LABELS = {
    "adult_female": 1,
    "adult_male": 2,
    "key_child": 3,
    "other_child": 4,
    "unclear": 5,
    "noise": 6,
}
PARAMS = {"annotation": {"n_segments_per_participant": 50, "random_seed": 42, "labels": LABELS}}

RTTM_CSV = (
    "uid,start_time_s,duration_s,label\n"
    "P1,0.0,0.5,FEM\n"
    "P1,1.0,0.5,KCHI\n"
    "P1,2.0,0.5,MAL\n"
    "P1,3.0,0.5,OCH\n"
    "P1,4.0,0.5,FEM\n"
)


def make_config(tmp_path: Path) -> Config:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data_dir: data\n", encoding="utf-8")
    return Config(config_file)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_participant(config: Config, participant: str = "P1") -> Path:
    """Synthetic VTC results and a 6-second prepared WAV. Returns the rttm.csv path."""
    rttm = config.path("vtc", participant, "rttm.csv")
    rttm.parent.mkdir(parents=True)
    rttm.write_text(RTTM_CSV, encoding="utf-8")
    wav = config.path("prepared_audio", participant, f"{participant}.wav")
    wav.parent.mkdir(parents=True)
    t = np.arange(6 * 16000) / 16000
    sf.write(str(wav), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000, subtype="PCM_16")
    return rttm


def answer_with(monkeypatch, answers: list[str]) -> list[Path]:
    """Replace the keyboard and the loudspeaker. Returns the list of played files."""
    played: list[Path] = []
    replies = iter(answers)
    monkeypatch.setattr(evaluation, "play_wav", lambda path: played.append(Path(path)))
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    return played


# ---------------------------------------------------------------------------
# Segment sampling
# ---------------------------------------------------------------------------

def test_segment_sampler_is_reproducible():
    df = pd.DataFrame(
        {
            "start_sec": list(range(100)),
            "end_sec": [value + 0.5 for value in range(100)],
            "predicted_class": ["adult_female"] * 100,
        }
    )

    first = sample_segments(df, n_segments=10, random_seed=42)
    second = sample_segments(df, n_segments=10, random_seed=42)

    assert first.equals(second)
    assert len(first) == 10
    assert first["start_sec"].is_monotonic_increasing
    # the same rule as before: pandas sample with the seed, then time order
    expected = df.sample(n=10, random_state=42).sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    assert first.equals(expected)


def test_segment_sampler_keeps_all_rows_of_a_small_table():
    df = pd.DataFrame({"start_sec": [2.0, 0.0], "end_sec": [2.5, 0.5], "predicted_class": ["a", "b"]})

    assert sample_segments(df, n_segments=10, random_seed=42).equals(df)
    assert sample_segments(df.iloc[0:0], n_segments=10, random_seed=42).empty


def test_label_shortcuts_come_from_params():
    shortcuts = label_shortcuts(LABELS)
    assert shortcuts["1"] == "adult_female"
    assert shortcuts["6"] == "noise"
    assert shortcuts["k"] == "key_child"
    assert shortcuts["m"] == "adult_male"


# ---------------------------------------------------------------------------
# Manual annotation
# ---------------------------------------------------------------------------

def test_run_annotation_saves_labels_and_continues(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    rttm = write_participant(config)
    before = sha256(rttm)
    output_csv = config.path("annotations", "P1", ANNOTATION_FILE)

    # first session: two labels, then quit
    played = answer_with(monkeypatch, ["1", "k", "q"])
    assert evaluation.run_annotation(config, PARAMS, "vtc", "P1") == 0

    table = pd.read_csv(output_csv)
    assert list(table.columns) == [
        "segment_index", "participant_id", "method", "start_sec", "end_sec", "predicted_class", "manual_label",
    ]
    assert len(table) == 5
    assert list(table["predicted_class"]) == ["adult_female", "key_child", "adult_male", "other_child", "adult_female"]
    assert list(table["manual_label"].fillna("")) == ["adult_female", "key_child", "", "", ""]
    assert set(table["method"]) == {"vtc"}
    assert len(played) == 3

    # second session: the two labels are kept, the other three segments are asked
    played = answer_with(monkeypatch, ["2", "o", "unclear"])
    assert evaluation.run_annotation(config, PARAMS, "vtc", "P1") == 0

    table = pd.read_csv(output_csv)
    assert list(table["manual_label"]) == ["adult_female", "key_child", "adult_male", "other_child", "unclear"]
    assert [path.name for path in played] == ["segment_0002.wav", "segment_0003.wav", "segment_0004.wav"]

    assert sha256(rttm) == before
    assert [p.name for p in rttm.parent.iterdir()] == ["rttm.csv"]


def test_run_annotation_does_not_replace_a_file_with_other_segments(tmp_path: Path, monkeypatch, capsys):
    config = make_config(tmp_path)
    write_participant(config)
    output_csv = config.path("annotations", "P1", ANNOTATION_FILE)

    answer_with(monkeypatch, ["1", "1", "1", "1", "1"])
    assert evaluation.run_annotation(config, PARAMS, "vtc", "P1") == 0
    before = sha256(output_csv)

    # other sampling settings give other segments
    other = {"annotation": {**PARAMS["annotation"], "n_segments_per_participant": 3}}
    answer_with(monkeypatch, [])
    assert evaluation.run_annotation(config, other, "vtc", "P1") == 1

    assert sha256(output_csv) == before
    assert "not replaced" in capsys.readouterr().out


def test_run_annotation_needs_results_and_audio(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    answer_with(monkeypatch, [])

    assert evaluation.run_annotation(config, PARAMS, "vtc", "P1") == 1
    assert not config.path("annotations").exists()
    assert not config.path("vtc").exists()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_metrics_on_a_hand_made_table():
    df = pd.DataFrame(
        {
            "manual_label": ["adult_female", "adult_female", "adult_male", "key_child", None, " "],
            "predicted_class": ["adult_female", "adult_male", "adult_male", "key_child", "adult_male", "adult_male"],
        }
    )
    labels = ["adult_female", "adult_male", "key_child"]

    # rows without a manual label are not counted
    assert compute_accuracy(df) == pytest.approx(0.75)
    assert compute_accuracy(df.iloc[4:]) == 0.0

    confusion = compute_confusion_matrix(df, labels=labels)
    assert confusion.index.name == "manual_label"
    assert confusion.columns.name == "predicted_class"
    assert confusion.to_numpy().tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]

    classwise = compute_classwise_metrics(df, labels=labels).set_index("label")
    assert list(classwise.columns) == ["tp", "fp", "fn", "tn", "precision", "recall", "f1"]
    assert classwise.loc["adult_female", ["tp", "fp", "fn", "tn"]].tolist() == [1, 0, 1, 2]
    assert classwise.loc["adult_female", "precision"] == pytest.approx(1.0)
    assert classwise.loc["adult_female", "recall"] == pytest.approx(0.5)
    assert classwise.loc["adult_female", "f1"] == pytest.approx(2 / 3)
    assert classwise.loc["adult_male", "precision"] == pytest.approx(0.5)
    assert classwise.loc["adult_male", "recall"] == pytest.approx(1.0)
    assert classwise.loc["key_child", "f1"] == pytest.approx(1.0)


def test_generate_evaluation_report_formats_markdown():
    classwise = pd.DataFrame(
        [
            {"label": "adult_female", "precision": 1.0, "recall": 0.5, "f1": 0.6667},
            {"label": "adult_male", "precision": 0.5, "recall": 1.0, "f1": 0.6667},
        ]
    )
    report = generate_evaluation_report(
        {"n_rows": 2, "accuracy": 0.5, "precision": 0.75, "recall": 0.75, "f1": 0.6667},
        classwise,
    )

    assert "# Model Evaluation Report" in report
    assert "| adult_female |" in report


def test_run_evaluation_writes_outputs(tmp_path: Path):
    config = make_config(tmp_path)
    participant_dir = config.path("annotations", "P1")
    participant_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"manual_label": "adult_female", "predicted_class": "adult_female"},
            {"manual_label": "adult_male", "predicted_class": "adult_female"},
        ]
    ).to_csv(participant_dir / ANNOTATION_FILE, index=False)
    before = sha256(participant_dir / ANNOTATION_FILE)

    assert evaluation.run_evaluation(config, PARAMS, "vtc", participant="P1") == 0

    output_dir = config.results("classification-evaluation", "vtc", "P1")
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "classwise_metrics.csv",
        "confusion_matrix.csv",
        "evaluated_annotations.csv",
        "metrics.json",
        "report.md",
    ]
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_rows"] == 2
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "n_rows"}
    classwise = pd.read_csv(output_dir / "classwise_metrics.csv")
    assert list(classwise["label"]) == list(LABELS)
    assert sha256(participant_dir / ANNOTATION_FILE) == before

    # existing results are not replaced
    metrics_before = sha256(output_dir / "metrics.json")
    assert evaluation.run_evaluation(config, PARAMS, "vtc", participant="P1") == 1
    assert sha256(output_dir / "metrics.json") == metrics_before


def test_run_evaluation_of_all_participants_uses_only_the_method(tmp_path: Path):
    config = make_config(tmp_path)
    rows = {
        "P1": ("vtc", "adult_female", "adult_female"),
        "P2": ("vtc", "adult_male", "adult_female"),
        "P3": ("xgb", "key_child", "key_child"),
    }
    for participant, (method, manual, predicted) in rows.items():
        folder = config.path("annotations", participant)
        folder.mkdir(parents=True)
        pd.DataFrame(
            [{"participant_id": participant, "method": method, "manual_label": manual, "predicted_class": predicted}]
        ).to_csv(folder / ANNOTATION_FILE, index=False)

    assert evaluation.run_evaluation(config, PARAMS, "vtc") == 0

    output_dir = config.results("classification-evaluation", "vtc", "summary")
    evaluated = pd.read_csv(output_dir / "evaluated_annotations.csv")
    assert list(evaluated["participant_id"]) == ["P1", "P2"]
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_rows"] == 2
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_run_evaluation_without_annotations_returns_1(tmp_path: Path):
    config = make_config(tmp_path)

    assert evaluation.run_evaluation(config, PARAMS, "vtc") == 1
    assert evaluation.run_evaluation(config, PARAMS, "vtc", participant="P1") == 1
    assert not config.path("results").exists()
