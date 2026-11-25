from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

# The whole file is skipped when the XGBoost extra is not installed.
joblib = pytest.importorskip("joblib")
pytest.importorskip("xgboost")
pytest.importorskip("opensmile")
pytest.importorskip("torch")
pytest.importorskip("praatio")

from hbn_paths import Config  # noqa: E402
from hindibabynet_audio import xgb  # noqa: E402

CLASS_ORDER = ["adult_male", "adult_female", "child", "background"]

# The xgb section of audio/params.yaml
PARAMS = {
    "xgb": {
        "class_names": CLASS_ORDER,
        "egemaps_dim": 88,
        "vad": {"aggressiveness": 2, "frame_ms": 30, "min_region_ms": 300},
        "diarization": {
            "model": "pyannote/speaker-diarization-3.1",
            "chunk_sec": 900.0,
            "overlap_sec": 10.0,
            "min_speakers": 2,
            "max_speakers": 4,
        },
        "segmentation": {"merge_gap_sec": 0.3, "min_segment_sec": 0.2},
        "classification": {"win_sec": 1.0, "hop_sec": 0.5},
    }
}


class ThreeClassModel:
    """Stand-in model with the wrong number of classes."""

    def predict_proba(self, X):
        return np.full((len(X), 3), 1 / 3, dtype=np.float32)


class FixedModel:
    """Stand-in model: every window gets the same probabilities."""

    def predict_proba(self, X):
        return np.tile(np.array([0.1, 0.2, 0.6, 0.1], dtype=np.float32), (len(X), 1))


def tone(seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Labels and model
# ---------------------------------------------------------------------------

def test_label_order():
    assert list(xgb.LABEL2ID) == CLASS_ORDER
    assert xgb.LABEL2ID == {"adult_male": 0, "adult_female": 1, "child": 2, "background": 3}
    assert [xgb.ID2LABEL[i] for i in range(4)] == CLASS_ORDER
    assert xgb.CLASS_NAMES == CLASS_ORDER
    assert xgb.TIER_MAP == {
        "adult_female": "FEM",
        "adult_male": "MAL",
        "child": "CHILD",
        "background": "BACKGROUND",
    }


def test_model_path_points_to_audio_models():
    assert xgb.MODEL_PATH.name == "xgb_egemaps.pkl"
    assert xgb.MODEL_PATH.parent.name == "models"
    assert xgb.MODEL_PATH.parent.parent == Path(xgb.__file__).resolve().parents[2]


def test_model_loads_and_gives_probabilities():
    if not xgb.MODEL_PATH.exists():
        pytest.skip(f"model file not available: {xgb.MODEL_PATH}")

    model = xgb.load_model()

    classes = getattr(model, "classes_", None)
    if classes is not None:
        names = [c if isinstance(c, str) else xgb.ID2LABEL[int(c)] for c in classes]
        assert names == CLASS_ORDER

    X = np.random.default_rng(0).normal(size=(5, 88)).astype(np.float32)
    proba = np.asarray(model.predict_proba(X))
    assert proba.shape == (5, 4)
    assert np.all(proba >= 0)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_load_model_checks_file_and_number_of_classes(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        xgb.load_model(tmp_path / "missing.pkl")

    wrong = tmp_path / "three_classes.pkl"
    joblib.dump(ThreeClassModel(), wrong)
    with pytest.raises(ValueError, match="3 classes"):
        xgb.load_model(wrong)

    good = tmp_path / "four_classes.pkl"
    joblib.dump(FixedModel(), good)
    assert isinstance(xgb.load_model(good), FixedModel)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_egemaps_features_of_a_tone_have_88_values():
    extractor = xgb.EGemapsExtractor()
    audio = tone(1.0)

    vec = xgb.extract_egemaps_for_window(extractor, audio, 16000, 0.0, 1.0)

    assert vec.shape == (88,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))
    assert np.any(vec != 0)

    # same values and same order as opensmile gives them
    names = list(extractor.smile.feature_names)
    assert len(names) == 88
    assert names[0] == "F0semitoneFrom27.5Hz_sma3nz_amean"
    assert names[-1] == "equivalentSoundLevel_dBp"
    direct = extractor.smile.process_signal(audio, 16000).values.flatten().astype(np.float32)
    assert np.array_equal(vec, direct)

    # a short window is padded with zeros to one second
    short = xgb.extract_egemaps_for_window(extractor, audio, 16000, 0.0, 0.4)
    padded = xgb.crop_or_pad(audio[:6400], 16000)
    direct_short = extractor.smile.process_signal(padded, 16000).values.flatten().astype(np.float32)
    assert short.shape == (88,)
    assert np.array_equal(short, direct_short)


def test_crop_or_pad():
    x = np.arange(5, dtype=np.float32)

    assert xgb.crop_or_pad(x, 5) is x
    assert xgb.crop_or_pad(x, 3).tolist() == [0.0, 1.0, 2.0]
    padded = xgb.crop_or_pad(x, 8)
    assert padded.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]
    assert padded.dtype == np.float32


def test_slice_audio():
    x = np.arange(100, dtype=np.float32)

    assert xgb.slice_audio(x, 10, 1.0, 2.0).tolist() == list(range(10, 20))
    assert xgb.slice_audio(x, 10, -1.0, 0.5).tolist() == list(range(0, 5))
    assert xgb.slice_audio(x, 10, 9.5, 20.0).tolist() == list(range(95, 100))
    assert len(xgb.slice_audio(x, 10, 12.0, 13.0)) == 0


def test_resample_audio():
    x = tone(1.0, sr=48000)

    assert xgb.resample_audio(x, 16000, 16000) is x
    y = xgb.resample_audio(x, 48000, 16000)
    assert len(y) == 16000
    assert y.dtype == np.float32


def test_generate_windows():
    # short segment: one window of full length
    assert xgb.generate_windows(2.0, 2.4) == [(2.0, 3.0, 1.0)]

    # long segment: hop 0.5 s and a last window that ends with the segment
    windows = xgb.generate_windows(0.0, 2.2)
    assert [w[:2] for w in windows] == [
        pytest.approx((0.0, 1.0)),
        pytest.approx((0.5, 1.5)),
        pytest.approx((1.0, 2.0)),
        pytest.approx((1.2, 2.2)),
    ]
    assert [w[2] for w in windows] == [1.0, 1.0, 1.0, 1.0]

    # no extra window when the last one ends exactly with the segment
    assert [w[:2] for w in xgb.generate_windows(0.0, 2.0)] == [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)]


def test_weighted_mean_probs():
    P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert xgb.weighted_mean_probs(P, [1.0, 1.0]) == pytest.approx([0.5, 0.5])
    assert xgb.weighted_mean_probs(P, [3.0, 1.0]) == pytest.approx([0.75, 0.25])


def test_classify_segments_with_a_fixed_model(monkeypatch):
    monkeypatch.setattr(
        xgb, "extract_egemaps_for_window", lambda extractor, audio, sr, start, end: np.zeros(88, dtype=np.float32)
    )
    df = pd.DataFrame({"start_sec": [0.0, 3.0], "end_sec": [2.2, 3.4], "duration_sec": [2.2, 0.4]})

    out = xgb.classify_segments(df, tone(4.0), 16000, FixedModel(), extractor=None, class_names=CLASS_ORDER)

    assert list(out["n_windows"]) == [4, 1]
    assert list(out["predicted_class"]) == ["child", "child"]
    assert list(out["predicted_confidence"]) == pytest.approx([0.6, 0.6])
    assert [c for c in out.columns if c.startswith("probs_")] == [f"probs_{name}" for name in CLASS_ORDER]
    assert list(out["probs_adult_female"]) == pytest.approx([0.2, 0.2])

    empty = xgb.classify_segments(df.iloc[0:0].copy(), tone(1.0), 16000, FixedModel(), None, CLASS_ORDER)
    assert empty.empty
    assert {"predicted_class", "predicted_confidence", "n_windows", "probs_child"} <= set(empty.columns)


# ---------------------------------------------------------------------------
# Voice activity detection, chunks, intersection, merging, streams
# ---------------------------------------------------------------------------

def test_vad_finds_no_speech_in_silence(tmp_path: Path):
    pytest.importorskip("webrtcvad")
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    assert xgb.webrtc_vad_regions(wav, aggressiveness=2, frame_ms=30, min_region_ms=300) == []

    wrong_rate = tmp_path / "wrong_rate.wav"
    sf.write(str(wrong_rate), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")
    with pytest.raises(ValueError):
        xgb.webrtc_vad_regions(wrong_rate)


def test_make_chunks():
    assert xgb._make_chunks(25.0, 10.0, 2.0) == [(0, 0.0, 10.0), (1, 8.0, 18.0), (2, 16.0, 25.0)]
    assert xgb._make_chunks(5.0, 10.0, 2.0) == [(0, 0.0, 5.0)]
    with pytest.raises(AssertionError):
        xgb._make_chunks(25.0, 10.0, 10.0)


def test_intersect_turns_with_vad():
    turns = pd.DataFrame(
        {
            "chunk_id": [0, 0, 0],
            "speaker_id_local": ["A", "B", "A"],
            "start_sec": [0.0, 3.0, 6.8],
            "end_sec": [2.0, 6.0, 7.5],
        }
    )
    vad = [(5.0, 7.0), (1.0, 4.0)]

    out = xgb.intersect_turns_with_vad(turns, vad, min_keep_sec=0.5)

    assert list(out.columns) == ["start_sec", "end_sec", "duration_sec", "chunk_id", "speaker_id_local"]
    assert out[["start_sec", "end_sec", "speaker_id_local"]].to_dict(orient="records") == [
        {"start_sec": 1.0, "end_sec": 2.0, "speaker_id_local": "A"},
        {"start_sec": 3.0, "end_sec": 4.0, "speaker_id_local": "B"},
        {"start_sec": 5.0, "end_sec": 6.0, "speaker_id_local": "B"},
    ]
    assert list(out["duration_sec"]) == [1.0, 1.0, 1.0]

    # without a minimum length the short part (6.8-7.0 s) is kept
    assert len(xgb.intersect_turns_with_vad(turns, vad)) == 4
    # no speech or no turns: empty table with the same columns
    assert xgb.intersect_turns_with_vad(turns, []).empty
    assert list(xgb.intersect_turns_with_vad(turns.iloc[0:0], vad).columns) == list(out.columns)


def test_merge_close_segments():
    df = pd.DataFrame(
        {
            "chunk_id": [0, 0, 0, 0, 1],
            "speaker_id_local": ["A", "B", "A", "A", "A"],
            "start_sec": [0.0, 2.1, 1.2, 3.0, 4.1],
            "end_sec": [1.0, 2.5, 2.0, 4.0, 5.0],
        }
    )
    df["duration_sec"] = df["end_sec"] - df["start_sec"]

    out = xgb.merge_close_segments(df, gap_thresh=0.3)

    # A 0.0-1.0 and A 1.2-2.0 are merged (gap 0.2 s). A 3.0-4.0 is too far away.
    # A 4.1-5.0 is in another chunk and is not merged.
    assert out[["chunk_id", "speaker_id_local", "start_sec", "end_sec", "n_merged"]].to_dict(orient="records") == [
        {"chunk_id": 0, "speaker_id_local": "A", "start_sec": 0.0, "end_sec": 2.0, "n_merged": 2},
        {"chunk_id": 0, "speaker_id_local": "B", "start_sec": 2.1, "end_sec": 2.5, "n_merged": 1},
        {"chunk_id": 0, "speaker_id_local": "A", "start_sec": 3.0, "end_sec": 4.0, "n_merged": 1},
        {"chunk_id": 1, "speaker_id_local": "A", "start_sec": 4.1, "end_sec": 5.0, "n_merged": 1},
    ]
    assert list(out["duration_sec"]) == pytest.approx([2.0, 0.4, 1.0, 0.9])
    assert xgb.merge_close_segments(df.iloc[0:0]).empty


def test_build_class_stream():
    audio = np.arange(100, dtype=np.float32)
    df = pd.DataFrame(
        {
            "start_sec": [4.0, 1.0, 6.0],
            "end_sec": [5.0, 2.0, 7.0],
            "predicted_class": ["child", "child", "adult_male"],
        }
    )

    stream, used, offsets = xgb.build_class_stream(df, audio, sr=10, target_class="child", gap_sec=0.15)

    # 10 samples, 1 sample of silence, 10 samples, 1 sample of silence
    assert stream.tolist() == list(range(10, 20)) + [0.0] + list(range(40, 50)) + [0.0]
    assert list(used["start_sec"]) == [1.0, 4.0]
    assert offsets == [pytest.approx((0.0, 1.0, 1.0, 2.0)), pytest.approx((1.1, 2.1, 4.0, 5.0))]

    stream, used, offsets = xgb.build_class_stream(df, audio, sr=10, target_class="background")
    assert len(stream) == 0 and used.empty and offsets == []


# ---------------------------------------------------------------------------
# run: existing results, incomplete folders, the vtc folder
# ---------------------------------------------------------------------------

class NoVtcConfig:
    """Config that fails the test when the vtc folder is asked for."""

    def __init__(self, config: Config):
        self._config = config

    def path(self, key: str, *parts: str) -> Path:
        assert key != "vtc", "the XGBoost classification must not use the vtc folder"
        return self._config.path(key, *parts)


def make_config(tmp_path: Path) -> Config:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data_dir: data\n", encoding="utf-8")
    return Config(config_file)


def add_prepared_audio(config: Config, *participants: str) -> None:
    for participant in participants:
        wav = config.path("prepared_audio", participant, f"{participant}.wav")
        wav.parent.mkdir(parents=True)
        sf.write(str(wav), tone(0.5), 16000, subtype="PCM_16")


def add_results(config: Config, participant: str, names: list[str]) -> Path:
    out_dir = config.path("xgb", participant)
    out_dir.mkdir(parents=True)
    for name in names:
        (out_dir / name.format(id=participant)).write_text("old", encoding="utf-8")
    return out_dir


def snapshot(folder: Path) -> dict[str, bytes]:
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


COMPLETE = ["{id}_main_female.wav", "{id}_main_male.wav", "{id}_summary.json", "{id}.TextGrid"]


@pytest.fixture
def mocked_steps(monkeypatch):
    """Replace the model loading and the heavy steps. Returns the list of classified participants."""
    calls: list[str] = []
    loaded: list[str] = []

    def fake_classify(wav_path, participant, out_dir, settings, model, diar_pipeline, extractor):
        calls.append(participant)
        if participant.startswith("BAD"):
            raise RuntimeError("synthetic failure")
        return {"n_classified_segments": 3, "total_speech_sec": 7200.0}

    monkeypatch.setattr(xgb, "load_model", lambda *args, **kwargs: loaded.append("model") or "model")
    monkeypatch.setattr(xgb, "load_diarization_pipeline", lambda name: loaded.append(name) or "pipeline")
    monkeypatch.setattr(xgb, "EGemapsExtractor", lambda **kwargs: "extractor")
    monkeypatch.setattr(xgb, "classify_recording", fake_classify)
    calls_and_loaded = {"calls": calls, "loaded": loaded}
    return calls_and_loaded


def test_is_complete(tmp_path: Path):
    out_dir = tmp_path / "P1"
    out_dir.mkdir()
    for name in COMPLETE[:-1]:
        (out_dir / name.format(id="P1")).write_text("x", encoding="utf-8")
    assert not xgb.is_complete("P1", out_dir)

    (out_dir / "P1.TextGrid").write_text("x", encoding="utf-8")
    assert xgb.is_complete("P1", out_dir)

    assert sorted(p.name for p in xgb.output_files("P1", out_dir).values()) == sorted(
        [
            "P1_main_female.wav",
            "P1_main_male.wav",
            "P1_child.wav",
            "P1_background.wav",
            "P1.TextGrid",
            "P1_segments.parquet",
            "P1_summary.json",
            "run_info.json",
        ]
    )


def test_run_skips_complete_refuses_incomplete_and_leaves_vtc_alone(tmp_path: Path, mocked_steps, capsys):
    config = make_config(tmp_path)
    add_prepared_audio(config, "P1", "P2", "P3")
    complete_dir = add_results(config, "P1", COMPLETE)
    incomplete_dir = add_results(config, "P2", ["{id}_summary.json", "run_info.json"])
    vtc_dir = config.path("vtc")
    (vtc_dir / "P1").mkdir(parents=True)
    (vtc_dir / "P1" / "rttm.csv").write_text("uid,start_time_s,duration_s,label\n", encoding="utf-8")
    before = {name: snapshot(folder) for name, folder in
              [("complete", complete_dir), ("incomplete", incomplete_dir), ("vtc", vtc_dir)]}

    result = xgb.run(NoVtcConfig(config), PARAMS)

    # P2 counts as failed, P3 is the only participant that is run
    assert result == 1
    assert mocked_steps["calls"] == ["P3"]
    assert mocked_steps["loaded"] == ["model", "pyannote/speaker-diarization-3.1"]
    assert snapshot(complete_dir) == before["complete"]
    assert snapshot(incomplete_dir) == before["incomplete"]
    assert snapshot(vtc_dir) == before["vtc"]

    out = capsys.readouterr().out
    assert "P1: results already available, not run again" in out
    assert "P2: ERROR" in out and "incomplete" in out
    assert "P3: started" in out and "P3: finished" in out
    assert "1 run, 1 already available, 1 failed" in out

    run_info = json.loads(config.path("xgb", "P3", "run_info.json").read_text(encoding="utf-8"))
    assert list(run_info) == [
        "participant_id", "backend", "input_wav", "output_dir",
        "started_at", "finished_at", "runtime_sec", "status",
    ]
    assert run_info["participant_id"] == "P3"
    assert run_info["backend"] == "xgb"
    assert run_info["status"] == "success"


def test_run_loads_nothing_when_all_results_are_available(tmp_path: Path, mocked_steps):
    config = make_config(tmp_path)
    add_prepared_audio(config, "P1")
    add_results(config, "P1", COMPLETE)

    assert xgb.run(NoVtcConfig(config), PARAMS) == 0
    assert mocked_steps["calls"] == []
    assert mocked_steps["loaded"] == []


def test_run_continues_after_a_failed_participant(tmp_path: Path, mocked_steps):
    config = make_config(tmp_path)
    add_prepared_audio(config, "BAD1", "P1")

    assert xgb.run(NoVtcConfig(config), PARAMS) == 1

    assert mocked_steps["calls"] == ["BAD1", "P1"]
    failed = json.loads(config.path("xgb", "BAD1", "run_info.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error"] == "synthetic failure"
    done = json.loads(config.path("xgb", "P1", "run_info.json").read_text(encoding="utf-8"))
    assert done["status"] == "success"


def test_run_selects_participant_and_limit(tmp_path: Path, mocked_steps):
    config = make_config(tmp_path)
    add_prepared_audio(config, "P3", "P1", "P2")

    assert xgb.run(NoVtcConfig(config), PARAMS, limit=2) == 0
    assert mocked_steps["calls"] == ["P1", "P2"]

    assert xgb.run(NoVtcConfig(config), PARAMS, participant="P3") == 0
    assert mocked_steps["calls"] == ["P1", "P2", "P3"]

    # unknown participant or no prepared audio: nothing to do, reported as an error
    assert xgb.run(NoVtcConfig(config), PARAMS, participant="P9") == 1
    assert not config.path("xgb", "P9").exists()


def test_run_stops_when_the_model_cannot_be_loaded(tmp_path: Path, mocked_steps, monkeypatch):
    config = make_config(tmp_path)
    add_prepared_audio(config, "P1")

    def no_model(*args, **kwargs):
        raise FileNotFoundError("Model not found")

    monkeypatch.setattr(xgb, "load_model", no_model)

    assert xgb.run(NoVtcConfig(config), PARAMS) == 1
    assert mocked_steps["calls"] == []
    assert not config.path("xgb").exists()


def test_run_with_other_file_names_and_places(tmp_path: Path, mocked_steps, capsys):
    config = make_config(tmp_path)
    root = config.path("prepared_audio")
    for place in ("P1.wav", "REC_0001.WAV", "P2/20250201/REC_0002.WAV"):
        (root / place).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(root / place), tone(0.5), 16000, format="WAV", subtype="PCM_16")
    (root / "P1_preparation.json").write_text("{}", encoding="utf-8")
    add_results(config, "P1", COMPLETE)  # existing results are accepted

    assert xgb.run(NoVtcConfig(config), PARAMS) == 0

    # The result files start with the file name; the folder is the place of the recording.
    assert mocked_steps["calls"] == ["REC_0002", "REC_0001"]
    assert config.path("xgb", "P2", "20250201", "REC_0002", "run_info.json").is_file()
    assert config.path("xgb", "REC_0001", "run_info.json").is_file()
    out = capsys.readouterr().out
    assert "P1: results already available" in out
    assert "P2/20250201/REC_0002: finished" in out

    # All recordings of one participant can be selected by the participant name.
    assert xgb.run(NoVtcConfig(config), PARAMS, participant="P2") == 1  # incomplete folder
    assert mocked_steps["calls"] == ["REC_0002", "REC_0001"]
