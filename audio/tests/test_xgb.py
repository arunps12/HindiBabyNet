from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

# The whole file is skipped when the XGBoost extra is not installed.
joblib = pytest.importorskip("joblib")
pytest.importorskip("xgboost")
pytest.importorskip("opensmile")
pytest.importorskip("praatio")

from hbn_paths import Config  # noqa: E402
from hindibabynet_audio import xgb  # noqa: E402
from hindibabynet_audio.params import load_params  # noqa: E402

CLASS_ORDER = ["adult_male", "adult_female", "child", "background"]

# Settings of these tests, with the same parts as the xgb section of audio/params.yaml
PARAMS = {
    "xgb": {
        "class_names": CLASS_ORDER,
        "egemaps_dim": 88,
        "vad": {"aggressiveness": 1, "frame_ms": 30, "min_region_ms": 150, "min_silence_ms": 30},
        "segmentation": {
            "merge_before_classification": False,
            "merge_after_classification": True,
            "merge_gap_sec": 0.3,
        },
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
# Voice activity detection, merging, streams
# ---------------------------------------------------------------------------

FRAME = 480  # samples of one 30 ms frame at 16 kHz


@pytest.fixture
def fake_vad(monkeypatch):
    """Replace WebRTC VAD: a frame is speech when it has a sample that is not zero."""

    class Vad:
        def __init__(self, aggressiveness):
            self.aggressiveness = aggressiveness

        def is_speech(self, data: bytes, sr: int) -> bool:
            return any(data)

    monkeypatch.setitem(sys.modules, "webrtcvad", types.SimpleNamespace(Vad=Vad))


def write_frames(path: Path, sounds: list[tuple[int, int, float]], n_frames: int) -> Path:
    """Write a 16 kHz recording of n_frames frames. sounds: (first frame, frame after the
    last one, sample value); all other frames are silence."""
    samples = np.zeros(n_frames * FRAME, dtype=np.float32)
    for first, after_last, value in sounds:
        samples[first * FRAME:after_last * FRAME] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, 16000, format="WAV", subtype="PCM_16")
    return path


def test_speech_runs_and_the_pause_setting():
    S, n = True, False
    #        0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
    flags = [n, S, S, n, S, n, n, S, n, n, n, S, n, n, n, n]

    # Every frame without speech ends a chunk (min_silence_ms up to one frame).
    each_frame = [(1, 3), (4, 5), (7, 8), (11, 12)]
    assert list(xgb.speech_runs(iter(flags), 30, 0)) == each_frame
    assert list(xgb.speech_runs(iter(flags), 30, 30)) == each_frame
    # A pause of one frame (30 ms) stays inside the chunk, two frames (60 ms) end it.
    assert list(xgb.speech_runs(iter(flags), 30, 60)) == [(1, 5), (7, 8), (11, 12)]
    # Pauses of 30, 60 and 90 ms stay inside the chunk; 120 ms ends it.
    assert list(xgb.speech_runs(iter(flags), 30, 100)) == [(1, 12)]
    # A chunk never ends with silence, and speech up to the last frame is kept.
    assert list(xgb.speech_runs(iter([S, n, S]), 30, 100)) == [(0, 3)]
    assert list(xgb.speech_runs(iter([n, n]), 30, 30)) == []


def test_vad_chunks_with_minimum_length_and_pause(tmp_path: Path, fake_vad):
    # Speech in the frames 10-19 (300 ms), 21-23 (90 ms) and 40-44 (150 ms).
    wav = write_frames(tmp_path / "rec.wav", [(10, 20, 0.5), (21, 24, 0.5), (40, 45, 0.5)], 60)

    chunks = xgb.webrtc_vad_regions(wav, aggressiveness=1, frame_ms=30, min_region_ms=150,
                                    min_silence_ms=30)
    # The chunk of 90 ms is shorter than min_region_ms and is not used.
    assert chunks == [pytest.approx((0.30, 0.60)), pytest.approx((1.20, 1.35))]

    # With a longer pause setting the pause of 30 ms stays inside the first chunk.
    chunks = xgb.webrtc_vad_regions(wav, aggressiveness=1, frame_ms=30, min_region_ms=150,
                                    min_silence_ms=60)
    assert chunks == [pytest.approx((0.30, 0.72)), pytest.approx((1.20, 1.35))]

    assert xgb.webrtc_vad_regions(wav, min_region_ms=300) == [pytest.approx((0.30, 0.60))]


def test_vad_finds_no_speech_in_silence(tmp_path: Path):
    pytest.importorskip("webrtcvad")
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    assert xgb.webrtc_vad_regions(wav, aggressiveness=2, frame_ms=30, min_region_ms=300) == []
    assert xgb.webrtc_vad_regions(wav, aggressiveness=1, frame_ms=30, min_region_ms=150,
                                  min_silence_ms=30) == []

    wrong_rate = tmp_path / "wrong_rate.wav"
    sf.write(str(wrong_rate), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")
    with pytest.raises(ValueError):
        xgb.webrtc_vad_regions(wrong_rate)


def test_chunk_statistics():
    statistics = xgb.chunk_statistics([0.5, 0.5, 1.0, 2.0])

    assert statistics["n_chunks"] == 4
    assert statistics["total_sec"] == pytest.approx(4.0)
    assert statistics["mean_sec"] == pytest.approx(1.0)
    assert statistics["median_sec"] == pytest.approx(0.75)
    assert (statistics["min_sec"], statistics["max_sec"]) == (0.5, 2.0)
    assert statistics["percent_shorter_than_1s"] == pytest.approx(50.0)
    assert xgb.chunk_statistics([]) == {"n_chunks": 0, "total_sec": 0.0}


def chunk_table(rows: list[tuple[float, float, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["start_sec", "end_sec", "predicted_class"])
    df["duration_sec"] = df["end_sec"] - df["start_sec"]
    return df


def test_merge_chunks_of_the_same_class():
    df = chunk_table([
        (0.0, 1.0, "adult_female"),
        (1.2, 2.0, "adult_female"),   # pause 0.2 s, same class: merged with the first
        (2.3, 3.0, "adult_female"),   # pause exactly 0.3 s: merged
        (3.33, 4.0, "adult_female"),  # pause 0.33 s: a new segment
        (4.1, 4.5, "child"),          # another class: a new segment
        (4.6, 5.0, "adult_female"),   # the child chunk lies between: not merged with 3.33-4.0
        (5.1, 5.5, "adult_female"),   # pause 0.1 s, same class: merged
    ])
    before = df.copy()

    number, merged = xgb.merge_chunks(df, gap_sec=0.3, same_class=True)

    assert list(number) == [0, 0, 0, 1, 2, 3, 3]
    assert list(merged.columns) == [
        "merged_segment", "start_sec", "end_sec", "predicted_class",
        "n_chunks", "speech_duration_sec", "span_sec",
    ]
    assert list(merged["merged_segment"]) == [0, 1, 2, 3]
    assert list(merged["predicted_class"]) == ["adult_female", "adult_female", "child", "adult_female"]
    assert list(merged["start_sec"]) == [0.0, 3.33, 4.1, 4.6]
    assert list(merged["end_sec"]) == [3.0, 4.0, 4.5, 5.5]
    assert list(merged["n_chunks"]) == [3, 1, 1, 2]
    # The speech duration is the sum of the chunks. The pauses are not counted.
    assert list(merged["speech_duration_sec"]) == pytest.approx([2.5, 0.67, 0.4, 0.8])
    assert list(merged["span_sec"]) == pytest.approx([3.0, 0.67, 0.4, 0.9])
    assert merged["speech_duration_sec"].sum() == pytest.approx(df["duration_sec"].sum())
    # The classified chunks are not changed.
    pd.testing.assert_frame_equal(df, before)

    # Another limit, and chunks that are not in time order.
    number, merged = xgb.merge_chunks(df.iloc[::-1], gap_sec=0.05, same_class=True)
    assert len(merged) == 7 and list(number.sort_index()) == [0, 1, 2, 3, 4, 5, 6]

    number, merged = xgb.merge_chunks(df.iloc[0:0], gap_sec=0.3, same_class=True)
    assert number.empty and merged.empty and "predicted_class" in merged.columns


def test_merge_chunks_before_the_classification_does_not_look_at_a_class():
    df = chunk_table([(0.0, 1.0, "a"), (1.2, 2.0, "b"), (2.5, 3.0, "a")]).drop(
        columns="predicted_class"
    )

    number, merged = xgb.merge_chunks(df, gap_sec=0.3, same_class=False)

    assert list(number) == [0, 0, 1]
    assert list(merged.columns) == [
        "merged_segment", "start_sec", "end_sec", "n_chunks", "speech_duration_sec", "span_sec"
    ]
    assert list(merged["speech_duration_sec"]) == pytest.approx([1.8, 0.5])
    assert list(merged["span_sec"]) == pytest.approx([2.0, 0.5])


# Not used at present: tests of the speaker diarization functions (see xgb.py).
#
# def test_make_chunks():
#     assert xgb._make_chunks(25.0, 10.0, 2.0) == [(0, 0.0, 10.0), (1, 8.0, 18.0), (2, 16.0, 25.0)]
#     assert xgb._make_chunks(5.0, 10.0, 2.0) == [(0, 0.0, 5.0)]
#     with pytest.raises(AssertionError):
#         xgb._make_chunks(25.0, 10.0, 10.0)
#
#
# def test_intersect_turns_with_vad():
#     turns = pd.DataFrame(
#         {
#             "chunk_id": [0, 0, 0],
#             "speaker_id_local": ["A", "B", "A"],
#             "start_sec": [0.0, 3.0, 6.8],
#             "end_sec": [2.0, 6.0, 7.5],
#         }
#     )
#     vad = [(5.0, 7.0), (1.0, 4.0)]
#
#     out = xgb.intersect_turns_with_vad(turns, vad, min_keep_sec=0.5)
#
#     assert list(out.columns) == ["start_sec", "end_sec", "duration_sec", "chunk_id", "speaker_id_local"]
#     assert out[["start_sec", "end_sec", "speaker_id_local"]].to_dict(orient="records") == [
#         {"start_sec": 1.0, "end_sec": 2.0, "speaker_id_local": "A"},
#         {"start_sec": 3.0, "end_sec": 4.0, "speaker_id_local": "B"},
#         {"start_sec": 5.0, "end_sec": 6.0, "speaker_id_local": "B"},
#     ]
#     assert list(out["duration_sec"]) == [1.0, 1.0, 1.0]
#
#     # without a minimum length the short part (6.8-7.0 s) is kept
#     assert len(xgb.intersect_turns_with_vad(turns, vad)) == 4
#     # no speech or no turns: empty table with the same columns
#     assert xgb.intersect_turns_with_vad(turns, []).empty
#     assert list(xgb.intersect_turns_with_vad(turns.iloc[0:0], vad).columns) == list(out.columns)
#
#
# def test_merge_close_segments():
#     df = pd.DataFrame(
#         {
#             "chunk_id": [0, 0, 0, 0, 1],
#             "speaker_id_local": ["A", "B", "A", "A", "A"],
#             "start_sec": [0.0, 2.1, 1.2, 3.0, 4.1],
#             "end_sec": [1.0, 2.5, 2.0, 4.0, 5.0],
#         }
#     )
#     df["duration_sec"] = df["end_sec"] - df["start_sec"]
#
#     out = xgb.merge_close_segments(df, gap_thresh=0.3)
#
#     # A 0.0-1.0 and A 1.2-2.0 are merged (gap 0.2 s). A 3.0-4.0 is too far away.
#     # A 4.1-5.0 is in another chunk and is not merged.
#     assert out[["chunk_id", "speaker_id_local", "start_sec", "end_sec", "n_merged"]].to_dict(orient="records") == [
#         {"chunk_id": 0, "speaker_id_local": "A", "start_sec": 0.0, "end_sec": 2.0, "n_merged": 2},
#         {"chunk_id": 0, "speaker_id_local": "B", "start_sec": 2.1, "end_sec": 2.5, "n_merged": 1},
#         {"chunk_id": 0, "speaker_id_local": "A", "start_sec": 3.0, "end_sec": 4.0, "n_merged": 1},
#         {"chunk_id": 1, "speaker_id_local": "A", "start_sec": 4.1, "end_sec": 5.0, "n_merged": 1},
#     ]
#     assert list(out["duration_sec"]) == pytest.approx([2.0, 0.4, 1.0, 0.9])
#     assert xgb.merge_close_segments(df.iloc[0:0]).empty


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


COMPLETE = ["{id}_segments.csv", "{id}_summary.json", "{id}.TextGrid"]


@pytest.fixture
def mocked_steps(monkeypatch):
    """Replace the model loading and the heavy steps. Returns the list of classified participants."""
    calls: list[str] = []
    loaded: list[str] = []

    def fake_classify(wav_path, participant, out_dir, settings, model, extractor):
        calls.append(participant)
        if participant.startswith("BAD"):
            raise RuntimeError("synthetic failure")
        return {"n_classified_segments": 3, "total_speech_sec": 7200.0}

    monkeypatch.setattr(xgb, "load_model", lambda *args, **kwargs: loaded.append("model") or "model")
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
            "P1.TextGrid",
            "P1_segments.csv",
            "P1_merged_segments.csv",
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
    assert mocked_steps["loaded"] == ["model"]  # no diarization model is loaded
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


# ---------------------------------------------------------------------------
# One complete run on a made-up recording
# ---------------------------------------------------------------------------

class LevelModel:
    """Stand-in model: the class is given by the sample value of the chunk
    (0.1 adult_male, 0.2 adult_female, 0.3 child, 0.4 background)."""

    def predict_proba(self, X):
        index = np.clip(np.rint(X[:, 0] * 10).astype(int) - 1, 0, 3)
        return np.eye(4, dtype=np.float32)[index]


FEM, CHILD = 0.2, 0.3
# (first frame, frame after the last one, sample value). One frame is 30 ms.
SOUNDS = [
    (10, 30, FEM),      # 1: 0.30-0.90 s
    (35, 50, FEM),      # 2: 1.05-1.50 s, pause 0.15 s            -> merged with 1
    (60, 80, CHILD),    # 3: 1.80-2.40 s, another class
    (83, 100, FEM),     # 4: 2.49-3.00 s
    (110, 120, FEM),    # 5: 3.30-3.60 s, pause exactly 0.30 s    -> merged with 4
    (121, 127, CHILD),  # 6: 3.63-3.81 s, another class
    (128, 140, FEM),    # 7: 3.84-4.20 s, chunk 6 lies between     -> not merged with 5
    (150, 160, FEM),    # 8: 4.50-4.80 s, pause 0.30 s            -> merged with 7
    (171, 180, FEM),    # 9: 5.13-5.40 s, pause 0.33 s            -> a new segment
    (190, 200, FEM),    # 10: 5.70-6.00 s, pause 0.30 s           -> merged with 9
    (220, 224, FEM),    # 120 ms: shorter than min_region_ms, not used
]
CHUNKS = [(first * 0.03, after_last * 0.03) for first, after_last, _ in SOUNDS[:10]]
SPEECH_SEC = sum(end - start for start, end in CHUNKS)  # 3.87 s


@pytest.fixture
def made_up_run(tmp_path: Path, fake_vad, monkeypatch):
    """A made-up recording and a function that classifies it with other settings."""
    wav = write_frames(tmp_path / "prepared" / "P1.wav", SOUNDS, 240)
    # Stand-in features: the sample value at the start of the window.
    monkeypatch.setattr(
        xgb, "extract_egemaps_for_window",
        lambda extractor, audio, sr, start, end: np.full(88, audio[int(round(start * sr))]),
    )

    def run(name: str, **segmentation) -> tuple[dict, Path]:
        settings = {**PARAMS["xgb"], "segmentation": {**PARAMS["xgb"]["segmentation"], **segmentation}}
        out_dir = tmp_path / "xgb" / name
        summary = xgb.classify_recording(wav, "P1", out_dir, settings, LevelModel(), None)
        return summary, out_dir

    return run


def read_textgrid_tiers(path: Path) -> dict[str, list[tuple[float, float]]]:
    from praatio import textgrid as tgio

    grid = tgio.openTextgrid(str(path), includeEmptyIntervals=False)
    return {
        name: [(float(start), float(end)) for start, end, _ in grid.getTier(name).entries]
        for name in grid.tierNames
    }


def test_every_vad_chunk_is_classified_alone_and_then_merged(made_up_run):
    summary, out_dir = made_up_run("default")

    # Table 1: one row for every VAD chunk, as it was classified.
    chunks = pd.read_csv(out_dir / "P1_segments.csv")
    assert list(chunks.columns) == [
        "start_sec", "end_sec", "duration_sec", "n_windows",
        "probs_adult_male", "probs_adult_female", "probs_child", "probs_background",
        "predicted_class", "predicted_confidence", "merged_segment",
    ]
    assert list(chunks["start_sec"]) == pytest.approx([start for start, _ in CHUNKS])
    assert list(chunks["end_sec"]) == pytest.approx([end for _, end in CHUNKS])
    assert list(chunks["duration_sec"]) == pytest.approx([end - start for start, end in CHUNKS])
    fem, child = "adult_female", "child"
    assert list(chunks["predicted_class"]) == [fem, fem, child, fem, fem, child, fem, fem, fem, fem]
    assert list(chunks["merged_segment"]) == [0, 0, 1, 2, 2, 3, 4, 4, 5, 5]
    # The numbers in the CSV file have no long tails like 0.6000000000000001.
    first_row = (out_dir / "P1_segments.csv").read_text(encoding="utf-8").splitlines()[1]
    assert first_row.startswith("0.3,0.9,0.6,1,0.0,1.0,0.0,0.0,adult_female,1.0,0")

    # Table 2: chunks of the same class that follow each other with a pause up to 0.3 s.
    merged = pd.read_csv(out_dir / "P1_merged_segments.csv")
    assert list(merged["merged_segment"]) == [0, 1, 2, 3, 4, 5]
    assert list(merged["predicted_class"]) == [fem, child, fem, child, fem, fem]
    assert list(merged["n_chunks"]) == [2, 1, 2, 1, 2, 2]
    assert list(merged["start_sec"]) == pytest.approx([0.30, 1.80, 2.49, 3.63, 3.84, 5.13])
    assert list(merged["end_sec"]) == pytest.approx([1.50, 2.40, 3.60, 3.81, 4.80, 6.00])
    # The pauses between merged chunks are not counted as speech.
    assert list(merged["speech_duration_sec"]) == pytest.approx([1.05, 0.60, 0.81, 0.18, 0.66, 0.57])
    assert list(merged["span_sec"]) == pytest.approx([1.20, 0.60, 1.11, 0.18, 0.96, 0.87])
    assert merged["speech_duration_sec"].sum() == pytest.approx(SPEECH_SEC)
    assert chunks["duration_sec"].sum() == pytest.approx(SPEECH_SEC)

    # The TextGrid has the chunks: a pause inside a merged segment is not labelled.
    tiers = read_textgrid_tiers(out_dir / "P1.TextGrid")
    assert list(tiers) == ["FEM", "MAL", "CHILD", "BACKGROUND"]
    assert tiers["FEM"] == [pytest.approx(CHUNKS[i]) for i in (0, 1, 3, 4, 6, 7, 8, 9)]
    assert tiers["CHILD"] == [pytest.approx(CHUNKS[i]) for i in (2, 5)]
    assert tiers["MAL"] == [] and tiers["BACKGROUND"] == []

    # Two CSV tables, the TextGrid and the summary. No audio files are written.
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "P1.TextGrid", "P1_merged_segments.csv", "P1_segments.csv", "P1_summary.json",
    ]
    assert xgb.is_complete("P1", out_dir)

    saved = json.loads((out_dir / "P1_summary.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(summary))
    assert summary["n_vad_chunks"] == 10 and summary["n_classified_segments"] == 10
    assert summary["n_merged_segments"] == 6
    assert summary["total_speech_sec"] == pytest.approx(SPEECH_SEC)
    assert summary["class_durations"] == pytest.approx(
        {"adult_male": 0.0, "adult_female": 3.09, "child": 0.78, "background": 0.0}
    )
    assert summary["class_counts"] == {"adult_male": 0, "adult_female": 8, "child": 2, "background": 0}
    assert summary["merged_class_counts"] == {
        "adult_male": 0, "adult_female": 4, "child": 2, "background": 0
    }
    assert summary["vad_chunk_statistics"]["percent_shorter_than_1s"] == pytest.approx(100.0)
    assert summary["settings"]["vad"] == PARAMS["xgb"]["vad"]
    assert summary["settings"]["segmentation"] == PARAMS["xgb"]["segmentation"]


def test_merging_can_be_switched_off_or_done_before_the_classification(made_up_run):
    # No merging: only the table of the classified chunks.
    summary, out_dir = made_up_run("no_merge", merge_after_classification=False)
    chunks = pd.read_csv(out_dir / "P1_segments.csv")
    assert len(chunks) == 10 and "merged_segment" not in chunks.columns
    assert not (out_dir / "P1_merged_segments.csv").exists()
    assert summary["n_merged_segments"] is None and summary["merged_segments"] is None
    assert xgb.is_complete("P1", out_dir)

    # Another limit for the pause: 0.15 s merges only the chunks 1 and 2.
    summary, _ = made_up_run("small_gap", merge_gap_sec=0.15)
    assert summary["n_merged_segments"] == 9

    # Merging before the classification joins chunks without looking at a class.
    summary, out_dir = made_up_run(
        "before", merge_before_classification=True, merge_after_classification=False
    )
    chunks = pd.read_csv(out_dir / "P1_segments.csv")
    assert list(chunks["n_chunks"]) == [8, 2]
    assert list(chunks["start_sec"]) == pytest.approx([0.30, 5.13])
    assert list(chunks["end_sec"]) == pytest.approx([4.80, 6.00])
    # Also here the duration is the speech of the chunks, without the pauses.
    assert list(chunks["duration_sec"]) == pytest.approx([3.30, 0.57])
    assert summary["n_vad_chunks"] == 10 and summary["n_classified_segments"] == 2
    assert summary["total_speech_sec"] == pytest.approx(SPEECH_SEC)


def test_recording_without_speech_gives_complete_empty_results(tmp_path: Path, fake_vad):
    wav = write_frames(tmp_path / "prepared" / "P1.wav", [], 100)
    out_dir = tmp_path / "xgb" / "P1"

    summary = xgb.classify_recording(wav, "P1", out_dir, PARAMS["xgb"], LevelModel(), None)

    assert summary["n_vad_chunks"] == 0 and summary["total_speech_sec"] == 0.0
    assert summary["n_merged_segments"] == 0
    assert pd.read_csv(out_dir / "P1_segments.csv").empty
    assert pd.read_csv(out_dir / "P1_merged_segments.csv").empty
    assert xgb.is_complete("P1", out_dir)
    assert not list(out_dir.glob("*.wav"))


def test_a_stopped_run_is_not_recorded_as_a_success(tmp_path: Path, monkeypatch):
    def stopped(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(xgb, "classify_recording", stopped)
    out_dir = tmp_path / "P1"

    with pytest.raises(KeyboardInterrupt):
        xgb.run_participant(tmp_path / "P1.wav", "P1", out_dir, PARAMS["xgb"], "model", "extractor")

    info = json.loads((out_dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["status"] == "interrupted"
    assert not xgb.is_complete("P1", out_dir)


# ---------------------------------------------------------------------------
# Settings and switched-off diarization
# ---------------------------------------------------------------------------

def test_settings_of_params_yaml():
    """The file has all settings and their values can be used. The values of the VAD
    and of the merging can be changed in the file, so they are not fixed here."""
    settings = load_params()["xgb"]

    xgb.check_settings(settings)
    assert {part: set(settings[part]) for part in ("vad", "segmentation", "classification")} == {
        part: set(PARAMS["xgb"][part]) for part in ("vad", "segmentation", "classification")
    }
    vad, segmentation = settings["vad"], settings["segmentation"]
    assert vad["aggressiveness"] in (0, 1, 2, 3)
    assert vad["frame_ms"] in (10, 20, 30)  # WebRTC VAD accepts only these frame lengths
    assert vad["min_region_ms"] >= 0 and vad["min_silence_ms"] >= 0
    assert isinstance(segmentation["merge_before_classification"], bool)
    assert isinstance(segmentation["merge_after_classification"], bool)
    assert segmentation["merge_gap_sec"] >= 0
    assert settings["classification"] == PARAMS["xgb"]["classification"]
    assert settings["class_names"] == CLASS_ORDER and settings["egemaps_dim"] == 88
    assert "diarization" not in settings  # kept as a comment in the file


def test_a_missing_setting_is_named(tmp_path: Path):
    config = make_config(tmp_path)
    add_prepared_audio(config, "P1")
    settings = {**PARAMS["xgb"], "vad": {"aggressiveness": 1, "frame_ms": 30, "min_region_ms": 150}}

    with pytest.raises(ValueError, match="xgb.vad.min_silence_ms"):
        xgb.run(config, {"xgb": settings})
    assert not config.path("xgb").exists()


def test_diarization_code_and_packages_are_switched_off():
    source = Path(xgb.__file__).read_text(encoding="utf-8")
    active = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
    used = [line for line in active if re.search(r"\b(torch|pyannote|dotenv|diar_pipeline)\b", line)]
    assert used == []
    # The code is kept as comments.
    for name in ("load_diarization_pipeline", "diarize_wav", "extract_main_speaker_audio"):
        assert f"# def {name}(" in source
        assert not hasattr(xgb, name)
