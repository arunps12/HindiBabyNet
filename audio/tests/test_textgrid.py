from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from hbn_paths import Config
from hindibabynet_audio import textgrid
from hindibabynet_audio.textgrid import intervals_to_df, load_segment_table, normalize_label

PARAMS = {"textgrid": {"tiers": ["adult_female", "adult_male", "key_child", "other_child"]}}

RTTM_CSV = (
    "uid,start_time_s,duration_s,label\n"
    "P1,0.0,0.5,FEM\n"
    "P1,0.5,0.5,KCHI\n"
    "P1,2.0,1.0,MAL\n"
    "P1,3.5,0.5,OCH\n"
)


def make_config(tmp_path: Path) -> Config:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("data_dir: data\n", encoding="utf-8")
    return Config(config_file)


def write_rttm(config: Config, participant: str = "P1") -> Path:
    path = config.path("vtc", participant, "rttm.csv")
    path.parent.mkdir(parents=True)
    path.write_text(RTTM_CSV, encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tiers(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Read a TextGrid back: tier name -> list of (start, end, label), without blank intervals."""
    from praatio import textgrid as tgio

    tg = tgio.openTextgrid(str(path), includeEmptyIntervals=False)
    names = list(tg.tierNames) if hasattr(tg, "tierNames") else list(tg.tierNameList)
    tiers = {}
    for name in names:
        tier = tg.getTier(name) if hasattr(tg, "getTier") else tg.tierDict[name]
        entries = tier.entries if hasattr(tier, "entries") else tier.entryList
        tiers[name] = [(float(start), float(end), str(label)) for start, end, label in entries]
    return tiers


# ---------------------------------------------------------------------------
# Reading classification results
# ---------------------------------------------------------------------------

def test_normalize_vtc_labels():
    assert normalize_label("FEM", backend="vtc") == "adult_female"
    assert normalize_label("MAL", backend="vtc") == "adult_male"
    assert normalize_label("KCHI", backend="vtc") == "key_child"
    assert normalize_label("OCH", backend="vtc") == "other_child"


def test_normalize_xgb_labels():
    assert normalize_label("adult_female", backend="xgb") == "adult_female"
    assert normalize_label("child", backend="xgb") == "key_child"
    assert normalize_label("child", backend="xgb", child_label="child") == "child"
    assert normalize_label("noise", backend="xgb") == "background"
    with pytest.raises(ValueError):
        normalize_label("FEM", backend="other")


def test_load_vtc_segment_table(tmp_path: Path):
    csv_path = tmp_path / "rttm.csv"
    csv_path.write_text(
        "uid,start_time_s,duration_s,label\n"
        "P1,0.0,0.5,FEM\n"
        "P1,0.5,0.5,KCHI\n",
        encoding="utf-8",
    )

    df = load_segment_table(csv_path, backend="vtc")

    assert list(df.columns) == ["start_sec", "end_sec", "predicted_class"]
    assert df.to_dict(orient="records") == [
        {"start_sec": 0.0, "end_sec": 0.5, "predicted_class": "adult_female"},
        {"start_sec": 0.5, "end_sec": 1.0, "predicted_class": "key_child"},
    ]


def test_load_xgb_segment_table_maps_child_label(tmp_path: Path):
    parquet_path = tmp_path / "segments.parquet"
    pd.DataFrame(
        {
            "start_sec": [0.0, 1.0],
            "end_sec": [0.5, 1.5],
            "predicted_class": ["adult_male", "child"],
        }
    ).to_parquet(parquet_path, index=False)

    df = load_segment_table(parquet_path, backend="xgb", child_label="child")

    assert df.to_dict(orient="records") == [
        {"start_sec": 0.0, "end_sec": 0.5, "predicted_class": "adult_male"},
        {"start_sec": 1.0, "end_sec": 1.5, "predicted_class": "child"},
    ]


def test_load_vtc_segment_table_needs_its_columns(tmp_path: Path):
    csv_path = tmp_path / "rttm.csv"
    csv_path.write_text("uid,start_time_s,label\nP1,0.0,FEM\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duration_s"):
        load_segment_table(csv_path, backend="vtc")


# ---------------------------------------------------------------------------
# Writing a TextGrid
# ---------------------------------------------------------------------------

def test_intervals_to_df():
    df = intervals_to_df([(0.0, 1.0), (2.0, 2.5)], "adult_female")
    assert df.to_dict(orient="records") == [
        {"start_sec": 0.0, "end_sec": 1.0, "predicted_class": "adult_female"},
        {"start_sec": 2.0, "end_sec": 2.5, "predicted_class": "adult_female"},
    ]

    empty = intervals_to_df([], "adult_male")
    assert empty.empty
    assert list(empty.columns) == ["start_sec", "end_sec", "predicted_class"]


def test_write_textgrid_one_tier_per_class_clipped_without_overlap(tmp_path: Path):
    pytest.importorskip("praatio")
    df = pd.DataFrame(
        {
            # two overlapping female intervals, one that ends after the recording,
            # one that lies fully after the recording
            "start_sec": [1.0, 0.0, 9.5, 11.0, 4.0],
            "end_sec": [3.0, 2.0, 12.0, 12.0, 5.0],
            "predicted_class": ["adult_female", "adult_female", "adult_female", "adult_female", "child"],
        }
    )
    tier_map = {"adult_female": "FEM", "adult_male": "MAL", "child": "CHILD"}
    out_path = tmp_path / "out" / "P1.TextGrid"

    result = textgrid.write_textgrid(df=df, duration_sec=10.0, out_path=out_path, tier_map=tier_map)

    assert result == out_path
    tiers = read_tiers(out_path)
    assert list(tiers) == ["FEM", "MAL", "CHILD"]

    fem = tiers["FEM"]
    assert [label for _, _, label in fem] == ["FEM", "FEM", "FEM"]
    assert [(s, e) for s, e, _ in fem] == [
        pytest.approx((0.0, 2.0)),
        pytest.approx((2.0, 3.0)),
        pytest.approx((9.5, 10.0)),
    ]
    assert tiers["MAL"] == []
    assert [(s, e) for s, e, _ in tiers["CHILD"]] == [pytest.approx((4.0, 5.0))]

    for entries in tiers.values():
        for (_, end, _), (next_start, _, _) in zip(entries, entries[1:]):
            assert next_start >= end
        for start, end, _ in entries:
            assert 0.0 <= start < end <= 10.0


# ---------------------------------------------------------------------------
# Optional tool
# ---------------------------------------------------------------------------

def test_run_writes_textgrid_and_leaves_rttm_unchanged(tmp_path: Path, capsys):
    pytest.importorskip("praatio")
    config = make_config(tmp_path)
    rttm = write_rttm(config)
    before = sha256(rttm)

    assert textgrid.run(config, PARAMS, "vtc") == 0

    out_path = config.path("textgrids", "vtc", "P1.TextGrid")
    assert out_path.is_file()
    tiers = read_tiers(out_path)
    assert list(tiers) == ["adult_female", "adult_male", "key_child", "other_child", "child", "background"]
    assert [(s, e) for s, e, _ in tiers["adult_female"]] == [pytest.approx((0.0, 0.5))]
    assert [(s, e) for s, e, _ in tiers["key_child"]] == [pytest.approx((0.5, 1.0))]
    assert [(s, e) for s, e, _ in tiers["adult_male"]] == [pytest.approx((2.0, 3.0))]
    assert [(s, e) for s, e, _ in tiers["other_child"]] == [pytest.approx((3.5, 4.0))]
    assert tiers["child"] == [] and tiers["background"] == []

    assert sha256(rttm) == before
    assert [p.name for p in rttm.parent.iterdir()] == ["rttm.csv"]
    assert "P1: TextGrid written" in capsys.readouterr().out


def test_run_reads_xgb_segments(tmp_path: Path):
    pytest.importorskip("praatio")
    config = make_config(tmp_path)
    segments = config.path("xgb", "P1", "P1_segments.csv")
    segments.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "start_sec": [0.0, 1.0, 2.0],
            "end_sec": [0.5, 1.5, 2.5],
            "predicted_class": ["adult_male", "child", "background"],
        }
    ).to_csv(segments, index=False)

    assert textgrid.run(config, PARAMS, "xgb", participant="P1") == 0

    tiers = read_tiers(config.path("textgrids", "xgb", "P1.TextGrid"))
    assert [(s, e) for s, e, _ in tiers["adult_male"]] == [pytest.approx((0.0, 0.5))]
    assert [(s, e) for s, e, _ in tiers["key_child"]] == [pytest.approx((1.0, 1.5))]
    assert [(s, e) for s, e, _ in tiers["background"]] == [pytest.approx((2.0, 2.5))]


def test_run_does_not_replace_existing_textgrid(tmp_path: Path, monkeypatch, capsys):
    config = make_config(tmp_path)
    rttm = write_rttm(config)
    before = sha256(rttm)
    out_path = config.path("textgrids", "vtc", "P1.TextGrid")
    out_path.parent.mkdir(parents=True)
    out_path.write_text("old TextGrid", encoding="utf-8")

    def must_not_write(**kwargs):
        raise AssertionError("an existing TextGrid must not be written again")

    monkeypatch.setattr(textgrid, "tgio", object())
    monkeypatch.setattr(textgrid, "write_textgrid", must_not_write)

    assert textgrid.run(config, PARAMS, "vtc") == 0

    assert out_path.read_text(encoding="utf-8") == "old TextGrid"
    assert sha256(rttm) == before
    assert "P1: TextGrid already available" in capsys.readouterr().out


def test_run_selects_participant_and_limit(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    for participant in ("P3", "P1", "P2"):
        write_rttm(config, participant)
    written = []

    def fake_write(df, duration_sec, out_path, tier_map):
        written.append((out_path.stem, duration_sec))
        out_path.write_text("TextGrid", encoding="utf-8")
        return out_path

    monkeypatch.setattr(textgrid, "tgio", object())
    monkeypatch.setattr(textgrid, "write_textgrid", fake_write)

    assert textgrid.run(config, PARAMS, "vtc", limit=2) == 0
    assert textgrid.run(config, PARAMS, "vtc", participant="P3") == 0

    # name order; the duration is the end of the last segment
    assert written == [("P1", 4.0), ("P2", 4.0), ("P3", 4.0)]


def test_run_without_results_returns_1(tmp_path: Path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(textgrid, "tgio", object())

    assert textgrid.run(config, PARAMS, "vtc") == 1
    with pytest.raises(ValueError):
        textgrid.run(config, PARAMS, "other")
    assert not config.path("vtc").exists()
    assert not config.path("textgrids").exists()
