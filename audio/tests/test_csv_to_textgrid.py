from __future__ import annotations

from pathlib import Path

import pandas as pd

from hindibabynet_pipeline.components.textgrid.csv_to_textgrid import load_segment_table, normalize_label


def test_normalize_vtc_labels():
    assert normalize_label("FEM", backend="vtc") == "adult_female"
    assert normalize_label("MAL", backend="vtc") == "adult_male"
    assert normalize_label("KCHI", backend="vtc") == "key_child"
    assert normalize_label("OCH", backend="vtc") == "other_child"


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