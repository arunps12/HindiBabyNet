from __future__ import annotations

from pathlib import Path

import pandas as pd


VTC_LABEL_MAP = {
    "FEM": "adult_female",
    "MAL": "adult_male",
    "KCHI": "key_child",
    "OCH": "other_child",
}

XGB_LABEL_MAP = {
    "adult_female": "adult_female",
    "adult_male": "adult_male",
    "child": "key_child",
    "background": "background",
    "noise": "background",
}


def normalize_label(label: str, backend: str, child_label: str = "key_child") -> str:
    backend_name = backend.lower()
    raw = str(label).strip()
    if backend_name == "vtc":
        return VTC_LABEL_MAP.get(raw, raw.lower())
    if backend_name == "xgb":
        if raw == "child":
            return child_label
        return XGB_LABEL_MAP.get(raw, raw)
    raise ValueError(f"Unsupported backend: {backend}")


def _load_vtc_csv(path: Path, child_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"start_time_s", "duration_s", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"VTC CSV missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "start_sec": df["start_time_s"].astype(float),
            "end_sec": df["start_time_s"].astype(float) + df["duration_s"].astype(float),
            "predicted_class": [normalize_label(label, "vtc", child_label) for label in df["label"].astype(str)],
        }
    )
    return out[out["end_sec"] > out["start_sec"]].reset_index(drop=True)


def _load_xgb_table(path: Path, child_label: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"start_sec", "end_sec", "predicted_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"XGB segment table missing columns: {sorted(missing)}")

    out = df.loc[:, ["start_sec", "end_sec", "predicted_class"]].copy()
    out["start_sec"] = out["start_sec"].astype(float)
    out["end_sec"] = out["end_sec"].astype(float)
    out["predicted_class"] = [normalize_label(label, "xgb", child_label) for label in out["predicted_class"].astype(str)]
    return out[out["end_sec"] > out["start_sec"]].reset_index(drop=True)


def load_segment_table(path: str | Path, backend: str, child_label: str = "key_child") -> pd.DataFrame:
    segment_path = Path(path)
    if backend.lower() == "vtc":
        return _load_vtc_csv(segment_path, child_label=child_label)
    if backend.lower() == "xgb":
        return _load_xgb_table(segment_path, child_label=child_label)
    raise ValueError(f"Unsupported backend: {backend}")