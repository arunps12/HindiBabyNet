"""Praat TextGrid files.

``write_textgrid`` is used by the XGBoost classification. ``run`` is an optional
tool: it makes a TextGrid from the classification results that already exist
(VTC ``rttm.csv`` or XGBoost ``<participant>_segments.parquet``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from praatio import textgrid as tgio
except ModuleNotFoundError:  # pragma: no cover - exercised in non-xgb environments
    tgio = None

from hbn_paths import new_output

from hindibabynet_audio.recordings import select

# ============================================================================
# Writing a TextGrid
# ============================================================================

def intervals_to_df(
    intervals: List[Tuple[float, float]],
    predicted_class: str,
) -> pd.DataFrame:
    """Convert a list of ``(start, end)`` intervals into a minimal DataFrame
    suitable for :func:`write_textgrid`."""
    if not intervals:
        return pd.DataFrame(columns=["start_sec", "end_sec", "predicted_class"])
    return pd.DataFrame(
        [{"start_sec": s, "end_sec": e, "predicted_class": predicted_class}
         for s, e in intervals]
    )


def _make_interval_tier(name: str, entries, xmin: float, xmax: float):
    """Create ``IntervalTier`` across different praatio versions."""
    if tgio is None:
        raise ModuleNotFoundError(
            "praatio is required for TextGrid generation. Install the xgb extra with 'uv sync --extra xgb'."
        )
    try:
        return tgio.IntervalTier(str(name), entries, xmin, xmax)
    except TypeError:
        pass
    try:
        return tgio.IntervalTier(name=str(name), entries=entries, minT=xmin, maxT=xmax)
    except TypeError:
        pass
    return tgio.IntervalTier(name=str(name), entryList=entries, minT=xmin, maxT=xmax)


def write_textgrid(
    df: pd.DataFrame,
    duration_sec: float,
    out_path: Path,
    tier_map: Dict[str, str],
) -> Path:
    """Write a TextGrid with one tier per predicted class.

    Parameters
    ----------
    df : DataFrame
        Must contain ``start_sec``, ``end_sec``, and ``predicted_class`` columns.
    duration_sec : float
        Total duration of the source audio (used as ``xmax``).
    out_path : Path
        Destination ``.TextGrid`` file.
    tier_map : dict
        Maps ``predicted_class`` values -> TextGrid tier names.
    """
    if tgio is None:
        raise ModuleNotFoundError(
            "praatio is required for TextGrid generation. Install the xgb extra with 'uv sync --extra xgb'."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xmin, xmax = 0.0, duration_sec

    tg = tgio.Textgrid()
    tg.minTimestamp = xmin
    tg.maxTimestamp = xmax

    for pred_class, tier_name in tier_map.items():
        sel = df[df["predicted_class"] == pred_class].copy()
        sel = sel.sort_values("start_sec")

        entries: List[Tuple[float, float, str]] = []
        for r in sel.itertuples(index=False):
            s = max(xmin, min(float(r.start_sec), xmax))
            e = max(xmin, min(float(r.end_sec), xmax))
            if e > s:
                entries.append((s, e, tier_name))

        # resolve overlaps within tier
        entries.sort(key=lambda x: (x[0], x[1]))
        cleaned: List[Tuple[float, float, str]] = []
        last_end = -1.0
        for s, e, lab in entries:
            if s < last_end:
                s = last_end
            if e > s:
                cleaned.append((s, e, lab))
                last_end = e

        tier = _make_interval_tier(tier_name, cleaned, xmin, xmax)
        tg.addTier(tier)

    tg.save(str(out_path), format="short_textgrid", includeBlankSpaces=True)
    return out_path


# ============================================================================
# Reading classification results
# ============================================================================

METHODS = ("vtc", "xgb")

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

# Tiers that are always written after the tiers of params.yaml, also when empty.
EXTRA_TIERS = ("child", "background")


def normalize_label(label: str, backend: str, child_label: str = "key_child") -> str:
    """Give one label of a classification method its common name."""
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
    """Read the segments of one participant: start_sec, end_sec, predicted_class."""
    segment_path = Path(path)
    if backend.lower() == "vtc":
        return _load_vtc_csv(segment_path, child_label=child_label)
    if backend.lower() == "xgb":
        return _load_xgb_table(segment_path, child_label=child_label)
    raise ValueError(f"Unsupported backend: {backend}")


def results_file(config, method: str, participant: str) -> Path:
    """Segment file of one recording (recording ID, for example P001). The file is only read."""
    parts = participant.split("/")
    if method == "vtc":
        return config.path("vtc", *parts, "rttm.csv")
    if method == "xgb":
        return config.path("xgb", *parts, f"{parts[-1]}_segments.parquet")
    raise ValueError(f"Unknown method: {method}. Use one of {METHODS}.")


def find_results(config, method: str) -> list[tuple[str, Path]]:
    """All recordings that have classification results of this method, in name order."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Use one of {METHODS}.")
    root = config.path(method)
    if not root.is_dir():
        return []
    pattern = "rttm.csv" if method == "vtc" else "*_segments.parquet"
    found = []
    for segment_path in root.rglob(pattern):
        name = segment_path.parent.relative_to(root).as_posix()
        if name != "." and segment_path == results_file(config, method, name):
            found.append((name, segment_path))
    return sorted(found)


# ============================================================================
# Optional tool: TextGrid from existing classification results
# ============================================================================

def run(config, params, method: str, participant: str | None = None, limit: int | None = None) -> int:
    """Write <textgrids>/<method>/<recording>.TextGrid. Return 0 if nothing failed, else 1."""
    if tgio is None:
        print("praatio is not installed. It is needed to write TextGrid files.")
        return 1

    tiers = list(params["textgrid"]["tiers"])
    tiers += [name for name in EXTRA_TIERS if name not in tiers]
    tier_map = {name: name for name in tiers}

    inputs = select(find_results(config, method), participant, limit)
    if not inputs:
        print(f"No {method} classification results found in {config.path(method)}.")
        return 1

    written = available = 0
    failed: list[str] = []
    for pid, segment_path in inputs:
        out_path = config.path("textgrids", method, f"{pid}.TextGrid")
        if out_path.exists():
            print(f"{pid}: TextGrid already available, not written again ({out_path})")
            available += 1
            continue
        try:
            df = load_segment_table(segment_path, backend=method)
            duration_sec = float(df["end_sec"].max()) if not df.empty else 0.0
            write_textgrid(df=df, duration_sec=duration_sec, out_path=new_output(out_path), tier_map=tier_map)
            print(f"{pid}: TextGrid written ({out_path})")
            written += 1
        except Exception as exc:
            print(f"{pid}: ERROR, no TextGrid written: {exc}")
            failed.append(pid)

    print(f"TextGrid files ({method}): {written} written, {available} already available, {len(failed)} failed.")
    if failed:
        print("Failed: " + ", ".join(failed))
    return 1 if failed else 0
