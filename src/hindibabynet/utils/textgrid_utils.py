"""Utility functions for Praat TextGrid generation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from praatio import textgrid as tgio

from src.hindibabynet.utils.io_utils import ensure_dir


# ============================================================================
# Helpers
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
    try:
        return tgio.IntervalTier(str(name), entries, xmin, xmax)
    except TypeError:
        pass
    try:
        return tgio.IntervalTier(name=str(name), entries=entries, minT=xmin, maxT=xmax)
    except TypeError:
        pass
    return tgio.IntervalTier(name=str(name), entryList=entries, minT=xmin, maxT=xmax)


# ============================================================================
# Public API
# ============================================================================

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
        Maps ``predicted_class`` values → TextGrid tier names.
    """
    ensure_dir(out_path.parent)
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
