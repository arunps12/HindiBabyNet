"""Aggregation helpers for participant-speaker summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def median_or_none(values: Iterable[float]) -> float | None:
    collected = [value for value in values if np.isfinite(value)]
    if not collected:
        return None
    return float(np.median(np.asarray(collected, dtype=np.float64)))
