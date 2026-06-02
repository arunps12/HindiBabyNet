from __future__ import annotations

from pathlib import Path

import pandas as pd


def textgrid_to_dataframe(path: str | Path) -> pd.DataFrame:
    try:
        from praatio import textgrid as tgio
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise ModuleNotFoundError(
            "praatio is required for TextGrid parsing. Install the xgb extra with 'uv sync --extra xgb'."
        ) from exc

    tg = tgio.openTextgrid(str(path), includeEmptyIntervals=False)
    rows: list[dict[str, object]] = []
    for tier_name in tg.tierNameList:
        tier = tg.tierDict[tier_name]
        for start_sec, end_sec, label in tier.entries:
            rows.append(
                {
                    "tier": tier_name,
                    "start_sec": float(start_sec),
                    "end_sec": float(end_sec),
                    "label": str(label),
                }
            )
    return pd.DataFrame(rows)