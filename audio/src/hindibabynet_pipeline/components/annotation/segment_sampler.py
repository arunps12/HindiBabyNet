from __future__ import annotations

import pandas as pd


def sample_segments(df: pd.DataFrame, n_segments: int, random_seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if len(df) <= n_segments:
        return df.reset_index(drop=True).copy()
    return df.sample(n=n_segments, random_state=random_seed).sort_values(["start_sec", "end_sec"]).reset_index(drop=True)