from __future__ import annotations

import pandas as pd


def compute_accuracy(df: pd.DataFrame, truth_col: str = "manual_label", pred_col: str = "predicted_class") -> float:
    valid = df[df[truth_col].notna() & (df[truth_col].astype(str).str.strip() != "")].copy()
    if valid.empty:
        return 0.0
    return float((valid[truth_col].astype(str) == valid[pred_col].astype(str)).mean())