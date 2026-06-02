from __future__ import annotations

import pandas as pd


def compute_confusion_matrix(df: pd.DataFrame, labels: list[str], truth_col: str = "manual_label", pred_col: str = "predicted_class") -> pd.DataFrame:
    valid = df[df[truth_col].notna() & (df[truth_col].astype(str).str.strip() != "")].copy()
    cm = pd.crosstab(valid[truth_col], valid[pred_col], dropna=False)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    cm.index.name = truth_col
    cm.columns.name = pred_col
    return cm