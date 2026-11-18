from __future__ import annotations

import pandas as pd

from hindibabynet_pipeline.components.evaluation.confusion_matrix import compute_confusion_matrix


def compute_classwise_metrics(df: pd.DataFrame, labels: list[str], truth_col: str = "manual_label", pred_col: str = "predicted_class") -> pd.DataFrame:
    cm = compute_confusion_matrix(df, labels=labels, truth_col=truth_col, pred_col=pred_col)
    total = int(cm.to_numpy().sum())
    rows: list[dict[str, float | str | int]] = []
    for label in labels:
        tp = int(cm.at[label, label])
        fp = int(cm[label].sum() - tp)
        fn = int(cm.loc[label].sum() - tp)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)