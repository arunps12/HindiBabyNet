from __future__ import annotations

import pandas as pd


def generate_evaluation_report(metrics: dict[str, float | int], classwise: pd.DataFrame) -> str:
    lines = [
        "# Model Evaluation Report",
        "",
        f"Rows evaluated: {int(metrics.get('n_rows', 0))}",
        f"Accuracy: {float(metrics.get('accuracy', 0.0)):.4f}",
        f"Macro precision: {float(metrics.get('precision', 0.0)):.4f}",
        f"Macro recall: {float(metrics.get('recall', 0.0)):.4f}",
        f"Macro F1: {float(metrics.get('f1', 0.0)):.4f}",
        "",
        "## Classwise Metrics",
        "",
        "| Label | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in classwise.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['f1']):.4f} |"
        )
    return "\n".join(lines) + "\n"