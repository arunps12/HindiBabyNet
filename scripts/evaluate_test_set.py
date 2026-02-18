#!/usr/bin/env python
"""
Evaluate speaker-type classifier performance from annotated test set.

Reads the annotation_sheet.csv (with human_label column filled in),
computes accuracy, UAR, per-class precision/recall/F1, and confusion matrix.

Usage
-----
uv run python scripts/evaluate_test_set.py \
    --csv artifacts/runs/20260217_133307/test_set/ABAN141223/annotation_sheet.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)

CLASSES = ["MAL", "FEM", "KCHI", "SIL"]


def main():
    ap = argparse.ArgumentParser(description="Evaluate speaker-type classifier from annotated test set")
    ap.add_argument("--csv", required=True, help="Path to annotation_sheet.csv with human_label filled")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ── Validate ──
    if "human_label" not in df.columns:
        raise ValueError("Missing 'human_label' column in CSV")

    # Drop rows where human_label is empty
    annotated = df[df["human_label"].notna() & (df["human_label"].str.strip() != "")].copy()
    n_total = len(df)
    n_annotated = len(annotated)

    if n_annotated == 0:
        print("ERROR: No annotations found. Fill in the 'human_label' column first.")
        return

    print(f"Annotation sheet : {csv_path}")
    print(f"Total clips      : {n_total}")
    print(f"Annotated clips  : {n_annotated}  ({100*n_annotated/n_total:.0f}%)")

    # Validate labels
    y_true = annotated["human_label"].str.strip().values
    y_pred = annotated["predicted_class"].str.strip().values

    invalid = set(y_true) - set(CLASSES)
    if invalid:
        print(f"\nWARNING: Unknown human labels: {invalid}")
        print(f"Valid labels: {CLASSES}")
        # Filter to valid labels only
        mask = np.isin(y_true, CLASSES)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        print(f"Using {len(y_true)} clips with valid labels")

    # ── Metrics ──
    acc = accuracy_score(y_true, y_pred)
    uar = recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  UAR (macro recall): {uar:.4f}  ({uar*100:.1f}%)")

    # ── Per-class report ──
    print(f"\n{classification_report(y_true, y_pred, labels=CLASSES, digits=4, zero_division=0)}")

    # ── Confusion matrix ──
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])
    print("Confusion Matrix:")
    print(cm_df.to_string())

    # ── Per-class accuracy ──
    print(f"\nPer-class accuracy:")
    for i, cls in enumerate(CLASSES):
        n_cls = cm[i].sum()
        if n_cls > 0:
            cls_acc = cm[i, i] / n_cls
            print(f"  {cls:<15}: {cls_acc:.4f}  ({cm[i,i]}/{n_cls})")
        else:
            print(f"  {cls:<15}: N/A (no samples)")

    # ── Confidence analysis ──
    if "predicted_confidence" in annotated.columns:
        annotated = annotated.copy()
        annotated["correct"] = (annotated["human_label"].str.strip() == annotated["predicted_class"].str.strip())
        print(f"\nConfidence analysis:")
        print(f"  Mean confidence (correct)  : {annotated[annotated['correct']]['predicted_confidence'].mean():.4f}")
        print(f"  Mean confidence (incorrect): {annotated[~annotated['correct']]['predicted_confidence'].mean():.4f}")

    # ── Save report ──
    report_path = csv_path.parent / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Evaluation Report\n{'='*60}\n")
        f.write(f"Annotation sheet: {csv_path}\n")
        f.write(f"Annotated clips : {n_annotated}\n")
        f.write(f"Accuracy        : {acc:.4f}\n")
        f.write(f"UAR             : {uar:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, labels=CLASSES, digits=4, zero_division=0))
        f.write(f"\nConfusion Matrix:\n{cm_df.to_string()}\n")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
