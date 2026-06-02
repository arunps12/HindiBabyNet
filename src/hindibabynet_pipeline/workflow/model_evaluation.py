from __future__ import annotations

from pathlib import Path

import pandas as pd

from hindibabynet_pipeline.components.annotation.annotation_schema import ANNOTATION_LABELS
from hindibabynet_pipeline.components.evaluation.accuracy import compute_accuracy
from hindibabynet_pipeline.components.evaluation.classwise_metrics import compute_classwise_metrics
from hindibabynet_pipeline.components.evaluation.confusion_matrix import compute_confusion_matrix
from hindibabynet_pipeline.components.evaluation.report_generator import generate_evaluation_report
from hindibabynet_pipeline.config.configuration import ConfigurationManager
from hindibabynet_pipeline.utils.io_utils import ensure_dir, write_json


def evaluate_models(participant_id: str | None = None) -> dict[str, Path]:
    cfg = ConfigurationManager()
    annotation_root = cfg.get_manual_annotation_root()
    evaluation_root = cfg.get_evaluation_output_root()

    if participant_id is not None:
        annotation_files = [annotation_root / participant_id / "speaker_class_annotations.csv"]
        output_dir = evaluation_root / participant_id
    else:
        annotation_files = sorted(annotation_root.glob("*/speaker_class_annotations.csv"))
        output_dir = evaluation_root / "summary"

    frames = [pd.read_csv(path) for path in annotation_files if path.exists()]
    if not frames:
        raise FileNotFoundError("No annotation files found for evaluation.")

    df = pd.concat(frames, ignore_index=True)
    ensure_dir(output_dir)

    accuracy = compute_accuracy(df)
    confusion = compute_confusion_matrix(df, labels=ANNOTATION_LABELS)
    classwise = compute_classwise_metrics(df, labels=ANNOTATION_LABELS)
    macro_precision = float(classwise["precision"].mean()) if not classwise.empty else 0.0
    macro_recall = float(classwise["recall"].mean()) if not classwise.empty else 0.0
    macro_f1 = float(classwise["f1"].mean()) if not classwise.empty else 0.0

    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix.csv"
    classwise_path = output_dir / "classwise_metrics.csv"
    evaluated_rows_path = output_dir / "evaluated_annotations.csv"
    report_path = output_dir / "report.md"

    write_json(
        {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "n_rows": int(len(df)),
        },
        metrics_path,
    )
    confusion.to_csv(confusion_path)
    classwise.to_csv(classwise_path, index=False)
    df.to_csv(evaluated_rows_path, index=False)
    report_path.write_text(
        generate_evaluation_report(
            metrics={
                "accuracy": accuracy,
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "n_rows": int(len(df)),
            },
            classwise=classwise,
        ),
        encoding="utf-8",
    )

    return {
        "metrics": metrics_path,
        "confusion_matrix": confusion_path,
        "classwise_metrics": classwise_path,
        "evaluated_annotations": evaluated_rows_path,
        "report": report_path,
    }