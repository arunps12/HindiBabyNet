from __future__ import annotations

import pandas as pd

from hindibabynet_pipeline.components.annotation.annotation_schema import ANNOTATION_LABELS
from hindibabynet_pipeline.components.evaluation.accuracy import compute_accuracy
from hindibabynet_pipeline.components.evaluation.classwise_metrics import compute_classwise_metrics
from hindibabynet_pipeline.components.evaluation.confusion_matrix import compute_confusion_matrix


def test_evaluation_metric_outputs():
    df = pd.DataFrame(
        {
            "predicted_class": ["adult_female", "adult_male", "noise"],
            "manual_label": ["adult_female", "noise", "noise"],
        }
    )

    accuracy = compute_accuracy(df)
    confusion = compute_confusion_matrix(df, labels=ANNOTATION_LABELS)
    classwise = compute_classwise_metrics(df, labels=ANNOTATION_LABELS)

    assert accuracy == 2 / 3
    assert confusion.at["adult_female", "adult_female"] == 1
    assert confusion.at["noise", "adult_male"] == 1
    assert set(classwise.columns) == {"label", "tp", "fp", "fn", "tn", "precision", "recall", "f1"}