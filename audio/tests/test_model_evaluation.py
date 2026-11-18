from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hindibabynet_pipeline.components.evaluation.report_generator import generate_evaluation_report
from hindibabynet_pipeline.workflow import model_evaluation as model_evaluation_module


def test_generate_evaluation_report_formats_markdown():
    classwise = pd.DataFrame(
        [
            {"label": "adult_female", "precision": 1.0, "recall": 0.5, "f1": 0.6667},
            {"label": "adult_male", "precision": 0.5, "recall": 1.0, "f1": 0.6667},
        ]
    )
    report = generate_evaluation_report(
        {"n_rows": 2, "accuracy": 0.5, "precision": 0.75, "recall": 0.75, "f1": 0.6667},
        classwise,
    )

    assert "# Model Evaluation Report" in report
    assert "| adult_female |" in report


def test_evaluate_models_writes_outputs(tmp_path: Path, monkeypatch):
    annotation_root = tmp_path / "annotations"
    evaluation_root = tmp_path / "evaluation"
    participant_dir = annotation_root / "P1"
    participant_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"manual_label": "adult_female", "predicted_class": "adult_female"},
            {"manual_label": "adult_male", "predicted_class": "adult_female"},
        ]
    ).to_csv(participant_dir / "speaker_class_annotations.csv", index=False)

    class StubConfig:
        def get_manual_annotation_root(self) -> Path:
            return annotation_root

        def get_evaluation_output_root(self) -> Path:
            return evaluation_root

    monkeypatch.setattr(model_evaluation_module, "ConfigurationManager", lambda: StubConfig())

    outputs = model_evaluation_module.evaluate_models(participant_id="P1")

    assert outputs["report"].exists()
    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    assert metrics["n_rows"] == 2