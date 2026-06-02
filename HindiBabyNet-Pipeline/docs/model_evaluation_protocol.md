# Model Evaluation Protocol

Model evaluation compares `automatic_label` against `manual_label` from the speaker-class annotation CSV.

Run it with:

```bash
uv run bash scripts/run_evaluate_models.sh --input-csv /path/to/ManualAnnotations/ABAN141223/ABAN141223_vtc_annotations.csv
```

Outputs written to `paths.evaluation_output_root` include:

- `summary.json`
- `confusion_matrix.csv`
- `classwise_metrics.csv`
- `annotated_segments.csv`

The evaluation currently reports:

- accuracy
- macro precision
- macro recall
- macro F1
- classwise metrics
- confusion matrix