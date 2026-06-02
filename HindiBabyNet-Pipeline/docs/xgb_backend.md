# XGB Backend

The XGB backend is the native HindiBabyNet-Pipeline speaker classifier.

It consumes prepared audio and produces:

- segment parquet tables
- class stream WAVs
- summary JSON files
- TextGrid outputs

Install the required stack with:

```bash
uv sync --extra xgb
```

Run it with:

```bash
uv run bash scripts/run_classify_xgb.sh --recordings-parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
```

Key parameters live under `xgb` in `configs/params.yaml`.