# VTC Backend

The VTC backend wraps an external VTC repository configured in `configs/config.yaml`.

Expected runtime settings:

- `speaker_classification.backend: vtc`
- `speaker_classification.vtc.repo_path`
- `speaker_classification.vtc.device`
- `speaker_classification.vtc.keep_inputs`

Run it with:

```bash
uv run bash scripts/run_classify_vtc.sh --recordings-parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
```

The VTC backend emits `rttm.csv` files using labels such as `FEM`, `MAL`, `KCHI`, and `OCH`. The pipeline normalizes those labels in downstream TextGrid and annotation workflows.