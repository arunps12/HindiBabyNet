# Audio Preparation

HindiBabyNet-Pipeline reads runtime path configuration from `configs/config.yaml` and audio-processing parameters from `configs/params.yaml`.

Use `uv run bash scripts/run_prepare_audio.sh` for task-based audio preparation.

Supported parameters in `params.yaml`:

- `join_multiple_files`
- `combine_gap_sec`
- `resample`
- `target_sr`
- `convert_to_mono`
- `normalize`
- `target_peak_dbfs`

Supported runtime flags in `config.yaml`:

- `audio_preparation.save_raw_joined_audio`
- `audio_preparation.save_prepared_audio`

Examples:

```bash
uv run bash scripts/run_prepare_audio.sh --recordings-parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
uv run bash scripts/run_prepare_audio.sh --wav /path/to/input.wav --recording-id REC001
```

Raw joined audio is written to the external `paths.raw_joined_audio_root`. Final prepared audio is written to `paths.prepared_audio_root` when `save_prepared_audio` is enabled.