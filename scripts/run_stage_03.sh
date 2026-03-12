#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Stage 03: Speaker Classification
#
# Supports two backends:
#   xgb  — HindiBabyNet (VAD → diarization → eGeMAPS → XGBoost)  [default]
#   vtc  — External VTC 2.0 voice-type classifier
#
# Usage:
#   # Single analysis WAV (default xgb backend):
#   uv run bash scripts/run_stage_03.sh --wav /path/to/audio_processed/<pid>/<pid>.wav
#
#   # Single analysis WAV with VTC backend:
#   uv run bash scripts/run_stage_03.sh --wav /path/to/audio_processed/<pid>/<pid>.wav --backend vtc
#
#   # All analysis WAVs under a directory (auto-discovers <pid>/<pid>.wav):
#   uv run bash scripts/run_stage_03.sh --analysis_dir /scratch/users/arunps/hindibabynet/audio_processed
#
#   # All analysis WAVs with VTC backend:
#   uv run bash scripts/run_stage_03.sh --analysis_dir /path/to/processed --backend vtc
#
#   # From recordings parquet (needs Stage 02 outputs to exist):
#   uv run bash scripts/run_stage_03.sh --recordings_parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
#
#   # Limit to first N participants (any mode):
#   uv run bash scripts/run_stage_03.sh --analysis_dir /path/to/processed --limit 3
# ===========================================================================

python -m src.hindibabynet.pipeline.stage_03_speaker_classification "$@"
