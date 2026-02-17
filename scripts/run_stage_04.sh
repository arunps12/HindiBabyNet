#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Stage 04: Speaker Diarization
#
# Usage:
#   # Single analysis WAV:
#   uv run bash scripts/run_stage_04.sh --wav /path/to/audio_processed/<pid>/<pid>.wav
#
#   # All analysis WAVs under a directory:
#   uv run bash scripts/run_stage_04.sh --analysis_dir /scratch/users/arunps/hindibabynet/audio_processed
#
#   # From recordings parquet:
#   uv run bash scripts/run_stage_04.sh --recordings_parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
#
#   # Limit to first N participants:
#   uv run bash scripts/run_stage_04.sh --analysis_dir /path/to/processed --limit 3
# ===========================================================================

python -m src.hindibabynet.pipeline.stage_04_diarization "$@"
