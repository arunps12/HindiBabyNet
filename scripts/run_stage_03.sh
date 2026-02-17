#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Stage 03: Voice Activity Detection (VAD)
#
# Usage:
#   # Single analysis WAV:
#   uv run bash scripts/run_stage_03.sh --wav /path/to/audio_processed/<pid>/<pid>.wav
#
#   # All analysis WAVs under a directory (auto-discovers <pid>/<pid>.wav):
#   uv run bash scripts/run_stage_03.sh --analysis_dir /scratch/users/arunps/hindibabynet/audio_processed
#
#   # From recordings parquet (needs Stage 02 outputs to exist):
#   uv run bash scripts/run_stage_03.sh --recordings_parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
#
#   # Limit to first N participants (any mode):
#   uv run bash scripts/run_stage_03.sh --analysis_dir /path/to/processed --limit 3
# ===========================================================================

python -m src.hindibabynet.pipeline.stage_03_vad "$@"
