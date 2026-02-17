#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Stage 05: VAD ∩ Diarization Intersection
#
# Requires Stage 03 + Stage 04 artifacts for the same --run_id.
#
# Usage:
#   # Single participant (explicit parquets):
#   uv run bash scripts/run_stage_05.sh \
#       --vad_parquet artifacts/runs/<run_id>/vad/<pid>_vad.parquet \
#       --diar_parquet artifacts/runs/<run_id>/diarization/<pid>_diarization.parquet
#
#   # All from analysis directory (needs --run_id):
#   uv run bash scripts/run_stage_05.sh \
#       --analysis_dir /scratch/users/arunps/hindibabynet/audio_processed \
#       --run_id <run_id>
#
#   # From recordings parquet:
#   uv run bash scripts/run_stage_05.sh \
#       --recordings_parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet \
#       --run_id <run_id>
# ===========================================================================

python -m src.hindibabynet.pipeline.stage_05_intersection "$@"
