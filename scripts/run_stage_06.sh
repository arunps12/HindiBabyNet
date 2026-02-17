#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Stage 06: Speaker-type Classification + Stream Export
#
# Requires Stage 05 speech-segments parquet + analysis WAV.
#
# Usage:
#   # Single participant (explicit paths):
#   uv run bash scripts/run_stage_06.sh \
#       --speech_segments_parquet artifacts/runs/<run_id>/intersection/<pid>_speech_segments.parquet \
#       --wav /path/to/audio_processed/<pid>/<pid>.wav
#
#   # All from analysis directory (needs --run_id):
#   uv run bash scripts/run_stage_06.sh \
#       --analysis_dir /scratch/users/arunps/hindibabynet/audio_processed \
#       --run_id <run_id>
#
#   # From recordings parquet:
#   uv run bash scripts/run_stage_06.sh \
#       --recordings_parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet \
#       --run_id <run_id>
# ===========================================================================

python -m src.hindibabynet.pipeline.stage_06_speaker_classification "$@"
