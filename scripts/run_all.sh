#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Run the full HindiBabyNet pipeline end-to-end:
#   Stage 01  →  Data Ingestion   (scan raw audio, produce recordings.parquet)
#   Stage 02  →  Audio Preparation (combine, mono, 16 kHz, normalize per participant)
#   Stage 03  →  VAD              (WebRTC voice activity detection → parquet)
#   Stage 04  →  Diarization      (pyannote speaker diarization → parquet)
#   Stage 05  →  Intersection     (VAD ∩ diarization → parquet)
#   Stage 06  →  Classification   (eGeMAPS + XGBoost → per-class WAVs + parquets)
#
# Usage:
#   uv run bash scripts/run_all.sh              # all participants
#   uv run bash scripts/run_all.sh --limit 3    # first 3 participants only
# ===========================================================================

LIMIT="${1:-}"   # optional: --limit N

# Generate a single RUN_ID to keep all stages' artifacts together
RUN_ID=$(date +%Y%m%d_%H%M%S)
echo "Run ID: $RUN_ID"

echo "===== Stage 01: Data Ingestion ====="
python -m src.hindibabynet.pipeline.stage_01_data_ingestion \
    --run_id "$RUN_ID"

# recordings.parquet is under the shared RUN_ID
REC_PARQUET="artifacts/runs/${RUN_ID}/data_ingestion/recordings.parquet"

if [[ ! -f "$REC_PARQUET" ]]; then
    echo "ERROR: recordings.parquet not found at $REC_PARQUET"
    exit 1
fi
echo "Recordings parquet: $REC_PARQUET"

# Build limit args
LIMIT_ARGS=()
if [[ "$LIMIT" == "--limit" ]]; then
    LIMIT_ARGS=(--limit "${2:?ERROR: --limit requires a number}")
elif [[ -n "$LIMIT" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

echo ""
echo "===== Stage 02: Audio Preparation ====="
python -m src.hindibabynet.pipeline.stage_02_audio_preparation_from_parquet \
    --recordings_parquet "$REC_PARQUET" --run_id "$RUN_ID" \
    "${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}"

echo ""
echo "===== Stage 03: VAD ====="
python -m src.hindibabynet.pipeline.stage_03_vad \
    --recordings_parquet "$REC_PARQUET" --run_id "$RUN_ID" \
    "${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}"

echo ""
echo "===== Stage 04: Diarization ====="
python -m src.hindibabynet.pipeline.stage_04_diarization \
    --recordings_parquet "$REC_PARQUET" --run_id "$RUN_ID" \
    "${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}"

echo ""
echo "===== Stage 05: Intersection ====="
python -m src.hindibabynet.pipeline.stage_05_intersection \
    --recordings_parquet "$REC_PARQUET" --run_id "$RUN_ID" \
    "${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}"

echo ""
echo "===== Stage 06: Speaker Classification ====="
python -m src.hindibabynet.pipeline.stage_06_speaker_classification \
    --recordings_parquet "$REC_PARQUET" --run_id "$RUN_ID" \
    "${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}"

echo ""
echo "===== All stages complete (run_id=$RUN_ID) ====="
