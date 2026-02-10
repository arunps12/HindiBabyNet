#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Run the full HindiBabyNet pipeline end-to-end:
#   Stage 01  →  Data Ingestion   (scan raw audio, produce recordings.parquet)
#   Stage 02  →  Audio Preparation (combine, mono, 16 kHz, normalize per participant)
#   Stage 03  →  Speaker Classification (VAD → diar → classify → export)
#
# Usage:
#   uv run bash scripts/run_all.sh              # all participants
#   uv run bash scripts/run_all.sh --limit 3    # first 3 participants only
# ===========================================================================

LIMIT="${1:-}"   # optional: --limit N

echo "===== Stage 01: Data Ingestion ====="
python -m src.hindibabynet.pipeline.stage_01_data_ingestion

# Find the latest recordings.parquet
LATEST_DI=$(ls -td artifacts/runs/*/data_ingestion 2>/dev/null | head -1)
REC_PARQUET="${LATEST_DI}/recordings.parquet"

if [[ ! -f "$REC_PARQUET" ]]; then
    echo "ERROR: recordings.parquet not found at $REC_PARQUET"
    exit 1
fi
echo "Recordings parquet: $REC_PARQUET"

echo ""
echo "===== Stage 02: Audio Preparation ====="
CMD_02=(python -m src.hindibabynet.pipeline.stage_02_audio_preparation_from_parquet
        --recordings_parquet "$REC_PARQUET")
if [[ "$LIMIT" == "--limit" ]]; then
    CMD_02+=(--limit "${2:?ERROR: --limit requires a number}")
elif [[ -n "$LIMIT" ]]; then
    # Allow: run_all.sh 3  (just a number)
    CMD_02+=(--limit "$LIMIT")
fi
"${CMD_02[@]}"

echo ""
echo "===== Stage 03: Speaker Classification ====="
CMD_03=(python -m src.hindibabynet.pipeline.stage_03_speaker_classification
        --recordings_parquet "$REC_PARQUET")
if [[ "$LIMIT" == "--limit" ]]; then
    CMD_03+=(--limit "${2}")
elif [[ -n "$LIMIT" ]]; then
    CMD_03+=(--limit "$LIMIT")
fi
"${CMD_03[@]}"

echo ""
echo "===== All stages complete ====="
