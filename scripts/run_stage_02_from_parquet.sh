#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   uv run bash scripts/run_stage_02_batch_from_parquet.sh <recordings.parquet> [limit]
#
# Example:
#   uv run bash scripts/run_stage_02_batch_from_parquet.sh artifacts/runs/<run_id>/data_ingestion/recordings.parquet
#   uv run bash scripts/run_stage_02_batch_from_parquet.sh artifacts/runs/<run_id>/data_ingestion/recordings.parquet 2

REC_PARQUET="${1:?ERROR: provide recordings.parquet path}"
LIMIT="${2:-}"

CMD=(python -m src.hindibabynet.pipeline.stage_02_audio_preparation_from_parquet
     --recordings_parquet "$REC_PARQUET")

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"
