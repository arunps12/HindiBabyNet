#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] scripts/run_stage_02_from_parquet.sh -> use scripts/run_prepare_audio.sh --recordings-parquet ..." >&2

REC_PARQUET="${1:?ERROR: provide recordings.parquet path}"
LIMIT="${2:-}"

CMD=(bash scripts/run_prepare_audio.sh --recordings-parquet "$REC_PARQUET")

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"
