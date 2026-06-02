#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] scripts/run_stage_02_single_wav.sh -> use scripts/run_prepare_audio.sh --wav ..." >&2

WAV_PATH="${1:?ERROR: provide wav path}"
RECORDING_ID="${2:-}"

CMD=(bash scripts/run_prepare_audio.sh --wav "$WAV_PATH")

if [[ -n "$RECORDING_ID" ]]; then
  CMD+=(--recording-id "$RECORDING_ID")
fi

"${CMD[@]}"
