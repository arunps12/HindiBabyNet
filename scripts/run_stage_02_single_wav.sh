#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   uv run bash scripts/run_stage_02_single_wav.sh /path/to/input.wav [recording_id]

WAV_PATH="${1:?ERROR: provide wav path}"
RECORDING_ID="${2:-}"

CMD=(python -m src.hindibabynet.pipeline.stage_02_audio_preparation_single_wav --wav "$WAV_PATH")

if [[ -n "$RECORDING_ID" ]]; then
  CMD+=(--recording_id "$RECORDING_ID")
fi

"${CMD[@]}"
