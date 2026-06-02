#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] scripts/run_stage_03.sh -> use scripts/run_classify_xgb.sh or scripts/run_classify_vtc.sh" >&2
uv run python -m hindibabynet_pipeline.cli.run_stage_03 "$@"
