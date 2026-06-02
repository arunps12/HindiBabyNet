#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] scripts/run_stage_01.sh -> use uv run python -m hindibabynet_pipeline.pipeline.stage_01_data_ingestion" >&2
uv run python -m hindibabynet_pipeline.pipeline.stage_01_data_ingestion "$@"
