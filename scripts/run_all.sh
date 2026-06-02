#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated] scripts/run_all.sh -> use scripts/run_pipeline.sh" >&2
exec bash scripts/run_pipeline.sh "$@"
