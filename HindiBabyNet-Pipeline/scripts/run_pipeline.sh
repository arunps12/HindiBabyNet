#!/usr/bin/env bash
set -euo pipefail

uv run python -m hindibabynet_pipeline.cli.pipeline "$@"