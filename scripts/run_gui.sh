#!/usr/bin/env bash
set -euo pipefail
# Launch the HindiBabyNet GUI
# Usage:
#   bash scripts/run_gui.sh
#   uv run bash scripts/run_gui.sh

cd "$(dirname "$0")/.."
exec python -m hindibabynet_gui
