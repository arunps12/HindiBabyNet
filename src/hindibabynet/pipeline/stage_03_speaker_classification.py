"""Backward-compatible Stage 03 entrypoint.

This module now delegates to the new config-driven CLI implementation:
`src.hindibabynet.cli.run_stage_03`.
"""
from __future__ import annotations

from src.hindibabynet.cli.run_stage_03 import main

_VALID_BACKENDS = ("xgb", "vtc")


if __name__ == "__main__":
    main()
