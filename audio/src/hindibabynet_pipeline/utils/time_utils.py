"""Time/run-id utilities."""
from __future__ import annotations

import time


def make_run_id() -> str:
    """Return a readable, sortable run id (``YYYYMMDD_HHMMSS``)."""
    return time.strftime("%Y%m%d_%H%M%S")
