# src/hindibabynet/exception/exception.py
from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class HindiBabyNetError(Exception):
    """
    Base exception for HindiBabyNet.

    Adds optional context so errors are informative in HPC logs.
    """
    message: str
    cause: Optional[BaseException] = None
    context: Optional[dict] = None

    def __str__(self) -> str:
        ctx = f" | context={self.context}" if self.context else ""
        if self.cause:
            return f"{self.message}{ctx} | cause={repr(self.cause)}"
        return f"{self.message}{ctx}"


def wrap_exception(message: str, exc: BaseException, context: Optional[dict] = None) -> HindiBabyNetError:
    """
    Helper to wrap any exception with a HindiBabyNetError, preserving traceback.
    """
    err = HindiBabyNetError(message=message, cause=exc, context=context)
    # Attach original traceback for debugging (keeps full trace in logs)
    err.__cause__ = exc
    return err


def format_traceback(exc: BaseException) -> str:
    """
    Convert an exception traceback to string.
    """
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
