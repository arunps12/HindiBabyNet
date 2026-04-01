"""Abstract base for Stage 03 speaker classification backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ClassificationBackend(ABC):
    """
    Protocol for a Stage 03 speaker-classification backend.

    Every backend receives a prepared analysis WAV (mono 16 kHz, from Stage 02)
    and writes classification outputs to a participant-specific output directory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short backend identifier, e.g. ``'xgb'`` or ``'vtc'``."""

    @abstractmethod
    def run_participant(
        self,
        wav_path: Path,
        participant_id: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        """
        Run classification on a single participant.

        Parameters
        ----------
        wav_path : Path
            Analysis-ready WAV from Stage 02
            (``<processed_audio_root>/<pid>/<pid>.wav``).
        participant_id : str
            Participant identifier.
        output_dir : Path
            Directory to write all outputs for this participant.

        Returns
        -------
        dict
            Run metadata (written as ``run_info.json``).
        """

    @abstractmethod
    def is_complete(self, participant_id: str, output_dir: Path) -> bool:
        """Return True if all expected outputs already exist."""

    @abstractmethod
    def expected_outputs(self, participant_id: str) -> list[str]:
        """Return a list of expected output file/directory names (for docs / diagnostics)."""
