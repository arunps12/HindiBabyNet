"""
Speaker Classification — backend-agnostic package.

Provides a unified interface for Stage 03 speaker classification
with pluggable backends (xgb, vtc).

Usage::

    from src.hindibabynet.components.speaker_classification import get_backend

    backend = get_backend(cfg)
    backend.run_participant(wav_path, participant_id, output_dir)
"""
from src.hindibabynet.components.speaker_classification.base import ClassificationBackend
from src.hindibabynet.components.speaker_classification.dispatcher import get_backend
from src.hindibabynet.components.speaker_classification.output_checks import is_stage03_complete

# Backward-compatible re-export so old imports still resolve:
#   from src.hindibabynet.components.speaker_classification import SpeakerClassification
from src.hindibabynet.components.speaker_classification._xgb_core import SpeakerClassification

__all__ = [
    "ClassificationBackend",
    "get_backend",
    "is_stage03_complete",
    "SpeakerClassification",
]
