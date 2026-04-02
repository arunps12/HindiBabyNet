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

# Backward-compatible lazy re-export so old imports still resolve without
# forcing XGB-only deps (opensmile/torch) in VTC-only environments.
class SpeakerClassification:  # noqa: N801
    def __new__(cls, *args, **kwargs):
        from src.hindibabynet.components.speaker_classification._xgb_core import (
            SpeakerClassification as _SpeakerClassification,
        )

        return _SpeakerClassification(*args, **kwargs)

__all__ = [
    "ClassificationBackend",
    "get_backend",
    "is_stage03_complete",
    "SpeakerClassification",
]
