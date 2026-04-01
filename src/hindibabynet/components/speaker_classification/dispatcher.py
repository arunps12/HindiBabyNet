"""
Backend dispatcher — select the right Stage 03 backend from config.

Usage::

    from src.hindibabynet.components.speaker_classification.dispatcher import get_backend

    backend = get_backend(cfg)          # reads speaker_classification.backend
    backend = get_backend(cfg, "vtc")   # explicit override
"""
from __future__ import annotations

from src.hindibabynet.components.speaker_classification.base import ClassificationBackend
from src.hindibabynet.config.configuration import ConfigurationManager

_VALID_BACKENDS = ("xgb", "vtc")


def get_backend(
    cfg: ConfigurationManager,
    override: str | None = None,
) -> ClassificationBackend:
    """
    Return the appropriate :class:`ClassificationBackend` instance.

    Parameters
    ----------
    cfg : ConfigurationManager
        Loaded project configuration.
    override : str, optional
        If given, overrides the ``speaker_classification.backend`` value
        from the config file.  Useful for CLI ``--backend`` flags.

    Raises
    ------
    ValueError
        If the requested backend name is not supported.
    """
    name = (override or cfg.get_speaker_classification_backend()).strip().lower()

    if name == "xgb":
        from src.hindibabynet.components.speaker_classification.xgb_backend import XGBBackend
        return XGBBackend(cfg)

    if name == "vtc":
        from src.hindibabynet.components.speaker_classification.vtc_backend import VTCBackend
        return VTCBackend(cfg)

    raise ValueError(
        f"Unknown speaker-classification backend '{name}'. "
        f"Valid backends: {_VALID_BACKENDS}"
    )
