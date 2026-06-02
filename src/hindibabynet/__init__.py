"""Compatibility package for the renamed hindibabynet_pipeline package."""

from warnings import warn

from hindibabynet_pipeline import *  # noqa: F401,F403

warn(
	"Package 'hindibabynet' is deprecated; use 'hindibabynet_pipeline' instead.",
	DeprecationWarning,
	stacklevel=2,
)
