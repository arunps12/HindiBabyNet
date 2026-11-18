from __future__ import annotations

from hindibabynet_prosody.features.aggregation import median_or_none


def test_median_or_none_is_deterministic() -> None:
    assert median_or_none([4.0, 2.0, 6.0]) == 4.0
    assert median_or_none([]) is None
