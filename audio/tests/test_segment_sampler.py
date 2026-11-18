from __future__ import annotations

import pandas as pd

from hindibabynet_pipeline.components.annotation.segment_sampler import sample_segments


def test_segment_sampler_is_reproducible():
    df = pd.DataFrame(
        {
            "start_sec": list(range(100)),
            "end_sec": [value + 0.5 for value in range(100)],
            "predicted_class": ["adult_female"] * 100,
        }
    )

    first = sample_segments(df, n_segments=10, random_seed=42)
    second = sample_segments(df, n_segments=10, random_seed=42)

    assert first.equals(second)
    assert len(first) == 10