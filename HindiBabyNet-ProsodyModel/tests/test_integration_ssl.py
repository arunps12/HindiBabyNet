from __future__ import annotations

import os

import numpy as np
import pytest

from hindibabynet_prosody.features.ssl import SSLExtractor


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("HINDIBABYNET_RUN_INTEGRATION") != "1",
    reason="Set HINDIBABYNET_RUN_INTEGRATION=1 to run real Hugging Face model integration tests.",
)
def test_ssl_extractor_downloads_real_model() -> None:
    extractor = SSLExtractor(
        model_id="hf-internal-testing/tiny-random-wav2vec2",
        layer=-1,
        device="cpu",
        minimum_segment_duration_s=0.1,
        maximum_segment_duration_s=None,
    )
    waveform = np.ones(3200, dtype=np.float32)
    result = extractor.extract_embedding(waveform, 16000)

    assert result.success is True
    assert result.embedding is not None
    assert result.hidden_size is not None