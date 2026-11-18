from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hindibabynet_prosody.features.ssl import SSLExtractor


class FakeProcessor:
    def __call__(self, waveform, sampling_rate, return_tensors, padding):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        assert padding is True
        if isinstance(waveform, list):
            return {
                "input_values": torch.tensor(
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]],
                    dtype=torch.float32,
                ),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.long),
            }
        return {
            "input_values": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        }


class FakeModel:
    def __call__(self, **kwargs):
        assert kwargs["output_hidden_states"] is True
        if kwargs["input_values"].shape[0] == 2:
            hidden_states = (
                torch.zeros((2, 3, 2), dtype=torch.float32),
                torch.tensor(
                    [
                        [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
                        [[2.0, 2.0], [4.0, 4.0], [100.0, 100.0]],
                    ],
                    dtype=torch.float32,
                ),
            )
            return SimpleNamespace(hidden_states=hidden_states)
        hidden_states = (
            torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
            torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]], dtype=torch.float32),
        )
        return SimpleNamespace(hidden_states=hidden_states)

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        assert length == 3
        return attention_mask[:, :length]


def test_ssl_extractor_uses_attention_mask_aware_pooling() -> None:
    extractor = SSLExtractor(
        model_id="fake/model",
        layer=-1,
        device="cpu",
        minimum_segment_duration_s=0.1,
        maximum_segment_duration_s=None,
        processor=FakeProcessor(),
        model=FakeModel(),
    )

    waveform = np.ones(3200, dtype=np.float32)
    result = extractor.extract_embedding(waveform, 16000)

    assert result.success is True
    assert result.hidden_size == 2
    assert result.embedding is not None
    assert result.embedding.tolist() == pytest.approx([2.0, 2.0])


def test_ssl_extractor_rejects_short_segments() -> None:
    extractor = SSLExtractor(
        model_id="fake/model",
        layer=-1,
        device="cpu",
        minimum_segment_duration_s=0.5,
        maximum_segment_duration_s=None,
        processor=FakeProcessor(),
        model=FakeModel(),
    )

    waveform = np.ones(1000, dtype=np.float32)
    result = extractor.extract_embedding(waveform, 16000)

    assert result.success is False
    assert result.failure_reason == "segment_too_short"


def test_ssl_extractor_batches_embeddings() -> None:
    extractor = SSLExtractor(
        model_id="fake/model",
        layer=-1,
        device="cpu",
        minimum_segment_duration_s=0.1,
        maximum_segment_duration_s=None,
        processor=FakeProcessor(),
        model=FakeModel(),
    )

    results = extractor.extract_embeddings(
        [np.ones(3200, dtype=np.float32), np.ones(3200, dtype=np.float32)],
        16000,
    )

    assert [result.success for result in results] == [True, True]
    assert results[0].embedding is not None
    assert results[1].embedding is not None
    assert results[0].embedding.tolist() == pytest.approx([2.0, 2.0])
    assert results[1].embedding.tolist() == pytest.approx([3.0, 3.0])
