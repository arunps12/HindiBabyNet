"""SSL embedding extraction wrappers for wav2vec2 and HuBERT models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor


@dataclass(frozen=True)
class SSLExtractionResult:
    embedding: np.ndarray | None
    success: bool
    failure_reason: str | None
    hidden_size: int | None


def resolve_torch_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    return requested_device


class SSLExtractor:
    def __init__(
        self,
        *,
        model_id: str,
        layer: int,
        device: str,
        minimum_segment_duration_s: float,
        maximum_segment_duration_s: float | None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.layer = layer
        self.device = resolve_torch_device(device)
        self.minimum_segment_duration_s = minimum_segment_duration_s
        self.maximum_segment_duration_s = maximum_segment_duration_s
        self._processor = processor
        self._model = model

    @property
    def processor(self) -> Any:
        if self._processor is None:
            try:
                self._processor = AutoProcessor.from_pretrained(self.model_id)
            except (TypeError, ValueError, OSError):
                self._processor = AutoFeatureExtractor.from_pretrained(self.model_id)
        return self._processor

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def extract_embedding(self, waveform: np.ndarray, sample_rate: int) -> SSLExtractionResult:
        return self.extract_embeddings([waveform], sample_rate)[0]

    def extract_embeddings(
        self,
        waveforms: list[np.ndarray],
        sample_rate: int,
    ) -> list[SSLExtractionResult]:
        if not waveforms:
            return []

        results: list[SSLExtractionResult | None] = [None] * len(waveforms)
        valid_indices: list[int] = []
        valid_waveforms: list[np.ndarray] = []

        for index, waveform in enumerate(waveforms):
            if waveform.size == 0:
                results[index] = SSLExtractionResult(None, False, "empty_waveform", None)
                continue
            duration_s = waveform.size / float(sample_rate)
            if duration_s < self.minimum_segment_duration_s:
                results[index] = SSLExtractionResult(None, False, "segment_too_short", None)
                continue
            if self.maximum_segment_duration_s is not None and duration_s > self.maximum_segment_duration_s:
                results[index] = SSLExtractionResult(None, False, "segment_too_long", None)
                continue
            valid_indices.append(index)
            valid_waveforms.append(waveform)

        if not valid_waveforms:
            return [result for result in results if result is not None]

        inputs = self.processor(
            valid_waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**model_inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[self.layer]
        if hidden_states.ndim != 3:
            failure = SSLExtractionResult(None, False, "unexpected_hidden_state_shape", None)
            for index in valid_indices:
                results[index] = failure
            return [result for result in results if result is not None]

        hidden_size = int(hidden_states.shape[-1])
        attention_mask = model_inputs.get("attention_mask")
        feature_mask = None
        if attention_mask is not None:
            if hasattr(self.model, "_get_feature_vector_attention_mask"):
                feature_mask = self.model._get_feature_vector_attention_mask(
                    hidden_states.shape[1],
                    attention_mask,
                )
            else:
                feature_mask = attention_mask[:, : hidden_states.shape[1]]

        for batch_index, result_index in enumerate(valid_indices):
            hidden = hidden_states[batch_index]
            if feature_mask is not None:
                mask = feature_mask[batch_index].to(hidden.device).bool()
                if not bool(mask.any()):
                    results[result_index] = SSLExtractionResult(None, False, "empty_feature_mask", hidden_size)
                    continue
                pooled = hidden[mask].mean(dim=0)
            else:
                pooled = hidden.mean(dim=0)
            results[result_index] = SSLExtractionResult(
                embedding=pooled.detach().cpu().numpy().astype(np.float32),
                success=True,
                failure_reason=None,
                hidden_size=hidden_size,
            )

        return [result for result in results if result is not None]