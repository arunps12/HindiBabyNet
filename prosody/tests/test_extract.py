from __future__ import annotations

import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from hindibabynet_prosody.features.ssl import SSLExtractionResult
from hindibabynet_prosody.pipeline import extract as extract_pipeline
from hindibabynet_prosody.pipeline.extract import run_extraction


class FakeSSLExtractor:
    def __init__(self, prefix: float) -> None:
        self.prefix = prefix
        self.calls = 0
        self.batch_calls = 0

    def extract_embedding(self, waveform: np.ndarray, sample_rate: int) -> SSLExtractionResult:
        self.calls += 1
        return SSLExtractionResult(
            embedding=np.asarray(
                [self.prefix + self.calls, self.prefix + self.calls + 1, self.prefix + self.calls + 2],
                dtype=np.float32,
            ),
            success=True,
            failure_reason=None,
            hidden_size=3,
        )

    def extract_embeddings(self, waveforms: list[np.ndarray], sample_rate: int) -> list[SSLExtractionResult]:
        self.batch_calls += 1
        results: list[SSLExtractionResult] = []
        for _waveform in waveforms:
            self.calls += 1
            results.append(
                SSLExtractionResult(
                    embedding=np.asarray(
                        [self.prefix + self.calls, self.prefix + self.calls + 1, self.prefix + self.calls + 2],
                        dtype=np.float32,
                    ),
                    success=True,
                    failure_reason=None,
                    hidden_size=3,
                )
            )
        return results


class FailingSSLExtractor:
    def extract_embedding(self, waveform: np.ndarray, sample_rate: int) -> SSLExtractionResult:
        raise AssertionError("cache was not used")

    def extract_embeddings(self, waveforms: list[np.ndarray], sample_rate: int) -> list[SSLExtractionResult]:
        raise AssertionError("cache was not used")


class BatchOnlySSLExtractor:
    def __init__(self, prefix: float) -> None:
        self.prefix = prefix
        self.calls = 0

    def extract_embedding(self, waveform: np.ndarray, sample_rate: int) -> SSLExtractionResult:
        raise AssertionError("single-item SSL path should not be used")

    def extract_embeddings(self, waveforms: list[np.ndarray], sample_rate: int) -> list[SSLExtractionResult]:
        results: list[SSLExtractionResult] = []
        for _waveform in waveforms:
            self.calls += 1
            results.append(
                SSLExtractionResult(
                    embedding=np.asarray(
                        [self.prefix + self.calls, self.prefix + self.calls + 1, self.prefix + self.calls + 2],
                        dtype=np.float32,
                    ),
                    success=True,
                    failure_reason=None,
                    hidden_size=3,
                )
            )
        return results


def _test_params() -> dict:
    """Settings of the tests. The folders come from the temporary config.yaml."""
    return {
        "project": {"name": "test-prosody", "random_seed": 42},
        "rttm": {
            "file_pattern": "**/rttm.csv",
            "participant_column": "uid",
            "start_column": "start_time_s",
            "duration_column": "duration_s",
            "speaker_column": "label",
        },
        "mapping": {"strict": True, "source_id_column": "original_par_id", "participant_id_column": "participant_id"},
        "speakers": {"include": {"FEM": "adult_female", "MAL": "adult_male", "OCH": "other_child"}},
        "audio": {
            "extensions": [".wav"],
            "preferred_extensions": [".wav"],
            "target_sample_rate": 16000,
            "mono": True,
            "end_tolerance_s": 0.05,
        },
        "pitch": {
            "method": "praat_ac",
            "speaker_settings": {
                "FEM": {"pitch_floor_hz": 75.0, "pitch_ceiling_hz": 700.0},
                "MAL": {"pitch_floor_hz": 50.0, "pitch_ceiling_hz": 400.0},
                "OCH": {"pitch_floor_hz": 100.0, "pitch_ceiling_hz": 800.0},
            },
            "time_step_s": 0.01,
            "lower_percentile": 5.0,
            "upper_percentile": 95.0,
            "minimum_voiced_frames": 3,
            "semitone_reference_hz": 1.0,
            "adaptive": {
                "enabled": True,
                "level": "segment",
                "lower_multiplier": 0.5,
                "upper_multiplier": 2.0,
                "minimum_floor_hz": 40.0,
                "maximum_ceiling_hz": 1000.0,
                "fallback_to_first_pass": True,
            },
            "praat": {
                "voicing_threshold": 0.45,
                "octave_cost": 0.01,
                "octave_jump_cost": 0.35,
                "voiced_unvoiced_cost": 0.14,
            },
        },
        "ssl": {
            "wav2vec2_model": "fake/wav2vec",
            "hubert_model": "fake/hubert",
            "wav2vec2_layer": -1,
            "hubert_layer": -1,
            "temporal_pooling": "mean",
            "segment_pooling": "mean",
            "minimum_segment_duration_s": 0.1,
            "maximum_segment_duration_s": None,
        },
        "pca": {"n_components": 3, "standardize": True, "random_seed": 42},
        "runtime": {"device": "cpu", "batch_size": 1, "num_workers": 0, "strict": False, "log_level": "INFO"},
        "output": {"final_filename": "participant_speaker_acoustic_features.csv", "save_pca_artifacts": True, "save_qc_reports": True},
    }


def _make_sine(frequency_hz: float, seconds: float = 0.6, sample_rate: int = 16000) -> np.ndarray:
    timeline = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    return (0.2 * np.sin(2.0 * np.pi * frequency_hz * timeline)).astype(np.float32)


def _build_fixture(make_config, params: dict | None = None):
    """One synthetic participant in the folders of a temporary config. Returns the config."""
    config = make_config(params=params or _test_params())
    audio_root = config.paths.audio_root
    rttm_root = config.paths.rttm_root
    (audio_root / "SRC001").mkdir(parents=True)
    (rttm_root / "SRC001").mkdir(parents=True)
    audio = np.concatenate([_make_sine(220.0), _make_sine(140.0), _make_sine(300.0), _make_sine(220.0)])
    sf.write(audio_root / "SRC001" / "SRC001.wav", audio, 16000)
    config.paths.participant_lookup.parent.mkdir(parents=True, exist_ok=True)
    config.paths.participant_lookup.write_text(
        "original_par_id,participant_id\nSRC001,P001\n", encoding="utf-8"
    )
    (rttm_root / "SRC001" / "rttm.csv").write_text(
        "uid,start_time_s,duration_s,label\n"
        "SRC001,0.0,0.5,FEM\n"
        "SRC001,0.6,0.5,MAL\n"
        "SRC001,1.2,0.5,OCH\n"
        "SRC001,1.8,0.4,KCHI\n",
        encoding="utf-8",
    )
    return config


def _build_multi_participant_fixture(make_config, params: dict | None = None):
    """Two synthetic participants in the folders of a temporary config. Returns the config."""
    config = make_config(params=params or _test_params())
    audio_root = config.paths.audio_root
    rttm_root = config.paths.rttm_root
    for source_id, frequencies in {
        "SRC001": [220.0, 140.0, 300.0, 220.0],
        "SRC002": [240.0, 160.0, 320.0, 240.0],
    }.items():
        (audio_root / source_id).mkdir(parents=True, exist_ok=True)
        (rttm_root / source_id).mkdir(parents=True, exist_ok=True)
        audio = np.concatenate([_make_sine(frequency_hz) for frequency_hz in frequencies])
        sf.write(audio_root / source_id / f"{source_id}.wav", audio, 16000)
        (rttm_root / source_id / "rttm.csv").write_text(
            "uid,start_time_s,duration_s,label\n"
            f"{source_id},0.0,0.5,FEM\n"
            f"{source_id},0.6,0.5,MAL\n"
            f"{source_id},1.2,0.5,OCH\n"
            f"{source_id},1.8,0.4,KCHI\n",
            encoding="utf-8",
        )
    config.paths.participant_lookup.parent.mkdir(parents=True, exist_ok=True)
    config.paths.participant_lookup.write_text(
        "original_par_id,participant_id\nSRC001,P001\nSRC002,P002\n",
        encoding="utf-8",
    )
    return config


class InlineProcessPoolExecutor:
    def __init__(self, *, max_workers: int, initializer=None, initargs=()) -> None:
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self) -> "InlineProcessPoolExecutor":
        if self.initializer is not None:
            self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


def test_run_extraction_writes_outputs_and_qc(make_config) -> None:
    config = _build_fixture(make_config)
    outputs = run_extraction(
        config,
        ssl_extractors={"wav2vec": FakeSSLExtractor(10.0), "hubert": FakeSSLExtractor(20.0)},
    )

    final_frame = pd.read_csv(outputs.final_csv)
    qc_frame = pd.read_csv(outputs.qc_summary_csv)
    skipped_frame = pd.read_csv(outputs.skipped_segments_csv)

    assert outputs.final_csv.exists()
    assert final_frame["speaker"].tolist() == ["adult_female", "adult_male", "other_child"]
    assert {"wav2vec_PC1", "wav2vec_PC2", "wav2vec_PC3", "hubert_PC1", "hubert_PC2", "hubert_PC3"} <= set(final_frame.columns)
    assert len(qc_frame) == 3
    assert "unsupported_speaker_label" in set(skipped_frame["failure_reason"].dropna())
    assert outputs.run_metadata_json.exists()
    assert (config.paths.cache_dir / "SRC001.joblib").exists()


def test_run_extraction_reuses_saved_participant_results(make_config, tmp_path: Path) -> None:
    config = _build_fixture(make_config)
    first = run_extraction(
        config,
        ssl_extractors={"wav2vec": FakeSSLExtractor(10.0), "hubert": FakeSSLExtractor(20.0)},
    )
    # Earlier results are never replaced, so the first results are moved away.
    first.final_csv.parent.rename(tmp_path / "first_features")

    outputs = run_extraction(
        config,
        ssl_extractors={"wav2vec": FailingSSLExtractor(), "hubert": FailingSSLExtractor()},
    )

    assert outputs.final_csv.exists()


def test_run_extraction_without_cache_calculates_again(make_config, tmp_path: Path) -> None:
    config = _build_fixture(make_config)
    first = run_extraction(
        config,
        ssl_extractors={"wav2vec": FakeSSLExtractor(10.0), "hubert": FakeSSLExtractor(20.0)},
    )
    first.final_csv.parent.rename(tmp_path / "first_features")

    with pytest.raises(AssertionError, match="cache was not used"):
        run_extraction(
            config,
            no_cache=True,
            ssl_extractors={"wav2vec": FailingSSLExtractor(), "hubert": FailingSSLExtractor()},
        )


def test_run_for_selected_speakers_does_not_save_participant_results(make_config) -> None:
    config = _build_fixture(make_config)
    run_extraction(
        config,
        speakers=["FEM", "MAL", "OCH"],
        ssl_extractors={"wav2vec": FakeSSLExtractor(10.0), "hubert": FakeSSLExtractor(20.0)},
    )

    assert not (config.paths.cache_dir / "SRC001.joblib").exists()


def test_run_extraction_supports_participant_speaker_adaptation(make_config) -> None:
    params = _test_params()
    params["pitch"]["adaptive"]["level"] = "participant_speaker"
    config = _build_fixture(make_config, params)

    outputs = run_extraction(
        config,
        ssl_extractors={"wav2vec": FakeSSLExtractor(10.0), "hubert": FakeSSLExtractor(20.0)},
    )

    final_frame = pd.read_csv(outputs.final_csv)
    qc_frame = pd.read_csv(outputs.qc_summary_csv)
    assert not final_frame.empty
    assert "median_adaptive_floor_Hz" in qc_frame.columns


def test_run_extraction_uses_batched_ssl_path(make_config) -> None:
    params = _test_params()
    params["runtime"]["batch_size"] = 2
    config = _build_fixture(make_config, params)

    wav2vec = BatchOnlySSLExtractor(10.0)
    hubert = BatchOnlySSLExtractor(20.0)
    outputs = run_extraction(
        config,
        ssl_extractors={"wav2vec": wav2vec, "hubert": hubert},
    )

    assert outputs.final_csv.exists()
    assert wav2vec.calls == 3
    assert hubert.calls == 3


def test_run_extraction_uses_participant_worker_pool(make_config, monkeypatch, caplog) -> None:
    params = _test_params()
    params["runtime"]["num_workers"] = 2
    config = _build_multi_participant_fixture(make_config, params)

    monkeypatch.setattr(extract_pipeline, "_PROCESS_POOL_EXECUTOR", InlineProcessPoolExecutor)
    caplog.set_level("INFO", logger="hindibabynet_prosody.pipeline.extract")
    outputs = run_extraction(
        config,
        ssl_extractors={"wav2vec": FakeSSLExtractor(10.0), "hubert": FakeSSLExtractor(20.0)},
    )

    final_frame = pd.read_csv(outputs.final_csv)
    assert outputs.final_csv.exists()
    assert set(final_frame["participant_id"]) == {"P001", "P002"}
    assert len(final_frame) == 6
    assert "worker=" in caplog.text
    assert "processed_segments=1/4" in caplog.text
    assert "completed_participants=2/2" in caplog.text

