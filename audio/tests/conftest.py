"""Helpers for the tests: a temporary config.yaml, the settings and small synthetic files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from hbn_paths import Config

from hindibabynet_audio.params import load_params


@pytest.fixture
def make_config(tmp_path: Path):
    """Return a function that writes a temporary root config.yaml and loads it.

    paths: folders stored in another place (the ``paths`` part of config.yaml).
    audio: local choices of the audio workflow (vtc_repo, vtc_device).

    The config file is saved as <tmp_path>/project/config.yaml and the main
    data folder is <tmp_path>/data. No folder is created.
    """

    def _make(paths: dict | None = None, audio: dict | None = None) -> Config:
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)
        values: dict = {"data_dir": str(tmp_path / "data")}
        if paths:
            values["paths"] = {key: str(value) for key, value in paths.items()}
        if audio:
            values["audio"] = {key: str(value) for key, value in audio.items()}
        config_file = project / "config.yaml"
        config_file.write_text(yaml.safe_dump(values), encoding="utf-8")
        return Config(config_file)

    return _make


@pytest.fixture
def params() -> dict:
    """The settings of audio/params.yaml as a dictionary that a test can change."""
    return load_params()


@pytest.fixture
def write_wav():
    """Return a function that writes a small WAV file with a tone (PCM 16 bit)."""

    def _write(
        path: Path,
        seconds: float = 0.5,
        sr: int = 44100,
        channels: int = 2,
        amplitude: float = 0.5,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(int(round(seconds * sr))) / sr
        tone = (amplitude * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
        if channels > 1:
            # The second channel is a little weaker than the first one.
            tone = np.stack([tone * (1.0 - 0.2 * c) for c in range(channels)], axis=1)
        sf.write(str(path), tone, sr, format="WAV", subtype="PCM_16")
        return path

    return _write


@pytest.fixture
def write_asr():
    """Return a function that writes an empty .ASR file (saved by the recorder)."""

    def _write(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    return _write
