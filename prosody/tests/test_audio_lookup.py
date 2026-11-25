"""The prepared audio of a participant is found in the places that hbn-audio prepare uses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hindibabynet_prosody.io.audio import resolve_audio_for_source


def _write_wav(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * 16000), dtype=np.float32), 16000,
             format="WAV", subtype="PCM_16")
    return path


@pytest.mark.parametrize(
    "place", ["SRC001.wav", "SRC001.WAV", "SRC001/SRC001.wav", "SRC001/recording.wav"]
)
def test_prepared_audio_is_found(make_config, place) -> None:
    config = make_config()
    wav = _write_wav(config.paths.audio_root.joinpath(*place.split("/")))
    # The file of another participant and the preparation record are not used.
    _write_wav(config.paths.audio_root / "SRC002.wav")
    (config.paths.audio_root / "SRC001_preparation.json").write_text("{}", encoding="utf-8")

    found = resolve_audio_for_source(config, "SRC001")

    assert found.issue is None
    assert found.path == wav
    assert found.duration_s == pytest.approx(0.5)


def test_missing_audio_is_reported(make_config) -> None:
    config = make_config()
    _write_wav(config.paths.audio_root / "SRC002.wav")

    assert resolve_audio_for_source(config, "SRC001").issue == "audio_file_missing"
    assert resolve_audio_for_source(make_config(paths={"prepared_audio": "missing"}),
                                    "SRC001").issue == "audio_file_missing"
