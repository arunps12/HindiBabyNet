"""Low-level audio read/write helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load an audio file to mono float32 array + sample rate."""
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False), sr


def write_wav_chunk(
    wav_path: Path,
    chunk_path: Path,
    start_sec: float,
    end_sec: float,
    logger=None,
) -> Path | None:
    """Write a time-slice of a WAV to disk. Returns ``None`` on failure."""
    try:
        info = sf.info(str(wav_path))
        sr = info.samplerate
        start_frame = max(0, int(start_sec * sr))
        end_frame = min(info.frames, int(end_sec * sr))
        n_frames = end_frame - start_frame
        if n_frames <= 0:
            return None
        audio, _ = sf.read(
            str(wav_path), start=start_frame, frames=n_frames, dtype="float32"
        )
        if audio is None or audio.size == 0:
            return None
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(chunk_path), audio, sr, format="WAV", subtype="PCM_16")
        return chunk_path
    except Exception as e:
        if logger is not None:
            logger.warning(f"write_wav_chunk failed ({chunk_path.name}): {e}")
        return None


def write_stream_wav(
    stream_audio: np.ndarray,
    sr: int,
    out_wav_path: Path,
) -> Path:
    """Write a class-stream audio array to WAV."""
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    if len(stream_audio) > 0:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
    return out_wav_path
