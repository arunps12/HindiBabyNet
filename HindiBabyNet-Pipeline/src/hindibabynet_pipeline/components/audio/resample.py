"""Resampling helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _resample_block(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    if x.ndim == 1:
        y = resample_poly(x, up, down).astype(np.float32, copy=False)
    else:
        ys = [resample_poly(x[:, c], up, down) for c in range(x.shape[1])]
        y = np.stack(ys, axis=1).astype(np.float32, copy=False)
    return y


def resample_audio(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample a 1-D float32 array."""
    if sr == target_sr:
        return x
    gcd = int(np.gcd(sr, target_sr))
    up = target_sr // gcd
    down = sr // gcd
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def ensure_mono_16k_wav_streaming(
    in_path: Path,
    out_path: Path,
    target_sr: int = 16000,
    to_mono: bool = True,
    block_frames: int = 262144,
) -> Dict[str, Any]:
    """Stream-read → (optional) mono → resample → write PCM_16 WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(in_path))
    sr_in = int(info.samplerate)
    ch_in = int(info.channels)

    with sf.SoundFile(str(in_path), "r") as fin, sf.SoundFile(
        str(out_path),
        "w",
        samplerate=target_sr,
        channels=(1 if to_mono else ch_in),
        subtype="PCM_16",
        format="WAV",
    ) as fout:
        while True:
            x = fin.read(frames=block_frames, dtype="float32", always_2d=True)
            if x.size == 0:
                break
            if to_mono and x.shape[1] > 1:
                x = x.mean(axis=1, keepdims=True)
            y = _resample_block(x, sr_in, target_sr)
            y = np.clip(y, -1.0, 1.0)
            fout.write(y)

    out_info = sf.info(str(out_path))
    return {
        "sr_in": sr_in,
        "sr_out": int(out_info.samplerate),
        "channels_in": ch_in,
        "channels_out": int(out_info.channels),
        "duration_sec": float(out_info.duration),
    }
