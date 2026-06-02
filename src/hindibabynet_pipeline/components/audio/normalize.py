"""Peak normalization helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf


def _dbfs_to_linear(dbfs: float) -> float:
    return float(10 ** (dbfs / 20.0))


def _stream_peak(path: Path, block_frames: int = 262144) -> float:
    peak = 0.0
    with sf.SoundFile(str(path), "r") as f:
        while True:
            x = f.read(frames=block_frames, dtype="float32", always_2d=True)
            if x.size == 0:
                break
            p = float(np.max(np.abs(x))) if x.size else 0.0
            if p > peak:
                peak = p
    return peak


def peak_normalize_wav_streaming(
    in_path: Path,
    out_path: Path,
    target_peak_dbfs: float = -1.0,
    block_frames: int = 262144,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Peak-normalize by streaming. Writes PCM_16 WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    peak = _stream_peak(in_path, block_frames=block_frames)
    target_peak = _dbfs_to_linear(target_peak_dbfs)
    gain = 1.0 if peak <= eps else (target_peak / peak)

    info = sf.info(str(in_path))
    sr = int(info.samplerate)
    ch = int(info.channels)

    with sf.SoundFile(str(in_path), "r") as fin, sf.SoundFile(
        str(out_path), "w", samplerate=sr, channels=ch, subtype="PCM_16", format="WAV"
    ) as fout:
        while True:
            x = fin.read(frames=block_frames, dtype="float32", always_2d=True)
            if x.size == 0:
                break
            y = np.clip(x * gain, -1.0, 1.0)
            fout.write(y)

    return {
        "input_peak": float(peak),
        "gain": float(gain),
        "target_peak": float(target_peak),
    }
