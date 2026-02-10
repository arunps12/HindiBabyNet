from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.hindibabynet.utils.io_utils import ensure_dir


def _dbfs_to_linear(dbfs: float) -> float:
    return float(10 ** (dbfs / 20.0))


def _stream_peak(path: Path, block_frames: int = 262144) -> float:
    peak = 0.0
    with sf.SoundFile(str(path), "r") as f:
        while True:
            x = f.read(frames=block_frames, dtype="float32", always_2d=True)
            if x.size == 0:
                break
            # peak over all channels
            p = float(np.max(np.abs(x))) if x.size else 0.0
            if p > peak:
                peak = p
    return peak


def _resample_block(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    # resample_poly expects (n,) or (n, ch)
    if x.ndim == 1:
        y = resample_poly(x, up, down).astype(np.float32, copy=False)
    else:
        # process each channel
        ys = [resample_poly(x[:, c], up, down) for c in range(x.shape[1])]
        y = np.stack(ys, axis=1).astype(np.float32, copy=False)
    return y


def ensure_mono_16k_wav_streaming(
    in_path: Path,
    out_path: Path,
    target_sr: int = 16000,
    to_mono: bool = True,
    block_frames: int = 262144,
) -> Dict[str, Any]:
    """
    Stream-read -> (optional) mono -> resample -> write PCM_16 WAV.
    Keeps timeline (no trimming).
    """
    ensure_dir(out_path.parent)

    info = sf.info(str(in_path))
    sr_in = int(info.samplerate)
    ch_in = int(info.channels)

    with sf.SoundFile(str(in_path), "r") as fin, sf.SoundFile(
        str(out_path), "w", samplerate=target_sr, channels=(1 if to_mono else ch_in),
        subtype="PCM_16", format="WAV"
    ) as fout:
        while True:
            x = fin.read(frames=block_frames, dtype="float32", always_2d=True)
            if x.size == 0:
                break

            if to_mono and x.shape[1] > 1:
                x = x.mean(axis=1, keepdims=True)

            # resample
            y = _resample_block(x, sr_in, target_sr)

            # safety clip
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


def peak_normalize_wav_streaming(
    in_path: Path,
    out_path: Path,
    target_peak_dbfs: float = -1.0,
    block_frames: int = 262144,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Peak-normalize by streaming. Writes PCM_16 WAV.
    """
    ensure_dir(out_path.parent)

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

    return {"input_peak": float(peak), "gain": float(gain), "target_peak": float(target_peak)}


def concatenate_wavs_streaming(
    wav_paths: List[Path],
    out_path: Path,
    gap_sec: float = 0.0,
    block_frames: int = 262144,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Concatenate WAV files in order (assumes same sr/ch across inputs).
    Returns (sr, ch, manifest_rows with combined offsets).
    """
    assert wav_paths, "No wav paths to concatenate."

    infos = [sf.info(str(p)) for p in wav_paths]
    sr = int(infos[0].samplerate)
    ch = int(infos[0].channels)

    for p, info in zip(wav_paths, infos):
        if int(info.samplerate) != sr:
            raise ValueError(f"Sample-rate mismatch: {p} has {info.samplerate}, expected {sr}")
        if int(info.channels) != ch:
            raise ValueError(f"Channel mismatch: {p} has {info.channels}, expected {ch}")

    ensure_dir(out_path.parent)

    gap_frames = int(round(gap_sec * sr))
    gap_buf = np.zeros((gap_frames, ch), dtype=np.float32) if gap_frames > 0 else None

    manifest: List[Dict[str, Any]] = []
    t = 0.0

    with sf.SoundFile(str(out_path), "w", samplerate=sr, channels=ch, subtype="PCM_16", format="WAV") as fout:
        for idx, p in enumerate(wav_paths):
            info = sf.info(str(p))
            dur = float(info.duration)
            s = t
            e = s + dur

            manifest.append(
                {
                    "source_index": idx,
                    "source_path": str(p),
                    "source_recording_id": p.stem,
                    "combined_start_sec": s,
                    "combined_end_sec": e,
                    "source_duration_sec": dur,
                    "sample_rate": sr,
                    "channels": ch,
                }
            )

            with sf.SoundFile(str(p), "r") as fin:
                while True:
                    x = fin.read(frames=block_frames, dtype="float32", always_2d=True)
                    if x.size == 0:
                        break
                    fout.write(np.clip(x, -1.0, 1.0))

            if gap_buf is not None:
                fout.write(gap_buf)
                t = e + gap_sec
            else:
                t = e

    return sr, ch, manifest
