from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import webrtcvad
from scipy.signal import resample_poly

from src.hindibabynet.utils.io_utils import ensure_dir


# ============================================================================
# Low-level primitives
# ============================================================================

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


# ============================================================================
# In-memory audio helpers
# ============================================================================

def load_audio_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load an audio file to mono float32 array + sample rate."""
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False), sr


def resample_audio(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample a 1-D float32 array."""
    if sr == target_sr:
        return x
    gcd = int(np.gcd(sr, target_sr))
    up = target_sr // gcd
    down = sr // gcd
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    """Crop or zero-pad a 1-D array to exactly *target_len* samples."""
    n = len(x)
    if n == target_len:
        return x
    if n > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = x
    return out


def slice_audio(
    x: np.ndarray, sr: int, start_sec: float, end_sec: float
) -> np.ndarray:
    """Return samples between *start_sec* and *end_sec*."""
    s = max(0, int(round(start_sec * sr)))
    e = min(len(x), int(round(end_sec * sr)))
    return x[s:e]


# ============================================================================
# VAD — WebRTC
# ============================================================================

def webrtc_vad_regions(
    path: Path,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    min_region_ms: int = 300,
) -> List[Tuple[float, float]]:
    """Return speech intervals ``(start_sec, end_sec)`` via WebRTC VAD."""
    vad = webrtcvad.Vad(aggressiveness)
    info = sf.info(str(path))
    sr = int(info.samplerate)

    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad needs sr in (8k,16k,32k,48k), got {sr}")

    frame_len = int(sr * frame_ms / 1000)

    speech_flags: List[bool] = []
    with sf.SoundFile(str(path), mode="r") as f:
        while True:
            frame = f.read(frames=frame_len, dtype="int16", always_2d=True)
            if frame.size == 0 or len(frame) < frame_len:
                break
            mono = frame[:, 0]
            speech_flags.append(vad.is_speech(mono.tobytes(), sr))

    # merge consecutive True flags into regions
    regions: List[Tuple[int, int]] = []
    in_speech = False
    start_i = 0
    for i, is_speech in enumerate(speech_flags):
        if is_speech and not in_speech:
            in_speech = True
            start_i = i
        elif (not is_speech) and in_speech:
            in_speech = False
            regions.append((start_i, i))
    if in_speech:
        regions.append((start_i, len(speech_flags)))

    out: List[Tuple[float, float]] = []
    for s_i, e_i in regions:
        s = (s_i * frame_len) / sr
        e = (e_i * frame_len) / sr
        if (e - s) * 1000 >= min_region_ms:
            out.append((float(s), float(e)))
    return out


# ============================================================================
# File-level WAV helpers
# ============================================================================

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
    """Write a concatenated class-stream audio array to WAV."""
    ensure_dir(out_wav_path.parent)
    if len(stream_audio) > 0:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
    return out_wav_path


# ============================================================================
# Streaming file-to-file operations (used by Stage 02)
# ============================================================================


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
