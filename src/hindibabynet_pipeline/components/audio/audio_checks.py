"""Audio validation and slicing helpers."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


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


def webrtc_vad_regions(
    path: Path,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    min_region_ms: int = 300,
) -> List[Tuple[float, float]]:
    """Return speech intervals ``(start_sec, end_sec)`` via WebRTC VAD."""
    try:
        import webrtcvad
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "webrtcvad is required only for the XGB backend VAD step. "
            "Install XGB dependencies with: uv sync --extra xgb"
        ) from exc

    import soundfile as sf

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
