"""WAV concatenation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf


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
            raise ValueError(
                f"Sample-rate mismatch: {p} has {info.samplerate}, expected {sr}"
            )
        if int(info.channels) != ch:
            raise ValueError(
                f"Channel mismatch: {p} has {info.channels}, expected {ch}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    gap_frames = int(round(gap_sec * sr))
    gap_buf = np.zeros((gap_frames, ch), dtype=np.float32) if gap_frames > 0 else None

    manifest: List[Dict[str, Any]] = []
    t = 0.0

    with sf.SoundFile(
        str(out_path), "w", samplerate=sr, channels=ch, subtype="PCM_16", format="WAV"
    ) as fout:
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
