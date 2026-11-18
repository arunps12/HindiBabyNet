from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def play_wav(path: str | Path) -> None:
    wav_path = str(path)
    if sys.platform.startswith("win"):
        import winsound

        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        return

    candidate_commands = [
        ["ffplay", "-nodisp", "-autoexit", wav_path],
        ["afplay", wav_path],
        ["aplay", wav_path],
    ]
    for cmd in candidate_commands:
        try:
            result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                return
        except FileNotFoundError:
            continue
    raise RuntimeError("No supported audio playback command is available for annotation playback.")