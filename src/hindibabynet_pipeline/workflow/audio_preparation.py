from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(cmd)}")


def run_prepare_audio(
    *,
    recordings_parquet: str | Path | None = None,
    wav: str | Path | None = None,
    recording_id: str | None = None,
    limit: int | None = None,
    run_id: str | None = None,
) -> None:
    if recordings_parquet is not None and wav is not None:
        raise ValueError("Provide either recordings_parquet or wav, not both.")
    if recordings_parquet is None and wav is None:
        raise ValueError("Provide either recordings_parquet or wav.")

    if recordings_parquet is not None:
        cmd = [
            sys.executable,
            "-m",
            "hindibabynet_pipeline.pipeline.stage_02_audio_preparation_from_parquet",
            "--recordings_parquet",
            str(recordings_parquet),
        ]
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        if run_id is not None:
            cmd.extend(["--run_id", run_id])
        _run_command(cmd)
        return

    cmd = [
        sys.executable,
        "-m",
        "hindibabynet_pipeline.pipeline.stage_02_audio_preparation_single_wav",
        "--wav",
        str(wav),
    ]
    if recording_id is not None:
        cmd.extend(["--recording_id", recording_id])
    _run_command(cmd)