from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_classify_xgb(
    *,
    recordings_parquet: str | Path | None = None,
    wav: str | Path | None = None,
    analysis_dir: str | Path | None = None,
    participant_id: str | None = None,
    limit: int | None = None,
) -> None:
    cmd = [sys.executable, "-m", "hindibabynet_pipeline.cli.run_stage_03", "--backend", "xgb"]
    if recordings_parquet is not None:
        cmd.extend(["--recordings_parquet", str(recordings_parquet)])
    if wav is not None:
        cmd.extend(["--wav", str(wav)])
    if analysis_dir is not None:
        cmd.extend(["--analysis_dir", str(analysis_dir)])
    if participant_id is not None:
        cmd.extend(["--participant_id", participant_id])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(cmd)}")