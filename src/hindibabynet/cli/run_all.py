"""Run full HindiBabyNet pipeline in config-driven mode.

Pipeline:
  Stage 01 -> Stage 02 -> Stage 03 (selected backend)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.logging.logger import add_file_handler, get_logger
from src.hindibabynet.pipeline.stage_01_data_ingestion import run_stage_01

logger = get_logger(__name__)


def _run_cmd(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit={result.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 01 -> Stage 02 -> Stage 03")
    ap.add_argument("legacy_limit", nargs="?", default=None, help="Legacy positional limit (kept for compatibility)")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N participants")
    ap.add_argument("--backend", choices=["xgb", "vtc"], default=None, help="Optional Stage 03 backend override")
    args = ap.parse_args()

    limit = args.limit
    if limit is None and args.legacy_limit is not None:
        try:
            limit = int(args.legacy_limit)
        except ValueError as exc:
            raise ValueError(
                f"Invalid positional limit '{args.legacy_limit}'. Use an integer or --limit N."
            ) from exc

    cfg = ConfigurationManager()
    run_id = cfg.make_run_id()
    add_file_handler(logger, cfg.get_logs_root() / run_id / "run_all.log")

    logger.info("Run-all started | run_id=%s", run_id)

    stage01_artifact = run_stage_01(run_id=run_id)
    rec_parquet = stage01_artifact.recordings_parquet_path
    logger.info("Stage 01 done | recordings_parquet=%s", rec_parquet)

    cmd_stage02 = [
        sys.executable,
        "-m",
        "src.hindibabynet.pipeline.stage_02_audio_preparation_from_parquet",
        "--recordings_parquet",
        str(rec_parquet),
        "--run_id",
        run_id,
    ]
    if limit:
        cmd_stage02.extend(["--limit", str(limit)])
    _run_cmd(cmd_stage02)

    cmd_stage03 = [
        sys.executable,
        "-m",
        "src.hindibabynet.cli.run_stage_03",
        "--recordings_parquet",
        str(rec_parquet),
    ]
    if limit:
        cmd_stage03.extend(["--limit", str(limit)])
    if args.backend:
        cmd_stage03.extend(["--backend", args.backend])
    _run_cmd(cmd_stage03)

    logger.info("Run-all finished successfully | run_id=%s", run_id)


if __name__ == "__main__":
    main()
