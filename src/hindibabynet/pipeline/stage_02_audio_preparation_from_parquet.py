from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.components.audio_preparation import AudioPreparation
from src.hindibabynet.exception.exception import format_traceback
from src.hindibabynet.logging.logger import get_logger, add_file_handler

logger = get_logger(__name__)


def _is_stage_02_complete(ap_cfg) -> bool:
    """Check the analysis WAV under processed_audio_root (fixed path, survives new run_ids)."""
    return ap_cfg.analysis_wav_path.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings_parquet", required=True, type=str)
    ap.add_argument("--limit", default=None, type=int, help="Optional: process only first N participants")
    ap.add_argument("--run_id", default=None, type=str, help="Shared run id for all stages")
    args = ap.parse_args()

    rec_path = Path(args.recordings_parquet)
    if not rec_path.exists():
        raise FileNotFoundError(f"recordings.parquet not found: {rec_path}")

    df = pd.read_parquet(rec_path)

    required_cols = {"participant_id", "path", "recording_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"recordings.parquet missing columns: {sorted(missing)}")

    participants = sorted(df["participant_id"].dropna().unique().tolist())
    if args.limit is not None:
        participants = participants[: int(args.limit)]

    cfg = ConfigurationManager()
    run_id = args.run_id or cfg.make_run_id()

    # One log file for the whole batch
    add_file_handler(logger, cfg.get_logs_root() / run_id / "stage_02_audio_preparation_batch.log")

    logger.info(
        f"Stage 02 batch started | recordings={len(df)} | participants={len(participants)} | run_id={run_id}"
    )

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for pid in participants:
        try:
            df_pid = df[df["participant_id"] == pid].copy()
            if df_pid.empty:
                raise ValueError(f"No rows for participant_id={pid}")

            # stable ordering when concatenating
            sort_cols = [c for c in ["session_date", "recording_id"] if c in df_pid.columns]
            df_pid = df_pid.sort_values(sort_cols if sort_cols else ["recording_id"]).reset_index(drop=True)

            recording_id = str(pid)  # output name = participant_id
            ap_cfg = cfg.get_audio_preparation_config(run_id=run_id, recording_id=recording_id)

            if _is_stage_02_complete(ap_cfg):
                logger.info(f"[{pid}] SKIP | Stage 02 outputs already exist at {ap_cfg.analysis_wav_path.parent}")
                n_skip += 1
                continue

            logger.info(f"[{pid}] files={len(df_pid)} -> {ap_cfg.analysis_wav_path}")

            artifact = AudioPreparation(ap_cfg).run(
                recordings_df=df_pid,
                participant_id=str(pid),
                recording_id=recording_id,
            )

            logger.info(
                f"[{pid}] DONE | dur={artifact.duration_sec/3600:.2f}h sr={artifact.sample_rate} ch={artifact.channels}"
            )
            n_ok += 1

        except Exception as e:
            n_fail += 1
            logger.error(f"[{pid}] FAILED: {e}")
            logger.error(format_traceback(e))

    logger.info(f"Stage 02 batch finished | ok={n_ok} skip={n_skip} fail={n_fail} run_id={run_id}")
    print(f"Stage 02 batch finished | ok={n_ok} skip={n_skip} fail={n_fail} run_id={run_id}")


if __name__ == "__main__":
    main()
