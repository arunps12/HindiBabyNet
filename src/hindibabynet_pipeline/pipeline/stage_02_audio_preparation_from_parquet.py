from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from hindibabynet_pipeline.config.configuration import ConfigurationManager
from hindibabynet_pipeline.components.audio_preparation import AudioPreparation
from hindibabynet_pipeline.exception.exception import format_traceback
from hindibabynet_pipeline.logging.logger import get_logger, add_file_handler

logger = get_logger(__name__)


def _is_stage_02_complete(ap_cfg) -> bool:
    """Check the analysis WAV under processed_audio_root (fixed path, survives new run_ids)."""
    return ap_cfg.analysis_wav_path.exists()


def _cleanup_after_processing(
    ap_cfg,
    pid: str,
    recording_id: str,
) -> None:
    """Delete only Stage 02 temp dirs under processed_audio_root.

    NOTE: raw_audio_root is treated as read-only source data and is never modified.
    """

    for tmp_name in ("_tmp_combine", "_tmp_prep"):
        tmp_dir = ap_cfg.processed_audio_root / tmp_name / recording_id
        if tmp_dir.is_dir():
            try:
                shutil.rmtree(tmp_dir)
                logger.info(f"[{pid}] CLEANUP | deleted temp dir: {tmp_dir}")
            except Exception as exc:
                logger.warning(f"[{pid}] CLEANUP | failed to delete temp dir {tmp_dir}: {exc}")


def _cleanup_temp_only(ap_cfg, recording_id: str, pid: str) -> None:
    """Delete only temp dirs (e.g. after a failure) to reclaim space."""
    for tmp_name in ("_tmp_combine", "_tmp_prep"):
        tmp_dir = ap_cfg.processed_audio_root / tmp_name / recording_id
        if tmp_dir.is_dir():
            try:
                shutil.rmtree(tmp_dir)
                logger.info(f"[{pid}] CLEANUP | deleted temp dir after failure: {tmp_dir}")
            except Exception as exc:
                logger.warning(f"[{pid}] CLEANUP | failed to delete temp dir {tmp_dir}: {exc}")


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

    cfg = ConfigurationManager()
    ap_params = cfg.get_audio_preparation_params()
    join_multiple_files = bool(ap_params.get("join_multiple_files", True))
    run_id = args.run_id or cfg.make_run_id()

    # One log file for the whole batch
    add_file_handler(logger, cfg.get_logs_root() / run_id / "stage_02_audio_preparation_batch.log")

    logger.info(
        f"Stage 02 batch started | recordings={len(df)} | join_multiple_files={join_multiple_files} | run_id={run_id}"
    )

    n_ok = 0
    n_skip = 0
    n_fail = 0

    if join_multiple_files:
        units: list[dict[str, object]] = [
            {
                "mode": "participant",
                "participant_id": str(pid),
                "recording_id": str(pid),
                "dataframe": df[df["participant_id"] == pid].copy(),
            }
            for pid in sorted(df["participant_id"].dropna().unique().tolist())
        ]
    else:
        sort_cols = [c for c in ["participant_id", "session_date", "recording_id"] if c in df.columns]
        sorted_df = df.sort_values(sort_cols if sort_cols else ["recording_id"]).reset_index(drop=True)
        units = []
        for row in sorted_df.itertuples(index=False):
            row_dict = row._asdict()
            recording_id = str(row_dict.get("recording_id") or Path(str(row_dict["path"])).stem)
            participant_id = str(row_dict.get("participant_id") or recording_id)
            units.append(
                {
                    "mode": "recording",
                    "participant_id": participant_id,
                    "recording_id": recording_id,
                    "wav_path": Path(str(row_dict["path"])),
                }
            )

    if args.limit is not None:
        units = units[: int(args.limit)]

    for unit in units:
        try:
            pid = str(unit["participant_id"])
            recording_id = str(unit["recording_id"])
            ap_cfg = cfg.get_audio_preparation_config(run_id=run_id, recording_id=recording_id)

            if _is_stage_02_complete(ap_cfg):
                logger.info(f"[{pid}] SKIP | Stage 02 outputs already exist at {ap_cfg.analysis_wav_path.parent}")
                n_skip += 1
                continue

            if unit["mode"] == "participant":
                df_pid = unit["dataframe"]
                logger.info(f"[{pid}] files={len(df_pid)} -> {ap_cfg.analysis_wav_path}")
                artifact = AudioPreparation(ap_cfg).run(
                    recordings_df=df_pid,
                    participant_id=str(pid),
                    recording_id=recording_id,
                )
            else:
                wav_path = unit["wav_path"]
                logger.info(f"[{pid}] wav={wav_path} -> {ap_cfg.analysis_wav_path}")
                artifact = AudioPreparation(ap_cfg).run(
                    wav_path=wav_path,
                    recording_id=recording_id,
                )

            logger.info(
                f"[{pid}] DONE | dur={artifact.duration_sec/3600:.2f}h sr={artifact.sample_rate} ch={artifact.channels}"
            )
            n_ok += 1

            # Free disk space: delete only temp dirs (never touch raw_audio_root)
            _cleanup_after_processing(ap_cfg, pid, recording_id)

        except Exception as e:
            n_fail += 1
            logger.error(f"[{pid}] FAILED: {e}")
            logger.error(format_traceback(e))
            # Still clean temp dirs on failure to reclaim space
            try:
                _cleanup_temp_only(ap_cfg, recording_id, pid)
            except Exception:
                pass

    logger.info(f"Stage 02 batch finished | ok={n_ok} skip={n_skip} fail={n_fail} run_id={run_id}")
    print(f"Stage 02 batch finished | ok={n_ok} skip={n_skip} fail={n_fail} run_id={run_id}")


if __name__ == "__main__":
    main()

