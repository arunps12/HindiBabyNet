"""
Stage 05 — VAD ∩ Diarization Intersection

Requires Stage 03 (VAD) and Stage 04 (Diarization) parquets as inputs.

CLI modes:
  --vad_parquet <p> --diar_parquet <p>            Single participant
  --analysis_dir <dir> --run_id <id>              Auto-discover from a run
  --recordings_parquet <p> --run_id <id>          Batch from recordings parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.components.intersection import Intersection
from src.hindibabynet.exception.exception import format_traceback
from src.hindibabynet.logging.logger import get_logger, add_file_handler

logger = get_logger(__name__)


def _is_complete(cfg: ConfigurationManager, run_id: str, pid: str) -> bool:
    i_cfg = cfg.get_intersection_config(run_id=run_id, participant_id=pid)
    return i_cfg.speech_segments_parquet_path.exists()


def _run_single(
    cfg: ConfigurationManager,
    run_id: str,
    vad_parquet: Path,
    diar_parquet: Path,
    pid: str,
):
    i_cfg = cfg.get_intersection_config(run_id=run_id, participant_id=pid)
    component = Intersection(i_cfg)
    return component.run(
        vad_parquet_path=vad_parquet,
        diarization_parquet_path=diar_parquet,
        participant_id=pid,
    )


def _find_parquets(artifacts_root: Path, run_id: str, pid: str):
    """Locate Stage 03 and Stage 04 parquets for a given run + participant."""
    vad_pq = artifacts_root / run_id / "vad" / f"{pid}_vad.parquet"
    diar_pq = artifacts_root / run_id / "diarization" / f"{pid}_diarization.parquet"
    return vad_pq, diar_pq


def main():
    ap = argparse.ArgumentParser(description="Stage 05: VAD ∩ Diarization Intersection")

    # --- Single mode ---
    ap.add_argument("--vad_parquet", type=str, help="VAD parquet from Stage 03")
    ap.add_argument("--diar_parquet", type=str, help="Diarization parquet from Stage 04")
    ap.add_argument("--participant_id", type=str, default=None, help="Override pid")

    # --- Batch modes ---
    ap.add_argument("--analysis_dir", type=str,
                    help="Auto-discover from analysis dir + run_id")
    ap.add_argument("--recordings_parquet", type=str,
                    help="Recordings parquet from Stage 01")
    ap.add_argument("--audio_processed_root", type=str, default=None)
    ap.add_argument("--run_id", type=str, default=None,
                    help="Run ID to find Stage 03/04 artifacts (required for batch)")
    ap.add_argument("--limit", type=int, default=None, help="Process first N")
    args = ap.parse_args()

    cfg = ConfigurationManager()
    run_id = args.run_id or cfg.make_run_id()
    add_file_handler(logger, cfg.get_logs_root() / run_id / "stage_05_intersection.log")

    artifacts_root = Path(cfg.config["artifacts_root"])

    # ======= Single: explicit parquets =======
    if args.vad_parquet and args.diar_parquet:
        vad_pq = Path(args.vad_parquet)
        diar_pq = Path(args.diar_parquet)
        if not vad_pq.exists():
            raise FileNotFoundError(f"VAD parquet not found: {vad_pq}")
        if not diar_pq.exists():
            raise FileNotFoundError(f"Diar parquet not found: {diar_pq}")

        pid = args.participant_id or vad_pq.stem.replace("_vad", "")
        logger.info(f"Stage 05 (single) | pid={pid} run_id={run_id}")
        artifact = _run_single(cfg, run_id, vad_pq, diar_pq, pid)
        logger.info(f"Stage 05 done: {artifact}")
        print(artifact)
        return

    # ======= Batch: analysis_dir =======
    if args.analysis_dir:
        analysis_dir = Path(args.analysis_dir)
        if not analysis_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {analysis_dir}")

        pids = sorted(
            p.parent.name
            for p in analysis_dir.rglob("*.wav")
            if p.stem == p.parent.name and not p.parent.name.startswith("_")
        )
        if not pids:
            raise FileNotFoundError(f"No <pid>/<pid>.wav under {analysis_dir}")
        if args.limit:
            pids = pids[: args.limit]

        logger.info(f"Stage 05 batch (analysis_dir) | n={len(pids)} run_id={run_id}")
        n_ok, n_fail, n_skip = 0, 0, 0
        for pid in pids:
            vad_pq, diar_pq = _find_parquets(artifacts_root, run_id, pid)
            if not vad_pq.exists() or not diar_pq.exists():
                logger.warning(f"[{pid}] Missing VAD/diar parquet(s), skipping")
                n_skip += 1
                continue
            if _is_complete(cfg, run_id, pid):
                logger.info(f"[{pid}] SKIP (already exists)")
                n_skip += 1
                continue
            try:
                logger.info(f"[{pid}] Processing")
                _run_single(cfg, run_id, vad_pq, diar_pq, pid)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                logger.error(f"[{pid}] FAILED: {e}")
                logger.error(format_traceback(e))

        logger.info(f"Stage 05 batch done | ok={n_ok} skip={n_skip} fail={n_fail}")
        print(f"Stage 05 batch done | ok={n_ok} skip={n_skip} fail={n_fail}")
        return

    # ======= Batch: recordings parquet =======
    if args.recordings_parquet:
        rec_path = Path(args.recordings_parquet)
        if not rec_path.exists():
            raise FileNotFoundError(f"Parquet not found: {rec_path}")

        df = pd.read_parquet(rec_path)
        pids = sorted(df["participant_id"].dropna().unique().tolist())
        if args.limit:
            pids = pids[: args.limit]

        logger.info(f"Stage 05 batch (parquet) | n={len(pids)} run_id={run_id}")
        n_ok, n_fail, n_skip = 0, 0, 0
        for pid in pids:
            vad_pq, diar_pq = _find_parquets(artifacts_root, run_id, pid)
            if not vad_pq.exists() or not diar_pq.exists():
                logger.warning(f"[{pid}] Missing VAD/diar parquet(s), skipping")
                n_skip += 1
                continue
            if _is_complete(cfg, run_id, pid):
                logger.info(f"[{pid}] SKIP (already exists)")
                n_skip += 1
                continue
            try:
                logger.info(f"[{pid}] Processing")
                _run_single(cfg, run_id, vad_pq, diar_pq, pid)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                logger.error(f"[{pid}] FAILED: {e}")
                logger.error(format_traceback(e))

        logger.info(f"Stage 05 batch done | ok={n_ok} fail={n_fail} skip={n_skip}")
        print(f"Stage 05 batch done | ok={n_ok} fail={n_fail} skip={n_skip}")
        return

    ap.error("Provide --vad_parquet + --diar_parquet, --analysis_dir, or --recordings_parquet")


if __name__ == "__main__":
    main()
