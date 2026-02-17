"""
Stage 06 — Speaker-type Classification + Stream Export

Requires Stage 05 speech-segments parquet and analysis WAV as inputs.

CLI modes:
  --speech_segments_parquet <p> --wav <p>         Single participant
  --analysis_dir <dir> --run_id <id>              Auto-discover from a run
  --recordings_parquet <p> --run_id <id>          Batch from recordings parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.components.speaker_classification_v2 import SpeakerClassification
from src.hindibabynet.exception.exception import format_traceback
from src.hindibabynet.logging.logger import get_logger, add_file_handler

logger = get_logger(__name__)


def _is_complete(cfg: ConfigurationManager, run_id: str, pid: str) -> bool:
    sc_cfg = cfg.get_speaker_classification_config(run_id=run_id, participant_id=pid)
    out_dir = sc_cfg.output_audio_root / pid
    return (
        (out_dir / f"{pid}_main_female.wav").exists()
        and (out_dir / f"{pid}_main_male.wav").exists()
        and (out_dir / f"{pid}_child.wav").exists()
        and (out_dir / f"{pid}_background.wav").exists()
    )


def _run_single(
    cfg: ConfigurationManager,
    run_id: str,
    speech_seg_pq: Path,
    wav_path: Path,
    pid: str,
):
    sc_cfg = cfg.get_speaker_classification_config(run_id=run_id, participant_id=pid)
    component = SpeakerClassification(sc_cfg)
    return component.run(
        speech_segments_parquet_path=speech_seg_pq,
        analysis_wav_path=wav_path,
        participant_id=pid,
    )


def _find_seg_parquet(artifacts_root: Path, run_id: str, pid: str) -> Path:
    return artifacts_root / run_id / "intersection" / f"{pid}_speech_segments.parquet"


def main():
    ap = argparse.ArgumentParser(description="Stage 06: Speaker Classification + Stream Export")

    # --- Single mode ---
    ap.add_argument("--speech_segments_parquet", type=str,
                    help="Speech segments parquet from Stage 05")
    ap.add_argument("--wav", type=str, help="Analysis-ready WAV")
    ap.add_argument("--participant_id", type=str, default=None, help="Override pid")

    # --- Batch modes ---
    ap.add_argument("--analysis_dir", type=str,
                    help="Auto-discover from analysis dir + run_id")
    ap.add_argument("--recordings_parquet", type=str,
                    help="Recordings parquet from Stage 01")
    ap.add_argument("--audio_processed_root", type=str, default=None)
    ap.add_argument("--run_id", type=str, default=None,
                    help="Run ID to find Stage 05 artifacts (required for batch)")
    ap.add_argument("--limit", type=int, default=None, help="Process first N")
    args = ap.parse_args()

    cfg = ConfigurationManager()
    run_id = args.run_id or cfg.make_run_id()
    add_file_handler(logger, cfg.get_logs_root() / run_id / "stage_06_speaker_classification.log")

    artifacts_root = Path(cfg.config["artifacts_root"])

    # ======= Single: explicit paths =======
    if args.speech_segments_parquet and args.wav:
        seg_pq = Path(args.speech_segments_parquet)
        wav_path = Path(args.wav)
        if not seg_pq.exists():
            raise FileNotFoundError(f"Segments parquet not found: {seg_pq}")
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        pid = args.participant_id or wav_path.stem
        logger.info(f"Stage 06 (single) | pid={pid} run_id={run_id}")
        artifact = _run_single(cfg, run_id, seg_pq, wav_path, pid)
        logger.info(f"Stage 06 done: {artifact}")
        print(artifact)
        return

    # ======= Batch: analysis_dir =======
    if args.analysis_dir:
        analysis_dir = Path(args.analysis_dir)
        if not analysis_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {analysis_dir}")

        wav_files = sorted(
            p for p in analysis_dir.rglob("*.wav")
            if p.stem == p.parent.name and not p.parent.name.startswith("_")
        )
        if not wav_files:
            raise FileNotFoundError(f"No <pid>/<pid>.wav under {analysis_dir}")
        if args.limit:
            wav_files = wav_files[: args.limit]

        logger.info(f"Stage 06 batch (analysis_dir) | n={len(wav_files)} run_id={run_id}")
        n_ok, n_fail, n_skip = 0, 0, 0
        for wav_path in wav_files:
            pid = wav_path.stem
            seg_pq = _find_seg_parquet(artifacts_root, run_id, pid)
            if not seg_pq.exists():
                logger.warning(f"[{pid}] Missing segments parquet, skipping")
                n_skip += 1
                continue
            if _is_complete(cfg, run_id, pid):
                logger.info(f"[{pid}] SKIP (already exists)")
                n_skip += 1
                continue
            try:
                logger.info(f"[{pid}] Processing")
                _run_single(cfg, run_id, seg_pq, wav_path, pid)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                logger.error(f"[{pid}] FAILED: {e}")
                logger.error(format_traceback(e))

        logger.info(f"Stage 06 batch done | ok={n_ok} skip={n_skip} fail={n_fail}")
        print(f"Stage 06 batch done | ok={n_ok} skip={n_skip} fail={n_fail}")
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

        if args.audio_processed_root:
            processed_root = Path(args.audio_processed_root)
        else:
            ap_cfg = cfg.config["audio_preparation"]
            processed_root = Path(ap_cfg["processed_audio_root"])

        logger.info(
            f"Stage 06 batch (parquet) | n={len(pids)} "
            f"processed_root={processed_root} run_id={run_id}"
        )
        n_ok, n_fail, n_skip = 0, 0, 0
        for pid in pids:
            wav = processed_root / pid / f"{pid}.wav"
            if not wav.exists():
                logger.warning(f"[{pid}] WAV not found: {wav}, skipping")
                n_skip += 1
                continue
            seg_pq = _find_seg_parquet(artifacts_root, run_id, pid)
            if not seg_pq.exists():
                logger.warning(f"[{pid}] Missing segments parquet, skipping")
                n_skip += 1
                continue
            if _is_complete(cfg, run_id, pid):
                logger.info(f"[{pid}] SKIP (already exists)")
                n_skip += 1
                continue
            try:
                logger.info(f"[{pid}] Processing")
                _run_single(cfg, run_id, seg_pq, wav, pid)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                logger.error(f"[{pid}] FAILED: {e}")
                logger.error(format_traceback(e))

        logger.info(f"Stage 06 batch done | ok={n_ok} fail={n_fail} skip={n_skip}")
        print(f"Stage 06 batch done | ok={n_ok} fail={n_fail} skip={n_skip}")
        return

    ap.error("Provide --speech_segments_parquet + --wav, --analysis_dir, or --recordings_parquet")


if __name__ == "__main__":
    main()
