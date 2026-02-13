"""
Stage 03 — Speaker Classification

CLI modes:
  --wav <path>                         Process a single analysis-ready WAV
  --analysis_dir <path>                Process all <pid>/<pid>.wav files in a directory
  --recordings_parquet <path>          Process all participants from a recordings parquet
                                       (expects Stage 02 outputs to already exist)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.components.speaker_classification import SpeakerClassification
from src.hindibabynet.exception.exception import format_traceback
from src.hindibabynet.logging.logger import get_logger, add_file_handler

logger = get_logger(__name__)


def _is_stage_03_complete(cfg: ConfigurationManager, run_id: str, participant_id: str) -> bool:
    """Check all 4 output WAVs under output_audio_root (fixed paths, survive new run_ids)."""
    sc_cfg = cfg.get_speaker_classification_config(run_id=run_id, participant_id=participant_id)
    out_dir = sc_cfg.output_audio_root / participant_id
    return (
        (out_dir / f"{participant_id}_main_female.wav").exists()
        and (out_dir / f"{participant_id}_main_male.wav").exists()
        and (out_dir / f"{participant_id}_child.wav").exists()
        and (out_dir / f"{participant_id}_background.wav").exists()
    )


def _run_single(cfg: ConfigurationManager, run_id: str, wav_path: Path, participant_id: str):
    """Process one analysis WAV end-to-end."""
    sc_cfg = cfg.get_speaker_classification_config(run_id=run_id, participant_id=participant_id)
    component = SpeakerClassification(sc_cfg)
    artifact = component.run(analysis_wav_path=wav_path, participant_id=participant_id)
    return artifact


def main():
    ap = argparse.ArgumentParser(description="Stage 03: Speaker Classification")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=str, help="Single analysis-ready WAV path")
    group.add_argument("--analysis_dir", type=str, help="Directory containing <pid>/<pid>.wav analysis files")
    group.add_argument(
        "--recordings_parquet",
        type=str,
        help="Recordings parquet from Stage 01 (looks for Stage 02 outputs)",
    )

    ap.add_argument("--participant_id", type=str, default=None, help="Override participant_id (single wav mode)")
    ap.add_argument("--audio_processed_root", type=str, default=None,
                    help="Root where Stage 02 wrote analysis WAVs (for --recordings_parquet mode)")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N participants")
    args = ap.parse_args()

    cfg = ConfigurationManager()
    run_id = cfg.make_run_id()

    add_file_handler(logger, cfg.get_logs_root() / run_id / "stage_03_speaker_classification.log")

    # ======= Mode 1: Single WAV =======
    if args.wav:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        pid = args.participant_id or wav_path.stem
        logger.info(f"Stage 03 (single) | wav={wav_path} participant_id={pid} run_id={run_id}")
        artifact = _run_single(cfg, run_id, wav_path, pid)
        logger.info(f"Stage 03 done: {artifact}")
        print(artifact)
        return

    # ======= Mode 2: Analysis directory =======
    if args.analysis_dir:
        analysis_dir = Path(args.analysis_dir)
        if not analysis_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {analysis_dir}")

        # Find analysis WAVs: each participant dir contains <pid>.wav
        wav_files = sorted(
            p for p in analysis_dir.rglob("*.wav")
            if p.parent != analysis_dir or True  # include all levels
        )
        # Filter: keep only WAVs whose stem matches their parent dir name
        # (i.e. ABAN141223/ABAN141223.wav, not tmp files)
        wav_files = [
            p for p in wav_files
            if p.stem == p.parent.name and not p.parent.name.startswith("_")
        ]
        if not wav_files:
            raise FileNotFoundError(f"No <pid>/<pid>.wav files found under {analysis_dir}")

        if args.limit:
            wav_files = wav_files[: args.limit]

        logger.info(f"Stage 03 batch (analysis_dir) | n_files={len(wav_files)} run_id={run_id}")
        n_ok, n_fail, n_skip = 0, 0, 0

        for wav_path in wav_files:
            pid = wav_path.stem
            if _is_stage_03_complete(cfg, run_id, pid):
                logger.info(f"[{pid}] SKIP | Stage 03 outputs already exist")
                n_skip += 1
                continue
            try:
                logger.info(f"[{pid}] Processing {wav_path}")
                _run_single(cfg, run_id, wav_path, pid)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                logger.error(f"[{pid}] FAILED: {e}")
                logger.error(format_traceback(e))

        logger.info(f"Stage 03 batch done | ok={n_ok} skip={n_skip} fail={n_fail}")
        print(f"Stage 03 batch done | ok={n_ok} skip={n_skip} fail={n_fail}")
        return

    # ======= Mode 3: From recordings parquet =======
    if args.recordings_parquet:
        rec_path = Path(args.recordings_parquet)
        if not rec_path.exists():
            raise FileNotFoundError(f"Parquet not found: {rec_path}")

        df = pd.read_parquet(rec_path)
        participants = sorted(df["participant_id"].dropna().unique().tolist())
        if args.limit:
            participants = participants[: args.limit]

        # Determine where Stage 02 wrote analysis WAVs
        if args.audio_processed_root:
            processed_root = Path(args.audio_processed_root)
        else:
            # Read from config
            ap_cfg = cfg.config["audio_preparation"]
            processed_root = Path(ap_cfg["processed_audio_root"])

        logger.info(
            f"Stage 03 batch (parquet) | participants={len(participants)} "
            f"processed_root={processed_root} run_id={run_id}"
        )

        n_ok, n_fail, n_skip = 0, 0, 0
        for pid in participants:
            analysis_wav = processed_root / pid / f"{pid}.wav"
            if not analysis_wav.exists():
                logger.warning(f"[{pid}] analysis wav not found: {analysis_wav}, skipping")
                n_skip += 1
                continue
            if _is_stage_03_complete(cfg, run_id, pid):
                logger.info(f"[{pid}] SKIP | Stage 03 outputs already exist")
                n_skip += 1
                continue
            try:
                logger.info(f"[{pid}] Processing {analysis_wav}")
                _run_single(cfg, run_id, analysis_wav, pid)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                logger.error(f"[{pid}] FAILED: {e}")
                logger.error(format_traceback(e))

        logger.info(f"Stage 03 batch done | ok={n_ok} fail={n_fail} skip={n_skip}")
        print(f"Stage 03 batch done | ok={n_ok} fail={n_fail} skip={n_skip}")


if __name__ == "__main__":
    main()
