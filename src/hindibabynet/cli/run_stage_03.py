"""
Stage 03 — Speaker Classification CLI.

Config-driven by default: reads backend, paths, and parameters from
``configs/config.yaml``.  All CLI arguments are optional overrides.

Usage::

    # Config-driven (processes all prepared participants):
    python -m hindibabynet.cli.run_stage_03

    # Override backend:
    python -m hindibabynet.cli.run_stage_03 --backend vtc

    # Single WAV:
    python -m hindibabynet.cli.run_stage_03 --wav /path/to/<pid>/<pid>.wav

    # Limit to first N participants:
    python -m hindibabynet.cli.run_stage_03 --limit 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.components.speaker_classification import get_backend
from src.hindibabynet.exception.exception import format_traceback
from src.hindibabynet.logging.logger import get_logger, add_file_handler

logger = get_logger(__name__)


def _discover_participants(processed_audio_root: Path) -> list[tuple[str, Path]]:
    """
    Find all ``<pid>/<pid>.wav`` under the processed audio root.

    Returns a sorted list of ``(participant_id, wav_path)`` tuples.
    """
    if not processed_audio_root.is_dir():
        logger.warning(f"Processed audio root not found: {processed_audio_root}")
        return []

    participants = []
    for pid_dir in sorted(processed_audio_root.iterdir()):
        if not pid_dir.is_dir():
            continue
        wav = pid_dir / f"{pid_dir.name}.wav"
        if wav.is_file():
            participants.append((pid_dir.name, wav))

    logger.info(f"Discovered {len(participants)} participants in {processed_audio_root}")
    return participants


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 03: Speaker Classification (config-driven)",
    )
    ap.add_argument(
        "--wav", type=str, default=None,
        help="Single analysis-ready WAV (overrides auto-discovery)",
    )
    ap.add_argument(
        "--analysis_dir", type=str, default=None,
        help="Directory containing <pid>/<pid>.wav (overrides config processed_audio_root)",
    )
    ap.add_argument(
        "--recordings_parquet", type=str, default=None,
        help="Optional Stage 01 parquet to select participant list; Stage 02 WAVs are read from processed root.",
    )
    ap.add_argument(
        "--audio_processed_root", type=str, default=None,
        help="Override processed audio root in recordings_parquet mode.",
    )
    ap.add_argument(
        "--participant_id", type=str, default=None,
        help="Override participant_id (single-wav mode only)",
    )
    ap.add_argument(
        "--backend", type=str, default=None, choices=["xgb", "vtc"],
        help="Override backend (default: from config.yaml)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N participants",
    )
    args = ap.parse_args()

    cfg = ConfigurationManager()
    run_id = cfg.make_run_id()
    backend = get_backend(cfg, override=args.backend)
    add_file_handler(
        logger,
        cfg.get_logs_root() / run_id / f"stage_03_speaker_classification_{backend.name}.log",
    )
    output_root = cfg.get_classification_output_root() / backend.name

    logger.info(f"Stage 03 | backend={backend.name} | output_root={output_root}")

    # --- Single WAV mode ---
    if args.wav:
        wav_path = Path(args.wav).resolve()
        if not wav_path.is_file():
            logger.error(f"WAV not found: {wav_path}")
            sys.exit(1)
        pid = args.participant_id or wav_path.stem
        out_dir = output_root / pid

        if backend.is_complete(pid, out_dir):
            logger.info(f"SKIP (already complete) | {pid}")
            return

        logger.info(f"Processing {pid} ...")
        backend.run_participant(wav_path, pid, out_dir)
        logger.info(f"Done | {pid}")
        return

    # --- Batch mode from recordings parquet ---
    if args.recordings_parquet:
        rec_path = Path(args.recordings_parquet)
        if not rec_path.is_file():
            logger.error(f"recordings_parquet not found: {rec_path}")
            sys.exit(1)

        df = pd.read_parquet(rec_path)
        if "participant_id" not in df.columns:
            logger.error("recordings_parquet missing required column: participant_id")
            sys.exit(1)

        processed_root = (
            Path(args.audio_processed_root).resolve()
            if args.audio_processed_root
            else cfg.get_processed_audio_root()
        )
        participants = []
        for pid in sorted(df["participant_id"].dropna().astype(str).unique().tolist()):
            wav = processed_root / pid / f"{pid}.wav"
            if wav.is_file():
                participants.append((pid, wav))
            else:
                logger.warning(f"Missing analysis wav, skipping {pid}: {wav}")
    else:
        # --- Batch mode: discover participants ---
        if args.analysis_dir:
            processed_root = Path(args.analysis_dir).resolve()
        else:
            processed_root = cfg.get_processed_audio_root()

        participants = _discover_participants(processed_root)

    if not participants:
        logger.error(
            f"No <pid>/<pid>.wav found under {processed_root}. "
            f"Run Stage 02 first, or check audio_preparation.processed_audio_root in config."
        )
        sys.exit(1)

    if args.limit:
        participants = participants[: args.limit]
        logger.info(f"Limited to first {args.limit} participants")

    ok = skip = fail = 0
    for pid, wav_path in participants:
        out_dir = output_root / pid
        if backend.is_complete(pid, out_dir):
            logger.info(f"SKIP (complete) | {pid}")
            skip += 1
            continue

        try:
            logger.info(f"Processing {pid} ...")
            backend.run_participant(wav_path, pid, out_dir)
            ok += 1
            logger.info(f"OK | {pid}")
        except Exception:
            fail += 1
            logger.error(f"FAIL | {pid}\n{format_traceback()}")

    logger.info(f"Stage 03 finished | ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
