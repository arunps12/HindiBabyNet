from __future__ import annotations

import argparse
from pathlib import Path

from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.components.audio_preparation import AudioPreparation
from src.hindibabynet.logging.logger import get_logger, add_file_handler

logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, type=str)
    ap.add_argument("--recording_id", default=None, type=str)
    args = ap.parse_args()

    wav_path = Path(args.wav)
    recording_id = args.recording_id or wav_path.stem

    cfg = ConfigurationManager()
    run_id = cfg.make_run_id()

    ap_cfg = cfg.get_audio_preparation_config(run_id=run_id, recording_id=recording_id)

    add_file_handler(logger, cfg.get_logs_root() / run_id / "stage_02_audio_preparation_single_wav.log")

    logger.info(f"Stage 02 (single wav) started: {wav_path}")
    artifact = AudioPreparation(ap_cfg).run(wav_path=wav_path, recording_id=recording_id)
    logger.info(f"Stage 02 done: {artifact}")
    print(artifact)


if __name__ == "__main__":
    main()
