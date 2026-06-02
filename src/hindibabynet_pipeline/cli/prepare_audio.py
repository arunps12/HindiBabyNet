from __future__ import annotations

import argparse

from hindibabynet_pipeline.workflow.audio_preparation import run_prepare_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare audio for HindiBabyNet-Pipeline.")
    parser.add_argument("--recordings-parquet", dest="recordings_parquet", default=None, help="Stage-01 recordings parquet for batch preparation.")
    parser.add_argument("--wav", default=None, help="Single raw WAV path.")
    parser.add_argument("--recording-id", default=None, help="Optional recording identifier for single-WAV mode.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N participants in batch mode.")
    parser.add_argument("--run-id", default=None, help="Optional shared run id.")
    args = parser.parse_args()
    run_prepare_audio(
        recordings_parquet=args.recordings_parquet,
        wav=args.wav,
        recording_id=args.recording_id,
        limit=args.limit,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()