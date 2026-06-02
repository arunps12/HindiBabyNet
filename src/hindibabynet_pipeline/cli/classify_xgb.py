from __future__ import annotations

import argparse

from hindibabynet_pipeline.workflow.xgb_classification import run_classify_xgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XGB speaker classification.")
    parser.add_argument("--recordings-parquet", dest="recordings_parquet", default=None)
    parser.add_argument("--wav", default=None)
    parser.add_argument("--analysis-dir", dest="analysis_dir", default=None)
    parser.add_argument("--participant-id", dest="participant_id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_classify_xgb(
        recordings_parquet=args.recordings_parquet,
        wav=args.wav,
        analysis_dir=args.analysis_dir,
        participant_id=args.participant_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()