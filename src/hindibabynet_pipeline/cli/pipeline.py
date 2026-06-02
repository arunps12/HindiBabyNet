from __future__ import annotations

import argparse

from hindibabynet_pipeline.workflow.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HindiBabyNet-Pipeline end-to-end.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N participants.")
    parser.add_argument("--backend", choices=["xgb", "vtc"], default=None, help="Override the configured classification backend.")
    args = parser.parse_args()
    run_pipeline(limit=args.limit, backend=args.backend)


if __name__ == "__main__":
    main()