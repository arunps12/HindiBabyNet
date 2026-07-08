"""Command-line interface for the HindiBabyNet vocal input statistics package."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hindibabynet-vocalinputstats",
        description="HindiBabyNet vocal input statistics workflow.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="hindibabynet-vocalinputstats 0.1.0",
    )
    return parser


def main() -> int:
    build_parser().parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())