"""Command-line interface for the HindiBabyNet vocal input statistics package."""

from __future__ import annotations

import argparse

from hindibabynet_vocalinputstats.build_master_dataset import run_build_master


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
    subparsers = parser.add_subparsers(dest="command")

    build_master_parser = subparsers.add_parser(
        "build-master",
        help="Build the final participant-level master dataset.",
    )
    build_master_parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the repository config file.",
    )
    build_master_parser.set_defaults(handler=_handle_build_master)
    return parser


def _handle_build_master(args: argparse.Namespace) -> int:
    run_build_master(config_path=args.config)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())