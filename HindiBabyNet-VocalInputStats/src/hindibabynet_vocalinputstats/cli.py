"""Command-line interface for the HindiBabyNet vocal input statistics package."""

from __future__ import annotations

import argparse

from hindibabynet_vocalinputstats.build_master_dataset import run_build_master
from hindibabynet_vocalinputstats.create_long_format import run_create_long
from hindibabynet_vocalinputstats.eda import run_eda
from hindibabynet_vocalinputstats.plots import run_plots


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

    create_long_parser = subparsers.add_parser(
        "create-long",
        help="Create long-format datasets from final_master.csv.",
    )
    create_long_parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the repository config file.",
    )
    create_long_parser.set_defaults(handler=_handle_create_long)

    eda_parser = subparsers.add_parser(
        "eda",
        help="Generate EDA summary tables.",
    )
    eda_parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the repository config file.",
    )
    eda_parser.set_defaults(handler=_handle_eda)

    plots_parser = subparsers.add_parser(
        "plots",
        help="Generate publication-style plots.",
    )
    plots_parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the repository config file.",
    )
    plots_parser.set_defaults(handler=_handle_plots)

    all_parser = subparsers.add_parser(
        "all",
        help="Run build-master, create-long, eda, and plots in sequence.",
    )
    all_parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the repository config file.",
    )
    all_parser.set_defaults(handler=_handle_all)
    return parser


def _handle_build_master(args: argparse.Namespace) -> int:
    run_build_master(config_path=args.config)
    return 0


def _handle_create_long(args: argparse.Namespace) -> int:
    run_create_long(config_path=args.config)
    return 0


def _handle_eda(args: argparse.Namespace) -> int:
    run_eda(config_path=args.config)
    return 0


def _handle_plots(args: argparse.Namespace) -> int:
    run_plots(config_path=args.config)
    return 0


def _handle_all(args: argparse.Namespace) -> int:
    run_build_master(config_path=args.config)
    run_create_long(config_path=args.config)
    run_eda(config_path=args.config)
    run_plots(config_path=args.config)
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