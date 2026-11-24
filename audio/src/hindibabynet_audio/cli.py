"""Command line of the audio workflow.

Main commands:
  hbn-audio prepare [--check]   one prepared WAV file per participant
  hbn-audio classify-vtc        speaker classification with VTC (in Ubuntu)
  hbn-audio classify-xgb        speaker classification with XGBoost

Optional tools:
  hbn-audio textgrid --method vtc|xgb                    Praat TextGrid files
  hbn-audio annotate --method vtc|xgb --participant ID   manual speaker labels
  hbn-audio evaluate --method vtc|xgb                    compare with the manual labels

The data folders are read from config.yaml in the project root (or --config).
"""

from __future__ import annotations

import argparse
import importlib

import yaml
from hbn_paths import Config

from hindibabynet_audio.params import load_params


def _module(name: str):
    """Import one part of the package only when its command is used."""
    return importlib.import_module(f"hindibabynet_audio.{name}")


def _prepare(config, params, args) -> int:
    return _module("prepare").run(
        config, params, participant=args.participant, limit=args.limit, check=args.check
    )


def _classify_vtc(config, params, args) -> int:
    return _module("vtc").run(config, params, participant=args.participant, limit=args.limit)


def _classify_xgb(config, params, args) -> int:
    try:
        xgb = _module("xgb")
    except ModuleNotFoundError:
        print("The XGBoost packages are not installed. "
              "Install them with: uv sync --project audio --extra xgb")
        return 1
    return xgb.run(config, params, participant=args.participant, limit=args.limit)


def _textgrid(config, params, args) -> int:
    return _module("textgrid").run(
        config, params, args.method, participant=args.participant, limit=args.limit
    )


def _annotate(config, params, args) -> int:
    return _module("evaluation").run_annotation(config, params, args.method, args.participant)


def _evaluate(config, params, args) -> int:
    return _module("evaluation").run_evaluation(
        config, params, args.method, participant=args.participant
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hbn-audio",
        description="HindiBabyNet audio: audio preparation and speaker classification.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add(name, handler, help_text, method=False, participant_required=False, limit=True):
        command = commands.add_parser(name, help=help_text, description=help_text)
        command.set_defaults(handler=handler)
        command.add_argument("--config", metavar="PATH", default=None,
                             help="config file (default: config.yaml in the project root)")
        if method:
            command.add_argument("--method", choices=["vtc", "xgb"], required=True,
                                 help="classification results to use")
        command.add_argument("--participant", metavar="ID", default=None,
                             required=participant_required, help="only this participant")
        if limit:
            command.add_argument("--limit", metavar="N", type=int, default=None,
                                 help="only the first N participants in name order")
        return command

    prepare = add("prepare", _prepare,
                  "Prepare one WAV file per participant from the first recording day.")
    prepare.add_argument("--check", action="store_true",
                         help="only report raw and prepared audio; nothing is written")
    add("classify-vtc", _classify_vtc, "Speaker classification with VTC (run it in Ubuntu).")
    add("classify-xgb", _classify_xgb, "Speaker classification with XGBoost.")
    add("textgrid", _textgrid, "Optional: write Praat TextGrid files.", method=True)
    add("annotate", _annotate, "Optional: label speech segments of one participant by hand.",
        method=True, participant_required=True, limit=False)
    add("evaluate", _evaluate, "Optional: compare the classification with the manual labels.",
        method=True, limit=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = Config(args.config)
        params = load_params()
        return args.handler(config, params, args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        # For example a missing config file: one clear line, no traceback.
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
