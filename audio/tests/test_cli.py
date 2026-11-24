"""Tests of the command line: options, the function behind every command, error messages."""

from __future__ import annotations

import sys
import types

import pytest

from hindibabynet_audio import cli


@pytest.fixture
def fake_part(monkeypatch):
    """Replace one part of the package by functions that only note how they are called."""
    calls = []

    def _fake(name: str, *functions: str, exit_code: int = 0):
        module = types.ModuleType(f"hindibabynet_audio.{name}")
        for function in functions:
            def note(*args, _function=function, **options):
                calls.append((f"{name}.{_function}", args, options))
                return exit_code
            setattr(module, function, note)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        return calls

    return _fake


def parse(*arguments: str):
    return cli.build_parser().parse_args(list(arguments))


# --- options ------------------------------------------------------------------


def test_prepare_options():
    args = parse("prepare")
    assert (args.config, args.participant, args.limit, args.check) == (None, None, None, False)

    args = parse("prepare", "--check", "--config", "my.yaml", "--participant", "P1",
                 "--limit", "3")
    assert (args.config, args.participant, args.limit, args.check) == ("my.yaml", "P1", 3, True)


@pytest.mark.parametrize("command", ["classify-vtc", "classify-xgb"])
def test_classification_options(command):
    args = parse(command)
    assert (args.config, args.participant, args.limit) == (None, None, None)

    args = parse(command, "--config", "my.yaml", "--participant", "P1", "--limit", "2")
    assert (args.config, args.participant, args.limit) == ("my.yaml", "P1", 2)
    with pytest.raises(SystemExit):
        parse(command, "--check")


def test_options_of_the_optional_tools():
    args = parse("textgrid", "--method", "vtc", "--participant", "P1", "--limit", "2")
    assert (args.method, args.participant, args.limit) == ("vtc", "P1", 2)
    args = parse("textgrid", "--method", "xgb")
    assert (args.method, args.participant, args.limit) == ("xgb", None, None)

    args = parse("annotate", "--method", "xgb", "--participant", "P1")
    assert (args.method, args.participant) == ("xgb", "P1")

    args = parse("evaluate", "--method", "vtc")
    assert (args.method, args.participant) == ("vtc", None)
    args = parse("evaluate", "--method", "vtc", "--participant", "P1", "--config", "my.yaml")
    assert (args.method, args.participant, args.config) == ("vtc", "P1", "my.yaml")


@pytest.mark.parametrize(
    "arguments",
    [
        [],                                           # no command
        ["other"],                                    # unknown command
        ["prepare", "--limit", "many"],               # the limit is a number
        ["textgrid"],                                 # the method is needed
        ["textgrid", "--method", "other"],            # unknown method
        ["annotate", "--method", "vtc"],              # the participant is needed
        ["annotate", "--method", "vtc", "--participant", "P1", "--limit", "2"],
        ["evaluate", "--method", "vtc", "--limit", "2"],
    ],
)
def test_wrong_options_are_refused(arguments, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.build_parser().parse_args(arguments)
    assert stop.value.code == 2


# --- the function behind every command ---------------------------------------


def test_prepare_calls_the_preparation(make_config, fake_part):
    config = make_config()
    calls = fake_part("prepare", "run")

    assert cli.main(["prepare", "--config", str(config.file)]) == 0
    assert cli.main(["prepare", "--config", str(config.file), "--check",
                     "--participant", "P1", "--limit", "2"]) == 0

    (name, args, options), (_, _, second) = calls
    assert name == "prepare.run"
    assert args[0].file == config.file  # the loaded config.yaml
    assert args[1]["vtc"]["batch_size"] == 32  # the settings of params.yaml
    assert options == {"participant": None, "limit": None, "check": False}
    assert second == {"participant": "P1", "limit": 2, "check": True}


@pytest.mark.parametrize("command, part", [("classify-vtc", "vtc"), ("classify-xgb", "xgb")])
def test_classification_commands_call_their_own_method(make_config, fake_part, command, part):
    config = make_config()
    calls = fake_part(part, "run", exit_code=1)

    code = cli.main([command, "--config", str(config.file), "--participant", "P1", "--limit", "4"])

    assert code == 1  # the exit code of the function is returned
    (name, args, options), = calls
    assert name == f"{part}.run"
    assert args[0].file == config.file and "audio_preparation" in args[1]
    assert options == {"participant": "P1", "limit": 4}


def test_optional_tools_call_their_functions(make_config, fake_part):
    config = make_config()
    fake_part("textgrid", "run")
    calls = fake_part("evaluation", "run_annotation", "run_evaluation")
    file = str(config.file)

    assert cli.main(["textgrid", "--config", file, "--method", "vtc", "--limit", "2"]) == 0
    assert cli.main(["annotate", "--config", file, "--method", "xgb", "--participant", "P1"]) == 0
    assert cli.main(["evaluate", "--config", file, "--method", "vtc"]) == 0

    textgrid, annotate, evaluate = calls
    assert textgrid[0] == "textgrid.run"
    assert textgrid[1][2] == "vtc" and textgrid[2] == {"participant": None, "limit": 2}
    assert annotate[0] == "evaluation.run_annotation"
    assert annotate[1][2:] == ("xgb", "P1") and annotate[2] == {}
    assert evaluate[0] == "evaluation.run_evaluation"
    assert evaluate[1][2] == "vtc" and evaluate[2] == {"participant": None}


# --- error messages -----------------------------------------------------------


def test_missing_config_gives_exit_code_1_and_one_line(tmp_path, fake_part, capsys):
    calls = fake_part("prepare", "run")

    code = cli.main(["prepare", "--config", str(tmp_path / "missing.yaml")])

    assert code == 1
    output = capsys.readouterr()
    assert output.err == ""
    assert len(output.out.splitlines()) == 1
    assert "Configuration file not found" in output.out
    assert calls == []


def test_classify_xgb_without_the_xgboost_packages_gives_the_install_hint(
    make_config, monkeypatch, capsys
):
    config = make_config()
    # None in sys.modules: the import fails with ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "hindibabynet_audio.xgb", None)

    code = cli.main(["classify-xgb", "--config", str(config.file)])

    assert code == 1
    output = capsys.readouterr().out
    assert "uv sync --project audio --extra xgb" in output
    assert len(output.splitlines()) == 1
