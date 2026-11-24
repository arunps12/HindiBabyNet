"""Tests of the VTC speaker classification. VTC itself is never started."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hindibabynet_audio import vtc

# VTC runs only in Ubuntu, so the tests that start a run are skipped on Windows.
ubuntu_only = pytest.mark.skipif(os.name == "nt", reason="VTC runs only in Ubuntu")


def must_not_run(*args, **kwargs):
    pytest.fail("VTC must not be started")


def prepared_wav(config, participant: str) -> Path:
    """Write a placeholder for the prepared WAV file of one participant."""
    path = config.path("prepared_audio", participant, f"{participant}.wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder for prepared audio")
    return path


def complete_results(folder: Path, name: str = "P1") -> None:
    """Write VTC results as they are found in the existing result folders."""
    (folder / "rttm").mkdir(parents=True)
    (folder / "rttm" / f"{name}.rttm").write_text("SPEAKER\n", encoding="utf-8")
    (folder / "rttm.csv").write_text("uid,start_time_s,duration_s,label\n", encoding="utf-8")
    (folder / "run_info.json").write_text("{}", encoding="utf-8")


def contents(folder: Path) -> dict:
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in folder.rglob("*") if path.is_file()
    }


@pytest.fixture
def vtc_repo(tmp_path, monkeypatch) -> Path:
    """A folder that looks like an installed VTC; uv is 'found' and git is not asked."""
    repo = tmp_path / "VTC"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "infer.py").write_text("# placeholder", encoding="utf-8")
    monkeypatch.setattr(vtc.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(vtc, "vtc_commit", lambda repo: "abc1234")
    return repo


@pytest.fixture
def fake_vtc(monkeypatch) -> list:
    """Replace the VTC program by a function that writes the two results. Returns the calls."""
    calls = []

    def fake_run(command, **options):
        calls.append((command, options))
        output = Path(command[command.index("--output") + 1])
        name = Path(command[command.index("--wavs") + 1]).name
        (output / "rttm").mkdir(parents=True, exist_ok=True)
        (output / "rttm" / f"{name}.rttm").write_text("SPEAKER\n", encoding="utf-8")
        (output / "rttm.csv").write_text(
            "uid,start_time_s,duration_s,label\n"
            f"{name},0.0,1.5,FEM\n{name},2.0,0.5,KCHI\n{name},3.0,0.7,MAL\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vtc.subprocess, "run", fake_run)
    return calls


def test_build_command():
    command = vtc.build_command(
        "/home/user/VTC", "/data/prepared-audio/P1", "/data/vtc-classification/P1", "cpu", 32
    )
    assert command == [
        "uv", "run", "scripts/infer.py",
        "--wavs", "/data/prepared-audio/P1",
        "--output", "/data/vtc-classification/P1",
        "--device", "cpu",
        "--batch_size", "32",
    ]


def test_results_complete(tmp_path):
    # Existing results: rttm/, rttm.csv and run_info.json, no raw_rttm.
    complete = tmp_path / "complete"
    complete_results(complete)
    assert vtc.results_complete(complete)

    assert not vtc.results_complete(tmp_path / "missing")

    no_csv = tmp_path / "no_csv"
    complete_results(no_csv)
    (no_csv / "rttm.csv").unlink()
    assert not vtc.results_complete(no_csv)

    no_rttm_file = tmp_path / "no_rttm_file"
    complete_results(no_rttm_file)
    (no_rttm_file / "rttm" / "P1.rttm").unlink()
    assert not vtc.results_complete(no_rttm_file)

    no_rttm_folder = tmp_path / "no_rttm_folder"
    no_rttm_folder.mkdir()
    (no_rttm_folder / "rttm.csv").write_text("uid,start_time_s,duration_s,label\n")
    assert not vtc.results_complete(no_rttm_folder)


@ubuntu_only
def test_complete_results_are_not_run_again(make_config, params, vtc_repo, monkeypatch, capsys):
    config = make_config(audio={"vtc_repo": vtc_repo})
    prepared_wav(config, "P1")
    results = config.path("vtc", "P1")
    complete_results(results)
    before = contents(results)
    monkeypatch.setattr(vtc.subprocess, "run", must_not_run)

    assert vtc.run(config, params) == 0

    output = capsys.readouterr().out
    assert "P1: results already available, not run again" in output
    assert "Summary: 0 classified, 1 already available, 0 error" in output
    assert contents(results) == before


@ubuntu_only
def test_incomplete_result_folder_is_an_error_and_is_not_touched(
    make_config, params, vtc_repo, monkeypatch, capsys
):
    config = make_config(audio={"vtc_repo": vtc_repo})
    prepared_wav(config, "P1")
    prepared_wav(config, "P2")
    results = config.path("vtc", "P1")
    complete_results(results)
    (results / "rttm.csv").unlink()
    before = contents(results)
    complete_results(config.path("vtc", "P2"), "P2")
    monkeypatch.setattr(vtc.subprocess, "run", must_not_run)

    assert vtc.run(config, params) == 1

    output = capsys.readouterr().out
    assert "P1: error" in output and "not complete" in output
    assert "P2: results already available" in output  # the other participant continues
    assert contents(results) == before


@ubuntu_only
def test_run_writes_run_info_and_uses_the_settings(
    make_config, params, vtc_repo, fake_vtc, capsys
):
    config = make_config(audio={"vtc_repo": vtc_repo, "vtc_device": "cuda"})
    wav = prepared_wav(config, "P1")
    results = config.path("vtc", "P1")

    assert vtc.run(config, params) == 0

    # One call: the folder of the participant as input, VTC folder as working directory.
    (command, options), = fake_vtc
    assert command == [
        "uv", "run", "scripts/infer.py",
        "--wavs", str(wav.parent),
        "--output", str(results),
        "--device", "cuda",
        "--batch_size", "32",
    ]
    assert Path(command[4]).is_absolute() and Path(command[6]).is_absolute()
    assert options["cwd"] == str(vtc_repo)
    # The VTC messages go to the console: nothing is captured.
    assert not {"capture_output", "stdout", "stderr"} & set(options)

    info = json.loads((results / "run_info.json").read_text(encoding="utf-8"))
    assert list(info) == [
        "participant_id", "backend", "input_wav", "output_dir", "started_at", "finished_at",
        "runtime_sec", "status", "command", "device", "vtc_repo_path", "outputs_present",
        "batch_size", "vtc_commit",
    ]
    assert info["participant_id"] == "P1"
    assert info["backend"] == "vtc"
    assert info["input_wav"] == str(wav)
    assert info["output_dir"] == str(results)
    assert info["status"] == "success"
    assert info["command"] == " ".join(command)
    assert info["device"] == "cuda"
    assert info["vtc_repo_path"] == str(vtc_repo)
    assert info["outputs_present"] == {"rttm": True, "rttm.csv": True}
    assert info["batch_size"] == 32
    assert info["vtc_commit"] == "abc1234"

    output = capsys.readouterr().out
    assert "P1: finished, 3 segments" in output
    assert "Summary: 1 classified, 0 already available, 0 error" in output

    # The prepared audio is not copied or changed, and nothing is written for XGBoost.
    assert sorted(path.name for path in wav.parent.iterdir()) == ["P1.wav"]
    assert not config.path("xgb").exists()
    assert sorted(path.name for path in config.data_dir.joinpath("derived").iterdir()) == [
        "prepared-audio", "vtc-classification"
    ]


@ubuntu_only
def test_default_device_and_other_batch_size(make_config, params, vtc_repo, fake_vtc):
    config = make_config(audio={"vtc_repo": vtc_repo})  # no vtc_device: cpu
    prepared_wav(config, "P1")
    params["vtc"]["batch_size"] = 8

    assert vtc.run(config, params) == 0

    (command, _), = fake_vtc
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--batch_size") + 1] == "8"
    info = json.loads(config.path("vtc", "P1", "run_info.json").read_text(encoding="utf-8"))
    assert info["device"] == "cpu" and info["batch_size"] == 8


@ubuntu_only
def test_participant_and_limit_select_the_participants(make_config, params, vtc_repo, fake_vtc):
    config = make_config(audio={"vtc_repo": vtc_repo})
    for name in ("P1", "P2", "P3"):
        prepared_wav(config, name)
    config.path("prepared_audio", "P0").mkdir()  # a folder without prepared audio

    assert vtc.run(config, params, limit=2) == 0
    assert sorted(path.name for path in config.path("vtc").iterdir()) == ["P1", "P2"]

    assert vtc.run(config, params, participant="P3") == 0
    assert vtc.results_complete(config.path("vtc", "P3"))
    assert len(fake_vtc) == 3

    assert vtc.run(config, params, participant="P9") == 1


@ubuntu_only
def test_vtc_error_gives_exit_code_1_and_no_run_info(
    make_config, params, vtc_repo, monkeypatch, capsys
):
    config = make_config(audio={"vtc_repo": vtc_repo})
    prepared_wav(config, "P1")
    monkeypatch.setattr(
        vtc.subprocess, "run", lambda command, **options: subprocess.CompletedProcess(command, 3)
    )

    assert vtc.run(config, params) == 1

    assert "P1: error" in capsys.readouterr().out
    assert not config.path("vtc", "P1", "run_info.json").exists()
    assert not config.path("xgb").exists()


@ubuntu_only
@pytest.mark.parametrize("missing", ["script", "uv"])
def test_missing_vtc_or_uv_is_a_clear_error(
    make_config, params, vtc_repo, monkeypatch, capsys, missing
):
    config = make_config(audio={"vtc_repo": vtc_repo})
    prepared_wav(config, "P1")
    if missing == "script":
        (vtc_repo / "scripts" / "infer.py").unlink()
    else:
        monkeypatch.setattr(vtc.shutil, "which", lambda name: None)
    monkeypatch.setattr(vtc.subprocess, "run", must_not_run)

    assert vtc.run(config, params) == 1

    output = capsys.readouterr().out
    assert "P1: error" in output
    assert ("infer.py" if missing == "script" else "uv was not found") in output
    assert not config.path("vtc").exists()


def test_refuses_on_windows(make_config, params, monkeypatch, capsys):
    config = make_config()
    monkeypatch.setattr(vtc.subprocess, "run", must_not_run)
    with monkeypatch.context() as patch:
        patch.setattr(vtc.os, "name", "nt")
        code = vtc.run(config, params)

    assert code == 1
    assert "VTC runs only in Ubuntu" in capsys.readouterr().out
