"""Synthetic launcher contracts. Never execute science, Git, installs or models."""

import ast
import argparse
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

import run as launcher


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected subprocess execution")
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher.shutil, "which", lambda value: str(value))
    for key in list(os.environ):
        if key.startswith("HBN_"):
            monkeypatch.delenv(key)


@pytest.fixture
def sandbox(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("metadata.csv", "lookup.csv", "master.csv", "long.csv", "io.csv", "features.csv", "rttm.csv", "synthetic.wav"):
        (inputs / name).write_text("synthetic,value\na,1\n", encoding="utf-8")
    settings = {"workflows": {}}
    for name in launcher.ORDER:
        project = tmp_path / launcher.PROJECTS[name]
        project.mkdir(exist_ok=True)
        for filename in ("pyproject.toml", "uv.lock", "src/science.py"):
            file = project / filename
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# synthetic", encoding="utf-8")
        python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()
        settings["workflows"][name] = {"enabled": True, "project": project.name}
        if name == "audio":
            repo = tmp_path / "vtc"
            for filename in ("scripts/infer.py", "uv.lock", "weights/model.pt", ".venv/" +
                             ("Scripts/python.exe" if os.name == "nt" else "bin/python")):
                file = repo / filename
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("synthetic", encoding="utf-8")
            data = {"paths": {"raw_audio_root": str(inputs), **{key: "generated/" + key for key in
                    ("raw_joined_audio_root", "prepared_audio_root", "classification_output_root", "textgrid_output_root",
                     "manual_annotation_root", "evaluation_output_root")}},
                    "speaker_classification": {"backend": "vtc", "vtc": {"repo_path": str(repo), "batch_size": 7}}}
            settings["audio_params"] = {"audio_preparation": {"target_sr": 12345}, "vtc": {"batch_size": 3}}
            write_yaml(project / "configs/params.yaml", settings["audio_params"])
        elif name == "vocal-stats":
            data = {"metadata_path": str(inputs / "metadata.csv"), "audio_root": str(inputs), "vtc_output_root": str(inputs),
                    "derived_data_dir": "data/derived", "private_data_dir": "private", "figures_dir": "figures",
                    "tables_dir": "tables", "results_dir": "results", "age_month_denominator": 37.25}
        elif name == "prosody":
            data = {"paths": {"audio_root": str(inputs), "rttm_root": str(inputs), "participant_lookup": str(inputs / "lookup.csv"),
                              "output_dir": "outputs", "cache_dir": "cache"}, "pitch": {"semitone_reference_hz": 123.0}}
        elif name == "cdi":
            (project / "metadata").mkdir()
            (project / "metadata/word_mapping.csv").write_text("word\nsynthetic", encoding="utf-8")
            data = {"paths": {"raw_data": str(inputs)}, "analysis": {"strict_hindi_threshold": 66.0}}
        elif name == "prosody-stats":
            data = {"paths": dict(zip(("acoustic_features", "final_master", "input_long", "input_output_long", "participant_lookup",
                                       "metadata_cleaned"), (str(inputs / f) for f in
                                       ("features.csv", "master.csv", "long.csv", "io.csv", "lookup.csv", "metadata.csv")))),
                    "analysis": {"monte_carlo_draws": 17}}
        else:
            data = None
        if data is not None:
            write_yaml(project / launcher.NATIVE[name], data)
            settings[launcher.SECTIONS[name]] = {**data, "project_root": str(project)}
        if name in {"vocal-models", "prosody-stats"}:
            for script in launcher.r_scripts(name) + (["04_final_report.R"] if name == "vocal-models" else []):
                file = project / "R" / script
                file.parent.mkdir(exist_ok=True)
                file.write_text("# synthetic", encoding="utf-8")
            template = project / ("R/report.qmd" if name == "vocal-models" else "HindiBabyNet_prosody_stat_model_report.Rmd")
            template.write_text("synthetic", encoding="utf-8")
    local = tmp_path / "config.local.yaml"
    write_yaml(local, settings)
    return tmp_path, local, settings


def select(sandbox, *names):
    root, local, settings = sandbox
    for name, entry in settings["workflows"].items():
        entry["enabled"] = name in names
    write_yaml(local, settings)
    return root, local, settings


def item_for(sandbox, name, stage=None, extra=()):
    _, local, settings = sandbox
    return launcher.workflow(name, settings["workflows"][name], local.parent, stage, extra, local)


def produce(item):
    for directory, pattern in item["expected"]:
        directory.mkdir(parents=True, exist_ok=True)
        filename = pattern.replace("*", "synthetic")
        (directory / filename).write_text("synthetic,value\na,1\n", encoding="utf-8")
    if item["name"] == "audio":
        root, data = item["root"], item["data"]
        prepared = root / data["paths"]["prepared_audio_root"] / "synthetic/synthetic.wav"
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_bytes(b"synthetic audio")
        sc = data["speaker_classification"]
        classification = root / sc.get("output_root", data["paths"]["classification_output_root"])
        directory = classification / item["backend"] / "synthetic"
        directory.mkdir(parents=True, exist_ok=True)
        if item["backend"] == "vtc":
            for part in ("rttm", "raw_rttm"):
                (directory / part).mkdir(exist_ok=True)
            for part in ("rttm.csv", "raw_rttm.csv"):
                (directory / part).write_text("synthetic,value\na,1\n", encoding="utf-8")


def mock_success(sandbox, monkeypatch, stage=None):
    calls = []

    def execute(command, **kw):
        name = next(name for name in launcher.ORDER if kw["cwd"].name == launcher.PROJECTS[name]
                    and (name != "vocal-stats" or command[0] == "uv")
                    and (name != "vocal-models" or command[0] != "uv"))
        calls.append((command, kw))
        produce(item_for(sandbox, name, stage))
    monkeypatch.setattr(launcher.subprocess, "run", execute)
    return calls


def records(local):
    return list((local.parent / ".local/runs").glob("*.json"))


def test_all_enabled_shared_commands_isolation_and_verified_skip(sandbox, monkeypatch):
    _, local, _ = sandbox
    monkeypatch.setenv("VIRTUAL_ENV", "foreign")
    monkeypatch.setenv("PYTHONPATH", "foreign")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "foreign")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["all", "--config", str(local)]) == 0
    assert len(calls) == 6
    assert len(records(local)) == 6
    assert [kw["cwd"].name for _, kw in calls] == [launcher.PROJECTS[n] for n in launcher.ORDER]
    for command, kw in calls:
        assert kw["check"] is True
        assert not {"VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT"} & kw["env"].keys()
        assert kw["env"]["HBN_CONFIG_PATH"] == str(local)
        assert kw["env"]["UV_NO_SYNC"] == "1"
        if command[0] == "uv":
            assert command[:4] == ["uv", "run", "--frozen", "--no-sync"]
            if kw["cwd"].name != launcher.PROJECTS["audio"]:
                assert str(local) in command
    assert launcher.main(["all", "--config", str(local)]) == 0
    assert len(calls) == 6


@pytest.mark.parametrize("command", ["check", "all"])
def test_read_only_no_state_or_processes(sandbox, command, capsys):
    root, local, _ = select(sandbox, "prosody")
    occupied = root / launcher.PROJECTS["prosody"] / "outputs/old.csv"
    occupied.parent.mkdir()
    occupied.write_text("private original", encoding="utf-8")
    assert launcher.main([command, "--config", str(local), "--dry-run"]) == 0
    assert "BLOCKED" in capsys.readouterr().out
    assert not (root / ".local").exists()
    assert occupied.read_text(encoding="utf-8") == "private original"


def test_check_without_dry_run(sandbox):
    _, local, _ = select(sandbox, "cdi")
    assert launcher.main(["check", "--config", str(local)]) == 0
    assert not local.parent.joinpath(".local").exists()


@pytest.mark.parametrize("change", ["config", "rttm", "audio", "metadata", "source", "lock", "env", "output", "extra-output", "missing-output"])
def test_stale_success_refuses_occupied_outputs(sandbox, monkeypatch, change):
    root, local, settings = select(sandbox, "prosody")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["prosody", "--config", str(local)]) == 0
    original = records(local)[0].read_bytes()
    project = root / launcher.PROJECTS["prosody"]
    if change == "config":
        settings["prosody"]["pitch"]["semitone_reference_hz"] = 124.0
        write_yaml(local, settings)
    elif change == "env":
        monkeypatch.setenv("HBN_SCIENTIFIC_SETTING", "different")
    elif change == "extra-output":
        (project / "outputs/unowned.csv").write_text("extra", encoding="utf-8")
    elif change == "missing-output":
        (project / "outputs/run_metadata.json").unlink()
    else:
        path = {"rttm": root / "inputs/rttm.csv", "audio": root / "inputs/synthetic.wav",
                "metadata": root / "inputs/lookup.csv", "source": project / "src/science.py",
                "lock": project / "uv.lock", "output": project / "outputs/run_metadata.json"}[change]
        path.write_text("changed bytes", encoding="utf-8")
    assert launcher.main(["prosody", "--config", str(local)]) == 1
    assert len(calls) == 1
    assert records(local)[0].read_bytes() == original


@pytest.mark.parametrize("change", ["weights/model.pt", "scripts/infer.py", "uv.lock"])
def test_external_vtc_content_identity(sandbox, monkeypatch, change):
    root, local, _ = select(sandbox, "audio")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["audio", "--config", str(local)]) == 0
    (root / "vtc" / change).write_text("changed", encoding="utf-8")
    assert launcher.main(["audio", "--config", str(local)]) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["child", "no-output", "empty-output", "partial-output", "input-during-run"])
def test_failure_never_records_completion(sandbox, monkeypatch, failure):
    root, local, _ = select(sandbox, "prosody")
    item = item_for(sandbox, "prosody")

    def execute(command, **kwargs):
        if failure == "child":
            produce(item)
            raise subprocess.CalledProcessError(17, command)
        if failure != "no-output":
            produce(item)
            if failure == "empty-output":
                (item["root"] / "outputs/run_metadata.json").write_text("")
            elif failure == "partial-output":
                (item["root"] / "outputs/run_metadata.json").unlink()
            else:
                (root / "inputs/rttm.csv").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(launcher.subprocess, "run", execute)
    assert launcher.main(["prosody", "--config", str(local)]) == (17 if failure == "child" else 1)
    assert not records(local)
    if failure != "no-output":
        monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("Partial resume"))
        assert launcher.main(["prosody", "--config", str(local)]) == 1


def test_no_resume_is_not_overwrite_permission(sandbox, monkeypatch):
    _, local, _ = select(sandbox, "cdi")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["cdi", "--config", str(local)]) == 0
    assert launcher.main(["cdi", "--config", str(local), "--no-resume"]) == 1
    assert len(calls) == 1


def test_forged_marker_cannot_adopt_outputs(sandbox):
    _, local, _ = select(sandbox, "prosody")
    item = item_for(sandbox, "prosody")
    path, _, _ = launcher.run_state(item, local, True)
    produce(item)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ownership": launcher.OWNER}), encoding="utf-8")
    assert launcher.main(["prosody", "--config", str(local)]) == 1


def test_configure_preserves_native_snapshot_and_refuses_overwrite(sandbox, monkeypatch):
    root, _, _ = sandbox
    monkeypatch.setattr(launcher, "ROOT", root)
    destination = root / "snapshot.local.yaml"
    native = {n: launcher.read_yaml(root / launcher.PROJECTS[n] / launcher.NATIVE[n])
              for n in launcher.ORDER if n != "vocal-models"}
    assert launcher.main(["configure", "--config", str(destination)]) == 0
    snapshot = launcher.read_yaml(destination)
    assert all(v["enabled"] is False for v in snapshot["workflows"].values())
    for name, mapping in native.items():
        section = snapshot[launcher.SECTIONS[name]]
        assert section == {**mapping, "project_root": str(root / launcher.PROJECTS[name])}
        assert launcher.workflow(name, snapshot["workflows"][name], root, shared=destination)["outputs"] == item_for(sandbox, name)["outputs"]
    assert snapshot["audio_params"]["audio_preparation"]["target_sr"] == 12345
    before = destination.read_bytes()
    assert launcher.main(["configure", "--config", str(destination)]) == 1
    assert destination.read_bytes() == before
    assert launcher.main(["configure", "--config", str(root / "absent.yaml"), "--dry-run"]) == 0
    assert not (root / "absent.yaml").exists()


def test_separate_native_configs_and_shared_relative_roots(sandbox):
    root, local, settings = select(sandbox, "prosody", "cdi", "audio")
    for name in ("prosody", "cdi", "audio"):
        shared = item_for(sandbox, name)
        settings["workflows"][name]["config"] = str(Path(launcher.PROJECTS[name]) / launcher.NATIVE[name])
        write_yaml(local, settings)
        assert item_for(sandbox, name)["outputs"] == shared["outputs"]
        settings["workflows"][name].pop("config")
    settings["prosody"]["project_root"] = launcher.PROJECTS["prosody"]
    write_yaml(local, settings)
    assert item_for(sandbox, "prosody")["outputs"][0] == root / launcher.PROJECTS["prosody"] / "outputs"
    assert item_for(sandbox, "audio")["data"]["speaker_classification"]["vtc"]["batch_size"] == 7


def test_explicit_config_wins_over_environment(sandbox, monkeypatch):
    _, local, _ = select(sandbox, "audio")
    monkeypatch.setenv("HBN_CONFIG_PATH", "nonexistent.yaml")
    assert launcher.main(["check", "--config", str(local)]) == 0
    monkeypatch.setenv("HBN_CONFIG_PATH", str(local))
    assert launcher.main(["check"]) == 0


def test_setup_explicit_sync_only(sandbox, monkeypatch):
    _, local, _ = sandbox
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    assert launcher.main(["setup", "--config", str(local)]) == 0
    assert len(calls) == 4
    assert all(cmd == ["uv", "sync", "--frozen"] for cmd, _ in calls)
    assert not records(local)


@pytest.mark.parametrize("name", ["vocal-models", "prosody-stats"])
def test_explicit_reports_not_default_and_no_overwrite(sandbox, monkeypatch, name):
    _, local, _ = select(sandbox, name)
    item = item_for(sandbox, name, "report")
    for path in item["inputs"]:
        if not path.exists():
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic", encoding="utf-8")
            else:
                path.mkdir(parents=True)
    assert "04_final_report.R" not in " ".join(item_for(sandbox, name)["command"])
    assert "rmarkdown::render" not in " ".join(item_for(sandbox, name)["command"])
    calls = mock_success(sandbox, monkeypatch, "report")
    assert launcher.main([name, "--config", str(local), "--stage", "report"]) == 0
    assert len(calls) == 1
    command, kw = calls[0]
    if name == "vocal-models":
        assert "04_final_report.R" in command[-1]
        assert kw["env"]["HBN_REPORT_FORMAT"] == "html"
    else:
        assert "rmarkdown::render" in command[-1]
        assert "run_workflow=FALSE" in command[-1]
    assert launcher.main([name, "--config", str(local), "--stage", "report"]) == 0
    assert len(calls) == 1
    assert launcher.main([name, "--config", str(local), "--stage", "report", "--no-resume"]) == 1


@pytest.mark.parametrize("option", ["--config=elsewhere", "--overwrite", "--backend", "--conf", "--resume"])
def test_passthrough_cannot_bypass_safety(sandbox, option):
    _, local, _ = select(sandbox, "prosody")
    assert launcher.main(["prosody", "--config", str(local), "--", option]) == 1


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_enabled_requires_boolean(sandbox, value):
    _, local, settings = sandbox
    settings["workflows"]["audio"]["enabled"] = value
    write_yaml(local, settings)
    assert launcher.main(["check", "--config", str(local)]) == 1


def test_disabled_and_malformed_configs(sandbox):
    _, local, _ = select(sandbox)
    assert launcher.main(["all", "--config", str(local)]) == 0
    assert launcher.main(["audio", "--config", str(local)]) == 1
    local.write_text("workflows: [unclosed", encoding="utf-8")
    assert launcher.main(["check", "--config", str(local)]) == 1


def test_xgb_cannot_feed_vtc_consumers(sandbox, capsys):
    root, local, settings = select(sandbox, "audio", "prosody")
    settings["audio"]["speaker_classification"].update(backend="xgb", xgb={"model_path": str(root / "inputs/master.csv")})
    settings["prosody"]["paths"]["rttm_root"] = str(root / launcher.PROJECTS["audio"] / "generated/classification_output_root")
    write_yaml(local, settings)
    assert launcher.main(["all", "--config", str(local), "--dry-run"]) == 1
    assert "XGB cannot feed VTC" in capsys.readouterr().err


def test_stage_requires_individual_workflow(sandbox):
    _, local, _ = sandbox
    with pytest.raises(SystemExit):
        launcher.main(["all", "--config", str(local), "--stage", "extract"])


@pytest.mark.parametrize("problem", ["swallowed-error", "missing-participant-file"])
def test_audio_partial_success_is_not_completion(sandbox, monkeypatch, problem):
    _, local, _ = select(sandbox, "audio")
    item = item_for(sandbox, "audio")

    def execute(*args, **kwargs):
        produce(item)
        if problem == "swallowed-error":
            log = item["root"] / "logs/synthetic.log"
            log.parent.mkdir()
            log.write_text("VTC classification finished | ok=1 skip=0 fail=1", encoding="utf-8")
        else:
            (item["root"] / "generated/classification_output_root/vtc/synthetic/raw_rttm.csv").unlink()
    monkeypatch.setattr(launcher.subprocess, "run", execute)
    assert launcher.main(["audio", "--config", str(local)]) == 1
    assert not records(local)


def test_every_output_file_is_fingerprinted(sandbox, monkeypatch):
    root, local, _ = select(sandbox, "prosody")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["prosody", "--config", str(local)]) == 0
    hidden = root / launcher.PROJECTS["prosody"] / "outputs/__pycache__/unexpected.bin"
    hidden.parent.mkdir()
    hidden.write_bytes(b"unowned")
    assert launcher.main(["prosody", "--config", str(local)]) == 1
    assert len(calls) == 1


def test_optional_cdi_contact_appearance_invalidates(sandbox, monkeypatch):
    root, local, _ = select(sandbox, "cdi")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["cdi", "--config", str(local)]) == 0
    contact = root / launcher.PROJECTS["cdi"] / "data/personal_info/synthetic.csv"
    contact.parent.mkdir(parents=True)
    contact.write_text("synthetic\nnew\n", encoding="utf-8")
    assert launcher.main(["cdi", "--config", str(local)]) == 1
    assert len(calls) == 1


def test_unknown_directory_only_output_is_refused(sandbox):
    root, local, _ = select(sandbox, "prosody")
    (root / launcher.PROJECTS["prosody"] / "cache/incomplete").mkdir(parents=True)
    assert launcher.main(["prosody", "--config", str(local)]) == 1


def test_local_ssl_model_change_invalidates(sandbox, monkeypatch):
    root, local, settings = select(sandbox, "prosody")
    model = root / "local-model/model.safetensors"
    model.parent.mkdir()
    model.write_bytes(b"original weights")
    settings["prosody"]["ssl"] = {"hubert_model": str(model.parent)}
    write_yaml(local, settings)
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["prosody", "--config", str(local)]) == 0
    model.write_bytes(b"changed weights")
    assert launcher.main(["prosody", "--config", str(local)]) == 1
    assert len(calls) == 1


def test_default_and_explicit_same_stage_share_resume(sandbox, monkeypatch):
    _, local, _ = select(sandbox, "prosody")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["prosody", "--config", str(local)]) == 0
    assert launcher.main(["prosody", "--config", str(local), "--stage", "extract"]) == 0
    assert len(calls) == 1


def test_extra_arguments_change_signature_and_refuse(sandbox, monkeypatch):
    _, local, _ = select(sandbox, "prosody")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["prosody", "--config", str(local)]) == 0
    assert launcher.main(["prosody", "--config", str(local), "--", "--participants", "SYNTHETIC"]) == 1
    assert len(calls) == 1


def test_changed_configuration_can_run_to_explicit_fresh_paths(sandbox, monkeypatch):
    _, local, settings = select(sandbox, "prosody")
    calls = mock_success(sandbox, monkeypatch)
    assert launcher.main(["prosody", "--config", str(local)]) == 0
    settings["prosody"]["paths"].update(output_dir="fresh-output", cache_dir="fresh-cache")
    write_yaml(local, settings)
    assert launcher.main(["prosody", "--config", str(local)]) == 0
    assert len(calls) == 2


def test_audio_pipeline_does_not_guard_unrelated_annotation_roots(sandbox):
    _, _, _ = select(sandbox, "audio")
    item = item_for(sandbox, "audio")
    unrelated = item["root"] / "generated/manual_annotation_root"
    unrelated.mkdir(parents=True)
    (unrelated / "original.txt").write_text("synthetic annotation", encoding="utf-8")
    launcher.guard_outputs(item)
    assert unrelated not in item["outputs"]


@pytest.mark.parametrize("stage,extra", [
    ("prepare-audio", ["--wav", "../inputs/synthetic.wav"]),
    ("prepare-audio", ["--recordings-parquet=../inputs/recordings.parquet"]),
    ("classify-vtc", ["--wav=../inputs/synthetic.wav", "--participant-id", "SYNTHETIC"]),
    ("classify-vtc", ["--recordings-parquet", "../inputs/recordings.parquet"]),
    ("generate-textgrids", ["--input-file", "../inputs/rttm.csv", "--participant-id", "SYNTHETIC"]),
    ("annotate-segments", ["--input-file=../inputs/rttm.csv"]),
    ("evaluate-models", ["--participant-id=SYNTHETIC"]),
])
def test_audio_stage_arguments_and_source_identity(sandbox, stage, extra):
    root, _, _ = sandbox
    manifest = root / "inputs/recordings.parquet"
    manifest.write_bytes(b"synthetic manifest; never decoded")
    item = item_for(sandbox, "audio", stage, extra)
    assert item["command"][1:1 + len(extra)] == extra
    if stage in {"generate-textgrids", "annotate-segments"}:
        assert item["command"][-2:] == ["--backend", "vtc"]
    for source in item["sources"]:
        assert source in item["inputs"]
        before = launcher.identity(item)
        source.write_bytes(b"changed source")
        assert launcher.identity(item) != before


@pytest.mark.parametrize("stage,extra", [
    ("pipeline", ["--wav=x.wav"]), ("prepare-audio", []),
    ("prepare-audio", ["--wav=x.wav", "--recordings-parquet=x.parquet"]),
    ("prepare-audio", ["--wav=x.wav", "--run-id=../../escape"]),
    ("classify-vtc", ["--participant-id=SYNTHETIC"]),
    ("classify-vtc", ["--wav=x.wav", "--participant-id=../escape"]),
    ("classify-vtc", ["--wav=...wav"]),
    ("classify-vtc", ["--analysis-dir=elsewhere"]),
    ("generate-textgrids", ["--backend=xgb"]),
    ("generate-textgrids", ["--input-f=x.csv"]),
    ("annotate-segments", ["--participant-id=C:\\escape"]),
    ("evaluate-models", ["--input-csv=x.csv"]),
    ("pipeline", ["--limit"]), ("pipeline", ["--limit=0"]),
    ("pipeline", ["--limit=bad"]), ("pipeline", ["positional"]),
])
def test_audio_rejects_unsupported_or_writable_overrides(sandbox, stage, extra):
    with pytest.raises(ValueError):
        item_for(sandbox, "audio", stage, extra)


@pytest.mark.parametrize("stage,extra", [
    ("validate", ["--participants=SYNTHETIC"]), ("inspect", []),
    ("inspect", ["--participant=SYNTHETIC", "--max-segments=0"]),
    ("extract", ["--participant=SYNTHETIC"]),
])
def test_prosody_arguments_match_selected_native_stage(sandbox, stage, extra):
    with pytest.raises(ValueError):
        item_for(sandbox, "prosody", stage, extra)


def test_audio_single_wav_checks_requested_participant_not_prepared_cohort(sandbox):
    item = item_for(sandbox, "audio", "classify-vtc", ["--wav", "../inputs/synthetic.wav", "--participant-id", "SELECTED"])
    directory = item["root"] / "generated/classification_output_root/vtc/SELECTED"
    for name in ("rttm", "raw_rttm"):
        (directory / name).mkdir(parents=True, exist_ok=True)
        (directory / (name + ".csv")).write_text("uid,label\nx,FEM\n", encoding="utf-8")
    launcher.validate_plan([item])
    assert launcher.completed_outputs(item)
    (directory / "raw_rttm.csv").unlink()
    with pytest.raises(ValueError, match="Incomplete native participant"):
        launcher.completed_outputs(item)


def test_manifest_stages_never_adopt_unverified_transitive_sources(sandbox):
    root, local, _ = sandbox
    (root / "inputs/recordings.parquet").write_bytes(b"synthetic")
    item = item_for(sandbox, "audio", "classify-vtc", ["--recordings-parquet=../inputs/recordings.parquet"])
    path, before, _ = launcher.run_state(item, local, True)
    produce(item)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"ownership": launcher.OWNER, "inputs": before,
                               "outputs": launcher.completed_outputs(item)}), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing existing"):
        launcher.run_state(item, local, True)


def test_evaluation_reads_annotations_and_annotation_reads_audio(sandbox):
    item = item_for(sandbox, "audio", "evaluate-models")
    annotations = item["root"] / "generated/manual_annotation_root"
    annotations.mkdir(parents=True)
    (annotations / "speaker_class_annotations.csv").write_text("label\nFEM\n", encoding="utf-8")
    launcher.validate_plan([item])  # Neither prepared audio nor classification is required.
    assert annotations in item["inputs"]
    assert item["root"] / "generated/classification_output_root" not in item["inputs"]
    annotated = item_for(sandbox, "audio", "annotate-segments", ["--input-file=../inputs/rttm.csv"])
    assert annotated["root"] / "generated/prepared_audio_root" in annotated["inputs"]


@pytest.mark.parametrize("stage", ["build-master", "create-long", "eda", "all"])
def test_literal_native_vocal_outputs(sandbox, stage):
    item = item_for(sandbox, "vocal-stats", stage)
    names = {filename for _, filename in item["expected"]}
    if stage in {"build-master", "all"}:
        assert {"final_master.csv", "participant_lookup.csv", "validation_report.csv",
                "recording_duration_audit.csv", "age_validation.csv", "dataset_build_report.txt"} <= names
    if stage in {"eda", "all"}:
        assert "participant_summary.csv" in names
    if stage == "create-long":
        master = item["root"] / "data/derived/final_master.csv"
        master.parent.mkdir(parents=True)
        master.write_text("participant_id\nSYNTHETIC\n", encoding="utf-8")
        launcher.validate_plan([item])
        launcher.guard_outputs(item)
        assert master in item["inputs"] and master.parent not in item["outputs"]
        report = item["root"] / "results/dataset_build_report.txt"
        report.parent.mkdir()
        report.write_text("Existing report must not be overwritten", encoding="utf-8")
        with pytest.raises(ValueError, match="refusing existing"):
            launcher.guard_outputs(item)


def test_literal_native_report_and_prosody_names(sandbox):
    vocal = item_for(sandbox, "vocal-models", "report")
    assert vocal["expected"] == [(vocal["root"] / "results/final_report", "report.html")]
    prosody = item_for(sandbox, "prosody")
    assert {filename for _, filename in prosody["expected"]} == {
        "participant_speaker_acoustic_features.csv", "run_metadata.json", "processing_summary.csv"}
    stats = item_for(sandbox, "prosody-stats", "report")
    assert stats["expected"] == [(stats["root"] / "outputs/reports", "HindiBabyNet_prosody_stat_model_report.html")]


def test_setup_does_not_read_native_configs_or_require_inputs_or_r(sandbox, monkeypatch):
    _, local, settings = sandbox
    write_yaml(local, {"workflows": settings["workflows"]})
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **_kw: calls.append(command))
    monkeypatch.setattr(launcher.shutil, "which", lambda value: str(value) if value == "uv" else None)
    assert launcher.main(["setup", "--config", str(local)]) == 0
    assert calls == [["uv", "sync", "--frozen"]] * 4
    assert launcher.main(["check", "--config", str(local)]) == 1


@pytest.mark.parametrize("filename", ["METADATA", "RECORD", "pyvenv.cfg", "python.exe"])
def test_basic_environment_identity_without_recursive_scan(sandbox, monkeypatch, filename):
    item = item_for(sandbox, "prosody")
    env = item["project"] / ".venv"
    path = (env / "Lib/site-packages/synthetic-1.dist-info" / filename if filename in {"METADATA", "RECORD"}
            else env / "pyvenv.cfg" if filename == "pyvenv.cfg"
            else env / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old", encoding="utf-8")
    original = Path.rglob

    def bounded(path, pattern):
        assert ".venv" not in path.parts, "Must not recursively scan the environment"
        return original(path, pattern)
    monkeypatch.setattr(Path, "rglob", bounded)
    before = launcher.identity(item)
    path.write_text("changed", encoding="utf-8")
    assert launcher.identity(item) != before


@pytest.mark.parametrize("stage,extra", [
    ("pipeline", ["--limit=1"]),
    ("prepare-audio", ["--wav=../inputs/synthetic.wav"]),
    ("prepare-audio", ["--recordings-parquet=manifest.parquet"]),
    ("classify-vtc", ["--wav=../inputs/synthetic.wav", "--participant-id=SYNTHETIC"]),
    ("classify-xgb", ["--recordings-parquet=manifest.parquet"]),
    ("generate-textgrids", ["--input-file=../inputs/rttm.csv", "--limit=1"]),
    ("annotate-segments", ["--participant-id=SYNTHETIC"]),
    ("evaluate-models", ["--participant-id=SYNTHETIC"]),
])
def test_forwarded_audio_arguments_parse_in_actual_native_cli(sandbox, monkeypatch, stage, extra):
    # Execute only each real parser's main AST, replacing its workflow callable.
    # No native imports, scientific computation, models or filesystem writes.
    _, local, settings = sandbox
    if stage == "classify-xgb":
        settings["audio"]["speaker_classification"]["backend"] = "xgb"
        write_yaml(local, settings)
    item = item_for(sandbox, "audio", stage, extra)
    source = launcher.ROOT / "HindiBabyNet-Pipeline/src/hindibabynet_pipeline/cli" / (stage.replace("-", "_") + ".py")
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return {}  # Both iterable paths and evaluation's items() are empty.
    names = {node.func.id for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    namespace = {name: execute for name in names - {"print"}}
    namespace["argparse"] = argparse
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.setattr(launcher.sys, "argv", item["command"])
    namespace["main"]()
    assert len(calls) == 1
    if stage in {"generate-textgrids", "annotate-segments"}:
        assert calls[0]["backend"] == "vtc"


def test_source_paths_allow_spaces_and_identifiers_cannot_redirect_outputs(sandbox):
    root, _, _ = sandbox
    source = root / "inputs/raw recording.wav"
    source.write_bytes(b"synthetic")
    item = item_for(sandbox, "audio", "classify-vtc", ["--wav", str(source), "--participant-id", "Safe ID"])
    launcher.validate_plan([item])
    assert source in item["inputs"]
    bad_source = item["root"] / "generated/classification_output_root/source.wav"
    bad_source.parent.mkdir(parents=True)
    bad_source.write_bytes(b"must not be overwritten")
    overlap = item_for(sandbox, "audio", "classify-vtc", ["--wav", str(bad_source)])
    with pytest.raises(ValueError, match="input/output overlap"):
        launcher.validate_plan([overlap])


def test_cdi_inventory_preserves_every_native_output_root(sandbox):
    item = item_for(sandbox, "cdi")
    expected = {"data/processed", "outputs/tables/qc", "outputs/tables/cdi1", "outputs/tables/cdi2",
                "outputs/tables/combined", "outputs/figures/cdi1/comprehension", "outputs/figures/cdi1/production",
                "outputs/figures/cdi1/items", "outputs/figures/cdi1/categories", "outputs/figures/cdi2/production",
                "outputs/figures/cdi2/items", "outputs/figures/cdi2/categories", "outputs/figures/combined",
                "outputs/figures/qc"}
    assert {p.relative_to(item["root"]).as_posix() for p in item["outputs"]} == expected