"""Tests for HindiBabyNet GUI — unit tests for services, utils, and imports."""

import json
import tempfile
from pathlib import Path

import pytest


# ── 1. Smoke test: imports ──────────────────────────────────────────────


def test_gui_package_imports():
    """All GUI modules should be importable without errors."""
    import hindibabynet_gui
    assert hasattr(hindibabynet_gui, "__version__")


def test_services_import():
    from hindibabynet_gui.services import ConfigService
    assert ConfigService is not None


def test_utils_import():
    from hindibabynet_gui.utils import PROJECT_ROOT, CONFIG_PATH
    from hindibabynet_gui.utils.yaml_utils import load_yaml, save_yaml
    from hindibabynet_gui.utils.command_builder import build_stage_01


def test_diagnostics_import():
    from hindibabynet_gui.services.diagnostics_service import run_all_checks


def test_history_import():
    from hindibabynet_gui.services.history_service import get_history


def test_outputs_import():
    from hindibabynet_gui.services.outputs_service import scan_outputs


def test_annotation_import():
    from hindibabynet_gui.services.annotation_service import list_annotatable_participants


# ── 2. YAML utils ──────────────────────────────────────────────────────


def test_yaml_roundtrip():
    from hindibabynet_gui.utils.yaml_utils import load_yaml, save_yaml
    data = {
        "artifacts_root": "artifacts/runs",
        "data_ingestion": {
            "raw_audio_root": "/tmp/test",
            "allowed_ext": [".wav", ".WAV"],
        },
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        path = Path(f.name)
    save_yaml(data, path)
    loaded = load_yaml(path)
    assert loaded["artifacts_root"] == "artifacts/runs"
    assert loaded["data_ingestion"]["raw_audio_root"] == "/tmp/test"
    assert loaded["data_ingestion"]["allowed_ext"] == [".wav", ".WAV"]
    path.unlink()


def test_nested_get_set():
    from hindibabynet_gui.utils.yaml_utils import nested_get, nested_set
    d = {"a": {"b": {"c": 42}}}
    assert nested_get(d, "a.b.c") == 42
    assert nested_get(d, "a.b.x", "default") == "default"
    nested_set(d, "a.b.c", 99)
    assert d["a"]["b"]["c"] == 99
    nested_set(d, "new.key.deep", "hello")
    assert d["new"]["key"]["deep"] == "hello"


# ── 3. Config service ──────────────────────────────────────────────────


def test_config_service_load_save():
    from hindibabynet_gui.services import ConfigService
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        f.write("artifacts_root: test\ndata_ingestion:\n  raw_audio_root: /tmp\n")
        path = Path(f.name)
    svc = ConfigService(config_path=path)
    assert svc.get("artifacts_root") == "test"
    assert svc.get("data_ingestion.raw_audio_root") == "/tmp"
    svc.set("artifacts_root", "changed")
    svc.save()
    svc2 = ConfigService(config_path=path)
    assert svc2.get("artifacts_root") == "changed"
    path.unlink()


def test_config_service_restore_backup():
    from hindibabynet_gui.services import ConfigService
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        f.write("key: original\n")
        path = Path(f.name)
    svc = ConfigService(config_path=path)
    svc.set("key", "modified")
    assert svc.get("key") == "modified"
    svc.restore_backup()
    assert svc.get("key") == "original"
    path.unlink()


def test_config_field_definitions():
    from hindibabynet_gui.services import ConfigService
    defs = ConfigService.field_definitions()
    assert len(defs) > 10
    keys = {d.key for d in defs}
    assert "speaker_classification.backend" in keys
    assert "data_ingestion.raw_audio_root" in keys


# ── 4. Command builder ──────────────────────────────────────────────────


def test_build_stage_01():
    from hindibabynet_gui.utils.command_builder import build_stage_01
    cmd = build_stage_01()
    assert cmd == ["python", "-m", "src.hindibabynet.pipeline.stage_01_data_ingestion"]


def test_build_stage_01_with_run_id():
    from hindibabynet_gui.utils.command_builder import build_stage_01
    cmd = build_stage_01(run_id="20260101_120000")
    assert "--run_id" in cmd
    assert "20260101_120000" in cmd


def test_build_stage_02_from_parquet():
    from hindibabynet_gui.utils.command_builder import build_stage_02_from_parquet
    cmd = build_stage_02_from_parquet("/path/to/rec.parquet", limit=5)
    assert "--recordings_parquet" in cmd
    assert "/path/to/rec.parquet" in cmd
    assert "--limit" in cmd
    assert "5" in cmd


def test_build_stage_02_single_wav():
    from hindibabynet_gui.utils.command_builder import build_stage_02_single_wav
    cmd = build_stage_02_single_wav("/path/to/audio.wav", recording_id="rec01")
    assert "--wav" in cmd
    assert "--recording_id" in cmd


def test_build_stage_03_modes():
    from hindibabynet_gui.utils.command_builder import build_stage_03
    # WAV mode
    cmd = build_stage_03(wav="/path/audio.wav", backend="vtc")
    assert "--wav" in cmd
    assert "--backend" in cmd
    assert "vtc" in cmd

    # Dir mode
    cmd = build_stage_03(analysis_dir="/path/dir")
    assert "--analysis_dir" in cmd

    # Parquet mode
    cmd = build_stage_03(recordings_parquet="/path/rec.parquet", limit=3)
    assert "--recordings_parquet" in cmd
    assert "--limit" in cmd


def test_build_full_pipeline():
    from hindibabynet_gui.utils.command_builder import build_full_pipeline
    cmd = build_full_pipeline(run_id="test123", limit=2)
    assert cmd[0] == "bash"
    assert "scripts/run_all.sh" in cmd[1]
    assert "--limit" in cmd
    assert "2" in cmd


def test_build_annotate():
    from hindibabynet_gui.utils.command_builder import build_annotate
    cmd = build_annotate(participant="ABAN141223", speaker="female", resume=True)
    assert "--participant" in cmd
    assert "--speaker" in cmd
    assert "--resume" in cmd

    cmd_status = build_annotate(show_status=True)
    assert "--status" in cmd_status


# ── 5. Diagnostics service ──────────────────────────────────────────────


def test_diagnostics_run():
    from hindibabynet_gui.services.diagnostics_service import run_all_checks
    results = run_all_checks()
    assert len(results) > 5
    names = {r.name for r in results}
    assert "Python Version" in names


def test_diagnostics_result_fields():
    from hindibabynet_gui.services.diagnostics_service import CheckResult
    r = CheckResult(name="Test", status="ok", message="Good")
    assert r.status == "ok"
    assert r.detail == ""


# ── 6. History service ──────────────────────────────────────────────────


def test_history_roundtrip(tmp_path, monkeypatch):
    from hindibabynet_gui.services import history_service
    # Point to temp dir
    monkeypatch.setattr(history_service, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(history_service, "GUI_DATA_DIR", tmp_path)

    rec = history_service.add_record(
        command="python -m test",
        mode="stage01",
        backend="xgb",
    )
    assert rec["status"] == "running"

    history_service.update_last("success", exit_code=0)
    records = history_service.get_history()
    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert records[0]["exit_code"] == 0


# ── 7. Path utils ──────────────────────────────────────────────────────


def test_project_root_exists():
    from hindibabynet_gui.utils import PROJECT_ROOT
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_resolve_path():
    from hindibabynet_gui.utils import resolve_path
    # Absolute stays absolute
    p = resolve_path("/tmp/test")
    assert str(p) == "/tmp/test"
    # Relative resolves against PROJECT_ROOT
    p = resolve_path("configs/config.yaml")
    assert p.is_absolute()
