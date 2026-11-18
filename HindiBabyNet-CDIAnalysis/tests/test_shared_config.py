from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hindibabynet_cdi import config as config_module
from hindibabynet_cdi.config import load_config


@pytest.fixture(autouse=True)
def clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HBN_CONFIG_PATH", raising=False)


def write_config(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_section_matches_native_config(tmp_path: Path) -> None:
    native = {
        "paths": {"raw_data": "../source/raw", "outputs": "results"},
        "forms": {"consent_form_id": "synthetic-consent"},
        "analysis": {
            "age_month_divisor": 30.44,
            "strict_hindi_threshold": 80.0,
            "younger_age_bins": ["8-10", "11-14"],
        },
        "eligibility": {"allow_preterm": True},
        "age": {"cdi1_min_month": 9},
        "education": {"analysis_groups": ["group-a", "group-b"]},
        "completeness": {"minimum_present_ratio": 0.9},
        "outputs": {"qc_dir": "results/qc"},
        "form_detection_rules": {"consent": ["synthetic consent column"]},
    }
    native_path = write_config(tmp_path / "cdi-project/configs/native.yaml", native)
    shared_path = write_config(
        tmp_path / "shared.yaml",
        {"audio": {}, "vocal": {}, "prosody": {}, "cdi": {**native, "project_root": "cdi-project"}},
    )

    assert load_config(shared_path) == load_config(native_path)
    assert load_config(shared_path).paths.raw_data == (tmp_path / "source/raw").resolve()


def test_section_root_is_relative_to_config_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_config(
        tmp_path / "settings/shared.yaml",
        {"cdi": {"project_root": "../project", "paths": {"raw_data": "../source"}}},
    )
    monkeypatch.chdir(tmp_path)

    config = load_config(source.relative_to(tmp_path))

    assert config.paths.repo_root == tmp_path / "project"
    assert config.paths.raw_data == tmp_path / "source"
    assert config.outputs.qc_dir == tmp_path / "project/outputs/tables/qc"


def test_absolute_section_root_and_paths_are_preserved(tmp_path: Path) -> None:
    root = tmp_path / "project"
    raw = tmp_path / "external/raw"
    path = write_config(
        tmp_path / "settings/shared.yaml",
        {"cdi": {"project_root": str(root), "paths": {"raw_data": str(raw)}}},
    )
    assert load_config(path).paths.repo_root == root
    assert load_config(path).paths.raw_data == raw


def test_flat_grandparent_and_section_package_root_defaults(tmp_path: Path) -> None:
    flat = write_config(tmp_path / "elsewhere/configs/native.yaml", {})
    shared = write_config(tmp_path / "shared.yaml", {"cdi": {}})

    assert load_config(flat).paths.repo_root == flat.parent.parent
    section_config = load_config(shared)
    assert section_config.paths.repo_root == Path(config_module.__file__).resolve().parents[2]
    assert section_config.analysis.age_month_divisor == 30.4375
    assert section_config.analysis.strict_hindi_threshold == 75.0
    assert section_config.completeness.minimum_present_ratio == 0.8


@pytest.mark.parametrize("sectioned", [False, True])
def test_explicit_config_beats_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sectioned: bool
) -> None:
    env = write_config(tmp_path / "env.yaml", {"cdi": {"analysis": {"strict_hindi_threshold": 91}}})
    native = {"analysis": {"strict_hindi_threshold": 82}}
    explicit = write_config(tmp_path / "explicit.yaml", {"cdi": native} if sectioned else native)
    monkeypatch.setenv("HBN_CONFIG_PATH", str(env))

    assert load_config().analysis.strict_hindi_threshold == 91
    assert load_config(explicit).analysis.strict_hindi_threshold == 82


def test_native_default_when_environment_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(tmp_path / "configs/config.yaml", {"analysis": {"strict_hindi_threshold": 83}})
    monkeypatch.chdir(tmp_path)
    assert load_config().analysis.strict_hindi_threshold == 83


def test_invalid_named_section_is_not_treated_as_flat(tmp_path: Path) -> None:
    path = write_config(tmp_path / "shared.yaml", {"cdi": []})
    with pytest.raises(TypeError, match="section 'cdi'"):
        load_config(path)