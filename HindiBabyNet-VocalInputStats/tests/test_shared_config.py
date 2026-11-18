from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from hindibabynet_vocalinputstats import config as config_module
from hindibabynet_vocalinputstats.config import get_repo_root, load_config


@pytest.fixture(autouse=True)
def clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HBN_CONFIG_PATH", raising=False)


@pytest.fixture
def native() -> dict:
    return {
        "metadata_csv": "../source/metadata.csv",
        "vtc_output_root": "../source/vtc",
        "audio_root": "../source/audio",
        "derived_data_dir": "data/derived",
        "private_data_dir": "data/private",
        "figures_dir": "figures",
        "tables_dir": "tables",
        "results_dir": "results",
        "participant_id_digits": 3,
        "age_month_denominator": 30.44,
        "ses_source": "mother_education",
        "minimum_recording_hours_warning": 1.0,
        "manual_mapping_csv": "metadata/manual.csv",
        "audio_extensions": [".wav", ".WAV"],
        "audio_layout": {"recursive": False, "expected_audio_files": "auto"},
        "vtc_layout": {"rttm_csv_name": "rttm.csv"},
        "recording_duration_source": "VTC",
        "session_selection": "EARLIEST",
    }


def write_config(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_section_matches_flat_and_keeps_code_root(tmp_path: Path, native: dict) -> None:
    root = get_repo_root()
    flat_path = write_config(tmp_path / "flat.yaml", native)
    shared_path = write_config(
        tmp_path / "shared.yaml",
        {"vocal": {**native, "project_root": str(root)}, "cdi": {}, "prosody": {}, "audio": {}},
    )

    flat = load_config(flat_path)
    shared = load_config(shared_path)

    assert shared == replace(flat, config_path=shared_path)
    assert get_repo_root() == root
    assert shared.recording_duration_source == "vtc"
    assert shared.session_selection == "earliest"
    assert shared.audio_extensions == (".wav", ".WAV")


def test_section_root_resolves_from_relative_config_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict
) -> None:
    code_root = tmp_path / "code"
    path = write_config(
        code_root / "settings/shared.yaml",
        {"vocal": {**native, "project_root": "../../research"}},
    )
    monkeypatch.setattr(config_module, "get_repo_root", lambda: code_root)
    monkeypatch.chdir(tmp_path)

    config = load_config("settings/shared.yaml")

    root = tmp_path / "research"
    assert config.config_path == path
    assert config.repo_root == root
    assert config.metadata_path == root / "../source/metadata.csv"
    assert config.audio_root == root / "../source/audio"
    assert config.manual_mapping_csv == root / "metadata/manual.csv"
    assert config_module.get_repo_root() == code_root


def test_section_without_root_matches_flat(tmp_path: Path, native: dict) -> None:
    flat = load_config(write_config(tmp_path / "flat.yaml", native))
    shared_path = write_config(tmp_path / "shared.yaml", {"vocal": native})
    assert load_config(shared_path) == replace(flat, config_path=shared_path)


@pytest.mark.parametrize("sectioned", [False, True])
def test_explicit_config_beats_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict, sectioned: bool
) -> None:
    env = write_config(tmp_path / "env.yaml", {"vocal": {**native, "age_month_denominator": 31}})
    explicit = write_config(tmp_path / "explicit.yaml", {"vocal": native} if sectioned else native)
    monkeypatch.setenv("HBN_CONFIG_PATH", str(env))

    assert load_config().age_month_denominator == 31
    assert load_config(explicit).age_month_denominator == 30.44


def test_native_default_and_metadata_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict
) -> None:
    native["metadata_path"] = str(tmp_path / "preferred.csv")
    native.pop("manual_mapping_csv")
    write_config(tmp_path / "configs/config.yaml", native)
    monkeypatch.setattr(config_module, "get_repo_root", lambda: tmp_path)

    config = load_config()

    assert config.metadata_path == tmp_path / "preferred.csv"
    assert config.manual_mapping_csv is None
    assert config.session_date_format == "%Y%m%d"


def test_invalid_named_section_is_not_treated_as_flat(tmp_path: Path) -> None:
    path = write_config(tmp_path / "shared.yaml", {"vocal": []})
    with pytest.raises(ValueError, match="section 'vocal'"):
        load_config(path)