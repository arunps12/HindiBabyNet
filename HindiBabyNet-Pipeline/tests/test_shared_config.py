from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from hindibabynet_pipeline.config import configuration as config_module
from hindibabynet_pipeline.config.configuration import ConfigurationManager


@pytest.fixture(autouse=True)
def clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HBN_CONFIG_PATH", raising=False)


@pytest.fixture
def native() -> dict:
    return {
        "paths": {
            "raw_audio_root": "../source/audio",
            "raw_joined_audio_root": "data/joined",
            "prepared_audio_root": "data/prepared",
            "classification_output_root": "classification",
            "textgrid_output_root": "textgrids",
            "manual_annotation_root": "annotation",
            "evaluation_output_root": "evaluation",
            "custom_filesystem_field": "metadata/lookup.csv",
        },
        "artifacts_root": "artifacts/runs",
        "logs_root": "logs",
        "data_ingestion": {"session_selection": "all", "allowed_ext": [".wav", ".WAV"]},
        "audio_preparation": {"save_raw_joined_audio": False},
        "speaker_classification": {
            "backend": "vtc",
            "output_root": "results/classification",
            "xgb": {
                "model_path": "models/runtime.pkl",
                "diarization_model": "synthetic/diarization",
                "vad_aggressiveness": 1,
                "chunk_sec": 321.0,
                "classify_win_sec": 4.0,
                "egemaps_dim": 88,
            },
            "vtc": {"repo_path": "external/VTC", "batch_size": 128, "device": "cpu"},
        },
    }


@pytest.fixture
def params() -> dict:
    return {
        "audio_preparation": {"target_sr": 16000, "target_peak_dbfs": -2.0},
        "xgb": {
            "model_path": "models/params.pkl",
            "vad": {"aggressiveness": 3},
            "diarization": {"chunk_sec": 900.0, "model": "synthetic/parameter-model"},
            "classification": {"win_sec": 1.5},
            "egemaps_dim": 77,
        },
        "vtc": {"batch_size": 32, "device": "cuda", "repo_path": "external/params-VTC"},
        "annotation": {"seed": 11},
        "textgrid": {"labels": ["FEM", "MAL", "CHI"]},
    }


def write_config(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def absolute_runtime(native: dict, root: Path) -> dict:
    payload = deepcopy(native)
    payload["paths"] = {key: str((root / value).resolve()) for key, value in payload["paths"].items()}
    for key in ("logs_root", "artifacts_root"):
        payload[key] = str(root / payload[key])
    sc = payload["speaker_classification"]
    sc["output_root"] = str(root / sc["output_root"])
    sc["xgb"]["model_path"] = str(root / sc["xgb"]["model_path"])
    sc["vtc"]["repo_path"] = str(root / sc["vtc"]["repo_path"])
    return payload


def absolute_params(params: dict, root: Path) -> dict:
    payload = deepcopy(params)
    payload["xgb"]["model_path"] = str(root / payload["xgb"]["model_path"])
    payload["vtc"]["repo_path"] = str(root / payload["vtc"]["repo_path"])
    return payload


def test_shared_sections_match_native_effective_values(tmp_path: Path, native: dict, params: dict) -> None:
    root = tmp_path / "audio-project"
    flat = write_config(tmp_path / "native/config.yaml", absolute_runtime(native, root))
    write_config(flat.with_name("params.yaml"), absolute_params(params, root))
    shared = write_config(
        tmp_path / "shared.yaml",
        {
            "audio": {**native, "project_root": "audio-project"},
            "audio_params": params,
            "cdi": {}, "vocal": {}, "prosody": {},
        },
    )

    old = ConfigurationManager(flat)
    new = ConfigurationManager(shared)

    assert new.runtime_config == old.runtime_config
    assert new.params == old.params
    assert new.config == old.config
    assert new.get_xgb_params() == old.get_xgb_params()
    assert new.get_vtc_params() == old.get_vtc_params()
    assert new.get_data_ingestion_config("run") == old.get_data_ingestion_config("run")
    assert new.get_audio_preparation_config("run", "synthetic") == old.get_audio_preparation_config("run", "synthetic")
    assert new.get_speaker_classification_config("run", "synthetic") == old.get_speaker_classification_config("run", "synthetic")
    assert new.get_vtc_config() == old.get_vtc_config()
    assert new.get_annotation_params() == params["annotation"]
    assert new.get_textgrid_params() == params["textgrid"]


def test_original_params_precedence(tmp_path: Path, native: dict, params: dict) -> None:
    shared = write_config(tmp_path / "shared.yaml", {"audio": native, "audio_params": params})
    cfg = ConfigurationManager(shared)

    assert cfg.get_vtc_params()["batch_size"] == 128
    assert cfg.get_vtc_params()["device"] == "cpu"
    xgb = cfg.get_xgb_params()
    assert xgb["vad_aggressiveness"] == 3
    assert xgb["chunk_sec"] == 900.0
    assert xgb["classify_win_sec"] == 1.5
    assert xgb["egemaps_dim"] == 88
    assert xgb["diarization_model"] == "synthetic/parameter-model"
    assert cfg.config["xgb"]["egemaps_dim"] == 77
    assert cfg.get_audio_preparation_params()["target_peak_dbfs"] == -2.0


def test_relative_source_and_all_native_path_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict
) -> None:
    absolute = tmp_path / "external/prepared"
    native["paths"]["prepared_audio_root"] = str(absolute)
    write_config(tmp_path / "settings/shared.yaml", {"audio": {**native, "project_root": "../project"}})
    monkeypatch.chdir(tmp_path)

    cfg = ConfigurationManager(Path("settings/shared.yaml"))

    assert cfg.get_processed_audio_root() == absolute
    assert cfg.get_data_ingestion_config("run").raw_audio_root == tmp_path / "source/audio"
    assert cfg.runtime_config["paths"]["custom_filesystem_field"] == str(tmp_path / "project/metadata/lookup.csv")
    assert cfg.get_logs_root() == tmp_path / "project/logs"
    assert cfg.get_artifacts_root() == tmp_path / "project/artifacts/runs"
    assert cfg.get_classification_output_root() == tmp_path / "project/results/classification"
    assert cfg.get_xgb_params()["model_path"] == str(tmp_path / "project/models/runtime.pkl")
    assert cfg.get_vtc_config().repo_path == tmp_path / "project/external/VTC"
    assert cfg.runtime_config["speaker_classification"]["xgb"]["diarization_model"] == "synthetic/diarization"
    assert cfg.runtime_config["data_ingestion"]["allowed_ext"] == [".wav", ".WAV"]


def test_native_flat_remains_cwd_relative_and_uses_sibling_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict, params: dict
) -> None:
    flat = write_config(tmp_path / "settings/config.yaml", native)
    sidecar = write_config(flat.with_name("params.yaml"), params)
    write_config(tmp_path / "configs/params.yaml", {"vtc": {"batch_size": 999}})
    monkeypatch.chdir(tmp_path)

    cfg = ConfigurationManager(flat)

    assert cfg.params_path == sidecar
    assert cfg.runtime_config == native
    assert cfg.params == params
    assert cfg.get_logs_root() == Path("logs")
    assert cfg.get_processed_audio_root() == Path("data/prepared")
    assert cfg.get_vtc_config().repo_path == Path("external/VTC")


def test_section_defaults_use_package_root_and_do_not_read_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(tmp_path / "params.yaml", {"audio_preparation": {"target_sr": 8000}})
    write_config(tmp_path / "configs/params.yaml", {"vtc": {"batch_size": 999}})
    shared = write_config(
        tmp_path / "shared.yaml",
        {"audio": {"paths": {"classification_output_root": "outputs"}}},
    )
    monkeypatch.chdir(tmp_path)

    cfg = ConfigurationManager(shared)
    root = Path(config_module.__file__).resolve().parents[3]

    assert cfg.params == {}
    assert cfg.get_logs_root() == root / "logs"
    assert cfg.get_artifacts_root() == root / "artifacts/runs"
    assert cfg.get_speaker_classification_config("run", "synthetic").model_path == root / "models/xgb_egemaps.pkl"
    assert cfg.get_vtc_config().repo_path == root / "external_models/VTC"
    assert cfg.get_speaker_classification_backend() == "xgb"


def test_explicit_params_source_and_same_shared_source(tmp_path: Path, native: dict, params: dict) -> None:
    shared = write_config(tmp_path / "shared.yaml", {"audio": native, "audio_params": params})
    sidecar = write_config(tmp_path / "custom.yaml", {"vtc": {"batch_size": 16}})

    assert ConfigurationManager(shared, sidecar).params == {"vtc": {"batch_size": 16}}
    assert ConfigurationManager(shared, shared).params["vtc"]["batch_size"] == 32


def test_params_section_root_resolves_only_filesystem_fields(tmp_path: Path, native: dict, params: dict) -> None:
    native["speaker_classification"]["xgb"].pop("model_path")
    native["speaker_classification"]["vtc"].pop("repo_path")
    shared = write_config(
        tmp_path / "settings/shared.yaml",
        {"audio": {**native, "project_root": "../audio"}, "audio_params": {**params, "project_root": "../models"}},
    )

    cfg = ConfigurationManager(shared)

    assert cfg.get_xgb_params()["model_path"] == str(tmp_path / "models/models/params.pkl")
    assert cfg.get_vtc_params()["repo_path"] == str(tmp_path / "models/external/params-VTC")
    assert cfg.get_xgb_params()["diarization_model"] == "synthetic/parameter-model"
    assert "project_root" not in cfg.params


@pytest.mark.parametrize("sectioned", [False, True])
def test_explicit_config_beats_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict, sectioned: bool
) -> None:
    env = write_config(tmp_path / "env.yaml", {"audio": {"speaker_classification": {"backend": "xgb"}}})
    explicit = write_config(tmp_path / "explicit/config.yaml", {"audio": native} if sectioned else native)
    write_config(explicit.with_name("params.yaml"), {})
    monkeypatch.setenv("HBN_CONFIG_PATH", str(env))

    assert ConfigurationManager().get_speaker_classification_backend() == "xgb"
    assert ConfigurationManager(explicit).get_speaker_classification_backend() == "vtc"


def test_native_default_when_environment_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict, params: dict
) -> None:
    write_config(tmp_path / "configs/config.yaml", native)
    write_config(tmp_path / "configs/params.yaml", params)
    monkeypatch.chdir(tmp_path)
    cfg = ConfigurationManager()
    assert cfg.runtime_config == native
    assert cfg.params == params


def test_explicit_native_default_path_beats_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict
) -> None:
    write_config(tmp_path / "configs/config.yaml", native)
    write_config(tmp_path / "configs/params.yaml", {})
    env = write_config(tmp_path / "shared.yaml", {"audio": {"speaker_classification": {"backend": "xgb"}}})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HBN_CONFIG_PATH", str(env))

    cfg = ConfigurationManager(Path("configs/config.yaml"))

    assert cfg.get_speaker_classification_backend() == "vtc"
    assert cfg.runtime_config == native


def test_flat_environment_source_retains_cwd_params_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict, params: dict
) -> None:
    flat = write_config(tmp_path / "elsewhere/config.yaml", native)
    write_config(tmp_path / "configs/params.yaml", params)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HBN_CONFIG_PATH", str(flat))

    cfg = ConfigurationManager()

    assert cfg.params_path == Path("configs/params.yaml")
    assert cfg.params == params
    assert cfg.get_xgb_params()["model_path"] == "models/runtime.pkl"


def test_separate_params_file_can_be_the_shared_root(tmp_path: Path, native: dict, params: dict) -> None:
    shared = write_config(
        tmp_path / "shared.yaml",
        {"audio": native, "audio_params": {**params, "project_root": "parameter-project"}},
    )
    flat = write_config(tmp_path / "native.yaml", native)

    cfg = ConfigurationManager(flat, shared)

    assert cfg.runtime_config == native
    assert cfg.params["xgb"]["model_path"] == str(tmp_path / "parameter-project/models/params.pkl")
    assert cfg.get_vtc_params()["batch_size"] == 128


@pytest.mark.parametrize("section", ["audio", "audio_params"])
def test_invalid_named_sections_are_not_treated_as_flat(tmp_path: Path, section: str) -> None:
    shared = write_config(tmp_path / "shared.yaml", {"audio": {}, section: []})
    with pytest.raises(TypeError, match=f"section '{section}'"):
        ConfigurationManager(shared)