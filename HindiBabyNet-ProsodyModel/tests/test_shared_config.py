from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hindibabynet_prosody import config as config_module
from hindibabynet_prosody.config import load_config


@pytest.fixture(autouse=True)
def clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HBN_CONFIG_PATH", raising=False)


@pytest.fixture
def native() -> dict:
    return {
        "project": {"name": "synthetic", "random_seed": 17},
        "paths": {
            "rttm_root": "../source/rttm",
            "audio_root": "../source/audio",
            "participant_lookup": "metadata/lookup.csv",
            "output_dir": "outputs",
            "cache_dir": "cache",
        },
        "rttm": {"file_pattern": "**/rttm.csv"},
        "mapping": {},
        "speakers": {"include": {"FEM": "adult_female"}},
        "audio": {"extensions": [".wav"], "preferred_extensions": [".wav"]},
        "pitch": {"speaker_settings": {"FEM": {"pitch_floor_hz": 75, "pitch_ceiling_hz": 700}}},
        "ssl": {"wav2vec2_model": "synthetic/wav2vec2", "hubert_model": "synthetic/hubert"},
        "pca": {"n_components": 2},
        "runtime": {"batch_size": 3},
        "output": {"final_filename": "synthetic.csv"},
    }


def write_config(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("reference_hz", [None, 100.0])
def test_section_matches_flat_effective_values(
    tmp_path: Path, native: dict, reference_hz: float | None
) -> None:
    root = tmp_path / "project"
    if reference_hz is not None:
        native["pitch"]["semitone_reference_hz"] = reference_hz
    flat_native = {**native, "paths": {key: str((root / value).resolve()) for key, value in native["paths"].items()}}
    flat = write_config(tmp_path / "flat.yaml", flat_native)
    shared = write_config(
        tmp_path / "settings/shared.yaml",
        {"audio": {}, "cdi": {}, "vocal": {}, "prosody": {**native, "project_root": "../project"}},
    )

    config = load_config(shared)

    assert config == load_config(flat)
    assert config.pitch.semitone_reference_hz == (1.0 if reference_hz is None else reference_hz)
    assert config.paths.audio_root == tmp_path / "source/audio"
    assert config.ssl.wav2vec2_model == "synthetic/wav2vec2"
    assert config.rttm.file_pattern == "**/rttm.csv"


def test_flat_paths_remain_relative_despite_native_audio_key(tmp_path: Path, native: dict) -> None:
    config = load_config(write_config(tmp_path / "flat.yaml", native))
    assert config.paths.audio_root == Path("../source/audio")
    assert config.paths.output_dir == Path("outputs")


def test_section_without_root_uses_package_root(tmp_path: Path, native: dict) -> None:
    shared = write_config(tmp_path / "shared.yaml", {"prosody": native})
    root = Path(config_module.__file__).resolve().parents[2]
    assert load_config(shared).paths.output_dir == root / "outputs"


def test_absolute_section_root_and_path(tmp_path: Path, native: dict) -> None:
    native["paths"]["audio_root"] = str(tmp_path / "external/audio")
    path = write_config(
        tmp_path / "settings/shared.yaml",
        {"prosody": {**native, "project_root": str(tmp_path / "project")}},
    )
    assert load_config(path).paths.audio_root == tmp_path / "external/audio"
    assert load_config(path).paths.cache_dir == tmp_path / "project/cache"


@pytest.mark.parametrize("sectioned", [False, True])
def test_explicit_config_beats_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict, sectioned: bool
) -> None:
    env = write_config(tmp_path / "env.yaml", {"prosody": {**native, "runtime": {"batch_size": 9}}})
    explicit = write_config(tmp_path / "explicit.yaml", {"prosody": native} if sectioned else native)
    monkeypatch.setenv("HBN_CONFIG_PATH", str(env))

    assert load_config().runtime.batch_size == 9
    assert load_config(explicit).runtime.batch_size == 3


def test_relative_config_path_uses_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: dict
) -> None:
    write_config(tmp_path / "settings/shared.yaml", {"prosody": {**native, "project_root": "../project"}})
    monkeypatch.chdir(tmp_path)
    assert load_config("settings/shared.yaml").paths.output_dir == tmp_path / "project/outputs"


def test_config_source_remains_required_without_environment() -> None:
    with pytest.raises(ValueError, match="config path or HBN_CONFIG_PATH"):
        load_config()


def test_invalid_named_section_is_not_treated_as_flat(tmp_path: Path) -> None:
    path = write_config(tmp_path / "shared.yaml", {"prosody": []})
    with pytest.raises(TypeError, match="section 'prosody'"):
        load_config(path)