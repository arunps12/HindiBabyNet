"""Folders from config.yaml, settings from params.yaml, and outputs that are never replaced."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from typer.testing import CliRunner

from hindibabynet_prosody.cli import app
from hindibabynet_prosody.config import DEFAULT_PARAMS_PATH, load_config
from hindibabynet_prosody.pipeline.extract import run_extraction


def _write_inputs(config) -> None:
    """One synthetic participant: VTC segments, a short recording and the lookup file."""
    (config.paths.rttm_root / "SRC001").mkdir(parents=True)
    (config.paths.rttm_root / "SRC001" / "rttm.csv").write_text(
        "uid,start_time_s,duration_s,label\nSRC001,0.0,0.5,FEM\nSRC001,0.6,0.5,KCHI\n",
        encoding="utf-8",
    )
    (config.paths.audio_root / "SRC001").mkdir(parents=True)
    sf.write(
        config.paths.audio_root / "SRC001" / "SRC001.wav",
        np.zeros(32000, dtype=np.float32),
        16000,
    )
    config.paths.participant_lookup.parent.mkdir(parents=True)
    config.paths.participant_lookup.write_text(
        "original_par_id,participant_id\nSRC001,P001\n", encoding="utf-8"
    )


def _everything_in(folder: Path) -> list[str]:
    return sorted(str(path) for path in folder.rglob("*"))


def test_default_folders_are_under_data_dir(make_config, tmp_path: Path) -> None:
    config = make_config()
    data = tmp_path / "data"

    assert config.paths.rttm_root == data / "derived" / "vtc-classification"
    assert config.paths.audio_root == data / "derived" / "prepared-audio"
    assert config.paths.participant_lookup == data / "raw" / "metadata" / "participant_lookup.csv"
    assert config.paths.output_dir == data / "derived" / "prosody-features"
    assert config.paths.cache_dir == data / "cache" / "prosody"


def test_folders_can_be_stored_in_other_places(make_config, tmp_path: Path) -> None:
    config = make_config(
        paths={
            "vtc": tmp_path / "other disk" / "VTC",
            "prepared_audio": tmp_path / "other disk" / "audio",
            "metadata": tmp_path / "metadata",
            "prosody_features": tmp_path / "features",
            "cache": tmp_path / "temporary",
        }
    )

    assert config.paths.rttm_root == tmp_path / "other disk" / "VTC"
    assert config.paths.audio_root == tmp_path / "other disk" / "audio"
    assert config.paths.participant_lookup == tmp_path / "metadata" / "participant_lookup.csv"
    assert config.paths.output_dir == tmp_path / "features"
    assert config.paths.cache_dir == tmp_path / "temporary" / "prosody"


def test_data_can_be_outside_the_project_folder(make_config, tmp_path: Path) -> None:
    # A relative data_dir starts from the folder where config.yaml is saved.
    config = make_config(data_dir="../study-data")
    project = tmp_path / "project"

    assert config.paths.output_dir == tmp_path / "study-data" / "derived" / "prosody-features"
    assert config.paths.audio_root == tmp_path / "study-data" / "derived" / "prepared-audio"
    assert project not in config.paths.output_dir.parents
    assert project not in config.paths.cache_dir.parents


def test_lookup_file_name_comes_from_the_settings(make_config, default_params: dict) -> None:
    default_params["mapping"]["lookup_file"] = "other_lookup.csv"
    config = make_config(params=default_params)

    assert config.paths.participant_lookup.name == "other_lookup.csv"
    assert config.paths.participant_lookup.parent.name == "metadata"


def test_config_file_can_be_set_with_the_environment_variable(
    make_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_config()
    monkeypatch.setenv("HBN_CONFIG", str(tmp_path / "project" / "config.yaml"))

    assert load_config().paths.output_dir == tmp_path / "data" / "derived" / "prosody-features"


def test_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_study_settings_are_read_from_params_yaml(make_config) -> None:
    config = make_config()
    settings = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))

    assert sorted(settings) == sorted(
        ["project", "rttm", "mapping", "speakers", "audio", "pitch", "ssl", "pca", "runtime", "output"]
    )
    assert config.pitch.semitone_reference_hz == 100.0
    assert config.mapping.lookup_file == "participant_lookup.csv"
    assert config.speakers.include == {
        "FEM": "adult_female",
        "MAL": "adult_male",
        "OCH": "other_child",
    }
    assert config.ssl.wav2vec2_model == "arunps/wav2vec2-home-hindibabynet-ssl"
    assert config.ssl.hubert_model == "arunps/hubert-home-hindibabynet-ssl"
    assert config.pca.n_components == 3
    assert config.output.final_filename == "participant_speaker_acoustic_features.csv"


def test_params_file_must_not_contain_paths(make_config, default_params: dict) -> None:
    default_params["paths"] = {"output_dir": "outputs"}

    with pytest.raises(ValueError, match="must not contain paths"):
        make_config(params=default_params)


def test_existing_features_file_is_never_replaced(make_config) -> None:
    config = make_config()
    old_file = config.paths.output_dir / config.output.final_filename
    old_file.parent.mkdir(parents=True)
    old_file.write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"paths\.prosody_features") as error:
        run_extraction(config)

    # The run stops before any work: nothing else is written.
    assert old_file.name in str(error.value)
    assert old_file.read_text(encoding="utf-8") == "old\n"
    assert sorted(path.name for path in config.paths.output_dir.iterdir()) == [old_file.name]
    assert not config.paths.cache_dir.exists()


def test_command_stops_when_the_features_file_exists(make_config, tmp_path: Path) -> None:
    config = make_config()
    old_file = config.paths.output_dir / config.output.final_filename
    old_file.parent.mkdir(parents=True)
    old_file.write_text("old\n", encoding="utf-8")
    before = _everything_in(tmp_path)

    result = CliRunner().invoke(app, ["--config", str(tmp_path / "project" / "config.yaml")])

    assert result.exit_code == 1
    assert "already exist and are not replaced" in result.output
    assert old_file.read_text(encoding="utf-8") == "old\n"
    assert _everything_in(tmp_path) == before


@pytest.mark.parametrize("option", ["--overwrite", "--resume"])
def test_command_has_no_option_to_replace_results(make_config, tmp_path: Path, option: str) -> None:
    make_config()

    result = CliRunner().invoke(
        app, ["--config", str(tmp_path / "project" / "config.yaml"), option]
    )

    assert result.exit_code == 2


def test_check_writes_nothing(make_config, tmp_path: Path) -> None:
    config = make_config()
    _write_inputs(config)
    before = _everything_in(tmp_path)

    result = CliRunner().invoke(
        app, ["--config", str(tmp_path / "project" / "config.yaml"), "--check"]
    )

    assert result.exit_code == 0, result.output
    assert "VTC result files" in result.output
    assert "not there yet" in result.output
    assert _everything_in(tmp_path) == before
    assert not config.paths.output_dir.exists()
    assert not config.paths.cache_dir.exists()


def test_check_reports_an_existing_features_file(make_config, tmp_path: Path) -> None:
    config = make_config()
    _write_inputs(config)
    old_file = config.paths.output_dir / config.output.final_filename
    old_file.parent.mkdir(parents=True)
    old_file.write_text("old\n", encoding="utf-8")
    before = _everything_in(tmp_path)

    result = CliRunner().invoke(
        app, ["--config", str(tmp_path / "project" / "config.yaml"), "--check"]
    )

    assert result.exit_code == 0, result.output
    assert "already exists" in result.output
    assert _everything_in(tmp_path) == before
