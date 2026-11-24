"""Tests of the settings (cdi/params.yaml) and the data folders (root config.yaml)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import case
from hindibabynet_cdi.cli import process_cdi
from hindibabynet_cdi.config import DEFAULT_PARAMS_PATH, FORM_ROLES, load_config


def same_path(first: Path, second: Path) -> bool:
    return Path(first).resolve() == Path(second).resolve()


def write_config(path: Path, data_dir: Path | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"data_dir": str(data_dir)}), encoding="utf-8")
    return path


# --- Data folders -----------------------------------------------------------------


def test_default_folders_are_inside_data_dir(make_study, tmp_path: Path) -> None:
    config = make_study()
    data_dir = tmp_path / "data"

    assert same_path(config.config_path, tmp_path / "project" / "config.yaml")
    assert same_path(config.params_path, DEFAULT_PARAMS_PATH)
    assert same_path(config.raw_dir, data_dir / "raw" / "cdi")
    assert same_path(config.word_mapping_path, data_dir / "raw" / "cdi" / "word_mapping.csv")
    assert same_path(config.output_dir, data_dir / "results" / "cdi")


def test_folders_can_be_stored_in_other_places(make_study, tmp_path: Path) -> None:
    paths = {
        "raw_cdi": tmp_path / "other disk" / "exports",
        "word_mapping": tmp_path / "lists" / "items.csv",
        "results": tmp_path / "out",
    }
    config = make_study([case("a")], paths=paths)

    assert same_path(config.raw_dir, paths["raw_cdi"])
    assert same_path(config.word_mapping_path, paths["word_mapping"])
    assert same_path(config.output_dir, paths["results"] / "cdi")
    # The processing works with these folders.
    assert process_cdi(config)["cdi1_summary"]["submission_id"].tolist() == ["cdi-a"]


def test_data_dir_can_be_relative_and_outside_the_project(make_study, tmp_path: Path) -> None:
    config = make_study([case("a")])
    # The data folder is next to the project folder, not inside it.
    config.config_path.write_text("data_dir: ../data\n", encoding="utf-8")

    relative = load_config(config.config_path)

    assert same_path(relative.raw_dir, tmp_path / "data" / "raw" / "cdi")
    assert same_path(relative.output_dir, tmp_path / "data" / "results" / "cdi")
    assert process_cdi(relative)["cdi1_summary"]["submission_id"].tolist() == ["cdi-a"]


def test_missing_input_stops_with_a_clear_message(make_study, tmp_path: Path) -> None:
    config = make_study()
    config.word_mapping_path.unlink()
    with pytest.raises(FileNotFoundError, match="word_mapping"):
        load_config(config.config_path)

    no_data = write_config(tmp_path / "empty" / "config.yaml", tmp_path / "no data here")
    with pytest.raises(FileNotFoundError, match="raw_cdi"):
        load_config(no_data)

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "no-config.yaml")


def test_config_file_from_the_environment(
    make_study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_study()
    monkeypatch.setenv("HBN_CONFIG", str(config.config_path))

    assert same_path(load_config().config_path, config.config_path)
    # A config file given to the command is used before the environment.
    other = write_config(tmp_path / "other" / "config.yaml", tmp_path / "data")
    assert same_path(load_config(other).config_path, other)


# --- Settings ---------------------------------------------------------------------


def test_settings_of_the_study(make_study) -> None:
    config = make_study()

    assert config.form_ids == {
        "consent": "539547",
        "eligibility": "539591",
        "background": "539576",
        "cdi_8_18": "539642",
        "cdi_19_36": "539644",
    }
    assert set(config.form_detection_rules) == set(FORM_ROLES)
    assert config.analysis.age_month_divisor == 30.4375
    assert config.analysis.strict_hindi_threshold == 75.0
    assert config.analysis.age_discrepancy_flag_months == 2.0
    assert config.analysis.younger_questionnaire == "8_18"
    assert config.analysis.older_questionnaire == "19_36"
    assert config.age.birthdate_column == "birthdate"
    assert (config.age.cdi1_min_month, config.age.cdi1_max_month) == (8, 18)
    assert (config.age.cdi2_min_month, config.age.cdi2_max_month) == (19, 36)
    assert config.eligibility.required_mother_tongue_response is True
    assert config.eligibility.allow_preterm is False
    assert config.eligibility.allow_impairment is False


def test_no_contact_form_and_no_completion_settings(make_study) -> None:
    config = make_study()
    text = DEFAULT_PARAMS_PATH.read_text(encoding="utf-8")
    params = yaml.safe_load(text)

    assert "540031" not in text
    assert not {"contact", "completeness", "paths", "outputs"} & set(params)
    assert "contact_form_id" not in params["forms"]
    assert "contact" not in params["form_detection_rules"]
    assert "contact" not in config.form_ids
    assert not hasattr(config, "contact")
    assert not hasattr(config, "completeness")


def test_other_params_file(make_study) -> None:
    config = make_study(params={"analysis": {"strict_hindi_threshold": 80}, "age": {"cdi1_min_month": 9}})

    assert same_path(config.params_path, config.config_path.parent / "params.yaml")
    assert config.analysis.strict_hindi_threshold == 80.0
    assert config.age.cdi1_min_month == 9
    assert config.analysis.age_month_divisor == 30.4375


def test_missing_setting_is_named(make_study, tmp_path: Path) -> None:
    config = make_study()
    settings = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
    del settings["age"]["cdi2_max_month"]
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match="cdi2_max_month"):
        load_config(config.config_path, params_file)
