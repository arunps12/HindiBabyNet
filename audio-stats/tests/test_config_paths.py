"""Folders from config.yaml, outputs that are never replaced, and the participant ID list."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hindibabynet_audio_stats.cli import prepare_dataset
from hindibabynet_audio_stats.config import load_config


CSV_METADATA = {"metadata_file": "metadata.csv"}


def _write_metadata(config) -> None:
    """Two synthetic participants. Audio and VTC files are not needed for these tests."""
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "par_id": ["BBB", "AAA"],
            "REC_date": ["2024-01-02", "2024-01-01"],
            "birthdate": ["2023-01-02", "2023-01-10"],
            "child_sex": ["F", "M"],
            "mother_education": ["bachelor", "master"],
            "father_education": ["bachelor", "master"],
            "Location": ["Delhi", "Jaipur"],
        }
    ).to_csv(config.metadata_path, index=False)


def test_default_folders_are_under_data_dir(make_config, tmp_path: Path) -> None:
    config = make_config()
    data = tmp_path / "data"

    assert config.metadata_path == data / "raw" / "metadata" / "metadata_cleaned.xlsx"
    assert config.lookup_path == data / "raw" / "metadata" / "participant_lookup.csv"
    assert config.audio_root == data / "raw" / "audio"
    assert config.vtc_output_root == data / "derived" / "vtc-classification"
    assert config.derived_data_dir == data / "derived" / "statistical-data"
    assert config.results_dir == config.derived_data_dir
    assert config.manual_mapping_csv is None
    assert config.config_path == tmp_path / "project" / "config.yaml"


def test_folders_can_be_stored_in_other_places(make_config, tmp_path: Path) -> None:
    config = make_config(
        paths={
            "metadata": tmp_path / "other disk" / "metadata",
            "raw_audio": tmp_path / "other disk" / "RawAudioData",
            "vtc": tmp_path / "vtc results",
            "statistical_data": tmp_path / "statistics",
        }
    )

    assert config.metadata_path == tmp_path / "other disk" / "metadata" / "metadata_cleaned.xlsx"
    assert config.lookup_path == tmp_path / "other disk" / "metadata" / "participant_lookup.csv"
    assert config.audio_root == tmp_path / "other disk" / "RawAudioData"
    assert config.vtc_output_root == tmp_path / "vtc results"
    assert config.derived_data_dir == tmp_path / "statistics"
    assert config.results_dir == tmp_path / "statistics"


def test_data_can_be_outside_the_project_folder(make_config, tmp_path: Path) -> None:
    # A relative data_dir starts from the folder where config.yaml is saved.
    config = make_config(data_dir="../study-data")
    project = tmp_path / "project"

    assert config.derived_data_dir == tmp_path / "study-data" / "derived" / "statistical-data"
    assert config.metadata_path.parent == tmp_path / "study-data" / "raw" / "metadata"
    assert project not in config.derived_data_dir.parents
    assert project not in config.metadata_path.parents


def test_recording_duration_source_selects_the_audio_folder(make_config, tmp_path: Path) -> None:
    data = tmp_path / "data"

    raw = make_config(params={"recording_duration_source": "raw_audio_sessions"})
    prepared = make_config(params={"recording_duration_source": "Audio_File", "session_selection": "EARLIEST"})

    assert raw.audio_root == data / "raw" / "audio"
    assert prepared.audio_root == data / "derived" / "prepared-audio"
    assert prepared.recording_duration_source == "audio_file"
    assert prepared.session_selection == "earliest"


def test_manual_mapping_file_is_in_the_metadata_folder(make_config, tmp_path: Path) -> None:
    config = make_config(params={"manual_mapping_file": "manual.csv"})

    assert config.manual_mapping_csv == tmp_path / "data" / "raw" / "metadata" / "manual.csv"


def test_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_existing_output_file_is_never_replaced(make_config) -> None:
    config = make_config(params=CSV_METADATA)
    _write_metadata(config)
    old_file = config.derived_data_dir / "input_long.csv"
    old_file.parent.mkdir(parents=True)
    old_file.write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"paths\.statistical_data") as error:
        prepare_dataset(config)

    # The run stops before any work: nothing else is written.
    assert "input_long.csv" in str(error.value)
    assert old_file.read_text(encoding="utf-8") == "old\n"
    assert sorted(path.name for path in config.derived_data_dir.iterdir()) == ["input_long.csv"]
    assert not config.lookup_path.exists()


def test_only_the_missing_wide_table_is_added_to_an_existing_dataset(make_config, capsys) -> None:
    config = make_config(params=CSV_METADATA)
    _write_metadata(config)
    prepare_dataset(config)
    data_dir = config.derived_data_dir
    first_wide = (data_dir / "input_output_wide.csv").read_bytes()

    # A dataset of an earlier version: all files but the wide table. The metadata is gone.
    (data_dir / "input_output_wide.csv").unlink()
    config.metadata_path.unlink()
    before = {path.name: path.read_bytes() for path in data_dir.iterdir()}
    capsys.readouterr()

    master, input_long, input_output_long = prepare_dataset(config)

    assert "Only input_output_wide.csv was missing" in capsys.readouterr().out
    after = {path.name: path.read_bytes() for path in data_dir.iterdir()}
    assert sorted(set(after) - set(before)) == ["input_output_wide.csv"]
    assert all(after[name] == content for name, content in before.items())
    assert after["input_output_wide.csv"] == first_wide
    assert (len(master), len(input_long), len(input_output_long)) == (2, 6, 6)

    # Now the dataset is complete: the next run stops and replaces nothing.
    with pytest.raises(FileExistsError, match="input_output_wide.csv"):
        prepare_dataset(config)
    assert {path.name: path.read_bytes() for path in data_dir.iterdir()} == after


def test_wide_table_has_the_numbers_of_final_master_exactly(make_config) -> None:
    config = make_config(params=CSV_METADATA)
    _write_metadata(config)
    prepare_dataset(config)
    data_dir = config.derived_data_dir

    def same_text_as_final_master() -> pd.DataFrame:
        master = pd.read_csv(data_dir / "final_master.csv", dtype=str)
        wide = pd.read_csv(data_dir / "input_output_wide.csv", dtype=str)
        for column in wide.columns:
            pd.testing.assert_series_equal(wide[column], master[column])
        return master

    master = same_text_as_final_master()

    # Numbers with 17 digits: a standard CSV reader can change their last digit.
    master["age_z"] = ["-0.24836162209524854", "0.10970639932180819"]
    master["adult_male_count_hour"] = ["1.6347830429585775", "0.27276877584472176"]
    master.to_csv(data_dir / "final_master.csv", index=False)
    (data_dir / "input_output_wide.csv").unlink()
    prepare_dataset(config)

    changed = same_text_as_final_master()
    assert changed["age_z"].tolist() == ["-0.24836162209524854", "0.10970639932180819"]


def test_identical_participant_lookup_is_reused(make_config, tmp_path: Path, capsys) -> None:
    first = make_config(params=CSV_METADATA)
    _write_metadata(first)
    prepare_dataset(first)
    saved_lookup = first.lookup_path.read_bytes()
    assert pd.read_csv(first.lookup_path).to_dict("records") == [
        {"original_par_id": "AAA", "participant_id": "P001"},
        {"original_par_id": "BBB", "participant_id": "P002"},
    ]
    capsys.readouterr()

    second = make_config(params=CSV_METADATA, paths={"statistical_data": tmp_path / "second-run"})
    master, _, _ = prepare_dataset(second)

    assert "reused" in capsys.readouterr().out
    assert second.lookup_path.read_bytes() == saved_lookup
    assert master["participant_id"].tolist() == ["P001", "P002"]
    assert (tmp_path / "second-run" / "final_master.csv").exists()


def test_different_participant_lookup_is_never_replaced(make_config, tmp_path: Path) -> None:
    first = make_config(params=CSV_METADATA)
    _write_metadata(first)
    prepare_dataset(first)
    changed_lookup = "original_par_id,participant_id\nAAA,P002\nBBB,P001\n"
    first.lookup_path.write_text(changed_lookup, encoding="utf-8")

    second = make_config(params=CSV_METADATA, paths={"statistical_data": tmp_path / "second-run"})
    with pytest.raises(FileExistsError, match="participant_lookup.csv"):
        prepare_dataset(second)

    assert second.lookup_path.read_text(encoding="utf-8") == changed_lookup
    assert not (tmp_path / "second-run").exists()