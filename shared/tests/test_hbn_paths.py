from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import hbn_paths
from hbn_paths import Config, local_path, new_output


def write_config(folder: Path, text: str) -> Path:
    path = folder / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_folders_are_under_data_dir(tmp_path):
    config = Config(write_config(tmp_path, "data_dir: study-data\n"))
    base = tmp_path / "study-data"
    assert config.path("raw_audio") == base / "raw" / "audio"
    assert config.path("vtc", "P1", "rttm.csv") == base / "derived" / "vtc-classification" / "P1" / "rttm.csv"
    assert config.results("cdi") == base / "results" / "cdi"
    assert config.path("word_mapping") == base / "raw" / "cdi" / "word_mapping.csv"


def test_data_can_be_outside_the_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "other-disk" / "data"
    config = Config(write_config(project, "data_dir: ../other-disk/data\n"))
    assert config.data_dir == outside
    absolute = Config(write_config(project, f"data_dir: {outside.as_posix()}\n"))
    assert absolute.data_dir == outside


def test_single_folder_can_be_stored_somewhere_else(tmp_path):
    existing = tmp_path / "old place" / "VTC"
    text = f"data_dir: data\npaths:\n  vtc: {existing.as_posix()}\n  word_mapping: lists/words.csv\n"
    config = Config(write_config(tmp_path, text))
    assert config.path("vtc") == existing
    assert config.path("word_mapping") == tmp_path / "lists" / "words.csv"
    assert config.path("prepared_audio") == tmp_path / "data" / "derived" / "prepared-audio"


def test_missing_input_names_the_config_key(tmp_path):
    config = Config(write_config(tmp_path, "data_dir: data\n"))
    with pytest.raises(FileNotFoundError, match="paths.metadata"):
        config.input("metadata", "metadata_cleaned.xlsx")
    with pytest.raises(KeyError):
        config.path("not_a_folder")


def test_missing_or_empty_config_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.example.yaml"):
        Config(tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="data_dir"):
        Config(write_config(tmp_path, "paths: {}\n"))


def test_settings_and_environment_variable(tmp_path, monkeypatch):
    path = write_config(tmp_path, "data_dir: data\naudio:\n  vtc_device: cuda\n")
    monkeypatch.setenv("HBN_CONFIG", str(path))
    config = Config()
    assert config.setting("audio", "vtc_device", "cpu") == "cuda"
    assert config.setting("audio", "vtc_repo", "~/VTC") == "~/VTC"


def test_windows_and_ubuntu_paths_are_converted(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert local_path("/mnt/d/study/raw", tmp_path).as_posix() == "D:/study/raw"
    assert local_path("/home/me/data", tmp_path, "Ubuntu-22.04").as_posix().endswith(
        "wsl.localhost/Ubuntu-22.04/home/me/data"
    )
    assert local_path("//server/share/raw", tmp_path).as_posix().endswith("server/share/raw")
    monkeypatch.setattr(os, "name", "posix")
    assert str(local_path("D:\\study\\raw", tmp_path)).replace("\\", "/") == "/mnt/d/study/raw"
    with pytest.raises(ValueError, match="Mount it"):
        local_path("//server/share/raw", tmp_path)


def test_existing_output_is_never_replaced(tmp_path):
    target = new_output(tmp_path / "results" / "cdi" / "cdi1_items.csv")
    assert target.parent.is_dir()
    target.write_text("kept", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not replaced"):
        new_output(target)
    assert target.read_text(encoding="utf-8") == "kept"


def test_data_folders_inside_the_project_are_ignored_by_git():
    """Any new folder or file at the project root must be ignored, whatever its name."""
    root = hbn_paths.REPO_ROOT
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    probes = [
        "config.yaml",
        "my-data/raw/audio/P1/20250101/a.WAV",
        "results/cdi/cdi1_items.csv",
        "anything/else/notes.txt",
        "audio/models/other_model.pkl",
        "cdi/output/cdi1_summary.csv",
        "audio-stats/audio-stats.html",
        "prosody-stats/figures/plot.png",
    ]
    for probe in probes:
        done = subprocess.run(["git", "check-ignore", "--no-index", "-q", probe], cwd=root)
        assert done.returncode == 0, f"not ignored: {probe}"
    allowed = ["README.md", "shared/hbn_paths.py", "cdi/pyproject.toml", "config.example.yaml"]
    for probe in allowed:
        done = subprocess.run(["git", "check-ignore", "--no-index", "-q", probe], cwd=root)
        assert done.returncode == 1, f"wrongly ignored: {probe}"
