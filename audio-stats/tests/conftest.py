"""Helper for the tests: a temporary config.yaml and params file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hindibabynet_audio_stats.config import DEFAULT_PARAMS_PATH, load_config


@pytest.fixture
def make_config(tmp_path: Path):
    """Return a function that writes a temporary root config.yaml and loads it.

    paths: folders stored in another place (the ``paths`` part of config.yaml).
    params: settings that differ from audio-stats/params.yaml.
    data_dir: main data folder. Default: <tmp_path>/data.
    """

    def _make(paths: dict | None = None, params: dict | None = None, data_dir: str | Path | None = None):
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        values: dict = {"data_dir": str(data_dir or tmp_path / "data")}
        if paths:
            values["paths"] = {key: str(value) for key, value in paths.items()}
        config_file = project / "config.yaml"
        config_file.write_text(yaml.safe_dump(values), encoding="utf-8")

        params_file = None
        if params:
            settings = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
            settings.update(params)
            params_file = project / "params.yaml"
            params_file.write_text(yaml.safe_dump(settings), encoding="utf-8")

        return load_config(config_file, params_file)

    return _make
