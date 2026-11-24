from __future__ import annotations

import pytest


def test_load_study_settings(make_config) -> None:
    config = make_config()

    assert config.rttm.participant_column == "uid"
    assert config.mapping.source_id_column == "original_par_id"
    assert config.pitch.speaker_settings["FEM"].pitch_floor_hz == 75.0


def _small_params(adaptive_level: str) -> dict:
    return {
        "project": {"name": "test", "random_seed": 42},
        "rttm": {"file_pattern": "**/rttm.csv"},
        "mapping": {"strict": True},
        "speakers": {"include": {"FEM": "adult_female"}},
        "audio": {"extensions": [".wav"], "preferred_extensions": [".wav"]},
        "pitch": {
            "method": "praat_ac",
            "speaker_settings": {"FEM": {"pitch_floor_hz": 75, "pitch_ceiling_hz": 700}},
            "semitone_reference_hz": 1.0,
            "adaptive": {
                "enabled": True,
                "level": adaptive_level,
                "lower_multiplier": 0.5,
                "upper_multiplier": 2.0,
                "minimum_floor_hz": 40,
                "maximum_ceiling_hz": 1000,
                "fallback_to_first_pass": True,
            },
        },
        "ssl": {"wav2vec2_model": "foo/bar", "hubert_model": "foo/baz"},
        "pca": {"n_components": 3},
        "runtime": {"device": "auto"},
        "output": {"final_filename": "out.csv"},
    }


def test_small_params_file_uses_the_other_defaults(make_config) -> None:
    config = make_config(params=_small_params("segment"))

    assert config.pitch.semitone_reference_hz == 1.0
    assert config.mapping.lookup_file == "participant_lookup.csv"
    assert config.output.final_filename == "out.csv"


def test_invalid_adaptive_level_raises(make_config) -> None:
    with pytest.raises(ValueError, match="pitch.adaptive.level"):
        make_config(params=_small_params("invalid"))


def test_semitone_reference_must_be_written_in_the_settings(make_config) -> None:
    params = _small_params("segment")
    del params["pitch"]["semitone_reference_hz"]

    with pytest.raises(ValueError, match="semitone_reference_hz"):
        make_config(params=params)
