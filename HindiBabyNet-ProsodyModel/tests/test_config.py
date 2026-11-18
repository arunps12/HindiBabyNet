from __future__ import annotations

from pathlib import Path

import pytest

from hindibabynet_prosody.config import load_config


def test_load_example_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "config.example.yaml")

    assert config.rttm.participant_column == "uid"
    assert config.mapping.source_id_column == "original_par_id"
    assert config.pitch.speaker_settings["FEM"].pitch_floor_hz == 75.0


def test_invalid_adaptive_level_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: test",
                "  random_seed: 42",
                "paths:",
                "  rttm_root: C:/tmp/rttm",
                "  audio_root: C:/tmp/audio",
                "  participant_lookup: C:/tmp/lookup.csv",
                "  output_dir: C:/tmp/output",
                "  cache_dir: C:/tmp/cache",
                "rttm:",
                "  file_pattern: '**/rttm.csv'",
                "mapping:",
                "  strict: true",
                "speakers:",
                "  include:",
                "    FEM: adult_female",
                "audio:",
                "  extensions: ['.wav']",
                "  preferred_extensions: ['.wav']",
                "pitch:",
                "  method: praat_ac",
                "  speaker_settings:",
                "    FEM:",
                "      pitch_floor_hz: 75",
                "      pitch_ceiling_hz: 700",
                "  adaptive:",
                "    enabled: true",
                "    level: invalid",
                "    lower_multiplier: 0.5",
                "    upper_multiplier: 2.0",
                "    minimum_floor_hz: 40",
                "    maximum_ceiling_hz: 1000",
                "    fallback_to_first_pass: true",
                "ssl:",
                "  wav2vec2_model: foo/bar",
                "  hubert_model: foo/baz",
                "pca:",
                "  n_components: 3",
                "runtime:",
                "  device: auto",
                "output:",
                "  final_filename: out.csv",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pitch.adaptive.level"):
        load_config(config_path)
