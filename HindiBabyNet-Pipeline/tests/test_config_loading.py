from __future__ import annotations

from pathlib import Path

from hindibabynet_pipeline.config.configuration import ConfigurationManager


def test_configuration_manager_loads_split_files(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    params_file = tmp_path / "params.yaml"
    config_file.write_text(
        "\n".join(
            [
                "artifacts_root: artifacts/runs",
                "logs_root: logs",
                "paths:",
                "  raw_audio_root: raw_audio",
                "  raw_joined_audio_root: raw_joined",
                "  prepared_audio_root: prepared",
                "  classification_output_root: classification_outputs",
                "  textgrid_output_root: textgrids",
                "  manual_annotation_root: annotations",
                "  evaluation_output_root: evaluation",
                "data_ingestion:",
                "  allowed_ext: ['.wav']",
                "speaker_classification:",
                "  backend: xgb",
            ]
        ),
        encoding="utf-8",
    )
    params_file.write_text(
        "\n".join(
            [
                "audio_preparation:",
                "  target_sr: 16000",
                "  convert_to_mono: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = ConfigurationManager(config_path=config_file, params_path=params_file)

    assert cfg.get_speaker_classification_backend() == "xgb"
    assert cfg.get_processed_audio_root() == Path("prepared")
    assert cfg.get_audio_preparation_params()["target_sr"] == 16000
    assert cfg.get_data_ingestion_config(run_id="r1").session_selection == "earliest"


def test_configuration_manager_reads_data_ingestion_session_selection(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    params_file = tmp_path / "params.yaml"
    config_file.write_text(
        "\n".join(
            [
                "artifacts_root: artifacts/runs",
                "logs_root: logs",
                "paths:",
                "  raw_audio_root: raw_audio",
                "  raw_joined_audio_root: raw_joined",
                "  prepared_audio_root: prepared",
                "  classification_output_root: classification_outputs",
                "  textgrid_output_root: textgrids",
                "  manual_annotation_root: annotations",
                "  evaluation_output_root: evaluation",
                "data_ingestion:",
                "  allowed_ext: ['.wav']",
                "  session_selection: all",
            ]
        ),
        encoding="utf-8",
    )
    params_file.write_text("audio_preparation: {}\n", encoding="utf-8")

    cfg = ConfigurationManager(config_path=config_file, params_path=params_file)

    assert cfg.get_data_ingestion_config(run_id="r1").session_selection == "all"