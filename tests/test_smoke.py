"""Smoke tests — verify all modules import cleanly."""
from __future__ import annotations


def test_import_io_utils():
    from src.hindibabynet.utils.io_utils import (
        ensure_dir,
        make_run_id,
        read_yaml,
        write_json,
        write_parquet,
    )
    rid = make_run_id()
    assert isinstance(rid, str) and len(rid) > 0


def test_import_entities():
    from src.hindibabynet.entity.config_entity import (
        AudioPreparationConfig,
        DataIngestionConfig,
        SpeakerClassificationConfig,
    )
    from src.hindibabynet.entity.artifact_entity import (
        AudioPreparationArtifact,
        DataIngestionArtifact,
        SpeakerClassificationArtifact,
    )


def test_import_configuration():
    from src.hindibabynet.config.configuration import ConfigurationManager


def test_import_components():
    from src.hindibabynet.components.data_ingestion import DataIngestion
    from src.hindibabynet.components.audio_preparation import AudioPreparation
    from src.hindibabynet.components.speaker_classification import SpeakerClassification


def test_import_exception():
    from src.hindibabynet.exception.exception import (
        HindiBabyNetError,
        format_traceback,
        wrap_exception,
    )


def test_import_logger():
    from src.hindibabynet.logging.logger import add_file_handler, get_logger
