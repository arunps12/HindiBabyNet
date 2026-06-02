"""Smoke tests — verify all modules import cleanly."""
from __future__ import annotations


def test_import_primary_package():
    import hindibabynet_pipeline

    assert hindibabynet_pipeline.__version__ == "0.1.0"


def test_import_io_utils():
    from hindibabynet_pipeline.utils.io_utils import (
        ensure_dir,
        make_run_id,
        read_yaml,
        write_json,
        write_parquet,
    )
    rid = make_run_id()
    assert isinstance(rid, str) and len(rid) > 0


def test_import_audio_components():
    from hindibabynet_pipeline.components.audio import (
        concatenate_wavs_streaming,
        crop_or_pad,
        ensure_mono_16k_wav_streaming,
        load_audio_mono,
        peak_normalize_wav_streaming,
        resample_audio,
        slice_audio,
        write_stream_wav,
        write_wav_chunk,
    )


def test_import_entities():
    from hindibabynet_pipeline.entity.config_entity import (
        AudioPreparationConfig,
        DataIngestionConfig,
        SpeakerClassificationConfig,
        VTCConfig,
    )
    from hindibabynet_pipeline.entity.artifact_entity import (
        AudioPreparationArtifact,
        DataIngestionArtifact,
        SpeakerClassificationArtifact,
        VTCInferenceArtifact,
    )


def test_import_configuration():
    from hindibabynet_pipeline.config.configuration import ConfigurationManager


def test_import_components():
    from hindibabynet_pipeline.workflow.data_ingestion import DataIngestion
    from hindibabynet_pipeline.workflow.audio_preparation import AudioPreparation
    from hindibabynet_pipeline.components.speaker_classification import SpeakerClassification
    from hindibabynet_pipeline.components.speaker_classification._vtc_core import VTCInferenceRunner


def test_import_exception():
    from hindibabynet_pipeline.exception.exception import (
        HindiBabyNetError,
        format_traceback,
        wrap_exception,
    )


def test_import_logger():
    from hindibabynet_pipeline.logging.logger import add_file_handler, get_logger
