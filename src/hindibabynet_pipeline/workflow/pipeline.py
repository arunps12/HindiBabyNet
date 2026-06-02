from __future__ import annotations

from hindibabynet_pipeline.config.configuration import ConfigurationManager
from hindibabynet_pipeline.logging.logger import add_file_handler, get_logger
from hindibabynet_pipeline.workflow.data_ingestion import run_data_ingestion
from hindibabynet_pipeline.workflow.audio_preparation import run_prepare_audio
from hindibabynet_pipeline.workflow.vtc_classification import run_classify_vtc
from hindibabynet_pipeline.workflow.xgb_classification import run_classify_xgb

logger = get_logger(__name__)


def run_pipeline(limit: int | None = None, backend: str | None = None) -> None:
    cfg = ConfigurationManager()
    run_id = cfg.make_run_id()
    add_file_handler(logger, cfg.get_logs_root() / run_id / "pipeline.log")

    stage01_artifact = run_data_ingestion(run_id=run_id)
    recordings_parquet = stage01_artifact.recordings_parquet_path

    run_prepare_audio(recordings_parquet=recordings_parquet, limit=limit, run_id=run_id)

    selected_backend = backend or cfg.get_speaker_classification_backend()
    if selected_backend == "xgb":
        run_classify_xgb(recordings_parquet=recordings_parquet, limit=limit)
        return
    if selected_backend == "vtc":
        run_classify_vtc(recordings_parquet=recordings_parquet, limit=limit)
        return
    raise ValueError(f"Unsupported backend: {selected_backend}")