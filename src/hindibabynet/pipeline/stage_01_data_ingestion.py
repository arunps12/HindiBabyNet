from __future__ import annotations

from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.components.data_ingestion import DataIngestion
from src.hindibabynet.logging.logger import get_logger, add_file_handler


def run_stage_01():
    cfg_mgr = ConfigurationManager()
    di_cfg = cfg_mgr.get_data_ingestion_config()
    logs_root = cfg_mgr.get_logs_root()

    logger = get_logger("hindibabynet.stage_01")

    # logs folder 
    run_id = di_cfg.artifacts_dir.parent.name  # artifacts/runs/<run_id>/data_ingestion -> <run_id>
    log_file = logs_root / run_id / "stage_01_data_ingestion.log"
    add_file_handler(logger, log_file)

    logger.info("Stage 01: Data ingestion started")
    artifact = DataIngestion(di_cfg).initiate_data_ingestion()
    logger.info(f"Stage 01 completed: {artifact}")
    return artifact


if __name__ == "__main__":
    run_stage_01()
