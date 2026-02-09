from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.hindibabynet.utils.common import read_yaml, make_run_id
from src.hindibabynet.entity.config_entity import DataIngestionConfig


@dataclass
class ConfigurationManager:
    config_path: Path = Path("configs/config.yaml")

    def __post_init__(self):
        self.config: Dict[str, Any] = read_yaml(self.config_path)

    def get_logs_root(self) -> Path:
        return Path(self.config.get("logs_root", "logs"))

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        run_id = make_run_id()
        artifacts_root = Path(self.config["artifacts_root"])
        di = self.config["data_ingestion"]

        artifacts_dir = artifacts_root / run_id / "data_ingestion"
        recordings_parquet_path = artifacts_dir / di.get("recordings_filename", "recordings.parquet")

        return DataIngestionConfig(
            raw_audio_root=Path(di["raw_audio_root"]),
            allowed_ext=list(di.get("allowed_ext", [".wav", ".WAV"])),
            artifacts_dir=artifacts_dir,
            recordings_parquet_path=recordings_parquet_path,
        )
