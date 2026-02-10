from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.hindibabynet.utils.common import read_yaml, make_run_id
from src.hindibabynet.entity.config_entity import DataIngestionConfig, AudioPreparationConfig


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
    
@dataclass
class ConfigurationManager:
    config_path: Path = Path("configs/config.yaml")

    def __post_init__(self):
        self.config: Dict[str, Any] = read_yaml(self.config_path)

    def get_logs_root(self) -> Path:
        return Path(self.config.get("logs_root", "logs"))

    def make_run_id(self) -> str:
        return make_run_id()

    def get_audio_preparation_config(self, run_id: str, recording_id: str) -> AudioPreparationConfig:
        ap = self.config["audio_preparation"]
        artifacts_root = Path(self.config["artifacts_root"])

        artifacts_dir = artifacts_root / run_id / "audio_preparation"

        processed_root = Path(ap["processed_audio_root"])
        processed_dir = processed_root / recording_id
        manifest_parquet_path = artifacts_dir / f"{recording_id}_audio_manifest.parquet"
        analysis_meta_json_path = artifacts_dir / f"{recording_id}_analysis_meta.json"
        analysis_wav_path = processed_dir / f"{recording_id}_analysis.wav"

        return AudioPreparationConfig(
            artifacts_dir=artifacts_dir,
            processed_audio_root=processed_root,
            target_sr=int(ap.get("target_sr", 16000)),
            to_mono=bool(ap.get("to_mono", True)),
            target_peak_dbfs=float(ap.get("target_peak_dbfs", -1.0)),
            combine_gap_sec=float(ap.get("combine_gap_sec", 0.0)),
            manifest_parquet_path=manifest_parquet_path,
            analysis_wav_path=analysis_wav_path,
            analysis_meta_json_path=analysis_meta_json_path,
        )