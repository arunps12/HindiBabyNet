from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.hindibabynet.utils.io_utils import read_yaml, make_run_id
from src.hindibabynet.entity.config_entity import (
    DataIngestionConfig,
    AudioPreparationConfig,
    SpeakerClassificationConfig,
)


@dataclass
class ConfigurationManager:
    config_path: Path = Path("configs/config.yaml")

    def __post_init__(self):
        self.config: Dict[str, Any] = read_yaml(self.config_path)

    # ---- helpers ----
    def get_logs_root(self) -> Path:
        return Path(self.config.get("logs_root", "logs"))

    def make_run_id(self) -> str:
        return make_run_id()

    # ---- Stage 01: Data Ingestion ----
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        run_id = make_run_id()
        artifacts_root = Path(self.config["artifacts_root"])
        di = self.config["data_ingestion"]

        artifacts_dir = artifacts_root / run_id / "data_ingestion"
        recordings_parquet_path = artifacts_dir / di.get(
            "recordings_filename", "recordings.parquet"
        )

        return DataIngestionConfig(
            raw_audio_root=Path(di["raw_audio_root"]),
            allowed_ext=list(di.get("allowed_ext", [".wav", ".WAV"])),
            artifacts_dir=artifacts_dir,
            recordings_parquet_path=recordings_parquet_path,
        )

    # ---- Stage 02: Audio Preparation ----
    def get_audio_preparation_config(
        self, run_id: str, recording_id: str
    ) -> AudioPreparationConfig:
        ap = self.config["audio_preparation"]
        artifacts_root = Path(self.config["artifacts_root"])

        artifacts_dir = artifacts_root / run_id / "audio_preparation"
        processed_root = Path(ap["processed_audio_root"])
        processed_dir = processed_root / recording_id

        return AudioPreparationConfig(
            artifacts_dir=artifacts_dir,
            processed_audio_root=processed_root,
            target_sr=int(ap.get("target_sr", 16000)),
            to_mono=bool(ap.get("to_mono", True)),
            target_peak_dbfs=float(ap.get("target_peak_dbfs", -1.0)),
            combine_gap_sec=float(ap.get("combine_gap_sec", 0.0)),
            manifest_parquet_path=artifacts_dir
            / f"{recording_id}_audio_manifest.parquet",
            analysis_wav_path=processed_dir / f"{recording_id}.wav",
            analysis_meta_json_path=artifacts_dir
            / f"{recording_id}_analysis_meta.json",
        )

    # ---- Stage 03: Speaker Classification ----
    def get_speaker_classification_config(
        self, run_id: str, participant_id: str
    ) -> SpeakerClassificationConfig:
        sc = self.config["speaker_classification"]
        artifacts_root = Path(self.config["artifacts_root"])

        artifacts_dir = artifacts_root / run_id / "speaker_classification"
        output_audio_root = Path(sc["output_audio_root"])
        output_dir = output_audio_root / participant_id

        return SpeakerClassificationConfig(
            artifacts_dir=artifacts_dir,
            model_path=Path(sc["model_path"]),
            class_names=list(
                sc.get(
                    "class_names",
                    ["adult_male", "adult_female", "child", "background"],
                )
            ),
            egemaps_dim=int(sc.get("egemaps_dim", 88)),
            vad_aggressiveness=int(sc.get("vad_aggressiveness", 2)),
            vad_frame_ms=int(sc.get("vad_frame_ms", 30)),
            vad_min_region_ms=int(sc.get("vad_min_region_ms", 300)),
            diarization_model=str(
                sc.get("diarization_model", "pyannote/speaker-diarization-3.1")
            ),
            chunk_sec=float(sc.get("chunk_sec", 900.0)),
            overlap_sec=float(sc.get("overlap_sec", 10.0)),
            min_speakers=int(sc.get("min_speakers", 2)),
            max_speakers=int(sc.get("max_speakers", 4)),
            merge_gap_sec=float(sc.get("merge_gap_sec", 0.7)),
            min_segment_sec=float(sc.get("min_segment_sec", 0.2)),
            classify_win_sec=float(sc.get("classify_win_sec", 1.0)),
            classify_hop_sec=float(sc.get("classify_hop_sec", 0.5)),
            output_audio_root=output_audio_root,
            segments_parquet_path=artifacts_dir
            / f"{participant_id}_segments.parquet",
            summary_json_path=artifacts_dir / f"{participant_id}_summary.json",
            textgrid_path=artifacts_dir / f"{participant_id}.TextGrid",
            main_female_wav_path=output_dir / f"{participant_id}_main_female.wav",
            main_male_wav_path=output_dir / f"{participant_id}_main_male.wav",
            child_wav_path=output_dir / f"{participant_id}_child.wav",
            background_wav_path=output_dir / f"{participant_id}_background.wav",
        )
