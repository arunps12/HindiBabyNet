from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.hindibabynet.utils.io_utils import read_yaml, make_run_id
from src.hindibabynet.entity.config_entity import (
    DataIngestionConfig,
    AudioPreparationConfig,
    VADConfig,
    DiarizationConfig,
    IntersectionConfig,
    SpeakerClassificationConfig,
    VTCConfig,
)


@dataclass
class ConfigurationManager:
    config_path: Path = Path("configs/config.yaml")

    def __post_init__(self):
        self.config: Dict[str, Any] = read_yaml(self.config_path)

    def _require(self, mapping: dict, key: str, section: str) -> Any:
        if key not in mapping:
            raise ValueError(f"Missing required config key: {section}.{key}")
        return mapping[key]

    # ---- helpers ----
    def get_logs_root(self) -> Path:
        return Path(self.config.get("logs_root", "logs"))

    def make_run_id(self) -> str:
        return make_run_id()

    # ---- Backend selector ----
    def get_speaker_classification_backend(self) -> str:
        """Return 'xgb' (default) or 'vtc'."""
        sc = self.config.get("speaker_classification", {})
        return str(sc.get("backend", "xgb")).lower()

    # ---- Unified param accessors (support old flat + new nested config) ---
    def get_xgb_params(self) -> dict:
        """Return the XGB algorithm parameters dict.

        New config:  ``speaker_classification.xgb.*``
        Old config:  ``speaker_classification.*``  (flat)
        """
        sc = self.config.get("speaker_classification", {})
        if "xgb" in sc and isinstance(sc["xgb"], dict):
            return sc["xgb"]
        # Fallback: old flat config — return the whole section
        return sc

    def get_vtc_params(self) -> dict:
        """Return the VTC parameters dict.

        New config:  ``speaker_classification.vtc.*``
        Old config:  top-level ``vtc.*``
        """
        sc = self.config.get("speaker_classification", {})
        if "vtc" in sc and isinstance(sc["vtc"], dict):
            return sc["vtc"]
        # Fallback: old top-level vtc section
        return self.config.get("vtc", {})

    def get_classification_output_root(self) -> Path:
        """Return the unified classification output root directory.

        New config:  ``speaker_classification.output_root``
        Old config:  ``speaker_classification.output_audio_root``
        """
        sc = self.config.get("speaker_classification", {})
        if "output_root" in sc:
            return Path(sc["output_root"])
        if "output_audio_root" in sc:
            return Path(sc["output_audio_root"])
        raise ValueError(
            "Missing 'speaker_classification.output_root' in config. "
            "Please update configs/config.yaml to the new format."
        )

    def get_processed_audio_root(self) -> Path:
        """Return the processed audio root from audio_preparation config."""
        ap = self.config.get("audio_preparation", {})
        if "processed_audio_root" not in ap:
            raise ValueError("Missing 'audio_preparation.processed_audio_root' in config.")
        return Path(ap["processed_audio_root"])

    # ---- Stage 01: Data Ingestion ----
    def get_data_ingestion_config(self, run_id: str | None = None) -> DataIngestionConfig:
        run_id = run_id or make_run_id()
        artifacts_root = Path(self.config.get("artifacts_root", "artifacts/runs"))
        di = self.config.get("data_ingestion", {})

        raw_audio_root = self._require(di, "raw_audio_root", "data_ingestion")

        artifacts_dir = artifacts_root / run_id / "data_ingestion"
        recordings_parquet_path = artifacts_dir / di.get(
            "recordings_filename", "recordings.parquet"
        )

        return DataIngestionConfig(
            raw_audio_root=Path(raw_audio_root),
            allowed_ext=list(di.get("allowed_ext", [".wav", ".WAV"])),
            artifacts_dir=artifacts_dir,
            recordings_parquet_path=recordings_parquet_path,
        )

    # ---- Stage 02: Audio Preparation ----
    def get_audio_preparation_config(
        self, run_id: str, recording_id: str
    ) -> AudioPreparationConfig:
        ap = self.config.get("audio_preparation", {})
        artifacts_root = Path(self.config.get("artifacts_root", "artifacts/runs"))

        processed_audio_root = self._require(
            ap, "processed_audio_root", "audio_preparation"
        )

        artifacts_dir = artifacts_root / run_id / "audio_preparation"
        processed_root = Path(processed_audio_root)
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

    # ---- Stage 03: VAD ----
    def get_vad_config(
        self, run_id: str, participant_id: str
    ) -> VADConfig:
        sc = self.get_xgb_params()
        artifacts_root = Path(self.config.get("artifacts_root", "artifacts/runs"))
        artifacts_dir = artifacts_root / run_id / "vad"

        return VADConfig(
            artifacts_dir=artifacts_dir,
            vad_aggressiveness=int(sc.get("vad_aggressiveness", 2)),
            vad_frame_ms=int(sc.get("vad_frame_ms", 30)),
            vad_min_region_ms=int(sc.get("vad_min_region_ms", 300)),
            vad_parquet_path=artifacts_dir / f"{participant_id}_vad.parquet",
            summary_json_path=artifacts_dir / f"{participant_id}_vad_summary.json",
        )

    # ---- Stage 04: Diarization ----
    def get_diarization_config(
        self, run_id: str, participant_id: str
    ) -> DiarizationConfig:
        sc = self.get_xgb_params()
        artifacts_root = Path(self.config.get("artifacts_root", "artifacts/runs"))
        artifacts_dir = artifacts_root / run_id / "diarization"
        output_audio_root = self.get_classification_output_root()

        return DiarizationConfig(
            artifacts_dir=artifacts_dir,
            diarization_model=str(
                sc.get("diarization_model", "pyannote/speaker-diarization-3.1")
            ),
            chunk_sec=float(sc.get("chunk_sec", 900.0)),
            overlap_sec=float(sc.get("overlap_sec", 10.0)),
            min_speakers=int(sc.get("min_speakers", 2)),
            max_speakers=int(sc.get("max_speakers", 4)),
            tmp_dir=output_audio_root / "_tmp_diar" / participant_id,
            diarization_parquet_path=artifacts_dir
            / f"{participant_id}_diarization.parquet",
            summary_json_path=artifacts_dir
            / f"{participant_id}_diarization_summary.json",
        )

    # ---- Stage 05: Intersection ----
    def get_intersection_config(
        self, run_id: str, participant_id: str
    ) -> IntersectionConfig:
        sc = self.get_xgb_params()
        artifacts_root = Path(self.config.get("artifacts_root", "artifacts/runs"))
        artifacts_dir = artifacts_root / run_id / "intersection"

        return IntersectionConfig(
            artifacts_dir=artifacts_dir,
            min_segment_sec=float(sc.get("min_segment_sec", 0.2)),
            speech_segments_parquet_path=artifacts_dir
            / f"{participant_id}_speech_segments.parquet",
            summary_json_path=artifacts_dir
            / f"{participant_id}_intersection_summary.json",
        )

    # ---- Stage 06: Speaker Classification (6-stage pipeline) ----
    def get_speaker_classification_config(
        self, run_id: str, participant_id: str
    ) -> SpeakerClassificationConfig:
        sc = self.get_xgb_params()
        artifacts_root = Path(self.config.get("artifacts_root", "artifacts/runs"))

        artifacts_dir = artifacts_root / run_id / "speaker_classification"
        output_audio_root = self.get_classification_output_root()
        output_dir = output_audio_root / "xgb" / participant_id

        return SpeakerClassificationConfig(
            artifacts_dir=artifacts_dir,
            model_path=Path(sc.get("model_path", "models/xgb_egemaps.pkl")),
            class_names=list(
                sc.get(
                    "class_names",
                    ["adult_male", "adult_female", "child", "background"],
                )
            ),
            egemaps_dim=int(sc.get("egemaps_dim", 88)),
            merge_gap_sec=float(sc.get("merge_gap_sec", 0.3)),
            min_segment_sec=float(sc.get("min_segment_sec", 0.2)),
            classify_win_sec=float(sc.get("classify_win_sec", 1.0)),
            classify_hop_sec=float(sc.get("classify_hop_sec", 0.5)),
            diarization_model=str(
                sc.get("diarization_model", "pyannote/speaker-diarization-3.1")
            ),
            min_speakers=1,
            max_speakers=3,
            output_audio_root=output_audio_root / "xgb",
            classified_segments_parquet_path=output_dir
            / f"{participant_id}_segments.parquet",
            main_female_parquet_path=output_dir
            / f"{participant_id}_main_female.parquet",
            main_male_parquet_path=output_dir
            / f"{participant_id}_main_male.parquet",
            child_parquet_path=output_dir
            / f"{participant_id}_child.parquet",
            background_parquet_path=output_dir
            / f"{participant_id}_background.parquet",
            summary_json_path=output_dir / f"{participant_id}_summary.json",
            textgrid_path=output_dir / f"{participant_id}.TextGrid",
            main_female_wav_path=output_dir / f"{participant_id}_main_female.wav",
            main_male_wav_path=output_dir / f"{participant_id}_main_male.wav",
            child_wav_path=output_dir / f"{participant_id}_child.wav",
            background_wav_path=output_dir / f"{participant_id}_background.wav",
        )

    # ---- VTC (Voice Type Classifier) ----
    def get_vtc_config(self) -> VTCConfig:
        """Build VTCConfig (kept for backward compat with old VTCInferenceRunner)."""
        vtc = self.get_vtc_params()
        output_root = self.get_classification_output_root() / "vtc"
        return VTCConfig(
            repo_path=Path(vtc.get("repo_path", "external_models/VTC")),
            device=str(vtc.get("device", "cuda")),
            output_root=output_root,
            input_root=output_root / "_tmp_vtc_inputs",
            keep_inputs=bool(vtc.get("keep_inputs", False)),
        )
