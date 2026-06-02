from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from hindibabynet_pipeline.utils.io_utils import read_yaml, make_run_id
from hindibabynet_pipeline.entity.config_entity import (
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
    params_path: Path = Path("configs/params.yaml")

    def __post_init__(self):
        self.config_path = Path(self.config_path)
        self.params_path = Path(self.params_path)
        if self.params_path == Path("configs/params.yaml") and self.config_path != Path("configs/config.yaml"):
            sibling_params = self.config_path.with_name("params.yaml")
            if sibling_params.exists():
                self.params_path = sibling_params

        self.runtime_config: Dict[str, Any] = read_yaml(self.config_path)
        self.params: Dict[str, Any] = read_yaml(self.params_path) if self.params_path.exists() else {}
        self.config: Dict[str, Any] = self._merge_dicts(self.runtime_config, self.params)

    def _merge_dicts(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _require(self, mapping: dict, key: str, section: str) -> Any:
        if key not in mapping:
            raise ValueError(f"Missing required config key: {section}.{key}")
        return mapping[key]

    def _require_path(self, key: str) -> Path:
        paths = self.runtime_config.get("paths", {})
        if key not in paths:
            raise ValueError(f"Missing required config key: paths.{key}")
        return Path(paths[key])

    # ---- helpers ----
    def get_logs_root(self) -> Path:
        return Path(self.runtime_config.get("logs_root", "logs"))

    def get_artifacts_root(self) -> Path:
        return Path(self.runtime_config.get("artifacts_root", "artifacts/runs"))

    def make_run_id(self) -> str:
        return make_run_id()

    # ---- Backend selector ----
    def get_speaker_classification_backend(self) -> str:
        """Return 'xgb' (default) or 'vtc'."""
        sc = self.runtime_config.get("speaker_classification", {})
        return str(sc.get("backend", "xgb")).lower()

    # ---- Unified param accessors (support old flat + new nested config) ---
    def get_xgb_params(self) -> dict:
        params = self.params.get("xgb", {})
        runtime = self.runtime_config.get("speaker_classification", {}).get("xgb", {})
        vad = params.get("vad", {})
        diarization = params.get("diarization", {})
        segmentation = params.get("segmentation", {})
        classification = params.get("classification", {})
        merged = {
            **params,
            **runtime,
            "vad_aggressiveness": vad.get("aggressiveness", runtime.get("vad_aggressiveness", 2)),
            "vad_frame_ms": vad.get("frame_ms", runtime.get("vad_frame_ms", 30)),
            "vad_min_region_ms": vad.get("min_region_ms", runtime.get("vad_min_region_ms", 300)),
            "diarization_model": diarization.get("model", runtime.get("diarization_model", "pyannote/speaker-diarization-3.1")),
            "chunk_sec": diarization.get("chunk_sec", runtime.get("chunk_sec", 900.0)),
            "overlap_sec": diarization.get("overlap_sec", runtime.get("overlap_sec", 10.0)),
            "min_speakers": diarization.get("min_speakers", runtime.get("min_speakers", 2)),
            "max_speakers": diarization.get("max_speakers", runtime.get("max_speakers", 4)),
            "merge_gap_sec": segmentation.get("merge_gap_sec", runtime.get("merge_gap_sec", 0.3)),
            "min_segment_sec": segmentation.get("min_segment_sec", runtime.get("min_segment_sec", 0.2)),
            "classify_win_sec": classification.get("win_sec", runtime.get("classify_win_sec", 1.0)),
            "classify_hop_sec": classification.get("hop_sec", runtime.get("classify_hop_sec", 0.5)),
        }
        return merged

    def get_vtc_params(self) -> dict:
        runtime = self.runtime_config.get("speaker_classification", {}).get("vtc", {})
        params = self.params.get("vtc", {})
        return {**params, **runtime}

    def get_audio_preparation_params(self) -> dict:
        return self.params.get("audio_preparation", {})

    def get_annotation_params(self) -> dict:
        return self.params.get("annotation", {})

    def get_textgrid_params(self) -> dict:
        return self.params.get("textgrid", {})

    def get_classification_output_root(self) -> Path:
        sc = self.runtime_config.get("speaker_classification", {})
        if "output_root" in sc:
            return Path(sc["output_root"])
        return self._require_path("classification_output_root")

    def get_processed_audio_root(self) -> Path:
        return self._require_path("prepared_audio_root")

    def get_raw_joined_audio_root(self) -> Path:
        return self._require_path("raw_joined_audio_root")

    def get_textgrid_output_root(self) -> Path:
        return self._require_path("textgrid_output_root")

    def get_manual_annotation_root(self) -> Path:
        return self._require_path("manual_annotation_root")

    def get_evaluation_output_root(self) -> Path:
        return self._require_path("evaluation_output_root")

    # ---- Stage 01: Data Ingestion ----
    def get_data_ingestion_config(self, run_id: str | None = None) -> DataIngestionConfig:
        run_id = run_id or make_run_id()
        artifacts_root = self.get_artifacts_root()
        di = self.runtime_config.get("data_ingestion", {})

        raw_audio_root = di.get("raw_audio_root") or str(self._require_path("raw_audio_root"))

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
        ap_runtime = self.runtime_config.get("audio_preparation", {})
        ap_params = self.get_audio_preparation_params()
        artifacts_root = self.get_artifacts_root()

        artifacts_dir = artifacts_root / run_id / "audio_preparation"
        processed_root = self.get_processed_audio_root()
        processed_dir = processed_root / recording_id

        return AudioPreparationConfig(
            artifacts_dir=artifacts_dir,
            processed_audio_root=processed_root,
            target_sr=int(ap_params.get("target_sr", 16000)),
            to_mono=bool(ap_params.get("convert_to_mono", True)),
            target_peak_dbfs=float(ap_params.get("target_peak_dbfs", -1.0)),
            combine_gap_sec=float(ap_params.get("combine_gap_sec", 0.0)),
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
        artifacts_root = self.get_artifacts_root()
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
        artifacts_root = self.get_artifacts_root()
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
        artifacts_root = self.get_artifacts_root()
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
        artifacts_root = self.get_artifacts_root()

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

