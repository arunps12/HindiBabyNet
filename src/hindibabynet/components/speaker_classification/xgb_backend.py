"""
XGB backend adapter for Stage 03.

Wraps the existing :class:`SpeakerClassification` monolithic pipeline
behind the :class:`ClassificationBackend` interface.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.hindibabynet.components.speaker_classification.base import ClassificationBackend
from src.hindibabynet.components.speaker_classification.metadata import (
    make_run_info,
    utcnow_iso,
    write_run_info,
)
from src.hindibabynet.components.speaker_classification.output_checks import is_xgb_complete
from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.entity.config_entity import SpeakerClassificationConfig
from src.hindibabynet.logging.logger import get_logger

logger = get_logger(__name__)


class XGBBackend(ClassificationBackend):
    """
    Native HindiBabyNet backend: VAD → diarization → eGeMAPS → XGBoost.

    Delegates to :class:`SpeakerClassification` from ``_xgb_core``.
    """

    def __init__(self, cfg: ConfigurationManager) -> None:
        self._cfg = cfg

    # -- protocol ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "xgb"

    def run_participant(
        self,
        wav_path: Path,
        participant_id: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        # Lazy import to avoid heavy torch/opensmile load at dispatch time
        from src.hindibabynet.components.speaker_classification._xgb_core import (
            SpeakerClassification,
        )

        sc_config = self._build_config(participant_id, output_dir)
        component = SpeakerClassification(sc_config)

        started = utcnow_iso()
        t0 = time.monotonic()
        status = "success"
        error_msg = ""

        try:
            artifact = component.run(
                analysis_wav_path=wav_path,
                participant_id=participant_id,
            )
        except Exception as exc:
            status = "failed"
            error_msg = str(exc)
            raise
        finally:
            runtime = round(time.monotonic() - t0, 2)
            info = make_run_info(
                participant_id=participant_id,
                backend="xgb",
                input_wav=wav_path,
                output_dir=output_dir,
                started_at=started,
                finished_at=utcnow_iso(),
                runtime_sec=runtime,
                status=status,
                **({"error": error_msg} if error_msg else {}),
            )
            write_run_info(output_dir, info)

        return info

    def is_complete(self, participant_id: str, output_dir: Path) -> bool:
        return is_xgb_complete(participant_id, output_dir)

    def expected_outputs(self, participant_id: str) -> list[str]:
        pid = participant_id
        return [
            f"{pid}_main_female.wav",
            f"{pid}_main_male.wav",
            f"{pid}_child.wav",
            f"{pid}_background.wav",
            f"{pid}.TextGrid",
            f"{pid}_segments.parquet",
            f"{pid}_summary.json",
            "run_info.json",
        ]

    # -- internal ----------------------------------------------------------

    def _build_config(
        self,
        participant_id: str,
        output_dir: Path,
    ) -> SpeakerClassificationConfig:
        """
        Construct the internal :class:`SpeakerClassificationConfig` so that
        all outputs land in *output_dir*.
        """
        xgb = self._cfg.get_xgb_params()
        pid = participant_id

        return SpeakerClassificationConfig(
            artifacts_dir=output_dir,
            model_path=Path(xgb.get("model_path", "models/xgb_egemaps.pkl")),
            class_names=list(
                xgb.get("class_names", ["adult_male", "adult_female", "child", "background"])
            ),
            egemaps_dim=int(xgb.get("egemaps_dim", 88)),
            merge_gap_sec=float(xgb.get("merge_gap_sec", 0.3)),
            min_segment_sec=float(xgb.get("min_segment_sec", 0.2)),
            classify_win_sec=float(xgb.get("classify_win_sec", 1.0)),
            classify_hop_sec=float(xgb.get("classify_hop_sec", 0.5)),
            diarization_model=str(
                xgb.get("diarization_model", "pyannote/speaker-diarization-3.1")
            ),
            min_speakers=1,
            max_speakers=3,
            output_audio_root=output_dir.parent,  # <output_root>/xgb
            classified_segments_parquet_path=output_dir / f"{pid}_segments.parquet",
            main_female_parquet_path=output_dir / f"{pid}_main_female.parquet",
            main_male_parquet_path=output_dir / f"{pid}_main_male.parquet",
            child_parquet_path=output_dir / f"{pid}_child.parquet",
            background_parquet_path=output_dir / f"{pid}_background.parquet",
            summary_json_path=output_dir / f"{pid}_summary.json",
            textgrid_path=output_dir / f"{pid}.TextGrid",
            main_female_wav_path=output_dir / f"{pid}_main_female.wav",
            main_male_wav_path=output_dir / f"{pid}_main_male.wav",
            child_wav_path=output_dir / f"{pid}_child.wav",
            background_wav_path=output_dir / f"{pid}_background.wav",
        )
