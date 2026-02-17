"""
Stage 03: Voice Activity Detection (VAD)

Runs WebRTC VAD on an analysis-ready WAV and outputs a parquet file
with speech regions (start_sec, end_sec, duration_sec).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import soundfile as sf

from src.hindibabynet.entity.artifact_entity import VADArtifact
from src.hindibabynet.entity.config_entity import VADConfig
from src.hindibabynet.exception.exception import wrap_exception
from src.hindibabynet.logging.logger import get_logger
from src.hindibabynet.utils.audio_utils import webrtc_vad_regions
from src.hindibabynet.utils.io_utils import ensure_dir, write_json, write_parquet

logger = get_logger(__name__)


class VAD:
    """Stage 03 — Voice Activity Detection."""

    def __init__(self, config: VADConfig):
        self.config = config

    def run(
        self,
        analysis_wav_path: Path,
        participant_id: str,
    ) -> VADArtifact:
        """Run VAD on one analysis WAV.

        Outputs
        -------
        ``<pid>_vad.parquet`` with columns:

        ============  =======  ==========================================
        Column        Dtype    Description
        ============  =======  ==========================================
        region_id     int64    0-based sequential index
        start_sec     float64  Region start time in analysis WAV (seconds)
        end_sec       float64  Region end time in analysis WAV (seconds)
        duration_sec  float64  ``end_sec - start_sec``
        ============  =======  ==========================================
        """
        try:
            ensure_dir(self.config.artifacts_dir)

            if not analysis_wav_path.exists():
                raise FileNotFoundError(f"analysis wav not found: {analysis_wav_path}")

            wav_info = sf.info(str(analysis_wav_path))
            full_duration = float(wav_info.duration)
            logger.info(
                f"[{participant_id}] VAD | wav={analysis_wav_path} "
                f"dur={full_duration / 3600:.2f}h sr={wav_info.samplerate}"
            )

            # Run WebRTC VAD
            logger.info(
                f"[{participant_id}] Running WebRTC VAD "
                f"(aggressiveness={self.config.vad_aggressiveness})"
            )
            vad_intervals: List[Tuple[float, float]] = webrtc_vad_regions(
                analysis_wav_path,
                aggressiveness=self.config.vad_aggressiveness,
                frame_ms=self.config.vad_frame_ms,
                min_region_ms=self.config.vad_min_region_ms,
            )
            logger.info(f"[{participant_id}] VAD regions: {len(vad_intervals)}")

            # Build DataFrame
            rows = []
            for i, (s, e) in enumerate(vad_intervals):
                rows.append({
                    "region_id": i,
                    "start_sec": float(s),
                    "end_sec": float(e),
                    "duration_sec": float(e - s),
                })
            vad_df = pd.DataFrame(
                rows,
                columns=["region_id", "start_sec", "end_sec", "duration_sec"],
            )

            # Save parquet
            write_parquet(vad_df, self.config.vad_parquet_path)

            # Summary
            total_speech = float(vad_df["duration_sec"].sum()) if not vad_df.empty else 0.0
            summary = {
                "participant_id": participant_id,
                "analysis_wav": str(analysis_wav_path),
                "duration_sec": full_duration,
                "n_vad_regions": len(vad_df),
                "total_speech_sec": total_speech,
                "speech_ratio": total_speech / full_duration if full_duration > 0 else 0.0,
            }
            write_json(summary, self.config.summary_json_path)

            logger.info(
                f"[{participant_id}] VAD DONE | regions={len(vad_df)} "
                f"speech={total_speech:.1f}s ({total_speech / 3600:.2f}h)"
            )

            return VADArtifact(
                vad_parquet_path=self.config.vad_parquet_path,
                summary_json_path=self.config.summary_json_path,
                n_regions=len(vad_df),
                total_speech_sec=total_speech,
            )

        except Exception as e:
            raise wrap_exception(
                "VAD failed",
                e,
                context={
                    "participant_id": participant_id,
                    "analysis_wav_path": str(analysis_wav_path),
                },
            )
