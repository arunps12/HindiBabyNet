"""
Stage 05: VAD ∩ Diarization Intersection

Takes VAD and diarization parquets, intersects them (sweep-line), and
outputs a speech-segments parquet with per-segment provenance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.hindibabynet.entity.artifact_entity import IntersectionArtifact
from src.hindibabynet.entity.config_entity import IntersectionConfig
from src.hindibabynet.exception.exception import wrap_exception
from src.hindibabynet.logging.logger import get_logger
from src.hindibabynet.utils.io_utils import ensure_dir, write_json, write_parquet

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core intersection logic (exact notebook sweep-line)
# ---------------------------------------------------------------------------
def intersect_turns_with_vad(
    turns_df: pd.DataFrame,
    vad_df: pd.DataFrame,
    min_keep_sec: float = 0.0,
) -> pd.DataFrame:
    """Intersect diarization turns with VAD regions (sweep-line).

    Returns a DataFrame with columns:

    =================  =======  ============================================
    Column             Dtype    Description
    =================  =======  ============================================
    start_sec          float64  Intersected segment start (seconds)
    end_sec            float64  Intersected segment end (seconds)
    duration_sec       float64  ``end_sec - start_sec``
    chunk_id           int64    Diarization chunk id (from diar parquet)
    speaker_id_local   object   Speaker label (from diar parquet)
    vad_region_id      int64    Source VAD region index
    =================  =======  ============================================
    """
    if turns_df.empty or vad_df.empty:
        return pd.DataFrame(
            columns=[
                "start_sec", "end_sec", "duration_sec",
                "chunk_id", "speaker_id_local", "vad_region_id",
            ]
        )

    turns_df = turns_df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    vad_df = vad_df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

    diar_arr = turns_df[["start_sec", "end_sec", "chunk_id", "speaker_id_local"]].to_numpy()

    # Prepare VAD array with region_id
    vad_starts = vad_df["start_sec"].to_numpy(dtype=float)
    vad_ends = vad_df["end_sec"].to_numpy(dtype=float)
    vad_ids = vad_df["region_id"].to_numpy() if "region_id" in vad_df.columns else np.arange(len(vad_df))

    i, j = 0, 0
    rows: List[Dict] = []

    while i < len(diar_arr) and j < len(vad_df):
        ds, de = float(diar_arr[i, 0]), float(diar_arr[i, 1])
        vs, ve = float(vad_starts[j]), float(vad_ends[j])

        s = max(ds, vs)
        e = min(de, ve)
        if s < e:
            dur = e - s
            if dur >= min_keep_sec:
                rows.append({
                    "start_sec": s,
                    "end_sec": e,
                    "duration_sec": dur,
                    "chunk_id": int(diar_arr[i, 2]),
                    "speaker_id_local": str(diar_arr[i, 3]),
                    "vad_region_id": int(vad_ids[j]),
                })
        if de <= ve:
            i += 1
        else:
            j += 1

    if not rows:
        return pd.DataFrame(
            columns=[
                "start_sec", "end_sec", "duration_sec",
                "chunk_id", "speaker_id_local", "vad_region_id",
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------
class Intersection:
    """Stage 05 — VAD ∩ Diarization intersection."""

    def __init__(self, config: IntersectionConfig):
        self.config = config

    def run(
        self,
        vad_parquet_path: Path,
        diarization_parquet_path: Path,
        participant_id: str,
    ) -> IntersectionArtifact:
        """Intersect VAD and diarization outputs.

        Outputs
        -------
        ``<pid>_speech_segments.parquet`` with columns:

        =================  =======  ============================================
        Column             Dtype    Description
        =================  =======  ============================================
        segment_id         int64    0-based sequential index
        start_sec          float64  Segment start time (seconds)
        end_sec            float64  Segment end time (seconds)
        duration_sec       float64  ``end_sec - start_sec``
        chunk_id           int64    Source diarization chunk id
        speaker_id_local   object   Source speaker label
        vad_region_id      int64    Source VAD region index
        =================  =======  ============================================
        """
        try:
            ensure_dir(self.config.artifacts_dir)

            # Load inputs
            vad_df = pd.read_parquet(vad_parquet_path)
            diar_df = pd.read_parquet(diarization_parquet_path)
            logger.info(
                f"[{participant_id}] Intersection | "
                f"vad_regions={len(vad_df)}  diar_turns={len(diar_df)}"
            )

            # Intersect
            speech_df = intersect_turns_with_vad(
                diar_df, vad_df, min_keep_sec=self.config.min_segment_sec,
            )

            # Add segment_id
            speech_df = speech_df.sort_values("start_sec").reset_index(drop=True)
            speech_df.insert(0, "segment_id", range(len(speech_df)))

            logger.info(f"[{participant_id}] Speech segments after intersection: {len(speech_df)}")

            # Save
            write_parquet(speech_df, self.config.speech_segments_parquet_path)

            total_speech = float(speech_df["duration_sec"].sum()) if not speech_df.empty else 0.0
            summary = {
                "participant_id": participant_id,
                "n_vad_regions": len(vad_df),
                "n_diarization_turns": len(diar_df),
                "n_speech_segments": len(speech_df),
                "total_speech_sec": total_speech,
                "min_segment_sec_filter": self.config.min_segment_sec,
            }
            write_json(summary, self.config.summary_json_path)

            logger.info(
                f"[{participant_id}] Intersection DONE | "
                f"segments={len(speech_df)}  speech={total_speech:.1f}s"
            )

            return IntersectionArtifact(
                speech_segments_parquet_path=self.config.speech_segments_parquet_path,
                summary_json_path=self.config.summary_json_path,
                n_segments=len(speech_df),
                total_speech_sec=total_speech,
            )

        except Exception as e:
            raise wrap_exception(
                "VAD-Diarization intersection failed",
                e,
                context={"participant_id": participant_id},
            )
