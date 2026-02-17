"""
Stage 04: Speaker Diarization

Runs chunked pyannote diarization on an analysis-ready WAV and outputs
a parquet file with speaker turns (chunk_id, speaker_id_local, start_sec,
end_sec, duration_sec).
"""
from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.hindibabynet.entity.artifact_entity import DiarizationArtifact
from src.hindibabynet.entity.config_entity import DiarizationConfig
from src.hindibabynet.exception.exception import wrap_exception
from src.hindibabynet.logging.logger import get_logger
from src.hindibabynet.utils.audio_utils import write_wav_chunk
from src.hindibabynet.utils.io_utils import ensure_dir, write_json, write_parquet

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chunking helper (same logic as old speaker_classification.py)
# ---------------------------------------------------------------------------
def _make_chunks(
    duration_sec: float, chunk_sec: float, overlap_sec: float,
) -> List[Tuple[int, float, float]]:
    """Yield (chunk_id, chunk_start, chunk_end) with overlap."""
    step = chunk_sec - overlap_sec
    assert step > 0, "chunk_sec must be > overlap_sec"
    chunks: List[Tuple[int, float, float]] = []
    t = 0.0
    chunk_id = 0
    while t < duration_sec:
        s = t
        e = min(t + chunk_sec, duration_sec)
        chunks.append((chunk_id, s, e))
        if e >= duration_sec:
            break
        t += step
        chunk_id += 1
    return chunks


def diarize_wav(
    wav_path: Path,
    diar_pipeline,
    chunk_sec: float,
    overlap_sec: float,
    min_speakers: int,
    max_speakers: int,
    tmp_dir: Path,
) -> pd.DataFrame:
    """Run chunked diarization on a single WAV, returning speaker turns."""
    info = sf.info(str(wav_path))
    full_duration = float(info.duration)
    chunks = _make_chunks(full_duration, chunk_sec, overlap_sec)

    all_rows: List[Dict[str, Any]] = []
    for chunk_id, chunk_start, chunk_end in chunks:
        chunk_wav = tmp_dir / f"{wav_path.stem}_chunk{chunk_id:04d}.wav"
        out = write_wav_chunk(wav_path, chunk_wav, chunk_start, chunk_end, logger=logger)
        if out is None:
            continue
        try:
            diar = diar_pipeline(
                {"audio": str(chunk_wav)},
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            for seg, _, spk in diar.itertracks(yield_label=True):
                s = float(seg.start) + chunk_start
                e = float(seg.end) + chunk_start
                if e <= s:
                    continue
                all_rows.append({
                    "chunk_id": int(chunk_id),
                    "speaker_id_local": spk,
                    "start_sec": s,
                    "end_sec": e,
                    "duration_sec": float(e - s),
                })
        except Exception as exc:
            logger.warning(f"Diarization failed chunk {chunk_id}: {exc}")
        finally:
            try:
                chunk_wav.unlink(missing_ok=True)
            except Exception:
                pass

    if not all_rows:
        return pd.DataFrame(
            columns=["chunk_id", "speaker_id_local", "start_sec", "end_sec", "duration_sec"]
        )
    return (
        pd.DataFrame(all_rows)
        .sort_values(["start_sec", "end_sec"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Component class
# ---------------------------------------------------------------------------
class Diarization:
    """Stage 04 — Speaker Diarization."""

    def __init__(self, config: DiarizationConfig):
        self.config = config
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        from dotenv import load_dotenv
        from pyannote.audio import Pipeline

        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")

        user = os.environ.get("USER", "user")
        scratch_cache = f"/scratch/users/{user}/.cache/huggingface"
        os.environ.setdefault("HF_HOME", scratch_cache)
        os.environ.setdefault("HF_HUB_CACHE", f"{scratch_cache}/hub")
        os.environ.setdefault("TRANSFORMERS_CACHE", f"{scratch_cache}/transformers")
        os.environ.setdefault("PYANNOTE_DISABLE_NOTEBOOK", "1")

        warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
        warnings.filterwarnings(
            "ignore",
            message=r".*std\(\): degrees of freedom is <= 0.*",
            category=UserWarning,
        )

        logger.info(f"Loading diarization pipeline: {self.config.diarization_model}")
        pipeline = Pipeline.from_pretrained(
            self.config.diarization_model, use_auth_token=hf_token
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        logger.info(f"Diarization pipeline loaded on {device}")
        self._pipeline = pipeline
        return self._pipeline

    def run(
        self,
        analysis_wav_path: Path,
        participant_id: str,
    ) -> DiarizationArtifact:
        """Run diarization on one analysis WAV.

        Outputs
        -------
        ``<pid>_diarization.parquet`` with columns:

        =================  =======  ====================================
        Column             Dtype    Description
        =================  =======  ====================================
        chunk_id           int64    Diarization chunk index
        speaker_id_local   object   Speaker label (e.g. SPEAKER_00)
        start_sec          float64  Turn start in analysis WAV (seconds)
        end_sec            float64  Turn end in analysis WAV (seconds)
        duration_sec       float64  ``end_sec - start_sec``
        =================  =======  ====================================
        """
        try:
            ensure_dir(self.config.artifacts_dir)
            ensure_dir(self.config.tmp_dir)

            if not analysis_wav_path.exists():
                raise FileNotFoundError(f"analysis wav not found: {analysis_wav_path}")

            wav_info = sf.info(str(analysis_wav_path))
            full_duration = float(wav_info.duration)
            logger.info(
                f"[{participant_id}] Diarization | wav={analysis_wav_path} "
                f"dur={full_duration / 3600:.2f}h  chunk={self.config.chunk_sec}s"
            )

            pipeline = self._load_pipeline()

            turns_df = diarize_wav(
                wav_path=analysis_wav_path,
                diar_pipeline=pipeline,
                chunk_sec=self.config.chunk_sec,
                overlap_sec=self.config.overlap_sec,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
                tmp_dir=self.config.tmp_dir,
            )
            logger.info(f"[{participant_id}] Diarization turns: {len(turns_df)}")

            # Save parquet
            write_parquet(turns_df, self.config.diarization_parquet_path)

            # Summary
            n_speakers = turns_df["speaker_id_local"].nunique() if not turns_df.empty else 0
            summary = {
                "participant_id": participant_id,
                "analysis_wav": str(analysis_wav_path),
                "duration_sec": full_duration,
                "n_turns": len(turns_df),
                "n_speakers": n_speakers,
                "speakers": sorted(turns_df["speaker_id_local"].unique().tolist()) if not turns_df.empty else [],
            }
            write_json(summary, self.config.summary_json_path)

            # Cleanup tmp
            try:
                shutil.rmtree(self.config.tmp_dir, ignore_errors=True)
            except Exception:
                pass

            logger.info(
                f"[{participant_id}] Diarization DONE | turns={len(turns_df)} speakers={n_speakers}"
            )

            return DiarizationArtifact(
                diarization_parquet_path=self.config.diarization_parquet_path,
                summary_json_path=self.config.summary_json_path,
                n_turns=len(turns_df),
                n_speakers=n_speakers,
            )

        except Exception as e:
            raise wrap_exception(
                "Diarization failed",
                e,
                context={
                    "participant_id": participant_id,
                    "analysis_wav_path": str(analysis_wav_path),
                },
            )
