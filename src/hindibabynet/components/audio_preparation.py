from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import soundfile as sf

from src.hindibabynet.entity.config_entity import AudioPreparationConfig
from src.hindibabynet.entity.artifact_entity import AudioPreparationArtifact
from src.hindibabynet.exception.exception import wrap_exception
from src.hindibabynet.logging.logger import get_logger
from src.hindibabynet.utils.io_utils import ensure_dir, write_parquet, write_json
from src.hindibabynet.utils.audio_utils import (
    concatenate_wavs_streaming,
    ensure_mono_16k_wav_streaming,
    peak_normalize_wav_streaming,
)

logger = get_logger(__name__)


class AudioPreparation:
    """
    Stage 02: Produce one analysis-ready WAV:
      - mono (optional)
      - 16 kHz
      - peak normalized (target_peak_dbfs)

    Supports two input modes:

    1) single wav:
        run(wav_path=..., recording_id=optional)

    2) recordings_df / parquet:
        run(recordings_df=..., participant_id=..., recording_id=optional)

       In this mode it combines ALL wavs for the participant_id (ignores session_date).
    """

    def __init__(self, config: AudioPreparationConfig):
        self.config = config

    @staticmethod
    def _infer_recording_id_from_wav(wav_path: Path) -> str:
        return wav_path.stem

    def _manifest_single_wav(self, wav_path: Path, recording_id: str) -> pd.DataFrame:
        info = sf.info(str(wav_path))
        return pd.DataFrame(
            [{
                "participant_id": recording_id,  
                "recording_id": recording_id,
                "source_index": 0,
                "source_path": str(wav_path),
                "source_recording_id": wav_path.stem,
                "combined_start_sec": 0.0,
                "combined_end_sec": float(info.duration),
                "source_duration_sec": float(info.duration),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
            }]
        )

    def _combine_for_participant(self, recordings_df: pd.DataFrame, participant_id: str, recording_id: str) -> pd.DataFrame:
        """
        Combine ALL wav files for the given participant_id (ignores session_date).
        Returns a manifest with per-source offsets and combined_raw_path.
        """
        required = {"participant_id", "recording_id", "path"}
        missing = required - set(recordings_df.columns)
        if missing:
            raise ValueError(f"recordings_df missing columns: {sorted(missing)}")

        g = recordings_df[recordings_df["participant_id"] == participant_id].copy()
        if g.empty:
            raise ValueError(f"No rows for participant_id={participant_id}")

        #  session_date then recording_id if present
        sort_cols = [c for c in ["session_date", "recording_id"] if c in g.columns]
        g = g.sort_values(sort_cols if sort_cols else ["recording_id"]).reset_index(drop=True)

        wavs = [Path(p) for p in g["path"].astype(str).tolist()]

        tmp_dir = self.config.processed_audio_root / "_tmp_combine" / recording_id
        ensure_dir(tmp_dir)
        combined_raw = tmp_dir / f"{recording_id}_combined_raw.wav"

        logger.info(f"[combine] participant_id={participant_id} | n_files={len(wavs)} -> {combined_raw}")

        sr, ch, manifest_rows = concatenate_wavs_streaming(
            wav_paths=wavs,
            out_path=combined_raw,
            gap_sec=self.config.combine_gap_sec,
        )

        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df["participant_id"] = participant_id
        manifest_df["recording_id"] = recording_id
        manifest_df["combined_raw_path"] = str(combined_raw)
        return manifest_df

    def run(
        self,
        wav_path: Optional[Union[str, Path]] = None,
        recordings_df: Optional[pd.DataFrame] = None,
        participant_id: Optional[str] = None,
        recording_id: Optional[str] = None,
        cleanup_tmp: bool = True,
    ) -> AudioPreparationArtifact:
        """
        Public users: provide wav_path.
        Your combine mode: provide recordings_df + participant_id.

        Returns: AudioPreparationArtifact (analysis_wav_path + manifest + meta)
        """
        try:
            ensure_dir(self.config.artifacts_dir)
            ensure_dir(self.config.analysis_wav_path.parent)

            # ---------- Mode 1: single wav ----------
            if wav_path is not None:
                wav_path = Path(wav_path)
                if not wav_path.exists():
                    raise FileNotFoundError(f"wav not found: {wav_path}")

                rec_id = recording_id or self._infer_recording_id_from_wav(wav_path)

                manifest_df = self._manifest_single_wav(wav_path, recording_id=rec_id)

                tmp_dir = self.config.processed_audio_root / "_tmp_prep" / rec_id
                ensure_dir(tmp_dir)
                tmp_16k = tmp_dir / f"{rec_id}_mono16k.wav"

                logger.info(f"[audio_prep] single wav -> mono={self.config.to_mono} sr={self.config.target_sr}")
                conv_stats = ensure_mono_16k_wav_streaming(
                    in_path=wav_path,
                    out_path=tmp_16k,
                    target_sr=self.config.target_sr,
                    to_mono=self.config.to_mono,
                )

                norm_stats = peak_normalize_wav_streaming(
                    in_path=tmp_16k,
                    out_path=self.config.analysis_wav_path,
                    target_peak_dbfs=self.config.target_peak_dbfs,
                )

                write_parquet(manifest_df, self.config.manifest_parquet_path)

                out_info = sf.info(str(self.config.analysis_wav_path))
                meta = {
                    "mode": "single_wav",
                    "input_wav": str(wav_path),
                    "analysis_wav": str(self.config.analysis_wav_path),
                    "convert": conv_stats,
                    "normalize": norm_stats,
                    "output_info": {
                        "duration_sec": float(out_info.duration),
                        "sample_rate": int(out_info.samplerate),
                        "channels": int(out_info.channels),
                        "frames": int(out_info.frames),
                        "format": str(out_info.format),
                        "subtype": str(out_info.subtype),
                    },
                }
                write_json(meta, self.config.analysis_meta_json_path)

                if cleanup_tmp:
                    try:
                        tmp_16k.unlink(missing_ok=True)
                    except Exception:
                        pass

                return AudioPreparationArtifact(
                    analysis_wav_path=self.config.analysis_wav_path,
                    manifest_parquet_path=self.config.manifest_parquet_path,
                    analysis_meta_json_path=self.config.analysis_meta_json_path,
                    duration_sec=float(out_info.duration),
                    sample_rate=int(out_info.samplerate),
                    channels=int(out_info.channels),
                )

            # ---------- Mode 2: combine by participant_id ----------
            if recordings_df is None:
                raise ValueError("Provide either wav_path (single wav) or recordings_df + participant_id (combine).")
            if participant_id is None:
                raise ValueError("Combine mode requires participant_id.")
            rec_id = recording_id or str(participant_id)

            manifest_df = self._combine_for_participant(
                recordings_df=recordings_df,
                participant_id=str(participant_id),
                recording_id=rec_id,
            )

            combined_raw_path = Path(manifest_df["combined_raw_path"].iloc[0])

            tmp_dir = self.config.processed_audio_root / "_tmp_prep" / rec_id
            ensure_dir(tmp_dir)
            tmp_16k = tmp_dir / f"{rec_id}_mono16k.wav"

            logger.info(f"[audio_prep] combine -> mono={self.config.to_mono} sr={self.config.target_sr}")
            conv_stats = ensure_mono_16k_wav_streaming(
                in_path=combined_raw_path,
                out_path=tmp_16k,
                target_sr=self.config.target_sr,
                to_mono=self.config.to_mono,
            )

            norm_stats = peak_normalize_wav_streaming(
                in_path=tmp_16k,
                out_path=self.config.analysis_wav_path,
                target_peak_dbfs=self.config.target_peak_dbfs,
            )

            write_parquet(manifest_df, self.config.manifest_parquet_path)

            out_info = sf.info(str(self.config.analysis_wav_path))
            meta = {
                "mode": "combine_by_participant_id",
                "participant_id": str(participant_id),
                "analysis_wav": str(self.config.analysis_wav_path),
                "combined_raw_path": str(combined_raw_path),
                "convert": conv_stats,
                "normalize": norm_stats,
                "output_info": {
                    "duration_sec": float(out_info.duration),
                    "sample_rate": int(out_info.samplerate),
                    "channels": int(out_info.channels),
                    "frames": int(out_info.frames),
                    "format": str(out_info.format),
                    "subtype": str(out_info.subtype),
                },
            }
            write_json(meta, self.config.analysis_meta_json_path)

            # cleanup (optional)
            if cleanup_tmp:
                try:
                    tmp_16k.unlink(missing_ok=True)
                except Exception:
                    pass

            return AudioPreparationArtifact(
                analysis_wav_path=self.config.analysis_wav_path,
                manifest_parquet_path=self.config.manifest_parquet_path,
                analysis_meta_json_path=self.config.analysis_meta_json_path,
                duration_sec=float(out_info.duration),
                sample_rate=int(out_info.samplerate),
                channels=int(out_info.channels),
            )

        except Exception as e:
            raise wrap_exception(
                "Audio preparation failed",
                e,
                context={
                    "analysis_wav_path": str(self.config.analysis_wav_path),
                    "manifest_parquet_path": str(self.config.manifest_parquet_path),
                },
            )
