from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import soundfile as sf

from hindibabynet_pipeline.entity.config_entity import AudioPreparationConfig
from hindibabynet_pipeline.entity.artifact_entity import AudioPreparationArtifact
from hindibabynet_pipeline.exception.exception import wrap_exception
from hindibabynet_pipeline.logging.logger import get_logger
from hindibabynet_pipeline.utils.io_utils import ensure_dir, write_parquet, write_json
from hindibabynet_pipeline.utils.audio_utils import (
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
        if not self.config.join_multiple_files and len(g) > 1:
            raise ValueError(
                "join_multiple_files is false, but multiple WAV files were provided for one participant. "
                "Use per-recording mode or enable joining."
            )

        #  session_date then recording_id if present
        sort_cols = [c for c in ["session_date", "recording_id"] if c in g.columns]
        g = g.sort_values(sort_cols if sort_cols else ["recording_id"]).reset_index(drop=True)

        wavs = [Path(p) for p in g["path"].astype(str).tolist()]

        if len(wavs) == 1:
            manifest_df = self._manifest_single_wav(wavs[0], recording_id=recording_id)
            manifest_df["participant_id"] = participant_id
            manifest_df["recording_id"] = recording_id
            manifest_df["combined_raw_path"] = str(wavs[0])
            return manifest_df

        tmp_dir = self.config.processed_audio_root / "_tmp_combine" / recording_id
        ensure_dir(tmp_dir)
        combined_raw = (
            self.config.raw_joined_wav_path
            if self.config.save_raw_joined_audio
            else tmp_dir / f"{recording_id}_combined_raw.wav"
        )
        ensure_dir(combined_raw.parent)

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

    def _prepare_audio(self, source_path: Path, recording_id: str) -> tuple[Path, dict, dict]:
        tmp_dir = self.config.processed_audio_root / "_tmp_prep" / recording_id
        ensure_dir(tmp_dir)

        converted_path = tmp_dir / f"{recording_id}_converted.wav"
        prepared_path = (
            self.config.analysis_wav_path
            if self.config.save_prepared_audio
            else tmp_dir / f"{recording_id}_prepared.wav"
        )
        ensure_dir(prepared_path.parent)

        source_info = sf.info(str(source_path))
        target_sr = self.config.target_sr if self.config.resample else int(source_info.samplerate)

        logger.info(
            "[audio_prep] process -> join=%s mono=%s resample=%s normalize=%s save_prepared=%s target_sr=%s",
            self.config.join_multiple_files,
            self.config.to_mono,
            self.config.resample,
            self.config.normalize,
            self.config.save_prepared_audio,
            target_sr,
        )
        conv_stats = ensure_mono_16k_wav_streaming(
            in_path=source_path,
            out_path=converted_path,
            target_sr=target_sr,
            to_mono=self.config.to_mono,
        )

        if self.config.normalize:
            norm_stats = peak_normalize_wav_streaming(
                in_path=converted_path,
                out_path=prepared_path,
                target_peak_dbfs=self.config.target_peak_dbfs,
            )
        else:
            shutil.copyfile(converted_path, prepared_path)
            norm_stats = {"skipped": True}

        return prepared_path, conv_stats, norm_stats

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
            ensure_dir(self.config.processed_audio_root)
            ensure_dir(self.config.raw_joined_audio_root)

            # ---------- Mode 1: single wav ----------
            if wav_path is not None:
                wav_path = Path(wav_path)
                if not wav_path.exists():
                    raise FileNotFoundError(f"wav not found: {wav_path}")

                rec_id = recording_id or self._infer_recording_id_from_wav(wav_path)

                manifest_df = self._manifest_single_wav(wav_path, recording_id=rec_id)
                prepared_path, conv_stats, norm_stats = self._prepare_audio(
                    source_path=wav_path,
                    recording_id=rec_id,
                )

                write_parquet(manifest_df, self.config.manifest_parquet_path)

                out_info = sf.info(str(prepared_path))
                meta = {
                    "mode": "single_wav",
                    "input_wav": str(wav_path),
                    "analysis_wav": str(prepared_path),
                    "raw_joined_wav": None,
                    "convert": conv_stats,
                    "normalize": norm_stats,
                    "save_prepared_audio": self.config.save_prepared_audio,
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
                    converted_tmp = self.config.processed_audio_root / "_tmp_prep" / rec_id / f"{rec_id}_converted.wav"
                    converted_tmp.unlink(missing_ok=True)

                return AudioPreparationArtifact(
                    raw_joined_wav_path=None,
                    analysis_wav_path=prepared_path,
                    manifest_parquet_path=self.config.manifest_parquet_path,
                    analysis_meta_json_path=self.config.analysis_meta_json_path,
                    duration_sec=float(out_info.duration),
                    sample_rate=int(out_info.samplerate),
                    channels=int(out_info.channels),
                    prepared_audio_saved=self.config.save_prepared_audio,
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
            prepared_path, conv_stats, norm_stats = self._prepare_audio(
                source_path=combined_raw_path,
                recording_id=rec_id,
            )

            write_parquet(manifest_df, self.config.manifest_parquet_path)

            out_info = sf.info(str(prepared_path))
            meta = {
                "mode": "combine_by_participant_id" if self.config.join_multiple_files else "single_recording_from_dataframe",
                "participant_id": str(participant_id),
                "analysis_wav": str(prepared_path),
                "combined_raw_path": str(combined_raw_path) if self.config.save_raw_joined_audio else None,
                "convert": conv_stats,
                "normalize": norm_stats,
                "save_raw_joined_audio": self.config.save_raw_joined_audio,
                "save_prepared_audio": self.config.save_prepared_audio,
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
                converted_tmp = self.config.processed_audio_root / "_tmp_prep" / rec_id / f"{rec_id}_converted.wav"
                converted_tmp.unlink(missing_ok=True)

            return AudioPreparationArtifact(
                raw_joined_wav_path=combined_raw_path if self.config.save_raw_joined_audio else None,
                analysis_wav_path=prepared_path,
                manifest_parquet_path=self.config.manifest_parquet_path,
                analysis_meta_json_path=self.config.analysis_meta_json_path,
                duration_sec=float(out_info.duration),
                sample_rate=int(out_info.samplerate),
                channels=int(out_info.channels),
                prepared_audio_saved=self.config.save_prepared_audio,
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

