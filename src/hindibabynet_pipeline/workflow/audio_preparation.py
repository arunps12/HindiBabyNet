from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import soundfile as sf

from hindibabynet_pipeline.components.audio.concatenate import concatenate_wavs_streaming
from hindibabynet_pipeline.components.audio.normalize import peak_normalize_wav_streaming
from hindibabynet_pipeline.components.audio.resample import ensure_mono_16k_wav_streaming
from hindibabynet_pipeline.entity.artifact_entity import AudioPreparationArtifact
from hindibabynet_pipeline.entity.config_entity import AudioPreparationConfig
from hindibabynet_pipeline.exception.exception import wrap_exception
from hindibabynet_pipeline.logging.logger import get_logger
from hindibabynet_pipeline.utils.io_utils import ensure_dir, write_json, write_parquet

logger = get_logger(__name__)


class AudioPreparation:
    """
    Produce one analysis-ready WAV per participant:
      - optional multi-file join
      - optional mono conversion
      - optional resampling
      - optional peak normalization
    """

    def __init__(self, config: AudioPreparationConfig):
        self.config = config

    @staticmethod
    def _infer_recording_id_from_wav(wav_path: Path) -> str:
        return wav_path.stem

    def _manifest_single_wav(self, wav_path: Path, recording_id: str) -> pd.DataFrame:
        info = sf.info(str(wav_path))
        return pd.DataFrame(
            [
                {
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
                }
            ]
        )

    def _combine_for_participant(
        self,
        recordings_df: pd.DataFrame,
        participant_id: str,
        recording_id: str,
    ) -> pd.DataFrame:
        required = {"participant_id", "recording_id", "path"}
        missing = required - set(recordings_df.columns)
        if missing:
            raise ValueError(f"recordings_df missing columns: {sorted(missing)}")

        g = recordings_df[recordings_df["participant_id"] == participant_id].copy()
        if g.empty:
            raise ValueError(f"No rows for participant_id={participant_id}")
        if not self.config.join_multiple_files and len(g) > 1:
            raise ValueError(
                "join_multiple_files is false, but multiple WAV files were provided "
                "for one participant."
            )

        sort_cols = [c for c in ["session_date", "recording_id"] if c in g.columns]
        g = g.sort_values(sort_cols if sort_cols else ["recording_id"]).reset_index(
            drop=True
        )
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
        logger.info(
            f"[combine] participant_id={participant_id} | n_files={len(wavs)} -> {combined_raw}"
        )
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

    def _prepare_audio(
        self, source_path: Path, recording_id: str
    ) -> tuple[Path, dict, dict]:
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
        target_sr = (
            self.config.target_sr
            if self.config.resample
            else int(source_info.samplerate)
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
        try:
            ensure_dir(self.config.artifacts_dir)
            ensure_dir(self.config.processed_audio_root)
            ensure_dir(self.config.raw_joined_audio_root)

            if wav_path is not None:
                wav_path = Path(wav_path)
                if not wav_path.exists():
                    raise FileNotFoundError(f"wav not found: {wav_path}")
                rec_id = recording_id or self._infer_recording_id_from_wav(wav_path)
                manifest_df = self._manifest_single_wav(wav_path, recording_id=rec_id)
                prepared_path, conv_stats, norm_stats = self._prepare_audio(
                    source_path=wav_path, recording_id=rec_id
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
                    },
                }
                write_json(meta, self.config.analysis_meta_json_path)
                if cleanup_tmp:
                    tmp = (
                        self.config.processed_audio_root
                        / "_tmp_prep"
                        / rec_id
                        / f"{rec_id}_converted.wav"
                    )
                    tmp.unlink(missing_ok=True)
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

            if recordings_df is None:
                raise ValueError(
                    "Provide either wav_path or recordings_df + participant_id."
                )
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
                source_path=combined_raw_path, recording_id=rec_id
            )
            write_parquet(manifest_df, self.config.manifest_parquet_path)
            out_info = sf.info(str(prepared_path))
            meta = {
                "mode": "combine_by_participant_id"
                if self.config.join_multiple_files
                else "single_recording_from_dataframe",
                "participant_id": str(participant_id),
                "analysis_wav": str(prepared_path),
                "combined_raw_path": str(combined_raw_path)
                if self.config.save_raw_joined_audio
                else None,
                "convert": conv_stats,
                "normalize": norm_stats,
                "save_prepared_audio": self.config.save_prepared_audio,
                "output_info": {
                    "duration_sec": float(out_info.duration),
                    "sample_rate": int(out_info.samplerate),
                    "channels": int(out_info.channels),
                },
            }
            write_json(meta, self.config.analysis_meta_json_path)
            if cleanup_tmp:
                tmp = (
                    self.config.processed_audio_root
                    / "_tmp_prep"
                    / rec_id
                    / f"{rec_id}_converted.wav"
                )
                tmp.unlink(missing_ok=True)
            return AudioPreparationArtifact(
                raw_joined_wav_path=combined_raw_path
                if self.config.save_raw_joined_audio
                else None,
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


def run_prepare_audio(
    *,
    recordings_parquet: str | Path | None = None,
    wav: str | Path | None = None,
    recording_id: str | None = None,
    limit: int | None = None,
    run_id: str | None = None,
) -> None:
    """Run audio preparation directly using the AudioPreparation component."""
    from hindibabynet_pipeline.config.configuration import ConfigurationManager

    if recordings_parquet is not None and wav is not None:
        raise ValueError("Provide either recordings_parquet or wav, not both.")
    if recordings_parquet is None and wav is None:
        raise ValueError("Provide either recordings_parquet or wav.")

    cfg = ConfigurationManager()

    if recordings_parquet is not None:
        df = pd.read_parquet(str(recordings_parquet))
        participants = df["participant_id"].dropna().astype(str).unique().tolist()
        if limit is not None:
            participants = participants[:limit]
        for pid in participants:
            prep_cfg = cfg.get_audio_preparation_config(participant_id=pid, run_id=run_id)
            AudioPreparation(prep_cfg).run(recordings_df=df, participant_id=pid)
        return

    wav = Path(wav)  # type: ignore[arg-type]
    rec_id = recording_id or wav.stem
    prep_cfg = cfg.get_audio_preparation_config(participant_id=rec_id, run_id=run_id)
    AudioPreparation(prep_cfg).run(wav_path=wav, recording_id=rec_id)