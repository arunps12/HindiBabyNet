from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import soundfile as sf

from src.hindibabynet.entity.config_entity import DataIngestionConfig
from src.hindibabynet.entity.artifact_entity import DataIngestionArtifact
from src.hindibabynet.exception.exception import wrap_exception
from src.hindibabynet.logging.logger import get_logger
from src.hindibabynet.utils.io_utils import write_parquet


logger = get_logger(__name__)


class DataIngestion:
    """
    Scans raw_audio_root:
      raw_audio_root/<participant_id>/<session_date>/*.wav

    Produces a recordings parquet manifest (no copying audio).
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _iter_recordings(self) -> List[Dict]:
        root = self.config.raw_audio_root
        allowed = set(self.config.allowed_ext)

        if not root.exists():
            raise FileNotFoundError(f"raw_audio_root not found: {root}")

        rows: List[Dict] = []

        # find wavs anywhere under root, but parse parent dirs if possible
        for wav_path in root.rglob("*"):
            if not wav_path.is_file():
                continue
            if wav_path.suffix not in allowed:
                continue

            try:
                info = sf.info(str(wav_path))
                # try parse participant_id/session_date from path
                # .../<participant>/<date>/<file>
                participant_id = wav_path.parents[1].name if len(wav_path.parents) >= 2 else None
                session_date = wav_path.parents[0].name if len(wav_path.parents) >= 1 else None

                rows.append(
                    {
                        "participant_id": participant_id,
                        "session_date": session_date,
                        "recording_id": wav_path.stem,
                        "path": str(wav_path),
                        "duration_sec": float(info.duration),
                        "sample_rate": int(info.samplerate),
                        "channels": int(info.channels),
                        "frames": int(info.frames),
                        "subtype": str(info.subtype),
                        "format": str(info.format),
                        "size_bytes": int(wav_path.stat().st_size),
                    }
                )
            except Exception as e:
                # log and skip bad files
                logger.warning(f"Failed reading audio info for {wav_path.name}: {e}")

        return rows

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info(f"Scanning raw audio root: {self.config.raw_audio_root}")
            rows = self._iter_recordings()

            if not rows:
                raise RuntimeError(
                    f"No recordings found under {self.config.raw_audio_root} with ext={self.config.allowed_ext}"
                )

            df = pd.DataFrame(rows).sort_values(
                ["participant_id", "session_date", "recording_id"], na_position="last"
            ).reset_index(drop=True)

            write_parquet(df, self.config.recordings_parquet_path)

            n_participants = df["participant_id"].nunique(dropna=True)
            n_sessions = df[["participant_id", "session_date"]].drop_duplicates().shape[0]

            logger.info(f"Wrote recordings manifest: {self.config.recordings_parquet_path}")
            logger.info(f"Recordings: {len(df)} | Participants: {n_participants} | Sessions: {n_sessions}")

            return DataIngestionArtifact(
                recordings_parquet_path=self.config.recordings_parquet_path,
                n_recordings=int(len(df)),
                n_sessions=int(n_sessions),
                n_participants=int(n_participants),
            )
        except Exception as e:
            raise wrap_exception(
                "Data ingestion failed",
                e,
                context={
                    "raw_audio_root": str(self.config.raw_audio_root),
                    "recordings_parquet_path": str(self.config.recordings_parquet_path),
                },
            )
