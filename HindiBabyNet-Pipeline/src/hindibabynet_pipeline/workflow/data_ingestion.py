"""Data ingestion workflow: scan raw audio root, produce recordings manifest."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import soundfile as sf

from hindibabynet_pipeline.entity.config_entity import DataIngestionConfig
from hindibabynet_pipeline.entity.artifact_entity import DataIngestionArtifact
from hindibabynet_pipeline.exception.exception import wrap_exception
from hindibabynet_pipeline.logging.logger import get_logger
from hindibabynet_pipeline.utils.io_utils import write_parquet

logger = get_logger(__name__)


class DataIngestion:
    """
    Scans raw_audio_root:
      raw_audio_root/<participant_id>/<session_date>/*.wav

    Produces a recordings parquet manifest (no audio copying).
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @staticmethod
    def _parse_session_date(session_name: str | None) -> datetime | None:
        if not session_name:
            return None
        try:
            return datetime.strptime(session_name, "%Y%m%d")
        except ValueError:
            return None

    @staticmethod
    def _format_file_list(rows: List[Dict]) -> str:
        if not rows:
            return "none"
        return ", ".join(Path(row["path"]).name for row in rows)

    def _log_session_selection(
        self,
        participant_id: str | None,
        available_dates: List[str],
        selected_date: str,
        retained_rows: List[Dict],
        excluded_rows: List[Dict],
    ) -> None:
        pid = participant_id or "<unknown>"
        available = ", ".join(available_dates) if available_dates else "none"
        logger.info(
            f"[session-selection] participant_id={pid} | available_dates={available} "
            f"| selected={selected_date} | retained_files={self._format_file_list(retained_rows)} "
            f"| excluded_files={self._format_file_list(excluded_rows)}"
        )

    def _filter_rows_by_session(self, rows: List[Dict]) -> List[Dict]:
        rows_by_participant: Dict[str | None, List[Dict]] = defaultdict(list)
        for row in rows:
            rows_by_participant[row.get("participant_id")].append(row)

        if self.config.session_selection == "all":
            for participant_id, participant_rows in rows_by_participant.items():
                available_dates = sorted(
                    {
                        row["session_date"]
                        for row in participant_rows
                        if self._parse_session_date(row.get("session_date")) is not None
                    }
                )
                self._log_session_selection(
                    participant_id=participant_id,
                    available_dates=available_dates,
                    selected_date="all",
                    retained_rows=participant_rows,
                    excluded_rows=[],
                )
            return rows

        filtered_rows: List[Dict] = []
        for participant_id, participant_rows in rows_by_participant.items():
            valid_dates = sorted(
                {
                    row["session_date"]
                    for row in participant_rows
                    if self._parse_session_date(row.get("session_date")) is not None
                }
            )

            if not valid_dates:
                pid = participant_id or "<unknown>"
                logger.warning(
                    f"[session-selection] participant_id={pid} | no valid YYYYMMDD "
                    "session folders found; preserving all recordings"
                )
                self._log_session_selection(
                    participant_id=participant_id,
                    available_dates=[],
                    selected_date="all",
                    retained_rows=participant_rows,
                    excluded_rows=[],
                )
                filtered_rows.extend(participant_rows)
                continue

            selected_date = valid_dates[0]
            retained_rows = [
                row for row in participant_rows if row.get("session_date") == selected_date
            ]
            excluded_rows = [
                row for row in participant_rows if row.get("session_date") != selected_date
            ]
            self._log_session_selection(
                participant_id=participant_id,
                available_dates=valid_dates,
                selected_date=selected_date,
                retained_rows=retained_rows,
                excluded_rows=excluded_rows,
            )
            filtered_rows.extend(retained_rows)

        return filtered_rows

    def _iter_recordings(self) -> List[Dict]:
        root = self.config.raw_audio_root
        allowed = set(self.config.allowed_ext)

        if not root.exists():
            raise FileNotFoundError(f"raw_audio_root not found: {root}")

        rows: List[Dict] = []

        for wav_path in root.rglob("*"):
            if not wav_path.is_file():
                continue
            if wav_path.suffix not in allowed:
                continue
            try:
                info = sf.info(str(wav_path))
                participant_id = (
                    wav_path.parents[1].name if len(wav_path.parents) >= 2 else None
                )
                session_date = (
                    wav_path.parents[0].name if len(wav_path.parents) >= 1 else None
                )
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
                logger.warning(f"Failed reading audio info for {wav_path.name}: {e}")

        return self._filter_rows_by_session(rows)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info(f"Scanning raw audio root: {self.config.raw_audio_root}")
            rows = self._iter_recordings()

            if not rows:
                raise RuntimeError(
                    f"No recordings found under {self.config.raw_audio_root} "
                    f"with ext={self.config.allowed_ext}"
                )

            df = (
                pd.DataFrame(rows)
                .sort_values(
                    ["participant_id", "session_date", "recording_id"],
                    na_position="last",
                )
                .reset_index(drop=True)
            )

            write_parquet(df, self.config.recordings_parquet_path)

            n_participants = df["participant_id"].nunique(dropna=True)
            n_sessions = df[["participant_id", "session_date"]].drop_duplicates().shape[0]

            logger.info(
                f"Wrote recordings manifest: {self.config.recordings_parquet_path}"
            )
            logger.info(
                f"Recordings: {len(df)} | Participants: {n_participants} | "
                f"Sessions: {n_sessions}"
            )

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
                    "recordings_parquet_path": str(
                        self.config.recordings_parquet_path
                    ),
                },
            )


def run_data_ingestion(run_id: str | None = None) -> DataIngestionArtifact:
    """Run data ingestion using ConfigurationManager."""
    from hindibabynet_pipeline.config.configuration import ConfigurationManager
    from hindibabynet_pipeline.logging.logger import add_file_handler

    cfg = ConfigurationManager()
    di_cfg = cfg.get_data_ingestion_config(run_id=run_id)
    logs_root = cfg.get_logs_root()

    actual_run_id = di_cfg.artifacts_dir.parent.name
    log_file = logs_root / actual_run_id / "data_ingestion.log"
    add_file_handler(logger, log_file)

    logger.info(f"Data ingestion started | run_id={actual_run_id}")
    artifact = DataIngestion(di_cfg).initiate_data_ingestion()
    logger.info(f"Data ingestion completed: {artifact}")
    return artifact
