from __future__ import annotations

from pathlib import Path


def run_classify_vtc(
    *,
    recordings_parquet: str | Path | None = None,
    wav: str | Path | None = None,
    analysis_dir: str | Path | None = None,
    participant_id: str | None = None,
    limit: int | None = None,
) -> None:
    """Run VTC speaker classification directly via the backend dispatcher."""
    from hindibabynet_pipeline.config.configuration import ConfigurationManager
    from hindibabynet_pipeline.components.speaker_classification import get_backend
    from hindibabynet_pipeline.exception.exception import format_traceback
    from hindibabynet_pipeline.logging.logger import get_logger, add_file_handler

    logger = get_logger(__name__)
    cfg = ConfigurationManager()
    run_id = cfg.make_run_id()
    backend = get_backend(cfg, override="vtc")
    add_file_handler(
        logger,
        cfg.get_logs_root() / run_id / f"vtc_classification_{run_id}.log",
    )
    output_root = cfg.get_classification_output_root() / backend.name

    if wav is not None:
        wav_path = Path(wav).resolve()
        pid = participant_id or wav_path.stem
        out_dir = output_root / pid
        if not backend.is_complete(pid, out_dir):
            backend.run_participant(wav_path, pid, out_dir)
        return

    if recordings_parquet is not None:
        import pandas as pd
        df = pd.read_parquet(str(recordings_parquet))
        processed_root = cfg.get_processed_audio_root()
        participants = [
            (pid, processed_root / pid / f"{pid}.wav")
            for pid in sorted(df["participant_id"].dropna().astype(str).unique())
            if (processed_root / pid / f"{pid}.wav").is_file()
        ]
    else:
        root = Path(analysis_dir).resolve() if analysis_dir else cfg.get_processed_audio_root()
        participants = [
            (d.name, d / f"{d.name}.wav")
            for d in sorted(root.iterdir())
            if d.is_dir() and (d / f"{d.name}.wav").is_file()
        ]

    if limit is not None:
        participants = participants[:limit]

    ok = skip = fail = 0
    for pid, wav_path in participants:
        out_dir = output_root / pid
        if backend.is_complete(pid, out_dir):
            skip += 1
            continue
        try:
            backend.run_participant(wav_path, pid, out_dir)
            ok += 1
        except Exception as exc:
            fail += 1
            logger.error(f"FAIL | {pid}\n{format_traceback(exc)}")

    logger.info(f"VTC classification finished | ok={ok} skip={skip} fail={fail}")