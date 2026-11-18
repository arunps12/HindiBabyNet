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
    from hindibabynet_pipeline.workflow.classification import run_classify

    run_classify(
        backend="vtc",
        recordings_parquet=recordings_parquet,
        wav=wav,
        analysis_dir=analysis_dir,
        participant_id=participant_id,
        limit=limit,
    )
