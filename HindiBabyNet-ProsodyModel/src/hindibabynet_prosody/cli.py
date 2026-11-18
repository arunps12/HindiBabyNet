"""Typer CLI entry point for the prosody pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hindibabynet_prosody.config import AppConfig, load_config
from hindibabynet_prosody.logging_utils import configure_logging
from hindibabynet_prosody.pipeline.extract import build_extract_plan, run_extraction
from hindibabynet_prosody.pipeline.inspect import run_inspect
from hindibabynet_prosody.pipeline.validate import run_validation


app = typer.Typer(help="HindiBabyNet prosodic and SSL feature extraction pipeline.")
console = Console()


def _load_and_log(config_path: Path, command_name: str, log_level: str | None = None) -> AppConfig:
    config = load_config(config_path)
    effective_log_level = log_level or config.runtime.log_level
    log_dir = config.paths.output_dir / "logs"
    log_file = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{command_name}.log"
    configure_logging(effective_log_level, log_file)
    return config


@app.command()
def validate(config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True)) -> None:
    """Validate configuration and input structure without running extraction."""
    loaded_config = _load_and_log(config, "validate")
    report = run_validation(loaded_config)

    table = Table(title="Validation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Project", loaded_config.project.name)
    table.add_row("RTTM files", str(len(report.rttm_files)))
    table.add_row("Resolved source participants", str(len(report.audio_resolutions)))
    table.add_row("Unmapped sources", str(len(report.unmapped_sources)))
    table.add_row("Skipped segment issues", str(len(report.skipped_segment_issues)))
    table.add_row(
        "Audio resolution issues",
        str(sum(1 for item in report.audio_resolutions if item.issue is not None)),
    )
    console.print(table)


@app.command()
def inspect(
    config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    participant: str = typer.Option(...),
    max_segments: int = typer.Option(10, min=1),
) -> None:
    """Inspect a small participant subset without running the full corpus."""
    loaded_config = _load_and_log(config, "inspect")
    report = run_inspect(loaded_config, participant=participant, max_segments=max_segments)
    table = Table(title=f"Inspect {participant}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Source participant", report.source_participant_id)
    table.add_row("Participant ID", report.participant_id or "<unmapped>")
    table.add_row("Audio path", report.audio_path or "<missing>")
    table.add_row("Segments inspected", str(report.n_segments_total))
    table.add_row("Included speaker segments", str(report.n_segments_included_speakers))
    table.add_row("Speaker counts", ", ".join(f"{k}={v}" for k, v in sorted(report.speaker_counts.items())))
    console.print(table)


@app.command()
def extract(
    config: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    participants: list[str] | None = typer.Option(None),
    speakers: list[str] | None = typer.Option(None),
    device: str | None = typer.Option(None),
    resume: bool = typer.Option(False),
    overwrite: bool = typer.Option(False),
    no_cache: bool = typer.Option(False),
    log_level: str | None = typer.Option(None),
) -> None:
    """Run the participant-by-speaker extraction pipeline."""
    loaded_config = _load_and_log(config, "extract", log_level=log_level)
    plan = build_extract_plan(loaded_config, device=device, no_cache=no_cache)
    effective_log_level = log_level or loaded_config.runtime.log_level
    table = Table(title="Extract Preflight")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Participants discovered", str(plan.participant_count))
    table.add_row("Device", plan.selected_device)
    table.add_row("Cache enabled", str(plan.cache_enabled))
    table.add_row("Resume", str(resume))
    table.add_row("Overwrite", str(overwrite))
    table.add_row("Log level", effective_log_level)
    table.add_row("Participant workers", str(loaded_config.runtime.num_workers))
    table.add_row("Participant filter", ", ".join(participants) if participants else "<all>")
    table.add_row("Speaker filter", ", ".join(speakers) if speakers else "<all>")
    console.print(table)

    outputs = run_extraction(
        loaded_config,
        participants=participants,
        speakers=speakers,
        device=device,
        resume=resume,
        overwrite=overwrite,
        no_cache=no_cache,
    )
    result_table = Table(title="Extract Outputs")
    result_table.add_column("Metric")
    result_table.add_column("Value")
    result_table.add_row("Participant count", str(outputs.participant_count))
    result_table.add_row("Row count", str(outputs.row_count))
    result_table.add_row("Final CSV", str(outputs.final_csv))
    result_table.add_row("QC summary", str(outputs.qc_summary_csv))
    result_table.add_row("Skipped segments", str(outputs.skipped_segments_csv))
    result_table.add_row("Run metadata", str(outputs.run_metadata_json))
    console.print(result_table)
