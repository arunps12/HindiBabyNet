"""Command hbn-prosody: extract pitch features and speech-model features."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from hindibabynet_prosody.config import AppConfig, load_config
from hindibabynet_prosody.pipeline.extract import (
    build_extract_plan,
    check_output_is_new,
    run_extraction,
)
from hindibabynet_prosody.pipeline.validate import run_validation


app = typer.Typer(add_completion=False)


def _show_progress_messages(level: str) -> None:
    """Show the progress messages of the extraction on the console."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("hindibabynet_prosody").setLevel(
        getattr(logging, level.upper(), logging.INFO)
    )


def _print_table(title: str, rows: list[tuple[str, object]]) -> None:
    width = max(len(label) for label, _ in rows)
    typer.echo(title)
    for label, value in rows:
        typer.echo(f"  {label.ljust(width)} : {value}")


def _check(config: AppConfig) -> None:
    """Check the settings and the input files. Nothing is written."""
    report = run_validation(config)
    final_csv = config.paths.output_dir / config.output.final_filename
    _print_table(
        "Check of the settings and the input files",
        [
            ("Project", config.project.name),
            ("VTC results folder", config.paths.rttm_root),
            ("Prepared audio folder", config.paths.audio_root),
            ("Participant lookup file", config.paths.participant_lookup),
            ("Features folder", config.paths.output_dir),
            ("VTC result files", len(report.rttm_files)),
            ("Participants with VTC results", len(report.audio_resolutions)),
            ("Participants not in the lookup file", len(report.unmapped_sources)),
            ("Segments not used (other speakers or invalid)", len(report.skipped_segment_issues)),
            (
                "Participants with an audio problem",
                sum(1 for item in report.audio_resolutions if item.issue is not None),
            ),
            (
                "Features file",
                "already exists, a new run stops" if final_csv.exists() else "not there yet",
            ),
        ],
    )


def _extract(
    config: AppConfig,
    participants: list[str] | None,
    speakers: list[str] | None,
    device: str | None,
    no_cache: bool,
) -> None:
    """Run the extraction and save the features file."""
    check_output_is_new(config)

    typer.echo("Step 1 of 2: checking the settings and the input files")
    plan = build_extract_plan(config, device=device, no_cache=no_cache)
    _print_table(
        "Extraction",
        [
            ("Participants with VTC results", plan.participant_count),
            ("Device", plan.selected_device),
            ("Saved participant results", "reused" if plan.cache_enabled else "not used"),
            ("Worker processes", config.runtime.num_workers),
            ("Selected participants", ", ".join(participants) if participants else "all"),
            ("Selected speakers", ", ".join(speakers) if speakers else "all"),
        ],
    )

    typer.echo("Step 2 of 2: extracting the features")
    outputs = run_extraction(
        config,
        participants=participants,
        speakers=speakers,
        device=device,
        no_cache=no_cache,
    )
    typer.echo(
        f"Finished: {outputs.participant_count} participants, "
        f"{outputs.row_count} rows (one row per participant and speaker)."
    )
    _print_table(
        "Saved files",
        [
            ("Features", outputs.final_csv),
            ("Quality checks", outputs.qc_summary_csv.parent),
            ("PCA files", outputs.final_csv.parent / "artifacts"),
            ("Run information", outputs.run_metadata_json),
        ],
    )


@app.command()
def main(
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to config.yaml. Default: config.yaml in the project root.",
    ),
    participants: list[str] | None = typer.Option(
        None,
        "--participants",
        help="Only this participant. Repeat the option for more participants. "
        "The PCA values are then calculated for the selected participants only.",
    ),
    speakers: list[str] | None = typer.Option(
        None,
        "--speakers",
        help="Only this speaker, for example FEM. Repeat the option for more speakers.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="cpu, cuda or auto. Default: runtime.device in params.yaml.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Do not reuse or save participant results; every participant is calculated again.",
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Only check the settings and the input files. Nothing is written.",
    ),
) -> None:
    """Extract pitch features and speech-model features per participant and speaker."""
    try:
        loaded_config = load_config(config)
        _show_progress_messages(loaded_config.runtime.log_level)
        if check:
            _check(loaded_config)
        else:
            _extract(loaded_config, participants, speakers, device, no_cache)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        typer.echo(f"Stopped: {error}", err=True)
        raise typer.Exit(code=1) from error
