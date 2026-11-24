"""Speaker classification with VTC (Voice Type Classifier 2.0).

VTC is a separate program (https://github.com/LAAC-LSCP/VTC) that is installed
in Ubuntu. This module starts VTC for the prepared WAV file of each participant
and checks that the results are complete. The VTC results are not changed.

Input:    <prepared_audio>/<participant>/<participant>.wav
Results:  <vtc>/<participant>/rttm/<participant>.rttm
          <vtc>/<participant>/rttm.csv      (uid, start_time_s, duration_s, label)
          <vtc>/<participant>/run_info.json (written by this module)

Labels of VTC: FEM (adult female), MAL (adult male), KCHI (key child),
OCH (other child).

Results that exist are never run again, replaced or deleted. VTC results and
XGBoost results are kept in two different folders and are never mixed.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from hbn_paths import local_path


def build_command(
    repo: str | Path,
    wav_dir: str | Path,
    output_dir: str | Path,
    device: str,
    batch_size: int,
) -> list[str]:
    """Command that runs VTC on all WAV files of one folder.

    The command must be started with the VTC folder (repo) as working directory:
    the script path in the command is relative to that folder.
    """
    return [
        "uv", "run", "scripts/infer.py",
        "--wavs", str(wav_dir),
        "--output", str(output_dir),
        "--device", str(device),
        "--batch_size", str(batch_size),
    ]


def results_complete(output_dir: str | Path) -> bool:
    """True if the folder has rttm/ with at least one .rttm file and rttm.csv."""
    output_dir = Path(output_dir)
    rttm_dir = output_dir / "rttm"
    return (
        rttm_dir.is_dir()
        and any(rttm_dir.glob("*.rttm"))
        and (output_dir / "rttm.csv").is_file()
    )


def count_segments(rttm_csv: Path) -> int:
    """Number of speech segments (rows) in rttm.csv."""
    with rttm_csv.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def vtc_commit(repo: Path) -> str | None:
    """Git commit of the installed VTC, or None if it cannot be read."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def setup_problem(repo: Path) -> str | None:
    """Return a message if VTC cannot be started on this computer, else None."""
    script = repo / "scripts" / "infer.py"
    if not script.is_file():
        return (
            f"VTC was not found: {script} is missing. Install VTC "
            "(https://github.com/LAAC-LSCP/VTC) and write its folder as "
            "audio.vtc_repo in config.yaml."
        )
    if shutil.which("uv") is None:
        return "The program uv was not found on PATH. Install uv in Ubuntu first."
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def classify_participant(
    participant: str,
    wav_path: Path,
    output_dir: Path,
    repo: Path,
    device: str,
    batch_size: int,
    problem: str | None,
) -> tuple[str, str]:
    """Run VTC for one participant. Returns the result and a short text."""
    if results_complete(output_dir):
        return "already available", "results already available, not run again"
    if output_dir.exists():
        raise ValueError(
            f"the result folder {output_dir} exists but is not complete "
            "(rttm/ with .rttm files and rttm.csv are needed); "
            "nothing was deleted or replaced"
        )
    if problem:
        raise ValueError(problem)

    # VTC reads all WAV files of the folder. The folder of a participant holds one WAV file.
    command = build_command(repo, wav_path.parent, output_dir, device, batch_size)
    print(f"{participant}: VTC started (device {device}, batch size {batch_size})")
    output_dir.mkdir(parents=True)
    # VTC has its own Python environment; the environment of this program is not passed on.
    own_environment = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_PROJECT", "PYTHONPATH")
    environment = {key: value for key, value in os.environ.items() if key not in own_environment}

    started = _now()
    clock = time.monotonic()
    # The messages of VTC are shown in the console as they come.
    finished = subprocess.run(command, cwd=str(repo), env=environment)
    runtime = round(time.monotonic() - clock, 2)

    if finished.returncode != 0:
        if not any(output_dir.iterdir()):
            output_dir.rmdir()  # the empty folder made above
        raise RuntimeError(
            f"VTC ended with exit code {finished.returncode}; see the VTC messages above"
        )
    if not results_complete(output_dir):
        raise RuntimeError(
            f"VTC ended without an error, but the results in {output_dir} are not complete"
        )

    info = {
        "participant_id": participant,
        "backend": "vtc",
        "input_wav": str(wav_path),
        "output_dir": str(output_dir),
        "started_at": started,
        "finished_at": _now(),
        "runtime_sec": runtime,
        "status": "success",
        "command": " ".join(command),
        "device": device,
        "vtc_repo_path": str(repo),
        "outputs_present": {
            name: (output_dir / name).exists() for name in ("rttm", "rttm.csv")
        },
        "batch_size": batch_size,
        "vtc_commit": vtc_commit(repo),
    }
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return "classified", f"finished, {count_segments(output_dir / 'rttm.csv')} segments"


def run(config, params: dict, participant: str | None = None, limit: int | None = None) -> int:
    """Run VTC for all participants with prepared audio. Returns the exit code."""
    if os.name == "nt":
        print("VTC runs only in Ubuntu (WSL). Open Ubuntu and run the same command there.")
        return 1

    prepared_root = config.path("prepared_audio")
    vtc_root = config.path("vtc")
    repo = local_path(
        config.setting("audio", "vtc_repo", "~/VTC"), config.file.parent, config.wsl_distro
    )
    device = str(config.setting("audio", "vtc_device", "cpu"))
    batch_size = int(params["vtc"]["batch_size"])

    if not prepared_root.is_dir():
        print(f"Prepared audio folder not found: {prepared_root}. "
              f"Check data_dir and paths.prepared_audio in {config.file}.")
        return 1
    names = sorted(
        entry.name for entry in prepared_root.iterdir()
        if (entry / f"{entry.name}.wav").is_file()
    )
    if participant is not None:
        names = [name for name in names if name == participant]
    if limit is not None:
        names = names[:limit]
    if not names:
        if participant is not None:
            print(f"No prepared audio of participant {participant} was found in {prepared_root}.")
        else:
            print(f"No prepared audio was found in {prepared_root}.")
        return 1

    problem = setup_problem(repo)
    counts = {"classified": 0, "already available": 0, "error": 0}
    for name in names:
        try:
            result, text = classify_participant(
                name, prepared_root / name / f"{name}.wav", vtc_root / name,
                repo, device, batch_size, problem,
            )
        except Exception as error:  # one participant with an error does not stop the others
            result, text = "error", f"error - {error}"
        counts[result] += 1
        print(f"{name}: {text}")

    print(f"VTC results folder: {vtc_root}")
    print("Summary: " + ", ".join(f"{number} {result}" for result, number in counts.items()))
    return 1 if counts["error"] else 0
