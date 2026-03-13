"""Outputs service: scan output directories and summarise participant completion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hindibabynet_gui.utils import resolve_path
from hindibabynet_gui.utils.yaml_utils import load_yaml
from hindibabynet_gui.utils import CONFIG_PATH


@dataclass
class ParticipantStatus:
    participant_id: str
    has_main_female: bool = False
    has_main_male: bool = False
    has_child: bool = False
    has_background: bool = False
    has_textgrid: bool = False
    has_segments_parquet: bool = False
    has_summary_json: bool = False
    output_dir: Optional[Path] = None
    summary: dict[str, Any] = field(default_factory=dict)


def scan_outputs(output_root: Path | str | None = None) -> list[ParticipantStatus]:
    """Scan output_audio_root for participant outputs and return completion status."""
    if output_root is None:
        cfg = load_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else {}
        output_root = (cfg.get("speaker_classification") or {}).get("output_audio_root", "")
    root = resolve_path(output_root) if output_root else None
    if not root or not root.exists():
        return []

    results: list[ParticipantStatus] = []

    for pid_dir in sorted(root.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        st = ParticipantStatus(participant_id=pid, output_dir=pid_dir)
        st.has_main_female = (pid_dir / f"{pid}_main_female.wav").exists()
        st.has_main_male = (pid_dir / f"{pid}_main_male.wav").exists()
        st.has_child = (pid_dir / f"{pid}_child.wav").exists()
        st.has_background = (pid_dir / f"{pid}_background.wav").exists()
        st.has_textgrid = (pid_dir / f"{pid}.TextGrid").exists()
        st.has_segments_parquet = (pid_dir / f"{pid}_segments.parquet").exists()

        # Summary JSON
        summary_path = pid_dir / f"{pid}_summary.json"
        if summary_path.exists():
            st.has_summary_json = True
            try:
                with open(summary_path, "r") as f:
                    st.summary = json.load(f)
            except Exception:
                pass

        results.append(st)

    return results


def scan_artifact_runs() -> list[dict[str, Any]]:
    """List available artifact runs with basic metadata."""
    cfg = load_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    artifacts_root = resolve_path(cfg.get("artifacts_root", "artifacts/runs"))
    if not artifacts_root.exists():
        return []

    runs = []
    for run_dir in sorted(artifacts_root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        stages = [p.name for p in run_dir.iterdir() if p.is_dir()]
        runs.append({
            "run_id": run_dir.name,
            "path": str(run_dir),
            "stages": stages,
        })
    return runs
