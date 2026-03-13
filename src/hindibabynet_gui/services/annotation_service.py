"""Annotation service: wrapper around scripts/annotate_ads_ids.py."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from hindibabynet_gui.utils import resolve_path
from hindibabynet_gui.utils.yaml_utils import load_yaml
from hindibabynet_gui.utils import CONFIG_PATH


def _get_roots() -> tuple[Optional[Path], Optional[Path]]:
    """Return (classified_root, annotation_root) from config or defaults."""
    cfg = load_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    classified = (cfg.get("speaker_classification") or {}).get("output_audio_root")
    classified_path = resolve_path(classified) if classified else None
    # Annotation root sits next to classified
    annotation_path = classified_path.parent / "annotations" if classified_path else None
    return classified_path, annotation_path


def list_annotatable_participants() -> list[dict]:
    """List participants that have classified outputs and their annotation progress."""
    classified_root, annotation_root = _get_roots()
    if not classified_root or not classified_root.exists():
        return []

    results = []
    for pid_dir in sorted(classified_root.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        has_female = (pid_dir / f"{pid}_main_female.wav").exists()
        has_male = (pid_dir / f"{pid}_main_male.wav").exists()
        if not (has_female or has_male):
            continue

        # Check annotation CSV progress
        ann_progress = {}
        if annotation_root and annotation_root.exists():
            ann_dir = annotation_root / pid
            for speaker in ("female", "male"):
                csv_path = ann_dir / f"{pid}_{speaker}_annotations.csv"
                if csv_path.exists():
                    try:
                        with open(csv_path, "r") as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            total = len(rows)
                            labeled = sum(1 for r in rows if r.get("label", "") != "")
                            ann_progress[speaker] = f"{labeled}/{total}"
                    except Exception:
                        ann_progress[speaker] = "error"
                else:
                    ann_progress[speaker] = "not started"

        results.append({
            "participant_id": pid,
            "has_female": has_female,
            "has_male": has_male,
            "female_progress": ann_progress.get("female", "n/a"),
            "male_progress": ann_progress.get("male", "n/a"),
            "annotation_dir": str(annotation_root / pid) if annotation_root else "",
        })

    return results
