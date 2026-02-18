#!/usr/bin/env python
"""
Create a test set for evaluating the speaker-type classifier.

Randomly samples segments from the classified_segments parquet, extracts the
corresponding audio clips from the analysis WAV, and writes them as individual
WAV files organised by predicted class.  A CSV annotation sheet is generated so
you can listen and manually label each clip.

Usage
-----
# Sample 50 segments per class from the latest run, participant ABAN141223
uv run python scripts/create_test_set.py \
    --pid ABAN141223 --per_class 50

# Specify a run ID explicitly
uv run python scripts/create_test_set.py \
    --pid ABAN141223 --per_class 50 --run_id 20260217_133307

# Use a different seed for reproducibility
uv run python scripts/create_test_set.py \
    --pid ABAN141223 --per_class 50 --seed 123

Output
------
<output_root>/<pid>/test_set/
    annotation_sheet.csv          ← fill in 'human_label' column
    clips/
        FEM/
            seg_0042_3.21s-5.67s.wav
            ...
        MAL/
            ...
        KCHI/
            ...
        SIL/
            ...
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# ── Label map (same as speaker-type-classifier) ────────
ID2LABEL = {0: "MAL", 1: "FEM", 2: "KCHI", 3: "SIL"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
CLASSES = list(LABEL2ID.keys())


def find_latest_run(artifacts_root: Path) -> str:
    runs = sorted(p.name for p in artifacts_root.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found in {artifacts_root}")
    return runs[-1]


def load_analysis_wav(summary_json: Path) -> tuple[np.ndarray, int]:
    with open(summary_json) as f:
        meta = json.load(f)
    wav_path = meta["analysis_wav"]
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def stratified_sample(
    df: pd.DataFrame,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Sample up to `per_class` segments from each predicted class."""
    rng = np.random.default_rng(seed)
    sampled = []
    for cls in CLASSES:
        cls_df = df[df["predicted_class"] == cls]
        n = min(per_class, len(cls_df))
        if n == 0:
            print(f"  WARNING: no segments for class '{cls}'")
            continue
        idx = rng.choice(len(cls_df), size=n, replace=False)
        sampled.append(cls_df.iloc[idx])
        print(f"  {cls}: sampled {n}/{len(cls_df)}")
    return pd.concat(sampled, ignore_index=True)


def extract_clips(
    df: pd.DataFrame,
    audio: np.ndarray,
    sr: int,
    clips_dir: Path,
) -> list[dict]:
    """Extract audio clips and return annotation rows."""
    rows = []
    for _, row in df.iterrows():
        cls = row["predicted_class"]
        start = row["start_sec"]
        end = row["end_sec"]
        dur = row["duration_sec"]
        conf = row["predicted_confidence"]

        s = max(0, int(round(start * sr)))
        e = min(len(audio), int(round(end * sr)))
        clip = audio[s:e]

        cls_dir = clips_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        fname = f"seg_{int(row['chunk_id']):04d}_{start:.2f}s-{end:.2f}s.wav"
        clip_path = cls_dir / fname

        sf.write(str(clip_path), clip, sr)

        rows.append({
            "clip_file": str(clip_path.relative_to(clips_dir.parent)),
            "predicted_class": cls,
            "predicted_confidence": round(conf, 4),
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(dur, 3),
            "human_label": "",           # ← annotator fills this
            "notes": "",                 # ← optional notes
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Create a test set for speaker-type classifier evaluation")
    ap.add_argument("--pid", required=True, help="Participant ID (e.g. ABAN141223)")
    ap.add_argument("--per_class", type=int, default=50, help="Segments to sample per class (default: 50)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--run_id", default=None, help="Run ID; defaults to latest")
    ap.add_argument("--artifacts_root", default="artifacts/runs", help="Artifacts root directory")
    ap.add_argument("--output_root", default=None,
                    help="Output root; defaults to <artifacts_root>/<run_id>/test_set")
    ap.add_argument("--min_duration", type=float, default=0.3,
                    help="Minimum segment duration in seconds (default: 0.3)")
    ap.add_argument("--max_duration", type=float, default=None,
                    help="Maximum segment duration in seconds (default: no limit)")
    ap.add_argument("--min_confidence", type=float, default=0.0,
                    help="Minimum prediction confidence to include (default: 0.0)")
    args = ap.parse_args()

    arts = Path(args.artifacts_root)
    run_id = args.run_id or find_latest_run(arts)
    run_dir = arts / run_id

    print(f"Run ID      : {run_id}")
    print(f"Participant : {args.pid}")
    print(f"Per class   : {args.per_class}")
    print(f"Seed        : {args.seed}")

    # ── Load classified segments ──
    seg_pq = run_dir / "speaker_classification" / f"{args.pid}_classified_segments.parquet"
    if not seg_pq.exists():
        raise FileNotFoundError(f"Not found: {seg_pq}")
    df = pd.read_parquet(seg_pq)
    print(f"\nTotal classified segments: {len(df)}")
    print(f"Class distribution:\n{df['predicted_class'].value_counts().to_string()}")

    # ── Filter ──
    mask = df["duration_sec"] >= args.min_duration
    if args.max_duration:
        mask &= df["duration_sec"] <= args.max_duration
    if args.min_confidence > 0:
        mask &= df["predicted_confidence"] >= args.min_confidence
    df = df[mask].reset_index(drop=True)
    print(f"\nAfter filters (min_dur={args.min_duration}s, max_dur={args.max_duration}, "
          f"min_conf={args.min_confidence}): {len(df)} segments")

    # ── Stratified random sample ──
    print("\nSampling:")
    sampled = stratified_sample(df, args.per_class, args.seed)
    print(f"\nTotal sampled: {len(sampled)}")

    # ── Load audio ──
    summary_json = run_dir / "speaker_classification" / f"{args.pid}_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"Not found: {summary_json}")
    print("\nLoading analysis WAV...")
    audio, sr = load_analysis_wav(summary_json)
    print(f"Audio: {len(audio)/sr:.1f}s @ {sr} Hz")

    # ── Output directory ──
    if args.output_root:
        out_dir = Path(args.output_root)
    else:
        out_dir = run_dir / "test_set" / args.pid
    clips_dir = out_dir / "clips"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract clips ──
    print(f"\nExtracting clips to {out_dir}...")
    rows = extract_clips(sampled, audio, sr, clips_dir)

    # ── Write annotation sheet ──
    ann_df = pd.DataFrame(rows)
    csv_path = out_dir / "annotation_sheet.csv"
    ann_df.to_csv(csv_path, index=False)
    print(f"\nAnnotation sheet: {csv_path}")
    print(f"Total clips: {len(rows)}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  TEST SET CREATED")
    print(f"{'='*60}")
    print(f"  Clips directory : {clips_dir}")
    print(f"  Annotation CSV  : {csv_path}")
    print(f"  Total clips     : {len(rows)}")
    for cls in CLASSES:
        n = sum(1 for r in rows if r["predicted_class"] == cls)
        print(f"    {cls:<15}: {n}")
    print(f"\nNext steps:")
    print(f"  1. Listen to the clips in each class folder")
    print(f"  2. Fill in the 'human_label' column in {csv_path.name}")
    print(f"     Valid labels: {CLASSES}")
    print(f"  3. Run the evaluation script to compare predictions vs human labels")


if __name__ == "__main__":
    main()
