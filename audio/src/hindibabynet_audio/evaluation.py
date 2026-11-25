"""Optional tools to check a classifier (VTC or XGBoost) against manual labels.

``run_annotation``  play sampled segments of one participant and type a label for each.
``run_evaluation``  compare the labels of the classifier with the manual labels.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

from hbn_paths import new_output
from hindibabynet_audio.recordings import prepared_recordings
from hindibabynet_audio.textgrid import load_segment_table, results_file

# One annotation file per participant: <annotations>/<participant>/speaker_class_annotations.csv
ANNOTATION_FILE = "speaker_class_annotations.csv"

LETTER_SHORTCUTS = {
    "f": "adult_female",
    "m": "adult_male",
    "k": "key_child",
    "o": "other_child",
    "u": "unclear",
    "n": "noise",
}

CONTROL_COMMANDS = {"r", "b", "q"}


# ============================================================================
# Manual annotation
# ============================================================================

def sample_segments(df: pd.DataFrame, n_segments: int, random_seed: int) -> pd.DataFrame:
    """Draw n_segments rows at random (fixed seed) and return them in time order."""
    if df.empty:
        return df.copy()
    if len(df) <= n_segments:
        return df.reset_index(drop=True).copy()
    return df.sample(n=n_segments, random_state=random_seed).sort_values(["start_sec", "end_sec"]).reset_index(drop=True)


def label_shortcuts(labels: dict) -> dict:
    """Keys that can be typed for a label: its number from params.yaml or its letter."""
    shortcuts = {str(number): name for name, number in labels.items()}
    shortcuts.update({letter: name for letter, name in LETTER_SHORTCUTS.items() if name in labels})
    return shortcuts


def play_wav(path: str | Path) -> None:
    """Play a WAV file and wait until it has ended."""
    wav_path = str(path)
    if sys.platform.startswith("win"):
        import winsound

        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        return

    candidate_commands = [
        ["ffplay", "-nodisp", "-autoexit", wav_path],
        ["afplay", wav_path],
        ["aplay", wav_path],
    ]
    for cmd in candidate_commands:
        try:
            result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                return
        except FileNotFoundError:
            continue
    raise RuntimeError("No supported audio playback command is available for annotation playback.")


def write_wav_chunk(
    wav_path: Path,
    chunk_path: Path,
    start_sec: float,
    end_sec: float,
) -> Path | None:
    """Write a time-slice of a WAV to disk. Returns ``None`` on failure.

    Same function as in xgb.py. It is repeated here because xgb.py needs the XGBoost extra.
    """
    try:
        info = sf.info(str(wav_path))
        sr = info.samplerate
        start_frame = max(0, int(start_sec * sr))
        end_frame = min(info.frames, int(end_sec * sr))
        n_frames = end_frame - start_frame
        if n_frames <= 0:
            return None
        audio, _ = sf.read(
            str(wav_path), start=start_frame, frames=n_frames, dtype="float32"
        )
        if audio is None or audio.size == 0:
            return None
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(chunk_path), audio, sr, format="WAV", subtype="PCM_16")
        return chunk_path
    except Exception as e:
        print(f"warning: write_wav_chunk failed ({chunk_path.name}): {e}")
        return None


def _load_existing(output_csv: Path) -> pd.DataFrame:
    if output_csv.exists():
        return pd.read_csv(output_csv)
    return pd.DataFrame()


def _merge_existing(sampled_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        out = sampled_df.copy()
        out["manual_label"] = ""
        return out
    merged = sampled_df.merge(
        existing_df[["segment_index", "manual_label"]],
        on="segment_index",
        how="left",
    )
    merged["manual_label"] = merged["manual_label"].fillna("")
    return merged


def _same_segments(sampled_df: pd.DataFrame, existing_df: pd.DataFrame, method: str) -> bool:
    """True when an existing annotation file holds exactly the sampled segments."""
    columns = ["segment_index", "start_sec", "end_sec", "predicted_class"]
    if len(existing_df) != len(sampled_df) or not set(columns) <= set(existing_df.columns):
        return False
    if "method" in existing_df.columns and not (existing_df["method"].astype(str) == method).all():
        return False
    old = existing_df[columns].reset_index(drop=True)
    new = sampled_df[columns].reset_index(drop=True)
    same_index = (old["segment_index"] == new["segment_index"]).all()
    same_class = (old["predicted_class"].astype(str) == new["predicted_class"].astype(str)).all()
    times = ["start_sec", "end_sec"]
    same_times = ((old[times].astype(float) - new[times].astype(float)).abs() < 1e-6).all().all()
    return bool(same_index and same_class and same_times)


def run_interactive_annotation(
    sampled_df: pd.DataFrame, audio_path: Path, output_csv: Path, shortcuts: dict
) -> Path:
    """Play every segment without a label, ask for the label, save after every answer."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_df = _load_existing(output_csv)
    working_df = _merge_existing(sampled_df, existing_df)
    temp_dir = output_csv.parent / "_tmp_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    while index < len(working_df):
        row = working_df.iloc[index]
        if str(row.get("manual_label", "")).strip():
            index += 1
            continue

        chunk_path = temp_dir / f"segment_{int(row.segment_index):04d}.wav"
        result = write_wav_chunk(audio_path, chunk_path, float(row.start_sec), float(row.end_sec))
        if result is None:
            raise RuntimeError(f"Failed to create playback segment for index {row.segment_index}")

        print(f"Segment {index + 1}/{len(working_df)} | predicted={row.predicted_class} | start={row.start_sec:.2f}s end={row.end_sec:.2f}s")
        play_wav(chunk_path)
        response = input("Label (r=repeat, b=back, q=quit): ").strip().lower()

        if response in CONTROL_COMMANDS:
            if response == "r":
                continue
            if response == "b":
                index = max(0, index - 1)
                working_df.at[index, "manual_label"] = ""
                working_df.to_csv(output_csv, index=False)
                continue
            if response == "q":
                working_df.to_csv(output_csv, index=False)
                return output_csv

        manual_label = shortcuts.get(response, response)
        working_df.at[index, "manual_label"] = manual_label
        working_df.to_csv(output_csv, index=False)
        index += 1

    working_df.to_csv(output_csv, index=False)
    return output_csv


def run_annotation(config, params, method: str, participant: str) -> int:
    """Label sampled segments of one participant by ear. Return 0 if it worked, else 1."""
    settings = params["annotation"]
    segment_path = results_file(config, method, participant)
    prepared = dict(prepared_recordings(config.path("prepared_audio")))
    audio_path = prepared.get(participant)
    if not segment_path.is_file():
        print(f"{participant}: no {method} classification results found ({segment_path}).")
        return 1
    if audio_path is None:
        print(f"{participant}: prepared recording not found "
              f"in {config.path('prepared_audio')}.")
        return 1

    df = load_segment_table(segment_path, backend=method).copy()
    df.insert(0, "segment_index", range(len(df)))
    df.insert(1, "participant_id", participant)
    df.insert(2, "method", method)
    sampled_df = sample_segments(
        df,
        n_segments=int(settings["n_segments_per_participant"]),
        random_seed=int(settings["random_seed"]),
    )

    # An existing file is continued only when it holds the same segments.
    output_csv = config.path("annotations", *participant.split("/"), ANNOTATION_FILE)
    if output_csv.exists():
        if not _same_segments(sampled_df, pd.read_csv(output_csv), method):
            print(
                f"{participant}: {output_csv} already exists and holds other segments "
                f"(another method, other settings or other classification results). "
                "It is not replaced. Move or rename the old file to start a new annotation."
            )
            return 1
        print(f"{participant}: annotation file found, labelling continues ({output_csv})")

    labels = settings["labels"]
    shortcuts = label_shortcuts(labels)
    letters = {name: letter for letter, name in LETTER_SHORTCUTS.items()}
    print("Labels: " + ", ".join(
        f"{number}" + (f" or {letters[name]}" if name in letters else "") + f" = {name}"
        for name, number in labels.items()
    ))
    run_interactive_annotation(sampled_df, audio_path=audio_path, output_csv=output_csv, shortcuts=shortcuts)
    print(f"{participant}: annotations saved in {output_csv}")
    return 0


# ============================================================================
# Evaluation
# ============================================================================

def compute_accuracy(df: pd.DataFrame, truth_col: str = "manual_label", pred_col: str = "predicted_class") -> float:
    """Share of labelled segments where the classifier gives the manual label."""
    valid = df[df[truth_col].notna() & (df[truth_col].astype(str).str.strip() != "")].copy()
    if valid.empty:
        return 0.0
    return float((valid[truth_col].astype(str) == valid[pred_col].astype(str)).mean())


def compute_confusion_matrix(df: pd.DataFrame, labels: list[str], truth_col: str = "manual_label", pred_col: str = "predicted_class") -> pd.DataFrame:
    """Counts of manual label (rows) by classifier label (columns)."""
    valid = df[df[truth_col].notna() & (df[truth_col].astype(str).str.strip() != "")].copy()
    cm = pd.crosstab(valid[truth_col], valid[pred_col], dropna=False)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    cm.index.name = truth_col
    cm.columns.name = pred_col
    return cm


def compute_classwise_metrics(df: pd.DataFrame, labels: list[str], truth_col: str = "manual_label", pred_col: str = "predicted_class") -> pd.DataFrame:
    """Precision, recall and F1 for every label."""
    cm = compute_confusion_matrix(df, labels=labels, truth_col=truth_col, pred_col=pred_col)
    total = int(cm.to_numpy().sum())
    rows: list[dict[str, float | str | int]] = []
    for label in labels:
        tp = int(cm.at[label, label])
        fp = int(cm[label].sum() - tp)
        fn = int(cm.loc[label].sum() - tp)
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def generate_evaluation_report(metrics: dict[str, float | int], classwise: pd.DataFrame) -> str:
    """Text of the short report (Markdown)."""
    lines = [
        "# Model Evaluation Report",
        "",
        f"Rows evaluated: {int(metrics.get('n_rows', 0))}",
        f"Accuracy: {float(metrics.get('accuracy', 0.0)):.4f}",
        f"Macro precision: {float(metrics.get('precision', 0.0)):.4f}",
        f"Macro recall: {float(metrics.get('recall', 0.0)):.4f}",
        f"Macro F1: {float(metrics.get('f1', 0.0)):.4f}",
        "",
        "## Classwise Metrics",
        "",
        "| Label | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in classwise.to_dict(orient="records"):
        lines.append(
            f"| {row['label']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['f1']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_evaluation(config, params, method: str, participant: str | None = None) -> int:
    """Compare classifier labels with manual labels and write tables and a report.

    One participant: results in <results>/classification-evaluation/<method>/<participant>.
    All participants together: .../<method>/summary. Return 0 if it worked, else 1.
    """
    labels = list(params["annotation"]["labels"])
    annotation_root = config.path("annotations")

    if participant is not None:
        parts = participant.split("/")
        annotation_files = [annotation_root.joinpath(*parts) / ANNOTATION_FILE]
        output_dir = config.results("classification-evaluation", method, *parts)
    else:
        annotation_files = sorted(annotation_root.rglob(ANNOTATION_FILE))
        output_dir = config.results("classification-evaluation", method, "summary")

    frames = []
    for path in annotation_files:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "method" not in table.columns:
            print(f"Note: {path} does not name the method. It is used as it is.")
        elif not (table["method"].astype(str) == method).all():
            print(f"Not used, the segments are from another method: {path}")
            continue
        frames.append(table)
    if not frames:
        print(f"No annotation files for {method} found in {annotation_root}.")
        return 1

    outputs = {
        "metrics": output_dir / "metrics.json",
        "confusion_matrix": output_dir / "confusion_matrix.csv",
        "classwise_metrics": output_dir / "classwise_metrics.csv",
        "evaluated_annotations": output_dir / "evaluated_annotations.csv",
        "report": output_dir / "report.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        print("Evaluation results already exist and are not replaced:")
        for path in existing:
            print(f"  {path}")
        print("Move or rename the old files to evaluate again.")
        return 1

    df = pd.concat(frames, ignore_index=True)

    accuracy = compute_accuracy(df)
    confusion = compute_confusion_matrix(df, labels=labels)
    classwise = compute_classwise_metrics(df, labels=labels)
    macro_precision = float(classwise["precision"].mean()) if not classwise.empty else 0.0
    macro_recall = float(classwise["recall"].mean()) if not classwise.empty else 0.0
    macro_f1 = float(classwise["f1"].mean()) if not classwise.empty else 0.0
    metrics = {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "n_rows": int(len(df)),
    }

    with open(new_output(outputs["metrics"]), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    confusion.to_csv(new_output(outputs["confusion_matrix"]))
    classwise.to_csv(new_output(outputs["classwise_metrics"]), index=False)
    df.to_csv(new_output(outputs["evaluated_annotations"]), index=False)
    new_output(outputs["report"]).write_text(
        generate_evaluation_report(metrics=metrics, classwise=classwise),
        encoding="utf-8",
    )

    print(f"Evaluation of {method}: {len(df)} rows, accuracy {accuracy:.4f}")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0
