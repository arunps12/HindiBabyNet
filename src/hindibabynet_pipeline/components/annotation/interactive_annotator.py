from __future__ import annotations

from pathlib import Path

import pandas as pd

from hindibabynet_pipeline.components.annotation.annotation_schema import CONTROL_COMMANDS, LABEL_SHORTCUTS
from hindibabynet_pipeline.components.annotation.audio_player import play_wav
from hindibabynet_pipeline.utils.audio_utils import write_wav_chunk
from hindibabynet_pipeline.utils.io_utils import ensure_dir


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


def run_interactive_annotation(sampled_df: pd.DataFrame, audio_path: Path, output_csv: Path) -> Path:
    ensure_dir(output_csv.parent)
    existing_df = _load_existing(output_csv)
    working_df = _merge_existing(sampled_df, existing_df)
    temp_dir = output_csv.parent / "_tmp_segments"
    ensure_dir(temp_dir)

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
        response = input("Label [1-6, f/m/k/o/u/n, r=repeat, b=back, q=quit]: ").strip().lower()

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

        manual_label = LABEL_SHORTCUTS.get(response, response)
        working_df.at[index, "manual_label"] = manual_label
        working_df.to_csv(output_csv, index=False)
        index += 1

    working_df.to_csv(output_csv, index=False)
    return output_csv