#!/usr/bin/env python3
"""
Standalone ADS / IDS annotation tool  (NOT part of the hindibabynet package).

Workflow
--------
1.  Load main_male.wav and/or main_female.wav for a participant.
2.  Detect speech segments via energy-based silence detection
    (recovers the ~0.15 s gaps that were inserted during stream concatenation).
3.  Play each segment; the annotator labels it from the terminal:
        0  =  Other
        1  =  ADS  (Adult-Directed Speech)
        2  =  IDS  (Infant-Directed Speech)
4.  Annotations are saved to a CSV (auto-resume if CSV already exists).
5.  After annotation, export four WAV files per participant:
        <PID>_female_ADS.wav   <PID>_female_IDS.wav
        <PID>_male_ADS.wav     <PID>_male_IDS.wav

Usage examples
--------------
    # Annotate one participant (both speakers)
    python scripts/annotate_ads_ids.py --participant ABAN141223

    # Annotate only the female speaker
    python scripts/annotate_ads_ids.py --participant ABAN141223 --speaker female

    # Resume a previously interrupted session
    python scripts/annotate_ads_ids.py --participant ABAN141223 --resume

    # Skip annotation, just export WAVs from an existing CSV
    python scripts/annotate_ads_ids.py --participant ABAN141223 --export-only

    # Annotate all participants sequentially
    python scripts/annotate_ads_ids.py --all

    # List participants that still need annotation
    python scripts/annotate_ads_ids.py --status

Controls during annotation
--------------------------
    0        →  Label as  Other
    1        →  Label as  ADS
    2        →  Label as  IDS
    r        →  Replay current segment
    b        →  Go back one segment and re-label
    q        →  Save progress and quit
    Ctrl+C   →  Save progress and quit

Requirements (install in your env, NOT in the package):
    uv pip install sounddevice
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASSIFIED_ROOT = Path("/scratch/users/arunps/hindibabynet/audio_classified")
ANNOTATION_ROOT = Path("/scratch/users/arunps/hindibabynet/annotations")

LABEL_MAP = {0: "Other", 1: "ADS", 2: "IDS"}
SPEAKERS = ("female", "male")

# Silence-detection defaults (tuned for 16 kHz, PCM-16 WAVs produced by the pipeline)
FRAME_MS = 20          # analysis frame length in ms
SILENCE_THRESH_DB = -40  # frames below this RMS (dBFS) are considered silence
MIN_SILENCE_SEC = 0.08   # minimum silence duration to split on (the gaps are ~0.15 s)
MIN_SEGMENT_SEC = 0.25   # discard segments shorter than this
MAX_SEGMENT_SEC = 15.0   # split segments longer than this at the quietest internal point


# ============================================================================
# Audio I/O  (stdlib wave + numpy — no soundfile dependency for reading)
# ============================================================================

def read_wav_mono_16(path: Path) -> Tuple[np.ndarray, int]:
    """Read a 16-bit mono WAV to float32 [-1, 1] using stdlib `wave`."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, f"Expected 16-bit WAV, got {wf.getsampwidth()*8}-bit"
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # if stereo, take first channel
    ch = 1  # our files are mono, but just in case
    if len(x) > n:
        ch = len(x) // n
        x = x.reshape(-1, ch)[:, 0]
    return x, sr


def write_wav_int16(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write float32 audio as 16-bit mono WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ============================================================================
# Silence-based segmentation
# ============================================================================

def _rms_frames(audio: np.ndarray, sr: int, frame_ms: int = 20) -> np.ndarray:
    """Compute per-frame RMS energy (dBFS). Returns 1-D array of dB values."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.array([], dtype=np.float32)
    trimmed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    return db.astype(np.float32)


def detect_segments(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = FRAME_MS,
    silence_thresh_db: float = SILENCE_THRESH_DB,
    min_silence_sec: float = MIN_SILENCE_SEC,
    min_segment_sec: float = MIN_SEGMENT_SEC,
    max_segment_sec: float = MAX_SEGMENT_SEC,
) -> List[Tuple[float, float]]:
    """
    Detect speech segments by finding silence gaps in the audio.

    Returns list of (start_sec, end_sec) tuples.
    """
    if len(audio) == 0:
        return []

    frame_len = int(sr * frame_ms / 1000)
    db = _rms_frames(audio, sr, frame_ms)
    if len(db) == 0:
        return []

    is_silence = db < silence_thresh_db
    min_sil_frames = max(1, int(min_silence_sec * 1000 / frame_ms))

    # Find silence regions (runs of True in is_silence)
    silence_regions: List[Tuple[int, int]] = []  # (start_frame, end_frame)
    in_sil = False
    sil_start = 0
    for i, s in enumerate(is_silence):
        if s and not in_sil:
            in_sil = True
            sil_start = i
        elif (not s) and in_sil:
            in_sil = False
            if i - sil_start >= min_sil_frames:
                silence_regions.append((sil_start, i))
    if in_sil and len(is_silence) - sil_start >= min_sil_frames:
        silence_regions.append((sil_start, len(is_silence)))

    # Convert silence regions to segment boundaries
    total_dur = len(audio) / sr
    segments: List[Tuple[float, float]] = []

    if not silence_regions:
        # No silence found → entire audio is one segment
        segments.append((0.0, total_dur))
    else:
        # Before first silence
        first_sil_start = silence_regions[0][0] * frame_len / sr
        if first_sil_start > 0:
            segments.append((0.0, first_sil_start))

        # Between silences
        for i in range(len(silence_regions) - 1):
            seg_start = silence_regions[i][1] * frame_len / sr
            seg_end = silence_regions[i + 1][0] * frame_len / sr
            if seg_end > seg_start:
                segments.append((seg_start, seg_end))

        # After last silence
        last_sil_end = silence_regions[-1][1] * frame_len / sr
        if last_sil_end < total_dur:
            segments.append((last_sil_end, total_dur))

    # Filter out very short segments
    segments = [(s, e) for s, e in segments if (e - s) >= min_segment_sec]

    # Split very long segments at their quietest internal point
    final_segments: List[Tuple[float, float]] = []
    for s, e in segments:
        if (e - s) <= max_segment_sec:
            final_segments.append((s, e))
        else:
            final_segments.extend(_split_long_segment(audio, sr, s, e, max_segment_sec, frame_ms))

    return final_segments


def _split_long_segment(
    audio: np.ndarray,
    sr: int,
    start: float,
    end: float,
    max_sec: float,
    frame_ms: int,
) -> List[Tuple[float, float]]:
    """Recursively split a long segment at its quietest internal point."""
    dur = end - start
    if dur <= max_sec:
        return [(start, end)]

    # Extract segment audio
    s_samp = int(start * sr)
    e_samp = int(end * sr)
    seg = audio[s_samp:e_samp]

    # Find the quietest frame (prefer the middle region)
    db = _rms_frames(seg, sr, frame_ms)
    if len(db) < 3:
        return [(start, end)]

    # Prefer splitting in the middle 60%
    n = len(db)
    lo = n // 5
    hi = n - n // 5
    search_db = db[lo:hi]
    if len(search_db) == 0:
        return [(start, end)]

    split_frame = lo + int(np.argmin(search_db))
    frame_len = int(sr * frame_ms / 1000)
    split_sec = start + split_frame * frame_len / sr

    left = _split_long_segment(audio, sr, start, split_sec, max_sec, frame_ms)
    right = _split_long_segment(audio, sr, split_sec, end, max_sec, frame_ms)
    return left + right


# ============================================================================
# Playback
# ============================================================================

def _try_import_sounddevice():
    """Try to import sounddevice; return module or None."""
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError):
        return None


def play_segment(
    audio: np.ndarray, sr: int, start_sec: float, end_sec: float, sd_module=None
) -> None:
    """Play an audio segment through the speakers."""
    s = max(0, int(start_sec * sr))
    e = min(len(audio), int(end_sec * sr))
    chunk = audio[s:e]

    if sd_module is not None:
        sd_module.play(chunk, sr)
        sd_module.wait()
    else:
        # Fallback: save to temp file and inform user
        tmp = Path("/tmp/_annotate_segment.wav")
        write_wav_int16(tmp, chunk, sr)
        print(f"    [no audio device] Segment saved to: {tmp}")
        print(f"    Play it manually (e.g., on your local machine via scp).")


# ============================================================================
# Annotation CSV persistence
# ============================================================================

def _csv_path(participant_id: str, speaker: str) -> Path:
    return ANNOTATION_ROOT / participant_id / f"{participant_id}_{speaker}_annotations.csv"


def load_annotations(participant_id: str, speaker: str) -> Dict[int, int]:
    """Load existing annotations {segment_index: label}."""
    p = _csv_path(participant_id, speaker)
    if not p.exists():
        return {}
    annotations: Dict[int, int] = {}
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["segment_index"])
            label = int(row["label"])
            annotations[idx] = label
    return annotations


def save_annotations(
    participant_id: str,
    speaker: str,
    segments: List[Tuple[float, float]],
    annotations: Dict[int, int],
) -> Path:
    """Save annotations to CSV. Returns the path."""
    p = _csv_path(participant_id, speaker)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_index", "start_sec", "end_sec", "duration_sec", "label", "label_name"])
        for idx in sorted(annotations.keys()):
            if idx < len(segments):
                s, e = segments[idx]
                label = annotations[idx]
                writer.writerow([idx, f"{s:.4f}", f"{e:.4f}", f"{e-s:.4f}", label, LABEL_MAP[label]])
    return p


# ============================================================================
# WAV export
# ============================================================================

def export_annotated_wavs(
    participant_id: str,
    speaker: str,
    audio: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
    annotations: Dict[int, int],
) -> Dict[str, Optional[Path]]:
    """
    Concatenate annotated segments into ADS and IDS WAV files.
    A small 0.15 s silence gap is inserted between segments.
    Returns {label_name: path_or_None}.
    """
    out_dir = ANNOTATION_ROOT / participant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    gap = np.zeros(int(sr * 0.15), dtype=np.float32)
    results: Dict[str, Optional[Path]] = {}

    for label_code, label_name in LABEL_MAP.items():
        # Gather segments for this label
        pieces: List[np.ndarray] = []
        for idx, lab in sorted(annotations.items()):
            if lab != label_code or idx >= len(segments):
                continue
            s, e = segments[idx]
            s_samp = max(0, int(s * sr))
            e_samp = min(len(audio), int(e * sr))
            chunk = audio[s_samp:e_samp]
            if len(chunk) > 0:
                pieces.append(chunk)
                pieces.append(gap)

        if pieces:
            combined = np.concatenate(pieces)
            out_path = out_dir / f"{participant_id}_{speaker}_{label_name}.wav"
            write_wav_int16(out_path, combined, sr)
            results[label_name] = out_path
        else:
            results[label_name] = None

    return results


# ============================================================================
# Interactive annotation loop
# ============================================================================

def annotate_speaker(
    participant_id: str,
    speaker: str,
    audio: np.ndarray,
    sr: int,
    segments: List[Tuple[float, float]],
    resume: bool = True,
) -> Dict[int, int]:
    """Run the interactive annotation loop for one speaker."""
    sd = _try_import_sounddevice()
    if sd is None:
        print("\n  WARNING: 'sounddevice' not installed or no audio device available.")
        print("  Install with:  pip install sounddevice")
        print("  Segments will be saved to /tmp for manual playback.\n")

    # Load existing annotations if resuming
    annotations: Dict[int, int] = {}
    if resume:
        annotations = load_annotations(participant_id, speaker)
        if annotations:
            print(f"  Loaded {len(annotations)} existing annotations (resuming).")

    total = len(segments)
    # Find first unannotated segment
    idx = 0
    if resume and annotations:
        unannotated = [i for i in range(total) if i not in annotations]
        if unannotated:
            idx = unannotated[0]
        else:
            print(f"  All {total} segments already annotated!")
            return annotations

    total_dur = sum(e - s for s, e in segments)
    annotated_dur = sum(
        segments[i][1] - segments[i][0] for i in annotations if i < len(segments)
    )

    print(f"\n{'='*60}")
    print(f"  Participant : {participant_id}")
    print(f"  Speaker     : {speaker}")
    print(f"  Segments    : {total}  (total duration: {total_dur:.1f} s = {total_dur/60:.1f} min)")
    print(f"  Annotated   : {len(annotations)}  ({annotated_dur:.1f} s)")
    print(f"  Remaining   : {total - len(annotations)}")
    print(f"{'='*60}")
    print(f"  Labels:  0 = Other  |  1 = ADS  |  2 = IDS")
    print(f"  Controls:  r = replay  |  b = back  |  q = save & quit")
    print(f"{'='*60}\n")

    try:
        while 0 <= idx < total:
            s, e = segments[idx]
            dur = e - s
            existing = annotations.get(idx)
            existing_str = f"  [current: {LABEL_MAP[existing]}]" if existing is not None else ""

            print(f"  [{idx+1}/{total}]  {s:.2f}s – {e:.2f}s  ({dur:.2f}s){existing_str}")

            # Play the segment
            play_segment(audio, sr, s, e, sd)

            # Get user input
            while True:
                try:
                    raw = input("    Label (0/1/2/r/b/q): ").strip().lower()
                except EOFError:
                    raw = "q"

                if raw == "q":
                    print("\n  Saving progress and quitting...")
                    save_annotations(participant_id, speaker, segments, annotations)
                    return annotations
                elif raw == "r":
                    print("    (replaying...)")
                    play_segment(audio, sr, s, e, sd)
                    continue
                elif raw == "b":
                    if idx > 0:
                        idx -= 1
                        print(f"    (going back to segment {idx+1})")
                    else:
                        print("    (already at first segment)")
                    break
                elif raw in ("0", "1", "2"):
                    label = int(raw)
                    annotations[idx] = label
                    print(f"    → {LABEL_MAP[label]}")
                    idx += 1

                    # Auto-save every 10 segments
                    if len(annotations) % 10 == 0:
                        save_annotations(participant_id, speaker, segments, annotations)

                    break
                else:
                    print("    Invalid input. Use 0, 1, 2, r, b, or q.")

    except KeyboardInterrupt:
        print("\n\n  Interrupted! Saving progress...")

    save_annotations(participant_id, speaker, segments, annotations)
    return annotations


# ============================================================================
# Per-participant orchestration
# ============================================================================

def process_participant(
    participant_id: str,
    speakers: Tuple[str, ...] = SPEAKERS,
    resume: bool = True,
    export_only: bool = False,
    silence_db: float = SILENCE_THRESH_DB,
    min_silence: float = MIN_SILENCE_SEC,
    min_segment: float = MIN_SEGMENT_SEC,
    max_segment: float = MAX_SEGMENT_SEC,
) -> None:
    """Annotate (or just export) one participant."""
    pdir = CLASSIFIED_ROOT / participant_id
    if not pdir.is_dir():
        print(f"ERROR: Participant directory not found: {pdir}")
        return

    for speaker in speakers:
        wav_path = pdir / f"{participant_id}_main_{speaker}.wav"
        if not wav_path.exists():
            print(f"  SKIP: {wav_path.name} not found.")
            continue

        print(f"\n{'#'*60}")
        print(f"  Loading {wav_path.name} ...")
        audio, sr = read_wav_mono_16(wav_path)
        dur = len(audio) / sr
        print(f"  Duration: {dur:.1f} s  ({dur/60:.1f} min)  |  SR: {sr} Hz")

        # Detect segments
        print(f"  Detecting segments (silence threshold={silence_db} dB) ...")
        segments = detect_segments(
            audio, sr,
            silence_thresh_db=silence_db,
            min_silence_sec=min_silence,
            min_segment_sec=min_segment,
            max_segment_sec=max_segment,
        )
        print(f"  Found {len(segments)} segments.")

        if len(segments) == 0:
            print(f"  No segments found — skipping.")
            continue

        # Show segment duration statistics
        durs = [e - s for s, e in segments]
        print(f"  Segment durations:  min={min(durs):.2f}s  "
              f"median={sorted(durs)[len(durs)//2]:.2f}s  "
              f"max={max(durs):.2f}s  "
              f"total={sum(durs):.1f}s")

        if export_only:
            annotations = load_annotations(participant_id, speaker)
            if not annotations:
                print(f"  No annotations found for {speaker} — nothing to export.")
                continue
        else:
            annotations = annotate_speaker(
                participant_id, speaker, audio, sr, segments, resume=resume
            )

        # Export WAVs
        n_ann = len(annotations)
        if n_ann == 0:
            print(f"  No annotations — skipping export.")
            continue

        n_ads = sum(1 for v in annotations.values() if v == 1)
        n_ids = sum(1 for v in annotations.values() if v == 2)
        n_other = sum(1 for v in annotations.values() if v == 0)
        print(f"\n  Summary:  ADS={n_ads}  IDS={n_ids}  Other={n_other}  "
              f"(total annotated: {n_ann}/{len(segments)})")

        if n_ann < len(segments):
            ans = input(f"  {len(segments) - n_ann} segments not yet annotated. Export anyway? (y/n): ").strip().lower()
            if ans != "y":
                print("  Skipping export.")
                continue

        print(f"  Exporting WAV files ...")
        results = export_annotated_wavs(participant_id, speaker, audio, sr, segments, annotations)
        for label_name, path in results.items():
            if path:
                print(f"    {label_name:6s} → {path}")
            else:
                print(f"    {label_name:6s} → (no segments)")

    print(f"\n  Done with {participant_id}.\n")


# ============================================================================
# Status view
# ============================================================================

def show_status() -> None:
    """Print annotation status for all participants."""
    participants = sorted(
        d.name for d in CLASSIFIED_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    print(f"\n{'Participant':<16} {'Female WAV':>11} {'Male WAV':>10} {'Fem Ann':>9} {'Mal Ann':>9}")
    print("-" * 60)
    for pid in participants:
        pdir = CLASSIFIED_ROOT / pid
        fem_exists = (pdir / f"{pid}_main_female.wav").exists()
        mal_exists = (pdir / f"{pid}_main_male.wav").exists()
        fem_ann = load_annotations(pid, "female")
        mal_ann = load_annotations(pid, "male")
        print(
            f"{pid:<16} "
            f"{'✓' if fem_exists else '✗':>11} "
            f"{'✓' if mal_exists else '✗':>10} "
            f"{len(fem_ann) if fem_ann else '-':>9} "
            f"{len(mal_ann) if mal_ann else '-':>9}"
        )
    print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Annotate main_male / main_female WAVs as ADS, IDS, or Other.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Labels:
    0 = Other (neither ADS nor IDS)
    1 = ADS   (Adult-Directed Speech)
    2 = IDS   (Infant-Directed Speech)

Examples:
    python scripts/annotate_ads_ids.py --participant ABAN141223
    python scripts/annotate_ads_ids.py --participant ABAN141223 --speaker female
    python scripts/annotate_ads_ids.py --all
    python scripts/annotate_ads_ids.py --status
    python scripts/annotate_ads_ids.py --participant ABAN141223 --export-only
""",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--participant", "-p", type=str, help="Participant ID (e.g. ABAN141223)")
    group.add_argument("--all", action="store_true", help="Annotate all participants sequentially")
    group.add_argument("--status", action="store_true", help="Show annotation status for all participants")

    parser.add_argument(
        "--speaker", "-s", choices=["female", "male", "both"], default="both",
        help="Which speaker to annotate (default: both)",
    )
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from previous annotations (default)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignoring previous annotations")
    parser.add_argument("--export-only", action="store_true", help="Skip annotation; just export WAVs from existing CSV")

    # Silence detection tuning
    parser.add_argument("--silence-db", type=float, default=SILENCE_THRESH_DB,
                        help=f"Silence threshold in dBFS (default: {SILENCE_THRESH_DB})")
    parser.add_argument("--min-silence", type=float, default=MIN_SILENCE_SEC,
                        help=f"Min silence duration to split on, seconds (default: {MIN_SILENCE_SEC})")
    parser.add_argument("--min-segment", type=float, default=MIN_SEGMENT_SEC,
                        help=f"Discard segments shorter than this, seconds (default: {MIN_SEGMENT_SEC})")
    parser.add_argument("--max-segment", type=float, default=MAX_SEGMENT_SEC,
                        help=f"Split segments longer than this, seconds (default: {MAX_SEGMENT_SEC})")

    args = parser.parse_args()

    resume = not args.no_resume
    speakers = SPEAKERS if args.speaker == "both" else (args.speaker,)

    if args.status:
        show_status()
        return

    seg_kwargs = dict(
        silence_db=args.silence_db,
        min_silence=args.min_silence,
        min_segment=args.min_segment,
        max_segment=args.max_segment,
    )

    if args.participant:
        process_participant(args.participant, speakers=speakers, resume=resume, export_only=args.export_only, **seg_kwargs)
    elif args.all:
        participants = sorted(
            d.name for d in CLASSIFIED_ROOT.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
        print(f"Found {len(participants)} participants.\n")
        for i, pid in enumerate(participants, 1):
            print(f"\n{'='*60}")
            print(f"  PARTICIPANT {i}/{len(participants)}: {pid}")
            print(f"{'='*60}")
            process_participant(pid, speakers=speakers, resume=resume, export_only=args.export_only, **seg_kwargs)


if __name__ == "__main__":
    main()
