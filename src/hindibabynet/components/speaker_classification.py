"""
Stage 03: Speaker Classification

Full pipeline per analysis WAV:
  Step 2  – WebRTC VAD  →  speech intervals
  Step 3  – Chunked pyannote diarization  →  speaker turns
  Step 4  – Intersect VAD ∩ diarization, drop < min_segment_sec, merge close
  Step 5  – eGeMAPS + XGBoost classification  →  adult_male / adult_female / child / background
  Step 6  – Aggregate class streams
  Step 7  – Secondary diarization on female/male streams  →  main speaker
  Step 8  – Export main_female.wav, main_male.wav
  Step 9  – TextGrid generation

All logic faithfully replicates notebooks/00_research.ipynb.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
import torch
import webrtcvad
from praatio import textgrid as tgio
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from src.hindibabynet.entity.artifact_entity import SpeakerClassificationArtifact
from src.hindibabynet.entity.config_entity import SpeakerClassificationConfig
from src.hindibabynet.exception.exception import wrap_exception
from src.hindibabynet.logging.logger import get_logger
from src.hindibabynet.utils.io_utils import ensure_dir, write_json, write_parquet

logger = get_logger(__name__)

# ============================================================================
# Label mapping  (must match training order in the XGBoost model)
# ============================================================================
LABEL2ID: Dict[str, int] = {
    "adult_male": 0,
    "adult_female": 1,
    "child": 2,
    "background": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# TextGrid tier names keyed by predicted class
TIER_MAP: Dict[str, str] = {
    "adult_female": "FEM",
    "adult_male": "MAL",
    "child": "CHILD",
    "background": "BACKGROUND",
}


# ============================================================================
# Audio helpers  (from notebook cells)
# ============================================================================
def load_audio_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False), sr


def resample_audio(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return x
    gcd = int(np.gcd(sr, target_sr))
    up = target_sr // gcd
    down = sr // gcd
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    n = len(x)
    if n == target_len:
        return x
    if n > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = x
    return out


def slice_audio(x: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    s = max(0, int(round(start_sec * sr)))
    e = min(len(x), int(round(end_sec * sr)))
    return x[s:e]


# ============================================================================
# Step 2 — VAD  (exact notebook: webrtc_vad_regions_streaming)
# ============================================================================
def webrtc_vad_regions(
    path: Path,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    min_region_ms: int = 300,
) -> List[Tuple[float, float]]:
    """Return speech intervals (start_sec, end_sec) via WebRTC VAD."""
    vad = webrtcvad.Vad(aggressiveness)
    info = sf.info(str(path))
    sr = int(info.samplerate)

    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad needs sr in (8k,16k,32k,48k), got {sr}")

    frame_len = int(sr * frame_ms / 1000)

    speech_flags: List[bool] = []
    with sf.SoundFile(str(path), mode="r") as f:
        while True:
            frame = f.read(frames=frame_len, dtype="int16", always_2d=True)
            if frame.size == 0 or len(frame) < frame_len:
                break
            mono = frame[:, 0]
            speech_flags.append(vad.is_speech(mono.tobytes(), sr))

    # merge consecutive True flags into regions
    regions: List[Tuple[int, int]] = []
    in_speech = False
    start_i = 0
    for i, is_speech in enumerate(speech_flags):
        if is_speech and not in_speech:
            in_speech = True
            start_i = i
        elif (not is_speech) and in_speech:
            in_speech = False
            regions.append((start_i, i))
    if in_speech:
        regions.append((start_i, len(speech_flags)))

    out: List[Tuple[float, float]] = []
    for s_i, e_i in regions:
        s = (s_i * frame_len) / sr
        e = (e_i * frame_len) / sr
        if (e - s) * 1000 >= min_region_ms:
            out.append((float(s), float(e)))
    return out


# ============================================================================
# Step 3 — Chunked diarization  (exact notebook logic)
# ============================================================================
def _make_chunks(
    duration_sec: float, chunk_sec: float, overlap_sec: float
) -> List[Tuple[int, float, float]]:
    """Yield (chunk_id, chunk_start, chunk_end) with overlap."""
    step = chunk_sec - overlap_sec
    assert step > 0, "chunk_sec must be > overlap_sec"
    chunks: List[Tuple[int, float, float]] = []
    t = 0.0
    chunk_id = 0
    while t < duration_sec:
        s = t
        e = min(t + chunk_sec, duration_sec)
        chunks.append((chunk_id, s, e))
        if e >= duration_sec:
            break
        t += step
        chunk_id += 1
    return chunks


def _write_wav_chunk(
    wav_path: Path, chunk_path: Path, start_sec: float, end_sec: float
) -> Optional[Path]:
    """Write a time-slice of a WAV to disk. Returns None on failure."""
    try:
        info = sf.info(str(wav_path))
        sr = info.samplerate
        start_frame = max(0, int(start_sec * sr))
        end_frame = min(info.frames, int(end_sec * sr))
        n_frames = end_frame - start_frame
        if n_frames <= 0:
            return None
        audio, _ = sf.read(str(wav_path), start=start_frame, frames=n_frames, dtype="float32")
        if audio is None or audio.size == 0:
            return None
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(chunk_path), audio, sr, format="WAV", subtype="PCM_16")
        return chunk_path
    except Exception as e:
        logger.warning(f"write_wav_chunk failed ({chunk_path.name}): {e}")
        return None


def diarize_wav(
    wav_path: Path,
    diar_pipeline,
    chunk_sec: float,
    overlap_sec: float,
    min_speakers: int,
    max_speakers: int,
    tmp_dir: Path,
) -> pd.DataFrame:
    """Run chunked diarization on a single WAV, returning speaker turns."""
    info = sf.info(str(wav_path))
    full_duration = float(info.duration)
    chunks = _make_chunks(full_duration, chunk_sec, overlap_sec)

    all_rows: List[Dict[str, Any]] = []
    for chunk_id, chunk_start, chunk_end in chunks:
        chunk_wav = tmp_dir / f"{wav_path.stem}_chunk{chunk_id:04d}.wav"
        out = _write_wav_chunk(wav_path, chunk_wav, chunk_start, chunk_end)
        if out is None:
            continue
        try:
            diar = diar_pipeline(
                {"audio": str(chunk_wav)},
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            for seg, _, spk in diar.itertracks(yield_label=True):
                s = float(seg.start) + chunk_start
                e = float(seg.end) + chunk_start
                if e <= s:
                    continue
                all_rows.append(
                    {
                        "chunk_id": int(chunk_id),
                        "speaker_id_local": spk,
                        "start_sec": s,
                        "end_sec": e,
                        "duration_sec": float(e - s),
                    }
                )
        except Exception as exc:
            logger.warning(f"Diarization failed chunk {chunk_id}: {exc}")
        finally:
            try:
                chunk_wav.unlink(missing_ok=True)
            except Exception:
                pass

    if not all_rows:
        return pd.DataFrame(
            columns=["chunk_id", "speaker_id_local", "start_sec", "end_sec", "duration_sec"]
        )
    return (
        pd.DataFrame(all_rows)
        .sort_values(["start_sec", "end_sec"])
        .reset_index(drop=True)
    )


# ============================================================================
# Step 4 — Intersect + merge  (exact notebook logic)
# ============================================================================
def intersect_turns_with_vad(
    turns_df: pd.DataFrame,
    vad_intervals: List[Tuple[float, float]],
    min_keep_sec: float = 0.0,
) -> pd.DataFrame:
    """Intersect diarization turns with VAD regions (sweep-line)."""
    if turns_df.empty or not vad_intervals:
        return pd.DataFrame(
            columns=["start_sec", "end_sec", "duration_sec", "chunk_id", "speaker_id_local"]
        )

    turns_df = turns_df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    diar_arr = turns_df[["start_sec", "end_sec", "chunk_id", "speaker_id_local"]].to_numpy()
    vad_arr = np.array(sorted(vad_intervals), dtype=float)

    i, j = 0, 0
    rows: List[Dict] = []

    while i < len(diar_arr) and j < len(vad_arr):
        ds, de = float(diar_arr[i, 0]), float(diar_arr[i, 1])
        vs, ve = float(vad_arr[j, 0]), float(vad_arr[j, 1])

        s = max(ds, vs)
        e = min(de, ve)
        if s < e:
            dur = e - s
            if dur >= min_keep_sec:
                rows.append(
                    {
                        "start_sec": s,
                        "end_sec": e,
                        "duration_sec": dur,
                        "chunk_id": int(diar_arr[i, 2]),
                        "speaker_id_local": str(diar_arr[i, 3]),
                    }
                )
        if de <= ve:
            i += 1
        else:
            j += 1

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["start_sec", "end_sec", "duration_sec", "chunk_id", "speaker_id_local"]
    )


def merge_close_segments(
    df: pd.DataFrame,
    gap_thresh: float = 0.5,
) -> pd.DataFrame:
    """Merge temporally close segments from the same speaker. (Notebook reference.)"""
    if df.empty:
        return df.copy()

    df = df.copy()
    df["_orig_row"] = df.index

    # sort columns that exist
    sort_cols = [c for c in [
        "wav_path", "chunk_id", "speaker_id_local", "start_sec", "end_sec"
    ] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    prev_wav = df["wav_path"].shift(1) if "wav_path" in df.columns else pd.Series(dtype=object)
    prev_chunk = df["chunk_id"].shift(1)
    prev_spk = df["speaker_id_local"].shift(1)
    prev_end = df["end_sec"].shift(1)
    gap = df["start_sec"] - prev_end

    new_group = (
        (df["chunk_id"] != prev_chunk)
        | (df["speaker_id_local"] != prev_spk)
        | (gap.isna())
        | (gap < 0)
        | (gap > gap_thresh)
    )
    if "wav_path" in df.columns:
        new_group = new_group | (df["wav_path"] != prev_wav)

    df["_merge_group"] = new_group.cumsum()

    agg_dict: Dict[str, Any] = {
        "chunk_id": ("chunk_id", "first"),
        "speaker_id_local": ("speaker_id_local", "first"),
        "start_sec": ("start_sec", "min"),
        "end_sec": ("end_sec", "max"),
        "n_merged": ("_orig_row", "count"),
    }
    if "wav_path" in df.columns:
        agg_dict["wav_path"] = ("wav_path", "first")

    out = df.groupby("_merge_group", as_index=False).agg(**agg_dict)
    out["duration_sec"] = out["end_sec"] - out["start_sec"]

    sort_out = [c for c in ["wav_path", "chunk_id", "start_sec"] if c in out.columns]
    return out.sort_values(sort_out).reset_index(drop=True)


# ============================================================================
# Step 5 — eGeMAPS feature extraction + classification  (exact notebook)
# ============================================================================
@dataclass
class EGemapsExtractor:
    """Wrapper around opensmile eGeMAPSv02 Functionals."""

    egemaps_dim: int = 88
    target_sr: int = 16000
    win_sec: float = 1.0

    def __post_init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def _fix_dim(self, vec: np.ndarray) -> np.ndarray:
        vec = vec.astype(np.float32).flatten()
        if vec.shape[0] == self.egemaps_dim:
            return vec
        out = np.zeros(self.egemaps_dim, dtype=np.float32)
        m = min(self.egemaps_dim, vec.shape[0])
        out[:m] = vec[:m]
        return out


def generate_windows(
    start: float, end: float, win: float = 1.0, hop: float = 0.5
) -> List[Tuple[float, float, float]]:
    """
    Notebook windowing strategy:
      - Short segments (< win) → single padded window.
      - Long segments → 1.0 s windows with 0.5 s hop + end-anchored last window.
    Returns list of (w_start, w_end, weight_duration).
    """
    dur = end - start
    if dur <= 0 or dur < win:
        return [(start, start + win, win)]

    windows: List[Tuple[float, float, float]] = []
    t = start
    while t + win <= end:
        windows.append((t, t + win, win))
        t += hop

    # end-anchored last window
    if not windows or windows[-1][1] < end:
        windows.append((end - win, end, win))
    return windows


def extract_egemaps_for_window(
    extractor: EGemapsExtractor,
    audio_16k: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Extract eGeMAPS from a window of pre-loaded 16 kHz mono audio."""
    s = max(0, int(round(start_sec * sr)))
    e = min(len(audio_16k), int(round(end_sec * sr)))
    seg = audio_16k[s:e]
    target_len = int(sr * extractor.win_sec)
    seg = crop_or_pad(seg, target_len)
    try:
        feats = extractor.smile.process_signal(seg, sr)
        vec = feats.values.flatten()
        return extractor._fix_dim(vec)
    except Exception:
        return np.zeros(extractor.egemaps_dim, dtype=np.float32)


def weighted_mean_probs(P: np.ndarray, weights: List[float]) -> np.ndarray:
    """Weighted mean of per-window probability vectors (notebook aggregation)."""
    W = np.array(weights, dtype=np.float32).reshape(-1, 1)
    return (P * W).sum(axis=0) / (W.sum() + 1e-12)


def classify_segments(
    df: pd.DataFrame,
    audio_16k: np.ndarray,
    sr: int,
    model,
    extractor: EGemapsExtractor,
    class_names: List[str],
    win: float = 1.0,
    hop: float = 0.5,
) -> pd.DataFrame:
    """
    Classify each segment using windowed eGeMAPS + XGBoost.
    Adds columns: probs_<class>, predicted_class, predicted_confidence, n_windows.
    """
    if df.empty:
        for cn in class_names:
            df[f"probs_{cn}"] = []
        df["predicted_class"] = []
        df["predicted_confidence"] = []
        df["n_windows"] = []
        return df

    probs_list: List[np.ndarray] = []
    nwin_list: List[int] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying", leave=False):
        windows = generate_windows(float(row.start_sec), float(row.end_sec), win=win, hop=hop)
        X_list = []
        weights = []
        for ws, we, wdur in windows:
            vec = extract_egemaps_for_window(extractor, audio_16k, sr, ws, we)
            X_list.append(vec)
            weights.append(wdur)

        Xw = np.stack(X_list, axis=0).astype(np.float32)
        Pw = model.predict_proba(Xw).astype(np.float32)
        p_final = weighted_mean_probs(Pw, weights)
        probs_list.append(p_final)
        nwin_list.append(len(windows))

    P = np.vstack(probs_list)
    df_out = df.copy()
    df_out["n_windows"] = nwin_list
    for i, cn in enumerate(class_names):
        df_out[f"probs_{cn}"] = P[:, i].astype(float)

    pred_idx = np.argmax(P, axis=1)
    df_out["predicted_class"] = [class_names[i] for i in pred_idx]
    df_out["predicted_confidence"] = P[np.arange(len(df_out)), pred_idx].astype(float)
    return df_out


# ============================================================================
# Step 6 — Build combined class-stream audio  (notebook: build_combined_audio)
# ============================================================================
def build_class_stream(
    df: pd.DataFrame,
    audio_16k: np.ndarray,
    sr: int,
    target_class: str,
    gap_sec: float = 0.15,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Concatenate all segments of *target_class* into a single audio array.
    Returns (audio_array, used_rows_df).
    """
    sel = df[df["predicted_class"] == target_class].copy()
    sel = sel.sort_values("start_sec").reset_index(drop=True)

    if sel.empty:
        return np.array([], dtype=np.float32), sel

    gap = np.zeros(int(sr * gap_sec), dtype=np.float32) if gap_sec > 0 else None
    pieces: List[np.ndarray] = []

    for _, r in sel.iterrows():
        seg = slice_audio(audio_16k, sr, float(r.start_sec), float(r.end_sec))
        if len(seg) == 0:
            continue
        pieces.append(seg)
        if gap is not None:
            pieces.append(gap)

    if not pieces:
        return np.array([], dtype=np.float32), sel

    y = np.concatenate(pieces).astype(np.float32, copy=False)
    return y, sel


# ============================================================================
# Step 7 — Secondary diarization → find main speaker
# ============================================================================
def find_main_speaker_segments(
    stream_audio: np.ndarray,
    sr: int,
    diar_pipeline,
    tmp_wav_path: Path,
    min_speakers: int = 1,
    max_speakers: int = 3,
) -> Optional[str]:
    """
    Diarize a class stream and return the speaker label with the most total time.
    Returns None if no speakers found.
    """
    if len(stream_audio) < sr:  # < 1 sec
        return None

    ensure_dir(tmp_wav_path.parent)
    sf.write(str(tmp_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")

    try:
        diar = diar_pipeline(
            {"audio": str(tmp_wav_path)},
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except Exception as e:
        logger.warning(f"Secondary diarization failed: {e}")
        return None
    finally:
        try:
            tmp_wav_path.unlink(missing_ok=True)
        except Exception:
            pass

    durations: Dict[str, float] = {}
    for seg, _, spk in diar.itertracks(yield_label=True):
        durations[spk] = durations.get(spk, 0.0) + float(seg.end - seg.start)

    if not durations:
        return None
    return max(durations, key=lambda k: durations[k])


def extract_main_speaker_audio(
    stream_audio: np.ndarray,
    sr: int,
    diar_pipeline,
    tmp_wav_path: Path,
    out_wav_path: Path,
    min_speakers: int = 1,
    max_speakers: int = 3,
    gap_sec: float = 0.15,
) -> Path:
    """
    Diarize the stream, keep only the dominant speaker's segments,
    and write to out_wav_path. Falls back to full stream if diarization
    finds nothing or only 1 speaker.
    """
    ensure_dir(out_wav_path.parent)

    # If stream too short, just write it directly
    if len(stream_audio) < sr:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
        return out_wav_path

    # Write temp wav for diarization
    ensure_dir(tmp_wav_path.parent)
    sf.write(str(tmp_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")

    try:
        diar = diar_pipeline(
            {"audio": str(tmp_wav_path)},
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except Exception as e:
        logger.warning(f"Secondary diarization failed, using full stream: {e}")
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
        return out_wav_path
    finally:
        try:
            tmp_wav_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Find dominant speaker
    durations: Dict[str, float] = {}
    segments_by_spk: Dict[str, List[Tuple[float, float]]] = {}
    for seg, _, spk in diar.itertracks(yield_label=True):
        s, e = float(seg.start), float(seg.end)
        durations[spk] = durations.get(spk, 0.0) + (e - s)
        segments_by_spk.setdefault(spk, []).append((s, e))

    if not durations:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
        return out_wav_path

    main_spk = max(durations, key=lambda k: durations[k])
    main_segs = sorted(segments_by_spk[main_spk])

    gap = np.zeros(int(sr * gap_sec), dtype=np.float32) if gap_sec > 0 else None
    pieces: List[np.ndarray] = []
    for seg_s, seg_e in main_segs:
        chunk = slice_audio(stream_audio, sr, seg_s, seg_e)
        if len(chunk) > 0:
            pieces.append(chunk)
            if gap is not None:
                pieces.append(gap)

    if not pieces:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
    else:
        y = np.concatenate(pieces).astype(np.float32, copy=False)
        sf.write(str(out_wav_path), y, sr, format="WAV", subtype="PCM_16")

    return out_wav_path


# ============================================================================
# Step 9 — TextGrid generation  (exact notebook: df_to_textgrid_by_speaker)
# ============================================================================
def _make_interval_tier(name: str, entries, xmin: float, xmax: float):
    """Create IntervalTier across praatio versions."""
    try:
        return tgio.IntervalTier(str(name), entries, xmin, xmax)
    except TypeError:
        pass
    try:
        return tgio.IntervalTier(name=str(name), entries=entries, minT=xmin, maxT=xmax)
    except TypeError:
        pass
    return tgio.IntervalTier(name=str(name), entryList=entries, minT=xmin, maxT=xmax)


def write_textgrid(
    df: pd.DataFrame,
    duration_sec: float,
    out_path: Path,
    tier_map: Dict[str, str],
) -> Path:
    """
    Write a TextGrid with one tier per predicted class.
    tier_map maps predicted_class → tier name.
    """
    ensure_dir(out_path.parent)
    xmin, xmax = 0.0, duration_sec

    tg = tgio.Textgrid()
    tg.minTimestamp = xmin
    tg.maxTimestamp = xmax

    for pred_class, tier_name in tier_map.items():
        sel = df[df["predicted_class"] == pred_class].copy()
        sel = sel.sort_values("start_sec")

        entries: List[Tuple[float, float, str]] = []
        for r in sel.itertuples(index=False):
            s = max(xmin, min(float(r.start_sec), xmax))
            e = max(xmin, min(float(r.end_sec), xmax))
            if e > s:
                entries.append((s, e, tier_name))

        # resolve overlaps within tier
        entries.sort(key=lambda x: (x[0], x[1]))
        cleaned: List[Tuple[float, float, str]] = []
        last_end = -1.0
        for s, e, lab in entries:
            if s < last_end:
                s = last_end
            if e > s:
                cleaned.append((s, e, lab))
                last_end = e

        tier = _make_interval_tier(tier_name, cleaned, xmin, xmax)
        tg.addTier(tier)

    tg.save(str(out_path), format="short_textgrid", includeBlankSpaces=True)
    return out_path


# ============================================================================
# Main component class
# ============================================================================
class SpeakerClassification:
    """
    Stage 03: Full speaker-classification pipeline for one analysis WAV.

    Steps: VAD → diarization → intersect → merge → classify → aggregate →
           secondary diar → export wavs → TextGrid
    """

    def __init__(self, config: SpeakerClassificationConfig):
        self.config = config
        self._diar_pipeline = None  # lazy-loaded
        self._model = None
        self._extractor: Optional[EGemapsExtractor] = None

    # ---- lazy loaders ----
    def _load_diarization_pipeline(self):
        if self._diar_pipeline is not None:
            return self._diar_pipeline

        from dotenv import load_dotenv
        from pyannote.audio import Pipeline

        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")

        # Set HuggingFace cache to scratch
        user = os.environ.get("USER", "user")
        scratch_cache = f"/scratch/users/{user}/.cache/huggingface"
        os.environ.setdefault("HF_HOME", scratch_cache)
        os.environ.setdefault("HF_HUB_CACHE", f"{scratch_cache}/hub")
        os.environ.setdefault("TRANSFORMERS_CACHE", f"{scratch_cache}/transformers")
        os.environ.setdefault("PYANNOTE_DISABLE_NOTEBOOK", "1")

        # Suppress known harmless warnings from pyannote / torch internals
        warnings.filterwarnings(
            "ignore",
            message=".*weights_only=False.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*weights_only=False.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*TensorFloat-32.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*std\(\): degrees of freedom is <= 0.*",
            category=UserWarning,
        )

        logger.info(f"Loading diarization pipeline: {self.config.diarization_model}")
        pipeline = Pipeline.from_pretrained(
            self.config.diarization_model, use_auth_token=hf_token
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        logger.info(f"Diarization pipeline loaded on {device}")
        self._diar_pipeline = pipeline
        return self._diar_pipeline

    def _load_model(self):
        if self._model is not None:
            return self._model
        model_path = self.config.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._model = joblib.load(model_path)

        # Force model to CPU — input is always numpy (CPU) arrays.
        # Without this, a model saved on cuda triggers a device-mismatch
        # warning and a slow DMatrix fallback on every predict_proba call.
        try:
            self._model.set_params(device="cpu")
        except Exception:
            pass  # older xgboost versions without device param

        logger.info(f"Loaded classifier: {type(self._model)} from {model_path}")

        # sanity check
        dummy = np.zeros((1, self.config.egemaps_dim), dtype=np.float32)
        proba = self._model.predict_proba(dummy)
        assert proba.shape[1] == len(self.config.class_names), (
            f"Model outputs {proba.shape[1]} classes, expected {len(self.config.class_names)}"
        )
        return self._model

    def _get_extractor(self) -> EGemapsExtractor:
        if self._extractor is None:
            self._extractor = EGemapsExtractor(
                egemaps_dim=self.config.egemaps_dim,
                target_sr=16000,
                win_sec=self.config.classify_win_sec,
            )
        return self._extractor

    # ---- main entry point ----
    def run(
        self,
        analysis_wav_path: Path,
        participant_id: str,
    ) -> SpeakerClassificationArtifact:
        """Process one analysis-ready WAV through the full classification pipeline."""
        try:
            ensure_dir(self.config.artifacts_dir)
            ensure_dir(self.config.output_audio_root / participant_id)

            if not analysis_wav_path.exists():
                raise FileNotFoundError(f"analysis wav not found: {analysis_wav_path}")

            wav_info = sf.info(str(analysis_wav_path))
            full_duration = float(wav_info.duration)
            logger.info(
                f"[{participant_id}] analysis wav: {analysis_wav_path} "
                f"| dur={full_duration/3600:.2f}h sr={wav_info.samplerate}"
            )

            # --- Step 2: VAD ---
            logger.info(f"[{participant_id}] Step 2: VAD (aggressiveness={self.config.vad_aggressiveness})")
            vad_intervals = webrtc_vad_regions(
                analysis_wav_path,
                aggressiveness=self.config.vad_aggressiveness,
                frame_ms=self.config.vad_frame_ms,
                min_region_ms=self.config.vad_min_region_ms,
            )
            logger.info(f"[{participant_id}] VAD regions: {len(vad_intervals)}")

            # --- Step 3: Chunked diarization ---
            logger.info(f"[{participant_id}] Step 3: Diarization (chunk={self.config.chunk_sec}s)")
            diar_pipeline = self._load_diarization_pipeline()
            tmp_dir = self.config.output_audio_root / "_tmp_diar" / participant_id
            ensure_dir(tmp_dir)

            turns_df = diarize_wav(
                wav_path=analysis_wav_path,
                diar_pipeline=diar_pipeline,
                chunk_sec=self.config.chunk_sec,
                overlap_sec=self.config.overlap_sec,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
                tmp_dir=tmp_dir,
            )
            logger.info(f"[{participant_id}] Diarization turns: {len(turns_df)}")

            # --- Step 4: Intersect + merge ---
            logger.info(f"[{participant_id}] Step 4: Intersect VAD ∩ diarization, merge")
            speech_df = intersect_turns_with_vad(
                turns_df, vad_intervals, min_keep_sec=self.config.min_segment_sec
            )
            speech_df.insert(0, "wav_path", str(analysis_wav_path))
            logger.info(f"[{participant_id}] Speech-only segments: {len(speech_df)}")

            merged_df = merge_close_segments(speech_df, gap_thresh=self.config.merge_gap_sec)
            logger.info(f"[{participant_id}] After merging: {len(merged_df)}")

            # --- Step 5: Classification ---
            logger.info(f"[{participant_id}] Step 5: eGeMAPS + XGBoost classification")
            audio_16k, sr_16k = load_audio_mono(str(analysis_wav_path))
            audio_16k = resample_audio(audio_16k, sr_16k, 16000)
            sr_16k = 16000

            model = self._load_model()
            extractor = self._get_extractor()

            classified_df = classify_segments(
                df=merged_df,
                audio_16k=audio_16k,
                sr=sr_16k,
                model=model,
                extractor=extractor,
                class_names=self.config.class_names,
                win=self.config.classify_win_sec,
                hop=self.config.classify_hop_sec,
            )

            # class distribution
            class_durations: Dict[str, float] = {}
            for cn in self.config.class_names:
                sel = classified_df[classified_df["predicted_class"] == cn]
                class_durations[cn] = float(sel["duration_sec"].sum()) if not sel.empty else 0.0
            logger.info(f"[{participant_id}] Class durations: {class_durations}")

            # --- Step 6 + 7 + 8: Streams, secondary diar, export ---
            logger.info(f"[{participant_id}] Steps 6-8: Stream export + secondary diarization")
            tmp_secondary = self.config.output_audio_root / "_tmp_secondary" / participant_id
            ensure_dir(tmp_secondary)

            # Female stream → main_female.wav
            fem_audio, _ = build_class_stream(classified_df, audio_16k, sr_16k, "adult_female")
            if len(fem_audio) > 0:
                extract_main_speaker_audio(
                    stream_audio=fem_audio,
                    sr=sr_16k,
                    diar_pipeline=diar_pipeline,
                    tmp_wav_path=tmp_secondary / "fem_stream_tmp.wav",
                    out_wav_path=self.config.main_female_wav_path,
                )
                logger.info(f"[{participant_id}] main_female.wav: {self.config.main_female_wav_path}")
            else:
                logger.warning(f"[{participant_id}] No adult_female segments, skipping main_female.wav")

            # Male stream → main_male.wav
            mal_audio, _ = build_class_stream(classified_df, audio_16k, sr_16k, "adult_male")
            if len(mal_audio) > 0:
                extract_main_speaker_audio(
                    stream_audio=mal_audio,
                    sr=sr_16k,
                    diar_pipeline=diar_pipeline,
                    tmp_wav_path=tmp_secondary / "mal_stream_tmp.wav",
                    out_wav_path=self.config.main_male_wav_path,
                )
                logger.info(f"[{participant_id}] main_male.wav: {self.config.main_male_wav_path}")
            else:
                logger.warning(f"[{participant_id}] No adult_male segments, skipping main_male.wav")

            # --- Step 9: TextGrid ---
            logger.info(f"[{participant_id}] Step 9: TextGrid")
            write_textgrid(
                df=classified_df,
                duration_sec=full_duration,
                out_path=self.config.textgrid_path,
                tier_map=TIER_MAP,
            )
            logger.info(f"[{participant_id}] TextGrid: {self.config.textgrid_path}")

            # --- Save artifacts ---
            # Drop wav_path before saving (redundant, large string)
            save_df = classified_df.drop(columns=["wav_path"], errors="ignore")
            write_parquet(save_df, self.config.segments_parquet_path)

            total_speech = float(classified_df["duration_sec"].sum())
            summary = {
                "participant_id": participant_id,
                "analysis_wav": str(analysis_wav_path),
                "duration_sec": full_duration,
                "n_vad_regions": len(vad_intervals),
                "n_diarization_turns": len(turns_df),
                "n_segments_after_merge": len(merged_df),
                "n_classified_segments": len(classified_df),
                "total_speech_sec": total_speech,
                "class_durations": class_durations,
                "main_female_wav": str(self.config.main_female_wav_path),
                "main_male_wav": str(self.config.main_male_wav_path),
                "textgrid": str(self.config.textgrid_path),
            }
            write_json(summary, self.config.summary_json_path)

            # cleanup tmp dirs
            import shutil
            for d in [tmp_dir, tmp_secondary]:
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass

            logger.info(f"[{participant_id}] Stage 03 DONE | segments={len(classified_df)} speech={total_speech/3600:.2f}h")

            return SpeakerClassificationArtifact(
                segments_parquet_path=self.config.segments_parquet_path,
                summary_json_path=self.config.summary_json_path,
                textgrid_path=self.config.textgrid_path,
                main_female_wav_path=self.config.main_female_wav_path,
                main_male_wav_path=self.config.main_male_wav_path,
                n_segments=len(classified_df),
                total_speech_sec=total_speech,
                class_durations=class_durations,
            )

        except Exception as e:
            raise wrap_exception(
                "Speaker classification failed",
                e,
                context={
                    "participant_id": participant_id,
                    "analysis_wav_path": str(analysis_wav_path),
                },
            )
