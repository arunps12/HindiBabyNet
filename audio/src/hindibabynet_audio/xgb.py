"""Speaker classification with the trained XGBoost model.

Steps for one prepared recording (16 kHz mono WAV):
  1. Voice activity detection (WebRTC VAD)  ->  speech chunks
  2. eGeMAPS features + XGBoost for every chunk alone
     ->  adult_male / adult_female / child / background
  3. Chunks that follow each other, have the same class and a short pause between
     them are merged in a second table. The classified chunks stay as they are.
  4. Write the two tables (CSV), the TextGrid and the summary

Speaker diarization is not used at present. Its two steps (speaker turns before the
classification, main female and main male speaker after it) are kept as comments in
this file. To use them again, remove the "# " of these blocks, of the diarization
settings in audio/params.yaml and of the diarization packages in audio/pyproject.toml.

This module needs the XGBoost extra (opensmile, xgboost, webrtcvad, joblib, tqdm).
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Not used at present (speaker diarization):
# import os
# import shutil
# import warnings

import joblib
import numpy as np
import opensmile
import pandas as pd
import praatio  # noqa: F401  (needed for the TextGrid at the end of a run)
import soundfile as sf
# import torch  # not used at present (speaker diarization)
import xgboost  # noqa: F401  (needed by joblib to read the model file)
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from hindibabynet_audio.recordings import prepared_recordings, select
from hindibabynet_audio.textgrid import write_textgrid
# from hindibabynet_audio.textgrid import intervals_to_df  # not used at present (speaker diarization)

# The one trained model: audio/models/xgb_egemaps.pkl
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "xgb_egemaps.pkl"

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
CLASS_NAMES: List[str] = list(LABEL2ID)
EGEMAPS_DIM = 88

# TextGrid tier names keyed by predicted class
TIER_MAP: Dict[str, str] = {
    "adult_female": "FEM",
    "adult_male": "MAL",
    "child": "CHILD",
    "background": "BACKGROUND",
}


# ============================================================================
# Audio helpers
# ============================================================================
def load_audio_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load an audio file to mono float32 array + sample rate."""
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False), sr


def resample_audio(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample a 1-D float32 array."""
    if sr == target_sr:
        return x
    gcd = int(np.gcd(sr, target_sr))
    up = target_sr // gcd
    down = sr // gcd
    return resample_poly(x, up, down).astype(np.float32, copy=False)


def crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    """Crop or zero-pad a 1-D array to exactly *target_len* samples."""
    n = len(x)
    if n == target_len:
        return x
    if n > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = x
    return out


def slice_audio(
    x: np.ndarray, sr: int, start_sec: float, end_sec: float
) -> np.ndarray:
    """Return samples between *start_sec* and *end_sec*."""
    s = max(0, int(round(start_sec * sr)))
    e = min(len(x), int(round(end_sec * sr)))
    return x[s:e]


def write_wav_chunk(
    wav_path: Path,
    chunk_path: Path,
    start_sec: float,
    end_sec: float,
) -> Path | None:
    """Write a time-slice of a WAV to disk. Returns ``None`` on failure."""
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
        print(f"    warning: write_wav_chunk failed ({chunk_path.name}): {e}")
        return None


def write_stream_wav(
    stream_audio: np.ndarray,
    sr: int,
    out_wav_path: Path,
) -> Path:
    """Write a class-stream audio array to WAV."""
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    if len(stream_audio) > 0:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
    return out_wav_path


# ============================================================================
# Step 1 - Voice activity detection
# ============================================================================
def speech_runs(decisions, frame_ms: int, min_silence_ms: int = 0):
    """Group the frame decisions of the VAD into speech chunks.

    decisions: True (speech) or False for every frame, one after the other.
    A chunk ends when the frames without speech after it are together at least
    min_silence_ms long. Shorter pauses stay inside the chunk. If min_silence_ms
    is not longer than one frame, every frame without speech ends a chunk.

    Gives (first frame, frame after the last speech frame) of every chunk.
    The decisions are used one by one and are not stored.
    """
    start = None
    after_last = 0
    for i, is_speech in enumerate(decisions):
        if is_speech:
            if start is None:
                start = i
            after_last = i + 1
        elif start is not None and (i + 1 - after_last) * frame_ms >= min_silence_ms:
            yield start, after_last
            start = None
    if start is not None:
        yield start, after_last


def webrtc_vad_regions(
    path: Path,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    min_region_ms: int = 300,
    min_silence_ms: int = 0,
) -> List[Tuple[float, float]]:
    """Return the speech chunks ``(start_sec, end_sec)`` of a recording via WebRTC VAD.

    Chunks shorter than min_region_ms are not returned. See speech_runs() for
    min_silence_ms.
    """
    try:
        import webrtcvad
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "webrtcvad is required for the voice activity detection of the XGBoost "
            "classification. Install it with: uv sync --project audio --extra xgb"
        ) from exc

    vad = webrtcvad.Vad(aggressiveness)
    info = sf.info(str(path))
    sr = int(info.samplerate)

    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad needs sr in (8k,16k,32k,48k), got {sr}")

    frame_len = int(sr * frame_ms / 1000)

    def decisions():
        with sf.SoundFile(str(path), mode="r") as f:
            while True:
                frame = f.read(frames=frame_len, dtype="int16", always_2d=True)
                if frame.size == 0 or len(frame) < frame_len:
                    break
                mono = frame[:, 0]
                yield vad.is_speech(mono.tobytes(), sr)

    out: List[Tuple[float, float]] = []
    for s_i, e_i in speech_runs(decisions(), frame_ms, min_silence_ms):
        s = (s_i * frame_len) / sr
        e = (e_i * frame_len) / sr
        if (e - s) * 1000 >= min_region_ms:
            out.append((float(s), float(e)))
    return out


# ============================================================================
# Not used at present: speaker diarization in chunks (pyannote)
# ============================================================================
# The functions of this part are kept as comments. See the text at the top of
# this file for how to use them again.
#
# def load_diarization_pipeline(model_name: str):
#     """Load the pyannote diarization pipeline. The access token is read from HF_TOKEN."""
#     from dotenv import load_dotenv
#     from pyannote.audio import Pipeline
#
#     load_dotenv()
#     hf_token = os.environ.get("HF_TOKEN")
#     os.environ.setdefault("PYANNOTE_DISABLE_NOTEBOOK", "1")
#
#     # Suppress known harmless warnings from pyannote / torch internals
#     warnings.filterwarnings(
#         "ignore",
#         message=".*weights_only=False.*",
#         category=FutureWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message=".*weights_only=False.*",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message=".*TensorFloat-32.*",
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message=r".*std\(\): degrees of freedom is <= 0.*",
#         category=UserWarning,
#     )
#
#     pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
#     if pipeline is None:
#         raise RuntimeError(
#             f"The diarization pipeline {model_name} could not be loaded. "
#             "Set the environment variable HF_TOKEN and accept the model conditions on Hugging Face."
#         )
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pipeline.to(device)
#     print(f"Diarization pipeline {model_name} loaded on {device}")
#     return pipeline
#
#
# def _make_chunks(
#     duration_sec: float, chunk_sec: float, overlap_sec: float
# ) -> List[Tuple[int, float, float]]:
#     """Yield (chunk_id, chunk_start, chunk_end) with overlap."""
#     step = chunk_sec - overlap_sec
#     assert step > 0, "chunk_sec must be > overlap_sec"
#     chunks: List[Tuple[int, float, float]] = []
#     t = 0.0
#     chunk_id = 0
#     while t < duration_sec:
#         s = t
#         e = min(t + chunk_sec, duration_sec)
#         chunks.append((chunk_id, s, e))
#         if e >= duration_sec:
#             break
#         t += step
#         chunk_id += 1
#     return chunks
#
#
# def diarize_wav(
#     wav_path: Path,
#     diar_pipeline,
#     chunk_sec: float,
#     overlap_sec: float,
#     min_speakers: int,
#     max_speakers: int,
#     tmp_dir: Path,
# ) -> pd.DataFrame:
#     """Run chunked diarization on a single WAV, returning speaker turns."""
#     info = sf.info(str(wav_path))
#     full_duration = float(info.duration)
#     chunks = _make_chunks(full_duration, chunk_sec, overlap_sec)
#
#     all_rows: List[Dict[str, Any]] = []
#     for chunk_id, chunk_start, chunk_end in chunks:
#         chunk_wav = tmp_dir / f"{wav_path.stem}_chunk{chunk_id:04d}.wav"
#         out = write_wav_chunk(wav_path, chunk_wav, chunk_start, chunk_end)
#         if out is None:
#             continue
#         try:
#             diar = diar_pipeline(
#                 {"audio": str(chunk_wav)},
#                 min_speakers=min_speakers,
#                 max_speakers=max_speakers,
#             )
#             for seg, _, spk in diar.itertracks(yield_label=True):
#                 s = float(seg.start) + chunk_start
#                 e = float(seg.end) + chunk_start
#                 if e <= s:
#                     continue
#                 all_rows.append(
#                     {
#                         "chunk_id": int(chunk_id),
#                         "speaker_id_local": spk,
#                         "start_sec": s,
#                         "end_sec": e,
#                         "duration_sec": float(e - s),
#                     }
#                 )
#         except Exception as exc:
#             print(f"    warning: diarization failed in chunk {chunk_id}: {exc}")
#         finally:
#             try:
#                 chunk_wav.unlink(missing_ok=True)
#             except Exception:
#                 pass
#
#     if not all_rows:
#         return pd.DataFrame(
#             columns=["chunk_id", "speaker_id_local", "start_sec", "end_sec", "duration_sec"]
#         )
#     return (
#         pd.DataFrame(all_rows)
#         .sort_values(["start_sec", "end_sec"])
#         .reset_index(drop=True)
#     )
#
#
# Not used at present: speech parts of the speaker turns, merging by speaker
# ----------------------------------------------------------------------------
# def intersect_turns_with_vad(
#     turns_df: pd.DataFrame,
#     vad_intervals: List[Tuple[float, float]],
#     min_keep_sec: float = 0.0,
# ) -> pd.DataFrame:
#     """Intersect diarization turns with VAD regions (sweep-line)."""
#     if turns_df.empty or not vad_intervals:
#         return pd.DataFrame(
#             columns=["start_sec", "end_sec", "duration_sec", "chunk_id", "speaker_id_local"]
#         )
#
#     turns_df = turns_df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
#     diar_arr = turns_df[["start_sec", "end_sec", "chunk_id", "speaker_id_local"]].to_numpy()
#     vad_arr = np.array(sorted(vad_intervals), dtype=float)
#
#     i, j = 0, 0
#     rows: List[Dict] = []
#
#     while i < len(diar_arr) and j < len(vad_arr):
#         ds, de = float(diar_arr[i, 0]), float(diar_arr[i, 1])
#         vs, ve = float(vad_arr[j, 0]), float(vad_arr[j, 1])
#
#         s = max(ds, vs)
#         e = min(de, ve)
#         if s < e:
#             dur = e - s
#             if dur >= min_keep_sec:
#                 rows.append(
#                     {
#                         "start_sec": s,
#                         "end_sec": e,
#                         "duration_sec": dur,
#                         "chunk_id": int(diar_arr[i, 2]),
#                         "speaker_id_local": str(diar_arr[i, 3]),
#                     }
#                 )
#         if de <= ve:
#             i += 1
#         else:
#             j += 1
#
#     return pd.DataFrame(rows) if rows else pd.DataFrame(
#         columns=["start_sec", "end_sec", "duration_sec", "chunk_id", "speaker_id_local"]
#     )
#
#
# def merge_close_segments(
#     df: pd.DataFrame,
#     gap_thresh: float = 0.5,
# ) -> pd.DataFrame:
#     """Merge temporally close segments from the same speaker."""
#     if df.empty:
#         return df.copy()
#
#     df = df.copy()
#     df["_orig_row"] = df.index
#
#     # sort columns that exist
#     sort_cols = [c for c in [
#         "wav_path", "chunk_id", "speaker_id_local", "start_sec", "end_sec"
#     ] if c in df.columns]
#     df = df.sort_values(sort_cols).reset_index(drop=True)
#
#     prev_wav = df["wav_path"].shift(1) if "wav_path" in df.columns else pd.Series(dtype=object)
#     prev_chunk = df["chunk_id"].shift(1)
#     prev_spk = df["speaker_id_local"].shift(1)
#     prev_end = df["end_sec"].shift(1)
#     gap = df["start_sec"] - prev_end
#
#     new_group = (
#         (df["chunk_id"] != prev_chunk)
#         | (df["speaker_id_local"] != prev_spk)
#         | (gap.isna())
#         | (gap < 0)
#         | (gap > gap_thresh)
#     )
#     if "wav_path" in df.columns:
#         new_group = new_group | (df["wav_path"] != prev_wav)
#
#     df["_merge_group"] = new_group.cumsum()
#
#     agg_dict: Dict[str, Any] = {
#         "chunk_id": ("chunk_id", "first"),
#         "speaker_id_local": ("speaker_id_local", "first"),
#         "start_sec": ("start_sec", "min"),
#         "end_sec": ("end_sec", "max"),
#         "n_merged": ("_orig_row", "count"),
#     }
#     if "wav_path" in df.columns:
#         agg_dict["wav_path"] = ("wav_path", "first")
#
#     out = df.groupby("_merge_group", as_index=False).agg(**agg_dict)
#     out["duration_sec"] = out["end_sec"] - out["start_sec"]
#
#     sort_out = [c for c in ["wav_path", "chunk_id", "start_sec"] if c in out.columns]
#     return out.sort_values(sort_out).reset_index(drop=True)


# ============================================================================
# Speech chunks: lengths and merging
# ============================================================================
def chunk_statistics(durations) -> Dict[str, float]:
    """Number, total duration and distribution of the chunk durations (seconds)."""
    values = np.asarray(list(durations), dtype=float)
    if values.size == 0:
        return {"n_chunks": 0, "total_sec": 0.0}
    p05, p25, median, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "n_chunks": int(values.size),
        "total_sec": float(values.sum()),
        "mean_sec": float(values.mean()),
        "min_sec": float(values.min()),
        "p05_sec": float(p05),
        "p25_sec": float(p25),
        "median_sec": float(median),
        "p75_sec": float(p75),
        "p95_sec": float(p95),
        "max_sec": float(values.max()),
        "percent_shorter_than_1s": float((values < 1.0).mean() * 100),
    }


MERGED_COLUMNS = ["merged_segment", "start_sec", "end_sec", "n_chunks", "speech_duration_sec", "span_sec"]


def merge_chunks(df: pd.DataFrame, gap_sec: float, same_class: bool) -> Tuple[pd.Series, pd.DataFrame]:
    """Merge chunks that follow each other with a pause of at most gap_sec.

    same_class=True: two chunks are merged only if they also have the same
    predicted_class. Only chunks that follow each other directly are compared, so a
    chunk of another class between two chunks always keeps them apart.

    Returns the number of the merged segment for every row of df, and the merged
    table. speech_duration_sec is the sum of the chunk durations: the pauses between
    the merged chunks are not counted. span_sec is the time from the first start to
    the last end, with the pauses. The rows of df are not changed.
    """
    columns = MERGED_COLUMNS[:3] + (["predicted_class"] if same_class else []) + MERGED_COLUMNS[3:]
    if df.empty:
        return pd.Series(dtype=int), pd.DataFrame(columns=columns)

    ordered = df.sort_values(["start_sec", "end_sec"])
    pause = ordered["start_sec"] - ordered["end_sec"].shift(1)
    # The chunk times are multiples of the frame length. The small tolerance keeps a
    # pause of exactly gap_sec inside the limit when the numbers are rounded.
    new_segment = pause.isna() | (pause > gap_sec + 1e-6)
    if same_class:
        new_segment |= ordered["predicted_class"] != ordered["predicted_class"].shift(1)
    number = (new_segment.cumsum() - 1).rename("merged_segment")

    parts = {
        "start_sec": ("start_sec", "min"),
        "end_sec": ("end_sec", "max"),
        "n_chunks": ("start_sec", "size"),
        "speech_duration_sec": ("duration_sec", "sum"),
    }
    if same_class:
        parts["predicted_class"] = ("predicted_class", "first")
    merged = ordered.groupby(number).agg(**parts).reset_index()
    merged["span_sec"] = merged["end_sec"] - merged["start_sec"]
    return number, merged[columns]


# ============================================================================
# Step 2 - eGeMAPS feature extraction + classification
# ============================================================================
def load_model(path: str | Path = MODEL_PATH, class_names: List[str] = CLASS_NAMES, egemaps_dim: int = EGEMAPS_DIM):
    """Load the trained XGBoost model and check its input and output size."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    # Force model to CPU - input is always numpy (CPU) arrays.
    # Without this, a model saved on cuda triggers a device-mismatch
    # warning and a slow DMatrix fallback on every predict_proba call.
    try:
        model.set_params(device="cpu")
    except Exception:
        pass  # older xgboost versions without device param

    # Check: one probability per class for one feature vector.
    dummy = np.zeros((1, egemaps_dim), dtype=np.float32)
    proba = model.predict_proba(dummy)
    if proba.shape[1] != len(class_names):
        raise ValueError(
            f"Model outputs {proba.shape[1]} classes, expected {len(class_names)}"
        )
    return model


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
    Windowing strategy:
      - Short segments (< win) -> single padded window.
      - Long segments -> 1.0 s windows with 0.5 s hop + end-anchored last window.
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
    """Weighted mean of per-window probability vectors."""
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
# Step 4 - Build combined class-stream audio
# ============================================================================
def build_class_stream(
    df: pd.DataFrame,
    audio_16k: np.ndarray,
    sr: int,
    target_class: str,
    gap_sec: float = 0.15,
) -> Tuple[np.ndarray, pd.DataFrame, List[Tuple[float, float, float, float]]]:
    """
    Concatenate all segments of *target_class* into a single audio array.

    Returns
    -------
    audio : np.ndarray          - concatenated stream
    used_df : pd.DataFrame      - rows that were included
    offset_table : list[(stream_start, stream_end, orig_start, orig_end)]
        Maps every original segment to its position inside the stream so
        secondary-diarization results can be mapped back.
    """
    sel = df[df["predicted_class"] == target_class].copy()
    sel = sel.sort_values("start_sec").reset_index(drop=True)

    if sel.empty:
        return np.array([], dtype=np.float32), sel, []

    gap_samples = int(sr * gap_sec) if gap_sec > 0 else 0
    gap = np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else None
    pieces: List[np.ndarray] = []
    offset_table: List[Tuple[float, float, float, float]] = []
    cursor = 0.0  # current position in stream (seconds)

    for _, r in sel.iterrows():
        seg = slice_audio(audio_16k, sr, float(r.start_sec), float(r.end_sec))
        if len(seg) == 0:
            continue
        seg_dur = len(seg) / sr
        offset_table.append((cursor, cursor + seg_dur, float(r.start_sec), float(r.end_sec)))
        pieces.append(seg)
        cursor += seg_dur
        if gap is not None:
            pieces.append(gap)
            cursor += gap_samples / sr

    if not pieces:
        return np.array([], dtype=np.float32), sel, []

    y = np.concatenate(pieces).astype(np.float32, copy=False)
    return y, sel, offset_table


# ============================================================================
# Not used at present: second diarization -> main speaker of a class stream
# ============================================================================
# def extract_main_speaker_audio(
#     stream_audio: np.ndarray,
#     sr: int,
#     diar_pipeline,
#     tmp_wav_path: Path,
#     out_wav_path: Path,
#     offset_table: List[Tuple[float, float, float, float]],
#     min_speakers: int = 1,
#     max_speakers: int = 3,
#     gap_sec: float = 0.15,
# ) -> Tuple[Path, List[Tuple[float, float]]]:
#     """
#     Diarize the stream, keep only the dominant speaker's segments,
#     and write to out_wav_path. Falls back to full stream if diarization
#     finds nothing or only 1 speaker.
#
#     Returns
#     -------
#     out_wav_path : Path
#     dominant_orig_intervals : list[(orig_start, orig_end)]
#         Original-recording-time intervals that belong to the dominant speaker.
#         If fallback (no filtering), returns ALL original intervals from offset_table.
#     """
#     all_orig = [(os, oe) for (_, _, os, oe) in offset_table]
#     out_wav_path.parent.mkdir(parents=True, exist_ok=True)
#
#     # If stream too short, just write it directly
#     if len(stream_audio) < sr:
#         sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
#         return out_wav_path, all_orig
#
#     # Write temp wav for diarization
#     tmp_wav_path.parent.mkdir(parents=True, exist_ok=True)
#     sf.write(str(tmp_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
#
#     try:
#         diar = diar_pipeline(
#             {"audio": str(tmp_wav_path)},
#             min_speakers=min_speakers,
#             max_speakers=max_speakers,
#         )
#     except Exception as e:
#         print(f"    warning: second diarization failed, the full stream is used: {e}")
#         sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
#         return out_wav_path, all_orig
#     finally:
#         try:
#             tmp_wav_path.unlink(missing_ok=True)
#         except Exception:
#             pass
#
#     # Find dominant speaker
#     durations: Dict[str, float] = {}
#     segments_by_spk: Dict[str, List[Tuple[float, float]]] = {}
#     for seg, _, spk in diar.itertracks(yield_label=True):
#         s, e = float(seg.start), float(seg.end)
#         durations[spk] = durations.get(spk, 0.0) + (e - s)
#         segments_by_spk.setdefault(spk, []).append((s, e))
#
#     if not durations:
#         sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
#         return out_wav_path, all_orig
#
#     main_spk = max(durations, key=lambda k: durations[k])
#     main_segs = sorted(segments_by_spk[main_spk])
#
#     # Write dominant-speaker audio
#     gap = np.zeros(int(sr * gap_sec), dtype=np.float32) if gap_sec > 0 else None
#     pieces: List[np.ndarray] = []
#     for seg_s, seg_e in main_segs:
#         chunk = slice_audio(stream_audio, sr, seg_s, seg_e)
#         if len(chunk) > 0:
#             pieces.append(chunk)
#             if gap is not None:
#                 pieces.append(gap)
#
#     if not pieces:
#         sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
#         return out_wav_path, all_orig
#     else:
#         y = np.concatenate(pieces).astype(np.float32, copy=False)
#         sf.write(str(out_wav_path), y, sr, format="WAV", subtype="PCM_16")
#
#     # Map dominant speaker segments back to original recording time.
#     # For each original segment in the offset table, check whether the
#     # dominant speaker covers the majority of its stream-time range.
#     dominant_orig: List[Tuple[float, float]] = []
#     for stream_s, stream_e, orig_s, orig_e in offset_table:
#         seg_dur = stream_e - stream_s
#         overlap = 0.0
#         for ms, me in main_segs:
#             o_s = max(stream_s, ms)
#             o_e = min(stream_e, me)
#             if o_e > o_s:
#                 overlap += o_e - o_s
#         # Keep if dominant speaker covers >= 50% of the segment
#         if overlap >= seg_dur * 0.5:
#             dominant_orig.append((orig_s, orig_e))
#
#     # If nothing survived, fall back to all
#     if not dominant_orig:
#         dominant_orig = all_orig
#
#     return out_wav_path, dominant_orig


# ============================================================================
# One recording: all steps
# ============================================================================
# Settings of the xgb part of params.yaml that must be present.
REQUIRED_SETTINGS = {
    "vad": ("aggressiveness", "frame_ms", "min_region_ms", "min_silence_ms"),
    "segmentation": ("merge_before_classification", "merge_after_classification", "merge_gap_sec"),
    "classification": ("win_sec", "hop_sec"),
}


def check_settings(settings: dict) -> None:
    """Raise a ValueError that names the settings that are missing in params.yaml."""
    missing = [
        f"xgb.{part}.{key}"
        for part, keys in REQUIRED_SETTINGS.items()
        for key in keys
        if key not in (settings.get(part) or {})
    ]
    if missing:
        raise ValueError("audio/params.yaml has no value for: " + ", ".join(missing))


def output_files(participant: str, out_dir: Path) -> Dict[str, Path]:
    """Names of the result files of one recording."""
    files = {
        "textgrid": out_dir / f"{participant}.TextGrid",
        "segments": out_dir / f"{participant}_segments.csv",
        "merged_segments": out_dir / f"{participant}_merged_segments.csv",
        "summary": out_dir / f"{participant}_summary.json",
        "run_info": out_dir / "run_info.json",
    }
    # Not used at present: one audio file per class (<name>_adult_male.wav, ...).
    # files.update({f"{name}_wav": out_dir / f"{participant}_{name}.wav" for name in CLASS_NAMES})
    return files


# Results are complete when these files exist. They are written for every recording.
REQUIRED_FILES = ("segments", "textgrid", "summary")


def is_complete(participant: str, out_dir: Path) -> bool:
    """True when the segment table, the TextGrid and the summary exist."""
    files = output_files(participant, out_dir)
    return all(files[name].exists() for name in REQUIRED_FILES)


def classify_recording(
    wav_path: Path,
    participant: str,
    out_dir: Path,
    settings: dict,
    model,
    # diar_pipeline,  # not used at present (speaker diarization)
    extractor: EGemapsExtractor,
) -> dict:
    """Process one prepared WAV through all steps and return the summary."""
    files = output_files(participant, out_dir)
    class_names = list(settings["class_names"])
    vad = settings["vad"]
    # diarization = settings["diarization"]  # not used at present
    segmentation = settings["segmentation"]
    classification = settings["classification"]
    merge_gap = float(segmentation["merge_gap_sec"])

    out_dir.mkdir(parents=True, exist_ok=True)

    if not wav_path.exists():
        raise FileNotFoundError(f"Prepared recording not found: {wav_path}")

    wav_info = sf.info(str(wav_path))
    full_duration = float(wav_info.duration)
    print(f"  recording: {wav_path} | {full_duration / 3600:.2f} h, {wav_info.samplerate} Hz")

    # --- Step 1: VAD chunks ---
    print("  step 1: voice activity detection")
    vad_intervals = webrtc_vad_regions(
        wav_path,
        aggressiveness=int(vad["aggressiveness"]),
        frame_ms=int(vad["frame_ms"]),
        min_region_ms=int(vad["min_region_ms"]),
        min_silence_ms=int(vad["min_silence_ms"]),
    )
    chunks_df = pd.DataFrame(vad_intervals, columns=["start_sec", "end_sec"])
    chunks_df["duration_sec"] = chunks_df["end_sec"] - chunks_df["start_sec"]
    statistics = chunk_statistics(chunks_df["duration_sec"])
    print(
        f"    speech chunks: {statistics['n_chunks']}, "
        f"{statistics['total_sec'] / 3600:.2f} h of speech"
    )

    # Optional: join chunks with a short pause before the classification. The class is
    # not known here, so chunks of different speakers can be joined.
    if bool(segmentation["merge_before_classification"]):
        _, joined = merge_chunks(chunks_df, merge_gap, same_class=False)
        chunks_df = joined.rename(columns={"speech_duration_sec": "duration_sec"})[
            ["start_sec", "end_sec", "duration_sec", "n_chunks"]
        ]
        print(f"    after joining chunks with a pause up to {merge_gap:g} s: {len(chunks_df)}")

    # Not used at present: speaker diarization and merging by speaker.
    # tmp_dir = out_dir.parent / "_tmp_diar" / participant
    # tmp_dir.mkdir(parents=True, exist_ok=True)
    # turns_df = diarize_wav(
    #     wav_path=wav_path,
    #     diar_pipeline=diar_pipeline,
    #     chunk_sec=float(diarization["chunk_sec"]),
    #     overlap_sec=float(diarization["overlap_sec"]),
    #     min_speakers=int(diarization["min_speakers"]),
    #     max_speakers=int(diarization["max_speakers"]),
    #     tmp_dir=tmp_dir,
    # )
    # speech_df = intersect_turns_with_vad(
    #     turns_df, vad_intervals, min_keep_sec=float(segmentation["min_segment_sec"])
    # )
    # speech_df.insert(0, "wav_path", str(wav_path))
    # chunks_df = merge_close_segments(speech_df, gap_thresh=float(segmentation["merge_gap_sec"]))

    # --- Step 2: Classification of every chunk ---
    print("  step 2: eGeMAPS features and XGBoost classification of every chunk")
    audio_16k, sr_16k = load_audio_mono(str(wav_path))
    audio_16k = resample_audio(audio_16k, sr_16k, 16000)
    sr_16k = 16000

    classified_df = classify_segments(
        df=chunks_df,
        audio_16k=audio_16k,
        sr=sr_16k,
        model=model,
        extractor=extractor,
        class_names=class_names,
        win=float(classification["win_sec"]),
        hop=float(classification["hop_sec"]),
    )

    # Speech seconds and number of chunks per class. Only the chunks are counted.
    class_durations: Dict[str, float] = {}
    class_counts: Dict[str, int] = {}
    for cn in class_names:
        sel = classified_df[classified_df["predicted_class"] == cn]
        class_durations[cn] = float(sel["duration_sec"].sum()) if not sel.empty else 0.0
        class_counts[cn] = int(len(sel))
    print(f"    seconds per class: {class_durations}")

    # --- Step 3: Merge chunks of the same class that follow each other ---
    # The classified chunks stay as they are. They get the number of their merged
    # segment, and the merged segments are saved in a second table.
    merged_df = None
    merged_counts: Dict[str, int] = {}
    if bool(segmentation["merge_after_classification"]):
        print(f"  step 3: merging chunks of the same class with a pause up to {merge_gap:g} s")
        number, merged_df = merge_chunks(classified_df, merge_gap, same_class=True)
        classified_df["merged_segment"] = number
        for cn in class_names:
            merged_counts[cn] = int((merged_df["predicted_class"] == cn).sum())
        print(f"    merged segments: {len(merged_df)}")

    # Not used at present: one audio file per class (all chunks of the class).
    # for cn in class_names:
    #     stream_audio, _, _ = build_class_stream(classified_df, audio_16k, sr_16k, cn)
    #     if len(stream_audio) > 0:
    #         write_stream_wav(stream_audio, sr_16k, files[f"{cn}_wav"])
    #     else:
    #         print(f"    note: no {cn} chunks, no {files[f'{cn}_wav'].name} written")

    # Not used at present: main female and main male speaker (second diarization).
    # tmp_secondary = out_dir.parent / "_tmp_secondary" / participant
    # fem_audio, _, fem_offsets = build_class_stream(classified_df, audio_16k, sr_16k, "adult_female")
    # _, fem_dominant_intervals = extract_main_speaker_audio(
    #     stream_audio=fem_audio,
    #     sr=sr_16k,
    #     diar_pipeline=diar_pipeline,
    #     tmp_wav_path=tmp_secondary / "fem_stream_tmp.wav",
    #     out_wav_path=out_dir / f"{participant}_main_female.wav",
    #     offset_table=fem_offsets,
    # )
    # mal_audio, _, mal_offsets = build_class_stream(classified_df, audio_16k, sr_16k, "adult_male")
    # _, mal_dominant_intervals = extract_main_speaker_audio(
    #     stream_audio=mal_audio,
    #     sr=sr_16k,
    #     diar_pipeline=diar_pipeline,
    #     tmp_wav_path=tmp_secondary / "mal_stream_tmp.wav",
    #     out_wav_path=out_dir / f"{participant}_main_male.wav",
    #     offset_table=mal_offsets,
    # )
    # The TextGrid then had only the main speakers on the FEM and MAL tiers:
    # textgrid_df = pd.concat(
    #     [
    #         intervals_to_df(fem_dominant_intervals, "adult_female"),
    #         intervals_to_df(mal_dominant_intervals, "adult_male"),
    #         classified_df[classified_df["predicted_class"].isin(["child", "background"])],
    #     ],
    #     ignore_index=True,
    # )

    # --- Step 4: Tables, TextGrid and summary ---
    # The TextGrid has the classified chunks, so a pause is never shown as speech.
    print("  step 4: segment tables (CSV), TextGrid and summary")
    write_textgrid(
        df=classified_df,
        duration_sec=full_duration,
        out_path=files["textgrid"],
        tier_map=TIER_MAP,
    )

    # CSV files, one row per chunk or segment. Numbers are written with 6 decimals.
    classified_df.round(6).to_csv(files["segments"], index=False)
    if merged_df is not None:
        merged_df.round(6).to_csv(files["merged_segments"], index=False)

    total_speech = float(classified_df["duration_sec"].sum())
    summary = {
        "participant_id": participant,
        "analysis_wav": str(wav_path),
        "duration_sec": full_duration,
        "settings": {"vad": dict(vad), "segmentation": dict(segmentation),
                     "classification": dict(classification)},
        "n_vad_chunks": len(vad_intervals),
        "vad_chunk_statistics": statistics,
        "n_classified_segments": len(classified_df),
        "total_speech_sec": total_speech,
        "class_durations": class_durations,
        "class_counts": class_counts,
        "n_merged_segments": None if merged_df is None else len(merged_df),
        "merged_class_counts": merged_counts,
        "segments": str(files["segments"]),
        "merged_segments": None if merged_df is None else str(files["merged_segments"]),
        # **{f"{cn}_wav": str(files[f"{cn}_wav"]) for cn in class_names},  # not used at present
        "textgrid": str(files["textgrid"]),
    }
    with open(files["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ============================================================================
# All participants
# ============================================================================
def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_participant(
    wav_path: Path,
    participant: str,
    out_dir: Path,
    settings: dict,
    model,
    extractor: EGemapsExtractor,
) -> dict:
    """Classify one recording and write run_info.json, also when the run fails or is stopped."""
    started = utcnow_iso()
    t0 = time.monotonic()
    status = "interrupted"  # stays when the run is stopped, for example with Ctrl+C
    error_msg = ""

    try:
        summary = classify_recording(wav_path, participant, out_dir, settings, model, extractor)
        status = "success"
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        raise
    finally:
        runtime = round(time.monotonic() - t0, 2)
        info = {
            "participant_id": participant,
            "backend": "xgb",
            "input_wav": str(wav_path),
            "output_dir": str(out_dir),
            "started_at": started,
            "finished_at": utcnow_iso(),
            "runtime_sec": runtime,
            "status": status,
            **({"error": error_msg} if error_msg else {}),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_info.json").write_text(
            json.dumps(info, indent=2, default=str), encoding="utf-8"
        )

    return summary


def find_prepared_audio(config) -> List[Tuple[str, Path]]:
    """All prepared WAV files as (recording ID, file), in the order of the IDs."""
    return prepared_recordings(config.path("prepared_audio"))


def run(config, params, participant: str | None = None, limit: int | None = None) -> int:
    """Classify the prepared recordings. Return 0 if nothing failed, else 1."""
    settings = params["xgb"]
    check_settings(settings)

    participants = select(find_prepared_audio(config), participant, limit)
    if not participants:
        print(f"No prepared recording found in {config.path('prepared_audio')}.")
        return 1

    # Existing results are never replaced. The result files start with the last
    # part of the recording ID (P001 for P001, REC_0001 for P001/REC_0001).
    to_run: List[Tuple[str, str, Path, Path]] = []
    available = 0
    failed: List[str] = []
    for pid, wav_path in participants:
        out_dir = config.path("xgb", *pid.split("/"))
        prefix = out_dir.name
        if is_complete(prefix, out_dir):
            print(f"{pid}: results already available, not run again ({out_dir})")
            available += 1
        elif out_dir.exists():
            files = output_files(prefix, out_dir)
            missing = [files[name].name for name in REQUIRED_FILES if not files[name].exists()]
            print(
                f"{pid}: ERROR, the output folder exists but is incomplete "
                f"(missing: {', '.join(missing)}). Nothing was changed. "
                f"Move or rename {out_dir} to run this participant again."
            )
            failed.append(pid)
        else:
            to_run.append((pid, prefix, wav_path, out_dir))

    done = 0
    if to_run:
        try:
            print(f"Loading the XGBoost model {MODEL_PATH}")
            model = load_model(MODEL_PATH, list(settings["class_names"]), int(settings["egemaps_dim"]))
            # Not used at present (speaker diarization):
            # diar_pipeline = load_diarization_pipeline(str(settings["diarization"]["model"]))
            extractor = EGemapsExtractor(
                egemaps_dim=int(settings["egemaps_dim"]),
                target_sr=16000,
                win_sec=float(settings["classification"]["win_sec"]),
            )
        except Exception as exc:
            traceback.print_exc()
            print(f"ERROR, the models could not be loaded, no participant was run: {exc}")
            return 1

        for pid, prefix, wav_path, out_dir in to_run:
            print(f"{pid}: started")
            try:
                summary = run_participant(wav_path, prefix, out_dir, settings, model, extractor)
            except Exception as exc:
                traceback.print_exc()
                print(f"{pid}: ERROR, classification failed: {exc}")
                failed.append(pid)
                continue
            print(
                f"{pid}: finished, {summary['n_classified_segments']} segments, "
                f"{summary['total_speech_sec'] / 3600:.2f} h speech"
            )
            print(f"{pid}: results in {out_dir}")
            done += 1

    print(
        f"XGBoost classification: {done} run, {available} already available, {len(failed)} failed."
    )
    if failed:
        print("Failed: " + ", ".join(failed))
    return 1 if failed else 0
