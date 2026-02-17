"""
Stage 06: Speaker-type Classification + Stream Export

Takes the speech-segments parquet from Stage 05 (VAD ∩ Diarization),
merges close segments, classifies with eGeMAPS + XGBoost, builds
per-class WAV streams, runs secondary diarization on female/male
streams, and exports per-class parquets with segment info.
"""
from __future__ import annotations

import os
import shutil
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
from tqdm.auto import tqdm

from src.hindibabynet.entity.artifact_entity import SpeakerClassificationArtifact
from src.hindibabynet.entity.config_entity import SpeakerClassificationConfig
from src.hindibabynet.exception.exception import wrap_exception
from src.hindibabynet.logging.logger import get_logger
from src.hindibabynet.utils.audio_utils import (
    crop_or_pad,
    load_audio_mono,
    resample_audio,
    slice_audio,
    write_stream_wav,
)
from src.hindibabynet.utils.io_utils import ensure_dir, write_json, write_parquet
from src.hindibabynet.utils.textgrid_utils import intervals_to_df, write_textgrid

logger = get_logger(__name__)

# ============================================================================
# Label mapping (must match training order in the XGBoost model)
# ============================================================================
LABEL2ID: Dict[str, int] = {
    "adult_male": 0,
    "adult_female": 1,
    "child": 2,
    "background": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

TIER_MAP: Dict[str, str] = {
    "adult_female": "FEM",
    "adult_male": "MAL",
    "child": "CHILD",
    "background": "BACKGROUND",
}


# ============================================================================
# Merge close segments
# ============================================================================
def merge_close_segments(
    df: pd.DataFrame,
    gap_thresh: float = 0.5,
) -> pd.DataFrame:
    """Merge temporally close segments from the same speaker.

    The output retains the ``source_segment_ids`` column that lists the
    original ``segment_id`` values (from the intersection parquet) that
    were merged into each output row.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df["_orig_row"] = df.index

    sort_cols = [c for c in [
        "wav_path", "chunk_id", "speaker_id_local", "start_sec", "end_sec",
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

    # Collect source segment_ids per merge group
    seg_id_col = "segment_id" if "segment_id" in df.columns else None

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

    # Build source_segment_ids list
    if seg_id_col is not None:
        src_ids = (
            df.groupby("_merge_group")[seg_id_col]
            .apply(lambda x: x.tolist())
            .reset_index(name="source_segment_ids")
        )
        out = out.merge(src_ids, on="_merge_group", how="left")
    else:
        out["source_segment_ids"] = out["_merge_group"].apply(lambda _: [])

    out = out.drop(columns=["_merge_group"], errors="ignore")
    sort_out = [c for c in ["wav_path", "chunk_id", "start_sec"] if c in out.columns]
    return out.sort_values(sort_out).reset_index(drop=True)


# ============================================================================
# eGeMAPS feature extraction + classification
# ============================================================================
@dataclass
class EGemapsExtractor:
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
    start: float, end: float, win: float = 1.0, hop: float = 0.5,
) -> List[Tuple[float, float, float]]:
    dur = end - start
    if dur <= 0 or dur < win:
        return [(start, start + win, win)]
    windows: List[Tuple[float, float, float]] = []
    t = start
    while t + win <= end:
        windows.append((t, t + win, win))
        t += hop
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
    """Classify each segment — adds probs_*, predicted_class, predicted_confidence, n_windows."""
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
# Build per-class stream + offset table
# ============================================================================
def build_class_stream(
    df: pd.DataFrame,
    audio_16k: np.ndarray,
    sr: int,
    target_class: str,
    gap_sec: float = 0.15,
) -> Tuple[np.ndarray, pd.DataFrame, List[Tuple[float, float, float, float]]]:
    sel = df[df["predicted_class"] == target_class].copy()
    sel = sel.sort_values("start_sec").reset_index(drop=True)

    if sel.empty:
        return np.array([], dtype=np.float32), sel, []

    gap_samples = int(sr * gap_sec) if gap_sec > 0 else 0
    gap = np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else None
    pieces: List[np.ndarray] = []
    offset_table: List[Tuple[float, float, float, float]] = []
    cursor = 0.0

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
    return np.concatenate(pieces).astype(np.float32, copy=False), sel, offset_table


def _build_stream_parquet(
    offset_table: List[Tuple[float, float, float, float]],
    target_class: str,
) -> pd.DataFrame:
    """Build a parquet-ready DataFrame for one per-class stream.

    Columns: stream_segment_id, stream_start_sec, stream_end_sec,
             stream_duration_sec, orig_start_sec, orig_end_sec,
             orig_duration_sec, predicted_class
    """
    rows = []
    for i, (ss, se, os_, oe) in enumerate(offset_table):
        rows.append({
            "stream_segment_id": i,
            "stream_start_sec": ss,
            "stream_end_sec": se,
            "stream_duration_sec": se - ss,
            "orig_start_sec": os_,
            "orig_end_sec": oe,
            "orig_duration_sec": oe - os_,
            "predicted_class": target_class,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "stream_segment_id", "stream_start_sec", "stream_end_sec",
        "stream_duration_sec", "orig_start_sec", "orig_end_sec",
        "orig_duration_sec", "predicted_class",
    ])


# ============================================================================
# Secondary diarization → main speaker extraction
# ============================================================================
def extract_main_speaker_audio(
    stream_audio: np.ndarray,
    sr: int,
    diar_pipeline,
    tmp_wav_path: Path,
    out_wav_path: Path,
    offset_table: List[Tuple[float, float, float, float]],
    min_speakers: int = 1,
    max_speakers: int = 3,
    gap_sec: float = 0.15,
) -> Tuple[Path, List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
    """Diarize stream, keep dominant speaker, return dominant offset_table."""
    all_orig = [(os_, oe) for (_, _, os_, oe) in offset_table]
    ensure_dir(out_wav_path.parent)

    if len(stream_audio) < sr:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
        return out_wav_path, all_orig, offset_table

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
        return out_wav_path, all_orig, offset_table
    finally:
        try:
            tmp_wav_path.unlink(missing_ok=True)
        except Exception:
            pass

    durations: Dict[str, float] = {}
    segments_by_spk: Dict[str, List[Tuple[float, float]]] = {}
    for seg, _, spk in diar.itertracks(yield_label=True):
        s, e = float(seg.start), float(seg.end)
        durations[spk] = durations.get(spk, 0.0) + (e - s)
        segments_by_spk.setdefault(spk, []).append((s, e))

    if not durations:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
        return out_wav_path, all_orig, offset_table

    main_spk = max(durations, key=lambda k: durations[k])
    main_segs = sorted(segments_by_spk[main_spk])

    # Write dominant-speaker audio & build new offset table
    gap_arr = np.zeros(int(sr * gap_sec), dtype=np.float32) if gap_sec > 0 else None
    pieces: List[np.ndarray] = []
    new_offset: List[Tuple[float, float, float, float]] = []
    cursor = 0.0

    for seg_s, seg_e in main_segs:
        chunk = slice_audio(stream_audio, sr, seg_s, seg_e)
        if len(chunk) > 0:
            chunk_dur = len(chunk) / sr
            # Map back to original time via offset_table
            for st_s, st_e, o_s, o_e in offset_table:
                ov_s = max(seg_s, st_s)
                ov_e = min(seg_e, st_e)
                if ov_e > ov_s:
                    ratio_s = (ov_s - st_s) / (st_e - st_s) if st_e > st_s else 0
                    ratio_e = (ov_e - st_s) / (st_e - st_s) if st_e > st_s else 1
                    orig_s = o_s + ratio_s * (o_e - o_s)
                    orig_e = o_s + ratio_e * (o_e - o_s)
                    new_offset.append((cursor, cursor + (ov_e - ov_s), orig_s, orig_e))

            pieces.append(chunk)
            cursor += chunk_dur
            if gap_arr is not None:
                pieces.append(gap_arr)
                cursor += len(gap_arr) / sr

    if not pieces:
        sf.write(str(out_wav_path), stream_audio, sr, format="WAV", subtype="PCM_16")
        return out_wav_path, all_orig, offset_table

    y = np.concatenate(pieces).astype(np.float32, copy=False)
    sf.write(str(out_wav_path), y, sr, format="WAV", subtype="PCM_16")

    # Dominant original intervals (for TextGrid)
    dominant_orig: List[Tuple[float, float]] = []
    for stream_s, stream_e, orig_s, orig_e in offset_table:
        seg_dur = stream_e - stream_s
        overlap = 0.0
        for ms, me in main_segs:
            o_s = max(stream_s, ms)
            o_e = min(stream_e, me)
            if o_e > o_s:
                overlap += o_e - o_s
        if overlap >= seg_dur * 0.5:
            dominant_orig.append((orig_s, orig_e))

    if not dominant_orig:
        dominant_orig = all_orig

    return out_wav_path, dominant_orig, new_offset if new_offset else offset_table


# ============================================================================
# Main component class
# ============================================================================
class SpeakerClassification:
    """Stage 06 — Speaker-type classification and stream export.

    Inputs: speech_segments parquet (from Stage 05) + analysis WAV.
    Outputs: classified parquets, per-class WAVs, per-class parquets, TextGrid.
    """

    def __init__(self, config: SpeakerClassificationConfig):
        self.config = config
        self._diar_pipeline = None
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

        user = os.environ.get("USER", "user")
        scratch_cache = f"/scratch/users/{user}/.cache/huggingface"
        os.environ.setdefault("HF_HOME", scratch_cache)
        os.environ.setdefault("HF_HUB_CACHE", f"{scratch_cache}/hub")
        os.environ.setdefault("TRANSFORMERS_CACHE", f"{scratch_cache}/transformers")
        os.environ.setdefault("PYANNOTE_DISABLE_NOTEBOOK", "1")

        warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
        warnings.filterwarnings(
            "ignore",
            message=r".*std\(\): degrees of freedom is <= 0.*",
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
        try:
            self._model.set_params(device="cpu")
        except Exception:
            pass
        logger.info(f"Loaded classifier: {type(self._model)} from {model_path}")
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
        speech_segments_parquet_path: Path,
        analysis_wav_path: Path,
        participant_id: str,
    ) -> SpeakerClassificationArtifact:
        """Classify speech segments and export per-class WAVs + parquets.

        Parameters
        ----------
        speech_segments_parquet_path
            Output of Stage 05 (VAD ∩ Diarization intersection).
        analysis_wav_path
            The original analysis-ready WAV (mono 16 kHz).
        participant_id
            Participant identifier.

        Outputs
        -------
        For each participant:

        **Classified segments parquet** (``<pid>_classified_segments.parquet``):

        ====================  =======  =============================================
        Column                Dtype    Description
        ====================  =======  =============================================
        chunk_id              int64    Diarization chunk
        speaker_id_local      object   Diarization speaker label
        start_sec             float64  Segment start in analysis WAV
        end_sec               float64  Segment end in analysis WAV
        n_merged              int64    Number of intersection segments merged
        source_segment_ids    object   List of Stage 05 segment_ids that were merged
        duration_sec          float64  Segment duration (seconds)
        n_windows             int64    eGeMAPS windows used
        probs_adult_male      float64  XGBoost probability
        probs_adult_female    float64  XGBoost probability
        probs_child           float64  XGBoost probability
        probs_background      float64  XGBoost probability
        predicted_class       object   Argmax class
        predicted_confidence  float64  Max probability
        ====================  =======  =============================================

        **Per-class parquets** (``<pid>_main_female.parquet``, etc.):

        ===================  =======  =============================================
        Column               Dtype    Description
        ===================  =======  =============================================
        stream_segment_id    int64    0-based index in the WAV stream
        stream_start_sec     float64  Start position in the class WAV
        stream_end_sec       float64  End position in the class WAV
        stream_duration_sec  float64  Duration in the class WAV
        orig_start_sec       float64  Corresponding start in analysis WAV
        orig_end_sec         float64  Corresponding end in analysis WAV
        orig_duration_sec    float64  Duration in analysis WAV
        predicted_class      object   Class label
        ===================  =======  =============================================
        """
        try:
            ensure_dir(self.config.artifacts_dir)
            ensure_dir(self.config.output_audio_root / participant_id)

            if not analysis_wav_path.exists():
                raise FileNotFoundError(f"analysis wav not found: {analysis_wav_path}")
            if not speech_segments_parquet_path.exists():
                raise FileNotFoundError(
                    f"speech segments parquet not found: {speech_segments_parquet_path}"
                )

            wav_info = sf.info(str(analysis_wav_path))
            full_duration = float(wav_info.duration)
            logger.info(
                f"[{participant_id}] Classification | wav={analysis_wav_path} "
                f"dur={full_duration / 3600:.2f}h"
            )

            # --- Load speech segments from Stage 05 ---
            speech_df = pd.read_parquet(speech_segments_parquet_path)
            speech_df.insert(0, "wav_path", str(analysis_wav_path))
            logger.info(f"[{participant_id}] Loaded {len(speech_df)} speech segments from Stage 05")

            # --- Merge close segments ---
            logger.info(f"[{participant_id}] Merging close segments (gap≤{self.config.merge_gap_sec}s)")
            merged_df = merge_close_segments(speech_df, gap_thresh=self.config.merge_gap_sec)

            # Drop segments shorter than threshold after merge
            if self.config.min_segment_sec > 0:
                before = len(merged_df)
                merged_df = merged_df[merged_df["duration_sec"] >= self.config.min_segment_sec].reset_index(drop=True)
                logger.info(
                    f"[{participant_id}] After merge: {before} → {len(merged_df)} "
                    f"(dropped {before - len(merged_df)} short segments)"
                )
            else:
                logger.info(f"[{participant_id}] After merge: {len(merged_df)} segments")

            # --- Classification ---
            logger.info(f"[{participant_id}] eGeMAPS + XGBoost classification")
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

            # Class distribution
            class_durations: Dict[str, float] = {}
            for cn in self.config.class_names:
                sel = classified_df[classified_df["predicted_class"] == cn]
                class_durations[cn] = float(sel["duration_sec"].sum()) if not sel.empty else 0.0
            logger.info(f"[{participant_id}] Class durations: {class_durations}")

            # --- Stream export + secondary diarization ---
            logger.info(f"[{participant_id}] Building per-class streams")
            tmp_secondary = self.config.output_audio_root / "_tmp_secondary" / participant_id
            ensure_dir(tmp_secondary)

            diar_pipeline = self._load_diarization_pipeline()

            # --- Female stream ---
            fem_audio, _, fem_offsets = build_class_stream(classified_df, audio_16k, sr_16k, "adult_female")
            fem_dominant_intervals: List[Tuple[float, float]] = []
            fem_final_offsets = fem_offsets
            if len(fem_audio) > 0:
                _, fem_dominant_intervals, fem_final_offsets = extract_main_speaker_audio(
                    stream_audio=fem_audio,
                    sr=sr_16k,
                    diar_pipeline=diar_pipeline,
                    tmp_wav_path=tmp_secondary / "fem_stream_tmp.wav",
                    out_wav_path=self.config.main_female_wav_path,
                    offset_table=fem_offsets,
                )
                logger.info(f"[{participant_id}] main_female.wav ({len(fem_dominant_intervals)} dominant segs)")
            else:
                logger.warning(f"[{participant_id}] No adult_female segments")

            fem_stream_df = _build_stream_parquet(fem_final_offsets, "adult_female")
            write_parquet(fem_stream_df, self.config.main_female_parquet_path)

            # --- Male stream ---
            mal_audio, _, mal_offsets = build_class_stream(classified_df, audio_16k, sr_16k, "adult_male")
            mal_dominant_intervals: List[Tuple[float, float]] = []
            mal_final_offsets = mal_offsets
            if len(mal_audio) > 0:
                _, mal_dominant_intervals, mal_final_offsets = extract_main_speaker_audio(
                    stream_audio=mal_audio,
                    sr=sr_16k,
                    diar_pipeline=diar_pipeline,
                    tmp_wav_path=tmp_secondary / "mal_stream_tmp.wav",
                    out_wav_path=self.config.main_male_wav_path,
                    offset_table=mal_offsets,
                )
                logger.info(f"[{participant_id}] main_male.wav ({len(mal_dominant_intervals)} dominant segs)")
            else:
                logger.warning(f"[{participant_id}] No adult_male segments")

            mal_stream_df = _build_stream_parquet(mal_final_offsets, "adult_male")
            write_parquet(mal_stream_df, self.config.main_male_parquet_path)

            # --- Child stream ---
            child_audio, _, child_offsets = build_class_stream(classified_df, audio_16k, sr_16k, "child")
            if len(child_audio) > 0:
                write_stream_wav(child_audio, sr_16k, self.config.child_wav_path)
                logger.info(f"[{participant_id}] child.wav")
            else:
                logger.warning(f"[{participant_id}] No child segments")

            child_stream_df = _build_stream_parquet(child_offsets, "child")
            write_parquet(child_stream_df, self.config.child_parquet_path)

            # --- Background stream ---
            bg_audio, _, bg_offsets = build_class_stream(classified_df, audio_16k, sr_16k, "background")
            if len(bg_audio) > 0:
                write_stream_wav(bg_audio, sr_16k, self.config.background_wav_path)
                logger.info(f"[{participant_id}] background.wav")
            else:
                logger.warning(f"[{participant_id}] No background segments")

            bg_stream_df = _build_stream_parquet(bg_offsets, "background")
            write_parquet(bg_stream_df, self.config.background_parquet_path)

            # --- TextGrid ---
            logger.info(f"[{participant_id}] TextGrid")
            child_df = classified_df[classified_df["predicted_class"] == "child"]
            bg_df = classified_df[classified_df["predicted_class"] == "background"]
            fem_dom_df = intervals_to_df(fem_dominant_intervals, "adult_female")
            mal_dom_df = intervals_to_df(mal_dominant_intervals, "adult_male")

            textgrid_df = pd.concat(
                [fem_dom_df, mal_dom_df, child_df, bg_df], ignore_index=True,
            )
            write_textgrid(
                df=textgrid_df,
                duration_sec=full_duration,
                out_path=self.config.textgrid_path,
                tier_map=TIER_MAP,
            )

            # --- Save classified segments parquet ---
            save_df = classified_df.drop(columns=["wav_path"], errors="ignore")
            write_parquet(save_df, self.config.classified_segments_parquet_path)

            total_speech = float(classified_df["duration_sec"].sum())
            summary = {
                "participant_id": participant_id,
                "analysis_wav": str(analysis_wav_path),
                "speech_segments_parquet": str(speech_segments_parquet_path),
                "duration_sec": full_duration,
                "n_input_segments": len(speech_df),
                "n_after_merge": len(merged_df),
                "n_classified_segments": len(classified_df),
                "total_speech_sec": total_speech,
                "class_durations": class_durations,
                "n_dominant_female_segments": len(fem_dominant_intervals),
                "n_dominant_male_segments": len(mal_dominant_intervals),
                "outputs": {
                    "classified_segments_parquet": str(self.config.classified_segments_parquet_path),
                    "main_female_wav": str(self.config.main_female_wav_path),
                    "main_female_parquet": str(self.config.main_female_parquet_path),
                    "main_male_wav": str(self.config.main_male_wav_path),
                    "main_male_parquet": str(self.config.main_male_parquet_path),
                    "child_wav": str(self.config.child_wav_path),
                    "child_parquet": str(self.config.child_parquet_path),
                    "background_wav": str(self.config.background_wav_path),
                    "background_parquet": str(self.config.background_parquet_path),
                    "textgrid": str(self.config.textgrid_path),
                },
            }
            write_json(summary, self.config.summary_json_path)

            # Cleanup
            try:
                shutil.rmtree(tmp_secondary, ignore_errors=True)
            except Exception:
                pass

            logger.info(
                f"[{participant_id}] Stage 06 DONE | "
                f"segments={len(classified_df)} speech={total_speech / 3600:.2f}h"
            )

            return SpeakerClassificationArtifact(
                classified_segments_parquet_path=self.config.classified_segments_parquet_path,
                main_female_parquet_path=self.config.main_female_parquet_path,
                main_male_parquet_path=self.config.main_male_parquet_path,
                child_parquet_path=self.config.child_parquet_path,
                background_parquet_path=self.config.background_parquet_path,
                summary_json_path=self.config.summary_json_path,
                textgrid_path=self.config.textgrid_path,
                main_female_wav_path=self.config.main_female_wav_path,
                main_male_wav_path=self.config.main_male_wav_path,
                child_wav_path=self.config.child_wav_path,
                background_wav_path=self.config.background_wav_path,
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
