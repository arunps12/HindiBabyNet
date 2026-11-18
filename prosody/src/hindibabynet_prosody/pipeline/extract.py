"""Participant-level extraction, caching, QC, and output orchestration."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import logging
import multiprocessing
import queue
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
import torch
import torchaudio.functional as ta_functional

from hindibabynet_prosody.config import AppConfig, config_to_dict
from hindibabynet_prosody.features.aggregation import median_or_none
from hindibabynet_prosody.features.pitch import (
    FirstPassPitchResult,
    SegmentPitchResult,
    extract_first_pass_pitch,
    extract_segment_pitch,
    extract_segment_pitch_with_shared_adaptive_range,
    summarize_participant_speaker_adaptive_range,
)
from hindibabynet_prosody.features.ssl import SSLExtractionResult, SSLExtractor, resolve_torch_device
from hindibabynet_prosody.io.audio import AudioResolution
from hindibabynet_prosody.io.outputs import write_csv, write_json
from hindibabynet_prosody.io.rttm import discover_rttm_files, normalize_segments
from hindibabynet_prosody.pipeline.validate import run_validation
from hindibabynet_prosody.reduction.pca import fit_embedding_pca
from hindibabynet_prosody.schemas import NormalizedSegment
from hindibabynet_prosody.utils.hashing import stable_hash
from hindibabynet_prosody.utils.paths import ensure_directory
from hindibabynet_prosody.utils.reproducibility import set_global_seed


logger = logging.getLogger("hindibabynet_prosody.pipeline.extract")
_PROCESS_POOL_EXECUTOR = concurrent.futures.ProcessPoolExecutor
_WORKER_CONFIG: AppConfig | None = None
_WORKER_EXTRACTORS: dict[str, SSLExtractor] | None = None
_WORKER_PROGRESS_QUEUE: Any | None = None
_WORKER_LABEL: str | None = None


@dataclass(frozen=True)
class ExtractPlan:
    participant_count: int
    selected_device: str
    cache_enabled: bool


@dataclass(frozen=True)
class ExtractOutputs:
    final_csv: Path
    qc_summary_csv: Path
    skipped_segments_csv: Path
    unmapped_participants_csv: Path
    audio_resolution_issues_csv: Path
    run_metadata_json: Path
    participant_count: int
    row_count: int


@dataclass(frozen=True)
class ParticipantTask:
    rttm_index: int
    total_rttm_files: int
    source_participant_id: str
    participant_id: str
    selected_segment_count: int
    audio_resolution: AudioResolution
    segments: list[NormalizedSegment]


@dataclass(frozen=True)
class ParticipantExtractResult:
    task: ParticipantTask
    final_rows: list[dict[str, Any]]
    qc_rows: list[dict[str, Any]]
    skipped_rows: list[dict[str, Any]]
    embedding_rows: list[dict[str, Any]]


def build_extract_plan(
    config: AppConfig,
    *,
    device: str | None,
    no_cache: bool,
) -> ExtractPlan:
    report = run_validation(config)
    return ExtractPlan(
        participant_count=len(report.audio_resolutions),
        selected_device=resolve_torch_device(device or config.runtime.device),
        cache_enabled=not no_cache,
    )


def _speaker_sort_order(config: AppConfig) -> dict[str, int]:
    ordered = list(dict.fromkeys(config.speakers.include.values()))
    return {speaker: index for index, speaker in enumerate(ordered)}


def _slice_audio_file(
    audio_file: sf.SoundFile,
    start_time_s: float,
    end_time_s: float,
    tolerance_s: float,
) -> np.ndarray | None:
    sample_rate = int(audio_file.samplerate)
    n_samples = len(audio_file)
    start_index = int(round(start_time_s * sample_rate))
    end_index = int(round(end_time_s * sample_rate))
    tolerated_max = n_samples + int(round(tolerance_s * sample_rate))
    if start_index < 0 or end_index <= start_index or end_index > tolerated_max:
        return None
    end_index = min(end_index, n_samples)
    if start_index >= n_samples or end_index <= start_index:
        return None
    audio_file.seek(start_index)
    frames = audio_file.read(frames=end_index - start_index, dtype="float32", always_2d=True)
    if frames.size == 0:
        return None
    return frames.mean(axis=1, dtype=np.float64).astype(np.float32)


def _resample_waveform(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    tensor = torch.from_numpy(waveform.astype(np.float32, copy=False)).unsqueeze(0)
    resampled = ta_functional.resample(tensor, source_rate, target_rate)
    return resampled.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _participant_matches(filters: list[str] | None, segment: NormalizedSegment) -> bool:
    if not filters:
        return True
    requested = set(filters)
    return segment.source_participant_id in requested or (
        segment.participant_id is not None and segment.participant_id in requested
    )


def _speaker_matches(filters: list[str] | None, segment: NormalizedSegment) -> bool:
    if not filters:
        return True
    requested = set(filters)
    return segment.speaker_label in requested or (segment.speaker is not None and segment.speaker in requested)


def _segment_issue_row(segment: NormalizedSegment, **fields: object) -> dict[str, object]:
    row: dict[str, object] = {
        "participant_id": segment.participant_id,
        "source_participant_id": segment.source_participant_id,
        "speaker": segment.speaker_label,
        "start_time_s": segment.start_time_s,
        "duration_s": segment.duration_s,
    }
    row.update(fields)
    return row


def _pitch_failure_row(segment: NormalizedSegment, result: SegmentPitchResult) -> dict[str, object]:
    return _segment_issue_row(
        segment,
        first_pass_floor_hz=result.first_pass_floor_hz,
        first_pass_ceiling_hz=result.first_pass_ceiling_hz,
        first_pass_voiced_frames=result.first_pass_voiced_frames,
        first_pass_median_F0_Hz=result.first_pass_median_f0_hz,
        adaptive_floor_hz=result.adaptive_floor_hz,
        adaptive_ceiling_hz=result.adaptive_ceiling_hz,
        second_pass_attempted=result.second_pass_attempted,
        second_pass_voiced_frames=result.second_pass_voiced_frames,
        final_pass_used=result.final_pass_used,
        final_voiced_frames=result.final_voiced_frames,
        voiced_proportion=result.voiced_proportion,
        pitch_extraction_success=result.pitch_extraction_success,
        fallback_used=result.fallback_used,
        failure_reason=result.failure_reason,
        warning_reason=result.warning_reason,
    )


def _ssl_failure_row(segment: NormalizedSegment, model_name: str, reason: str) -> dict[str, object]:
    return _segment_issue_row(
        segment,
        model=model_name,
        failure_reason=reason,
    )


def _make_group_state(participant_id: str, source_participant_id: str, speaker: str) -> dict[str, Any]:
    return {
        "participant_id": participant_id,
        "source_participant_id": source_participant_id,
        "speaker": speaker,
        "durations": [],
        "pitch_values": {
            "mean_F0_Hz": [],
            "median_F0_Hz": [],
            "sd_F0_Hz": [],
            "robust_pitch_range_Hz": [],
            "mean_F0_ST": [],
            "median_F0_ST": [],
            "sd_F0_ST": [],
            "robust_pitch_range_ST": [],
            "first_pass_median_f0_hz": [],
            "adaptive_floor_hz": [],
            "adaptive_ceiling_hz": [],
        },
        "wav2vec_embeddings": [],
        "hubert_embeddings": [],
        "counts": {
            "n_segments_total": 0,
            "n_segments_valid_audio": 0,
            "n_segments_valid_pitch": 0,
            "n_segments_valid_wav2vec": 0,
            "n_segments_valid_hubert": 0,
            "n_segments_failed_pitch": 0,
            "n_segments_failed_wav2vec": 0,
            "n_segments_failed_hubert": 0,
            "n_segments_pitch_first_pass_valid": 0,
            "n_segments_pitch_second_pass_valid": 0,
            "n_segments_pitch_first_pass_fallback": 0,
            "n_segments_pitch_failed": 0,
        },
        "durations_summary": {
            "total_segment_duration_s": 0.0,
            "total_valid_pitch_duration_s": 0.0,
            "total_valid_wav2vec_duration_s": 0.0,
            "total_valid_hubert_duration_s": 0.0,
        },
        "pending_segments": [],
    }


def _finalize_group_rows(
    group_states: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    for state in group_states.values():
        counts = state["counts"]
        durations_summary = state["durations_summary"]
        durations = state["durations"]
        pitch_values = state["pitch_values"]
        final_rows.append(
            {
                "participant_id": state["participant_id"],
                "source_participant_id": state["source_participant_id"],
                "speaker": state["speaker"],
                "n_segments_total": counts["n_segments_total"],
                "n_segments_pitch_valid": counts["n_segments_valid_pitch"],
                "n_segments_wav2vec_valid": counts["n_segments_valid_wav2vec"],
                "n_segments_hubert_valid": counts["n_segments_valid_hubert"],
                "total_segment_duration_s": durations_summary["total_segment_duration_s"],
                "median_segment_duration_s": median_or_none(durations),
                "mean_F0_Hz": median_or_none(pitch_values["mean_F0_Hz"]),
                "median_F0_Hz": median_or_none(pitch_values["median_F0_Hz"]),
                "sd_F0_Hz": median_or_none(pitch_values["sd_F0_Hz"]),
                "robust_pitch_range_Hz": median_or_none(pitch_values["robust_pitch_range_Hz"]),
                "mean_F0_ST": median_or_none(pitch_values["mean_F0_ST"]),
                "median_F0_ST": median_or_none(pitch_values["median_F0_ST"]),
                "sd_F0_ST": median_or_none(pitch_values["sd_F0_ST"]),
                "robust_pitch_range_ST": median_or_none(pitch_values["robust_pitch_range_ST"]),
            }
        )
        qc_rows.append(
            {
                "participant_id": state["participant_id"],
                "source_participant_id": state["source_participant_id"],
                "speaker": state["speaker"],
                **counts,
                **durations_summary,
                "median_first_pass_F0_Hz": median_or_none(pitch_values["first_pass_median_f0_hz"]),
                "median_adaptive_floor_Hz": median_or_none(pitch_values["adaptive_floor_hz"]),
                "median_adaptive_ceiling_Hz": median_or_none(pitch_values["adaptive_ceiling_hz"]),
            }
        )
        embedding_rows.append(
            {
                "participant_id": state["participant_id"],
                "source_participant_id": state["source_participant_id"],
                "speaker": state["speaker"],
                "wav2vec_mean_embedding": (
                    np.mean(np.vstack(state["wav2vec_embeddings"]), axis=0).astype(np.float32)
                    if state["wav2vec_embeddings"]
                    else None
                ),
                "hubert_mean_embedding": (
                    np.mean(np.vstack(state["hubert_embeddings"]), axis=0).astype(np.float32)
                    if state["hubert_embeddings"]
                    else None
                ),
            }
        )
    return final_rows, qc_rows, embedding_rows


def _cache_path(config: AppConfig, source_participant_id: str) -> Path:
    return config.paths.cache_dir / f"{source_participant_id}.joblib"


def _load_cache(path: Path, config_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = joblib.load(path)
    if payload.get("config_hash") != config_hash:
        return None
    return payload


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    joblib.dump(payload, path)


def _make_ssl_extractors(
    config: AppConfig,
    *,
    device: str,
    overrides: dict[str, SSLExtractor] | None,
) -> dict[str, SSLExtractor]:
    if overrides is not None:
        return overrides
    return {
        "wav2vec": SSLExtractor(
            model_id=config.ssl.wav2vec2_model,
            layer=config.ssl.wav2vec2_layer,
            device=device,
            minimum_segment_duration_s=config.ssl.minimum_segment_duration_s,
            maximum_segment_duration_s=config.ssl.maximum_segment_duration_s,
        ),
        "hubert": SSLExtractor(
            model_id=config.ssl.hubert_model,
            layer=config.ssl.hubert_layer,
            device=device,
            minimum_segment_duration_s=config.ssl.minimum_segment_duration_s,
            maximum_segment_duration_s=config.ssl.maximum_segment_duration_s,
        ),
    }


def _initialize_extract_worker(
    config_payload: dict[str, Any],
    device: str,
    ssl_extractors: dict[str, SSLExtractor] | None,
    progress_queue: Any | None = None,
) -> None:
    global _WORKER_CONFIG, _WORKER_EXTRACTORS, _WORKER_PROGRESS_QUEUE, _WORKER_LABEL
    _WORKER_CONFIG = AppConfig.model_validate(config_payload)
    set_global_seed(_WORKER_CONFIG.project.random_seed)
    _WORKER_EXTRACTORS = _make_ssl_extractors(
        _WORKER_CONFIG,
        device=device,
        overrides=ssl_extractors,
    )
    _WORKER_PROGRESS_QUEUE = progress_queue
    worker_name = multiprocessing.current_process().name
    match = re.search(r"(\d+)$", worker_name)
    _WORKER_LABEL = match.group(1) if match else str(multiprocessing.current_process().pid)


def _emit_segment_progress(task: ParticipantTask, processed_segments: int) -> None:
    if _WORKER_PROGRESS_QUEUE is not None:
        _WORKER_PROGRESS_QUEUE.put(
            {
                "worker": _WORKER_LABEL or str(multiprocessing.current_process().pid),
                "source_participant_id": task.source_participant_id,
                "participant_id": task.participant_id,
                "processed_segments": processed_segments,
                "selected_segment_count": task.selected_segment_count,
            }
        )
        return
    logger.info(
        "[%s/%s] source=%s participant=%s processed_segments=%s/%s",
        task.rttm_index,
        task.total_rttm_files,
        task.source_participant_id,
        task.participant_id,
        processed_segments,
        task.selected_segment_count,
    )


def _progress_consumer(progress_queue: Any, stop_event: threading.Event) -> None:
    while True:
        try:
            item = progress_queue.get(timeout=0.2)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue
        if item is None:
            break
        logger.info(
            "worker=%s source=%s participant=%s processed_segments=%s/%s",
            item["worker"],
            item["source_participant_id"],
            item["participant_id"],
            item["processed_segments"],
            item["selected_segment_count"],
        )


def _prepare_participant_tasks(
    config: AppConfig,
    *,
    validation_lookup_mapping: dict[str, str],
    audio_map: dict[str, AudioResolution],
    participants: list[str] | None,
    speakers: list[str] | None,
) -> tuple[list[ParticipantTask], list[dict[str, Any]]]:
    skipped_rows: list[dict[str, Any]] = []
    tasks: list[ParticipantTask] = []
    rttm_paths = list(discover_rttm_files(config))
    total_rttm_files = len(rttm_paths)

    for rttm_index, rttm_path in enumerate(rttm_paths, start=1):
        segments, normalize_issues = normalize_segments(config, rttm_path, validation_lookup_mapping)
        skipped_rows.extend(normalize_issues)
        if not segments:
            continue
        source_participant_id = segments[0].source_participant_id
        participant_id = segments[0].participant_id
        if participant_id is None:
            continue
        if not any(_participant_matches(participants, segment) for segment in segments):
            continue

        selected_segment_count = _count_selected_segments(
            segments,
            participants=participants,
            speakers=speakers,
        )
        if selected_segment_count == 0:
            continue

        audio_resolution = audio_map.get(source_participant_id)
        if audio_resolution is None or audio_resolution.path is None or audio_resolution.issue is not None:
            continue

        tasks.append(
            ParticipantTask(
                rttm_index=rttm_index,
                total_rttm_files=total_rttm_files,
                source_participant_id=source_participant_id,
                participant_id=participant_id,
                selected_segment_count=selected_segment_count,
                audio_resolution=audio_resolution,
                segments=[
                    segment
                    for segment in segments
                    if _participant_matches(participants, segment) and _speaker_matches(speakers, segment)
                ],
            )
        )

    return tasks, skipped_rows


def _extract_participant_task(
    task: ParticipantTask,
    *,
    config: AppConfig,
    model_extractors: dict[str, SSLExtractor],
    log_progress: bool,
) -> ParticipantExtractResult:
    group_states: dict[tuple[str, str, str], dict[str, Any]] = {}
    local_skipped: list[dict[str, Any]] = []
    pending_ssl_items: list[dict[str, Any]] = []
    selected_segments_seen = 0
    with sf.SoundFile(task.audio_resolution.path) as audio_file:
        sample_rate = int(audio_file.samplerate)
        for segment in task.segments:
            selected_segments_seen += 1
            if log_progress and (selected_segments_seen == 1 or selected_segments_seen % 250 == 0):
                _emit_segment_progress(task, selected_segments_seen)
            if segment.speaker is None:
                local_skipped.append(_segment_issue_row(segment, failure_reason="unsupported_speaker_label"))
                continue

            group_key = (segment.participant_id or "", segment.source_participant_id, segment.speaker)
            state = group_states.setdefault(
                group_key,
                _make_group_state(segment.participant_id or "", segment.source_participant_id, segment.speaker),
            )
            counts = state["counts"]
            durations_summary = state["durations_summary"]
            counts["n_segments_total"] += 1
            durations_summary["total_segment_duration_s"] += segment.duration_s
            state["durations"].append(segment.duration_s)

            segment_waveform = _slice_audio_file(
                audio_file,
                segment.start_time_s,
                segment.end_time_s,
                config.audio.end_tolerance_s,
            )
            if segment_waveform is None:
                counts["n_segments_failed_pitch"] += 1
                counts["n_segments_failed_wav2vec"] += 1
                counts["n_segments_failed_hubert"] += 1
                local_skipped.append(_segment_issue_row(segment, failure_reason="segment_outside_audio_duration"))
                continue

            counts["n_segments_valid_audio"] += 1
            pitch_sound = parselmouth.Sound(segment_waveform.astype(np.float64), sample_rate)
            if config.pitch.adaptive.level == "segment":
                pitch_result = extract_segment_pitch(
                    pitch_sound,
                    segment.speaker_label,
                    config.pitch,
                    default_range=config.pitch.default_speaker_settings,
                )
                _update_state_with_pitch_success(
                    state,
                    segment,
                    pitch_result,
                    local_skipped,
                    config.pitch.minimum_voiced_frames,
                )
                _queue_ssl_segment(
                    config=config,
                    pending_items=pending_ssl_items,
                    sample_rate=sample_rate,
                    segment_waveform=segment_waveform,
                    segment=segment,
                    state=state,
                    model_extractors=model_extractors,
                    local_skipped=local_skipped,
                )
                continue

            first_pass_result = extract_first_pass_pitch(
                pitch_sound,
                segment.speaker_label,
                config.pitch,
                default_range=config.pitch.default_speaker_settings,
            )
            state["pending_segments"].append(
                {
                    "segment": segment,
                    "segment_waveform": segment_waveform,
                    "sample_rate": sample_rate,
                    "first_pass_result": first_pass_result,
                    "pitch_sound": pitch_sound,
                }
            )

    if config.pitch.adaptive.level == "participant_speaker":
        for state in group_states.values():
            pending_segments = state["pending_segments"]
            first_pass_results: list[FirstPassPitchResult] = [item["first_pass_result"] for item in pending_segments]
            shared_adaptive_range = summarize_participant_speaker_adaptive_range(
                first_pass_results,
                config.pitch,
            )
            for item in pending_segments:
                segment = item["segment"]
                pitch_result = extract_segment_pitch_with_shared_adaptive_range(
                    item["pitch_sound"],
                    segment.speaker_label,
                    config.pitch,
                    shared_adaptive_range,
                    default_range=config.pitch.default_speaker_settings,
                )
                _update_state_with_pitch_success(
                    state,
                    segment,
                    pitch_result,
                    local_skipped,
                    config.pitch.minimum_voiced_frames,
                )
                _queue_ssl_segment(
                    config=config,
                    pending_items=pending_ssl_items,
                    sample_rate=item["sample_rate"],
                    segment_waveform=item["segment_waveform"],
                    segment=segment,
                    state=state,
                    model_extractors=model_extractors,
                    local_skipped=local_skipped,
                )
            state["pending_segments"] = []

    _flush_ssl_batch(
        config=config,
        pending_items=pending_ssl_items,
        model_extractors=model_extractors,
        local_skipped=local_skipped,
    )

    participant_final_rows, participant_qc_rows, participant_embedding_rows = _finalize_group_rows(group_states)
    return ParticipantExtractResult(
        task=task,
        final_rows=participant_final_rows,
        qc_rows=participant_qc_rows,
        skipped_rows=local_skipped,
        embedding_rows=participant_embedding_rows,
    )


def _extract_participant_task_in_worker(task: ParticipantTask) -> ParticipantExtractResult:
    if _WORKER_CONFIG is None or _WORKER_EXTRACTORS is None:
        raise RuntimeError("participant worker was not initialized")
    return _extract_participant_task(
        task,
        config=_WORKER_CONFIG,
        model_extractors=_WORKER_EXTRACTORS,
        log_progress=True,
    )


def _merge_participant_result(
    final_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    embedding_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    result: ParticipantExtractResult,
) -> None:
    final_rows.extend(result.final_rows)
    qc_rows.extend(result.qc_rows)
    embedding_rows.extend(result.embedding_rows)
    skipped_rows.extend(result.skipped_rows)


def _parallel_worker_count(config: AppConfig, selected_device: str, task_count: int) -> int:
    if selected_device != "cpu":
        return 0
    if config.runtime.num_workers <= 1:
        return 0
    if task_count <= 1:
        return 0
    return min(config.runtime.num_workers, task_count)


def _merge_pca_scores(
    final_frame: pd.DataFrame,
    embedding_rows: list[dict[str, Any]],
    config: AppConfig,
    artifact_dir: Path,
) -> pd.DataFrame:
    enriched = final_frame.copy()
    id_columns = ["participant_id", "source_participant_id", "speaker"]
    for model_name in ["wav2vec", "hubert"]:
        usable_rows = [row for row in embedding_rows if row.get(f"{model_name}_mean_embedding") is not None]
        if not usable_rows:
            for index in range(1, config.pca.n_components + 1):
                enriched[f"{model_name}_PC{index}"] = np.nan
            continue
        dimension = len(usable_rows[0][f"{model_name}_mean_embedding"])
        wide_rows: list[dict[str, Any]] = []
        for row in usable_rows:
            wide_row = {column: row[column] for column in id_columns}
            for dim_index, value in enumerate(row[f"{model_name}_mean_embedding"]):
                wide_row[f"d{dim_index}"] = value
            wide_rows.append(wide_row)
        wide_frame = pd.DataFrame(wide_rows)
        value_columns = [f"d{dim_index}" for dim_index in range(dimension)]
        artifacts = fit_embedding_pca(
            embeddings=wide_frame,
            id_columns=id_columns,
            value_columns=value_columns,
            pca_config=config.pca,
            component_prefix=model_name,
            artifact_dir=artifact_dir,
        )
        enriched = enriched.merge(artifacts.projected, on=id_columns, how="left")
    return enriched


def _git_commit_or_none(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "torch", "torchaudio", "transformers", "praat-parselmouth", "scikit-learn"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _count_selected_segments(
    segments: list[NormalizedSegment],
    *,
    participants: list[str] | None,
    speakers: list[str] | None,
) -> int:
    return sum(
        1
        for segment in segments
        if _participant_matches(participants, segment) and _speaker_matches(speakers, segment)
    )


def _update_state_with_pitch_success(
    state: dict[str, Any],
    segment: NormalizedSegment,
    pitch_result: SegmentPitchResult,
    local_skipped: list[dict[str, Any]],
    minimum_voiced_frames: int,
) -> None:
    counts = state["counts"]
    durations_summary = state["durations_summary"]
    if pitch_result.first_pass_voiced_frames >= minimum_voiced_frames:
        counts["n_segments_pitch_first_pass_valid"] += 1
        if pitch_result.first_pass_median_f0_hz is not None:
            state["pitch_values"]["first_pass_median_f0_hz"].append(pitch_result.first_pass_median_f0_hz)
    if pitch_result.second_pass_attempted and pitch_result.second_pass_voiced_frames >= minimum_voiced_frames:
        counts["n_segments_pitch_second_pass_valid"] += 1
    if pitch_result.final_pass_used == "first_pass_fallback" and pitch_result.pitch_extraction_success:
        counts["n_segments_pitch_first_pass_fallback"] += 1
    if pitch_result.pitch_extraction_success and pitch_result.statistics is not None:
        counts["n_segments_valid_pitch"] += 1
        durations_summary["total_valid_pitch_duration_s"] += segment.duration_s
        stats = pitch_result.statistics
        state["pitch_values"]["mean_F0_Hz"].append(stats.mean_f0_hz)
        state["pitch_values"]["median_F0_Hz"].append(stats.median_f0_hz)
        state["pitch_values"]["sd_F0_Hz"].append(stats.sd_f0_hz)
        state["pitch_values"]["robust_pitch_range_Hz"].append(stats.robust_pitch_range_hz)
        state["pitch_values"]["mean_F0_ST"].append(stats.mean_f0_st)
        state["pitch_values"]["median_F0_ST"].append(stats.median_f0_st)
        state["pitch_values"]["sd_F0_ST"].append(stats.sd_f0_st)
        state["pitch_values"]["robust_pitch_range_ST"].append(stats.robust_pitch_range_st)
        if pitch_result.adaptive_floor_hz is not None:
            state["pitch_values"]["adaptive_floor_hz"].append(pitch_result.adaptive_floor_hz)
        if pitch_result.adaptive_ceiling_hz is not None:
            state["pitch_values"]["adaptive_ceiling_hz"].append(pitch_result.adaptive_ceiling_hz)
        if pitch_result.warning_reason is not None:
            local_skipped.append(_pitch_failure_row(segment, pitch_result))
        return
    counts["n_segments_failed_pitch"] += 1
    counts["n_segments_pitch_failed"] += 1
    local_skipped.append(_pitch_failure_row(segment, pitch_result))


def _process_ssl_for_segment(
    *,
    config: AppConfig,
    sample_rate: int,
    segment_waveform: np.ndarray,
    segment: NormalizedSegment,
    state: dict[str, Any],
    model_extractors: dict[str, SSLExtractor],
    local_skipped: list[dict[str, Any]],
) -> None:
    _flush_ssl_batch(
        config=config,
        pending_items=[
            {
                "sample_rate": sample_rate,
                "segment_waveform": segment_waveform,
                "segment": segment,
                "state": state,
            }
        ],
        model_extractors=model_extractors,
        local_skipped=local_skipped,
    )


def _flush_ssl_batch(
    *,
    config: AppConfig,
    pending_items: list[dict[str, Any]],
    model_extractors: dict[str, SSLExtractor],
    local_skipped: list[dict[str, Any]],
) -> None:
    if not pending_items:
        return

    resampled_waveforms = [
        _resample_waveform(
            item["segment_waveform"],
            item["sample_rate"],
            config.audio.target_sample_rate,
        )
        for item in pending_items
    ]
    for model_name, extractor in model_extractors.items():
        results = extractor.extract_embeddings(
            resampled_waveforms,
            config.audio.target_sample_rate,
        )
        for item, result in zip(pending_items, results, strict=True):
            counts = item["state"]["counts"]
            durations_summary = item["state"]["durations_summary"]
            segment = item["segment"]
            valid_key = f"n_segments_valid_{model_name}"
            failed_key = f"n_segments_failed_{model_name}"
            duration_key = f"total_valid_{model_name}_duration_s"
            embedding_key = f"{model_name}_embeddings"
            if result.success and result.embedding is not None:
                counts[valid_key] += 1
                durations_summary[duration_key] += segment.duration_s
                item["state"][embedding_key].append(result.embedding)
            else:
                counts[failed_key] += 1
                local_skipped.append(
                    _ssl_failure_row(segment, model_name, result.failure_reason or "ssl_extraction_failed")
                )


def _queue_ssl_segment(
    *,
    config: AppConfig,
    pending_items: list[dict[str, Any]],
    sample_rate: int,
    segment_waveform: np.ndarray,
    segment: NormalizedSegment,
    state: dict[str, Any],
    model_extractors: dict[str, SSLExtractor],
    local_skipped: list[dict[str, Any]],
) -> None:
    pending_items.append(
        {
            "sample_rate": sample_rate,
            "segment_waveform": segment_waveform,
            "segment": segment,
            "state": state,
        }
    )
    batch_size = max(1, int(config.runtime.batch_size))
    if len(pending_items) >= batch_size:
        _flush_ssl_batch(
            config=config,
            pending_items=pending_items,
            model_extractors=model_extractors,
            local_skipped=local_skipped,
        )
        pending_items.clear()


def run_extraction(
    config: AppConfig,
    *,
    participants: list[str] | None = None,
    speakers: list[str] | None = None,
    device: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
    no_cache: bool = False,
    ssl_extractors: dict[str, SSLExtractor] | None = None,
) -> ExtractOutputs:
    set_global_seed(config.project.random_seed)
    validation = run_validation(config)
    config_hash = stable_hash(config_to_dict(config))
    output_dir = ensure_directory(config.paths.output_dir)
    qc_dir = ensure_directory(output_dir / "qc")
    artifact_dir = ensure_directory(output_dir / "artifacts")
    final_csv = output_dir / config.output.final_filename
    if final_csv.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing final output: {final_csv}")

    selected_device = resolve_torch_device(device or config.runtime.device)
    audio_map = {item.source_participant_id: item for item in validation.audio_resolutions}
    participant_tasks, normalize_skipped_rows = _prepare_participant_tasks(
        config,
        validation_lookup_mapping=validation.lookup.mapping,
        audio_map=audio_map,
        participants=participants,
        speakers=speakers,
    )

    final_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = list(normalize_skipped_rows)
    audio_issue_rows = [
        {
            "source_participant_id": item.source_participant_id,
            "audio_path": str(item.path) if item.path is not None else None,
            "duration_s": item.duration_s,
            "issue": item.issue,
        }
        for item in validation.audio_resolutions
        if item.issue is not None
    ]
    completed_participants = 0
    total_rttm_files = participant_tasks[0].total_rttm_files if participant_tasks else len(discover_rttm_files(config))
    worker_count = _parallel_worker_count(config, selected_device, len(participant_tasks))
    if config.runtime.num_workers > 1 and worker_count == 0:
        logger.info(
            "Participant multiprocessing disabled for this run (device=%s, selected_participants=%s)",
            selected_device,
            len(participant_tasks),
        )

    def consume_result(result: ParticipantExtractResult, elapsed_s: float) -> None:
        nonlocal completed_participants
        _merge_participant_result(final_rows, qc_rows, embedding_rows, skipped_rows, result)
        if not no_cache:
            _save_cache(
                _cache_path(config, result.task.source_participant_id),
                {
                    "config_hash": config_hash,
                    "final_rows": result.final_rows,
                    "qc_rows": result.qc_rows,
                    "skipped_rows": result.skipped_rows,
                    "embedding_rows": result.embedding_rows,
                },
            )
        completed_participants += 1
        logger.info(
            "[%s/%s] Finished extraction for source=%s participant=%s rows=%s skipped=%s elapsed_s=%.1f completed_participants=%s/%s",
            completed_participants,
            total_rttm_files,
            result.task.source_participant_id,
            result.task.participant_id,
            len(result.final_rows),
            len(result.skipped_rows),
            elapsed_s,
            completed_participants,
            total_rttm_files,
        )

    pending_tasks: list[tuple[ParticipantTask, float]] = []
    for task in participant_tasks:
        cache_path = _cache_path(config, task.source_participant_id)
        if resume and not no_cache:
            cached = _load_cache(cache_path, config_hash)
            if cached is not None:
                completed_participants += 1
                logger.info(
                    "[%s/%s] Loaded cached outputs for source=%s participant=%s completed_participants=%s/%s",
                    completed_participants,
                    total_rttm_files,
                    task.source_participant_id,
                    task.participant_id,
                    completed_participants,
                    total_rttm_files,
                )
                final_rows.extend(cached["final_rows"])
                qc_rows.extend(cached["qc_rows"])
                skipped_rows.extend(cached["skipped_rows"])
                embedding_rows.extend(cached["embedding_rows"])
                continue
        logger.info(
            "[%s/%s] Starting extraction for source=%s participant=%s selected_segments=%s",
            task.rttm_index,
            total_rttm_files,
            task.source_participant_id,
            task.participant_id,
            task.selected_segment_count,
        )
        pending_tasks.append((task, time.monotonic()))

    if worker_count > 1:
        logger.info("Using participant-level multiprocessing with workers=%s", worker_count)
        with multiprocessing.Manager() as manager:
            progress_queue = manager.Queue()
            stop_event = threading.Event()
            consumer_thread = threading.Thread(
                target=_progress_consumer,
                args=(progress_queue, stop_event),
                daemon=True,
            )
            consumer_thread.start()
            try:
                with _PROCESS_POOL_EXECUTOR(
                    max_workers=worker_count,
                    initializer=_initialize_extract_worker,
                    initargs=(config_to_dict(config), selected_device, ssl_extractors, progress_queue),
                ) as executor:
                    future_context: dict[Any, tuple[ParticipantTask, float]] = {}
                    for task, started_at in pending_tasks:
                        future = executor.submit(_extract_participant_task_in_worker, task)
                        future_context[future] = (task, started_at)
                    for future in concurrent.futures.as_completed(future_context):
                        task, started_at = future_context[future]
                        result = future.result()
                        consume_result(result, time.monotonic() - started_at)
            finally:
                stop_event.set()
                progress_queue.put(None)
                consumer_thread.join()
    else:
        model_extractors = _make_ssl_extractors(config, device=selected_device, overrides=ssl_extractors)
        for task, started_at in pending_tasks:
            result = _extract_participant_task(
                task,
                config=config,
                model_extractors=model_extractors,
                log_progress=True,
            )
            consume_result(result, time.monotonic() - started_at)

    final_frame = pd.DataFrame(final_rows)
    if final_frame.empty:
        raise ValueError("No participant-speaker rows were produced by extraction")

    final_frame = _merge_pca_scores(final_frame, embedding_rows, config, artifact_dir)
    speaker_order = _speaker_sort_order(config)
    final_frame["_speaker_order"] = final_frame["speaker"].map(speaker_order).fillna(999)
    final_frame = final_frame.sort_values(["participant_id", "_speaker_order", "speaker"]).drop(columns=["_speaker_order"]).reset_index(drop=True)

    qc_frame = pd.DataFrame(qc_rows).sort_values(["participant_id", "speaker"]).reset_index(drop=True)
    skipped_frame = pd.DataFrame(skipped_rows)
    unmapped_frame = pd.DataFrame({"source_participant_id": validation.unmapped_sources})
    audio_issues_frame = pd.DataFrame(audio_issue_rows)

    write_csv(final_frame, final_csv)
    qc_summary_csv = qc_dir / "processing_summary.csv"
    skipped_segments_csv = qc_dir / "skipped_segments.csv"
    unmapped_participants_csv = qc_dir / "unmapped_participants.csv"
    audio_resolution_issues_csv = qc_dir / "audio_resolution_issues.csv"
    write_csv(qc_frame, qc_summary_csv)
    write_csv(skipped_frame, skipped_segments_csv)
    write_csv(unmapped_frame, unmapped_participants_csv)
    write_csv(audio_issues_frame, audio_resolution_issues_csv)

    run_metadata_json = output_dir / "run_metadata.json"
    write_json(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit_or_none(output_dir),
            "config_hash": config_hash,
            "python_version": sys.version,
            "package_versions": _package_versions(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_used": selected_device,
            "participant_workers_used": worker_count if worker_count > 1 else 1,
            "wav2vec_model_id": config.ssl.wav2vec2_model,
            "hubert_model_id": config.ssl.hubert_model,
            "wav2vec_layer": config.ssl.wav2vec2_layer,
            "hubert_layer": config.ssl.hubert_layer,
            "n_participants_processed": int(final_frame["participant_id"].nunique()),
            "n_participant_speaker_rows": int(len(final_frame)),
        },
        run_metadata_json,
    )

    return ExtractOutputs(
        final_csv=final_csv,
        qc_summary_csv=qc_summary_csv,
        skipped_segments_csv=skipped_segments_csv,
        unmapped_participants_csv=unmapped_participants_csv,
        audio_resolution_issues_csv=audio_resolution_issues_csv,
        run_metadata_json=run_metadata_json,
        participant_count=int(final_frame["participant_id"].nunique()),
        row_count=len(final_frame),
    )
