"""Prepare one WAV file per participant from the raw recordings of the first day.

Raw recordings:  <raw_audio>/<participant>/<YYYYMMDD>/*.WAV
Prepared audio:  <prepared_audio>/<participant>/<participant>.wav
Record:          <prepared_audio>/<participant>/<participant>_preparation.json

Only the earliest recording day of a participant is used. The WAV files of that
day are joined in name order, changed to one channel, resampled and normalised.
Other files of the day (for example the .ASR files of the recorder) are not read.
The raw recordings are only read. Prepared audio that exists is never replaced.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from hbn_paths import new_output
from scipy.signal import resample_poly

# Settings of the audio_preparation part of params.yaml and their types.
SETTING_TYPES = {
    "join_multiple_files": bool,
    "combine_gap_sec": float,
    "resample": bool,
    "target_sr": int,
    "convert_to_mono": bool,
    "normalize": bool,
    "target_peak_dbfs": float,
}
# The check reports a difference between raw and prepared duration above this value.
DURATION_TOLERANCE_SEC = 1.0


# ---------------------------------------------------------------------------
# Audio functions (block-wise, so that long recordings fit in memory)
# ---------------------------------------------------------------------------


def concatenate_wavs_streaming(
    wav_paths: List[Path],
    out_path: Path,
    gap_sec: float = 0.0,
    block_frames: int = 262144,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Concatenate WAV files in order (assumes same sr/ch across inputs).
    Returns (sr, ch, manifest_rows with combined offsets).
    """
    assert wav_paths, "No wav paths to concatenate."

    infos = [sf.info(str(p)) for p in wav_paths]
    sr = int(infos[0].samplerate)
    ch = int(infos[0].channels)

    for p, info in zip(wav_paths, infos):
        if int(info.samplerate) != sr:
            raise ValueError(
                f"Sample-rate mismatch: {p} has {info.samplerate}, expected {sr}"
            )
        if int(info.channels) != ch:
            raise ValueError(
                f"Channel mismatch: {p} has {info.channels}, expected {ch}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    gap_frames = int(round(gap_sec * sr))
    gap_buf = np.zeros((gap_frames, ch), dtype=np.float32) if gap_frames > 0 else None

    manifest: List[Dict[str, Any]] = []
    t = 0.0

    with sf.SoundFile(
        str(out_path), "w", samplerate=sr, channels=ch, subtype="PCM_16", format="WAV"
    ) as fout:
        for idx, p in enumerate(wav_paths):
            info = sf.info(str(p))
            dur = float(info.duration)
            s = t
            e = s + dur

            manifest.append(
                {
                    "source_index": idx,
                    "source_path": str(p),
                    "source_recording_id": p.stem,
                    "combined_start_sec": s,
                    "combined_end_sec": e,
                    "source_duration_sec": dur,
                    "sample_rate": sr,
                    "channels": ch,
                }
            )

            with sf.SoundFile(str(p), "r") as fin:
                while True:
                    x = fin.read(frames=block_frames, dtype="float32", always_2d=True)
                    if x.size == 0:
                        break
                    fout.write(np.clip(x, -1.0, 1.0))

            if gap_buf is not None:
                fout.write(gap_buf)
                t = e + gap_sec
            else:
                t = e

    return sr, ch, manifest


def _resample_block(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    if x.ndim == 1:
        y = resample_poly(x, up, down).astype(np.float32, copy=False)
    else:
        ys = [resample_poly(x[:, c], up, down) for c in range(x.shape[1])]
        y = np.stack(ys, axis=1).astype(np.float32, copy=False)
    return y


def ensure_mono_16k_wav_streaming(
    in_path: Path,
    out_path: Path,
    target_sr: int = 16000,
    to_mono: bool = True,
    block_frames: int = 262144,
) -> Dict[str, Any]:
    """Stream-read → (optional) mono → resample → write PCM_16 WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(in_path))
    sr_in = int(info.samplerate)
    ch_in = int(info.channels)

    with sf.SoundFile(str(in_path), "r") as fin, sf.SoundFile(
        str(out_path),
        "w",
        samplerate=target_sr,
        channels=(1 if to_mono else ch_in),
        subtype="PCM_16",
        format="WAV",
    ) as fout:
        while True:
            x = fin.read(frames=block_frames, dtype="float32", always_2d=True)
            if x.size == 0:
                break
            if to_mono and x.shape[1] > 1:
                x = x.mean(axis=1, keepdims=True)
            y = _resample_block(x, sr_in, target_sr)
            y = np.clip(y, -1.0, 1.0)
            fout.write(y)

    out_info = sf.info(str(out_path))
    return {
        "sr_in": sr_in,
        "sr_out": int(out_info.samplerate),
        "channels_in": ch_in,
        "channels_out": int(out_info.channels),
        "duration_sec": float(out_info.duration),
    }


def _dbfs_to_linear(dbfs: float) -> float:
    return float(10 ** (dbfs / 20.0))


def _stream_peak(path: Path, block_frames: int = 262144) -> float:
    peak = 0.0
    with sf.SoundFile(str(path), "r") as f:
        while True:
            x = f.read(frames=block_frames, dtype="float32", always_2d=True)
            if x.size == 0:
                break
            p = float(np.max(np.abs(x))) if x.size else 0.0
            if p > peak:
                peak = p
    return peak


def peak_normalize_wav_streaming(
    in_path: Path,
    out_path: Path,
    target_peak_dbfs: float = -1.0,
    block_frames: int = 262144,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Peak-normalize by streaming. Writes PCM_16 WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    peak = _stream_peak(in_path, block_frames=block_frames)
    target_peak = _dbfs_to_linear(target_peak_dbfs)
    gain = 1.0 if peak <= eps else (target_peak / peak)

    info = sf.info(str(in_path))
    sr = int(info.samplerate)
    ch = int(info.channels)

    with sf.SoundFile(str(in_path), "r") as fin, sf.SoundFile(
        str(out_path), "w", samplerate=sr, channels=ch, subtype="PCM_16", format="WAV"
    ) as fout:
        while True:
            x = fin.read(frames=block_frames, dtype="float32", always_2d=True)
            if x.size == 0:
                break
            y = np.clip(x * gain, -1.0, 1.0)
            fout.write(y)

    return {
        "input_peak": float(peak),
        "gain": float(gain),
        "target_peak": float(target_peak),
    }


def prepare_audio(participant: str, wavs: list[Path], settings: dict, work_dir: Path) -> Path:
    """Join, convert and normalise the WAV files. Returns the prepared file in work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Join the files of the day. A single file is used as it is.
    if len(wavs) > 1:
        if not settings["join_multiple_files"]:
            raise ValueError(
                "join_multiple_files is false, but the first day has more than one WAV file"
            )
        source = work_dir / f"{participant}_combined_raw.wav"
        concatenate_wavs_streaming(wavs, source, gap_sec=settings["combine_gap_sec"])
    else:
        source = wavs[0]

    # 2. One channel and the target sample rate.
    if settings["resample"]:
        target_sr = settings["target_sr"]
    else:
        target_sr = int(sf.info(str(source)).samplerate)
    converted = work_dir / f"{participant}_converted.wav"
    ensure_mono_16k_wav_streaming(
        in_path=source,
        out_path=converted,
        target_sr=target_sr,
        to_mono=settings["convert_to_mono"],
    )

    # 3. Peak normalisation.
    prepared = work_dir / f"{participant}_prepared.wav"
    if settings["normalize"]:
        peak_normalize_wav_streaming(
            in_path=converted,
            out_path=prepared,
            target_peak_dbfs=settings["target_peak_dbfs"],
        )
    else:
        shutil.copyfile(converted, prepared)
    return prepared


# ---------------------------------------------------------------------------
# Raw recordings: first day and file checks
# ---------------------------------------------------------------------------


def is_date(name: str, date_format: str) -> bool:
    """True for a folder name of 8 digits that is a real date, for example 20250131."""
    if len(name) != 8 or not name.isascii() or not name.isdigit():
        return False
    try:
        datetime.strptime(name, date_format)
    except ValueError:
        return False
    return True


def date_folders(participant_dir: Path, date_format: str) -> list[str]:
    """Names of the date folders of one participant, earliest first."""
    names = [
        entry.name
        for entry in Path(participant_dir).iterdir()
        if entry.is_dir() and is_date(entry.name, date_format)
    ]
    return sorted(names, key=lambda name: datetime.strptime(name, date_format))


def select_first_day(
    participant_dir: str | Path, extensions: list[str], date_format: str
) -> tuple[str, list[Path]]:
    """Return the earliest recording date of a participant and its WAV files in name order.

    Later days are never used. If the first day cannot be used, a ValueError
    gives the reason.
    """
    participant_dir = Path(participant_dir)
    if not participant_dir.is_dir():
        raise ValueError(f"raw folder not found: {participant_dir}")
    dates = date_folders(participant_dir, date_format)
    if not dates:
        raise ValueError(
            f"no date folder (8 digits, for example 20250131) in {participant_dir}"
        )
    # A recording outside the date folders (for example in a folder with a wrongly
    # typed date) could belong to an earlier day, so the first day is then not sure.
    allowed = {extension.lower() for extension in extensions}
    outside = [
        path
        for path in participant_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and not (path.parent.parent == participant_dir and path.parent.name in dates)
    ]
    if outside:
        raise ValueError(
            f"{len(outside)} WAV file(s) are not directly inside a date folder, for example "
            f"{outside[0].relative_to(participant_dir)}; correct the folder names first"
        )
    folder = participant_dir / dates[0]
    wavs = [
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in allowed
    ]
    if not wavs:
        raise ValueError(
            f"the first day folder {folder} has no WAV file; later days are not used"
        )
    return dates[0], sorted(wavs, key=lambda path: (path.stem, path.name))


def check_wav(path: Path) -> None:
    """Raise an error if a WAV file is incomplete, empty or cannot be read."""
    with path.open("rb") as stream:
        header = stream.read(12)
    if (
        header[:4] in {b"RIFF", b"RIFX"}
        and header[8:12] == b"WAVE"
        and int.from_bytes(header[4:8], "little" if header[:4] == b"RIFF" else "big") + 8
        > path.stat().st_size
    ):
        raise ValueError("incomplete WAV container")
    info = sf.info(str(path))
    if info.frames <= 0:
        raise ValueError("empty WAV")
    # Read all samples: a valid header does not show that the file is complete.
    with sf.SoundFile(str(path)) as audio:
        frames = sum(len(block) for block in audio.blocks(blocksize=262144))
    if frames != info.frames:
        raise ValueError("incomplete WAV data")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def header_duration(paths: list[Path]) -> float | None:
    """Total duration in seconds from the file headers. None if a header cannot be read."""
    try:
        return float(sum(sf.info(str(path)).duration for path in paths))
    except (RuntimeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Prepared audio and its record
# ---------------------------------------------------------------------------


def preparation_settings(params: dict) -> dict:
    """The audio_preparation settings of params.yaml. Every setting must be present."""
    section = params.get("audio_preparation") or {}
    missing = [key for key in SETTING_TYPES if key not in section]
    if missing:
        raise ValueError(f"params.yaml: audio_preparation has no value for {', '.join(missing)}")
    return {key: kind(section[key]) for key, kind in SETTING_TYPES.items()}


def output_paths(prepared_root: Path, participant: str) -> tuple[Path, Path]:
    """Prepared WAV file and preparation record of one participant."""
    folder = prepared_root / participant
    return folder / f"{participant}.wav", folder / f"{participant}_preparation.json"


def read_record(record_path: Path) -> dict:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"the preparation record cannot be read: {record_path}") from error
    if not isinstance(record, dict):
        raise ValueError(f"the preparation record cannot be read: {record_path}")
    return record


def record_differences(record: dict, date: str, wavs: list[Path], settings: dict) -> list[str]:
    """Compare a preparation record with the first day folder and the settings of today."""
    differences = []
    try:
        if record.get("selected_date") != date:
            differences.append(
                f"date in the record {record.get('selected_date')}, first day now {date}"
            )
        recorded = [(item["name"], item["size_bytes"]) for item in record["source_files"]]
        present = [(path.name, path.stat().st_size) for path in wavs]
        if [name for name, _ in recorded] != [name for name, _ in present]:
            differences.append(
                "source files in the record: "
                + ", ".join(name for name, _ in recorded)
                + "; WAV files in the first day folder now: "
                + ", ".join(name for name, _ in present)
            )
        elif recorded != present:
            changed = [name for (name, size), (_, now) in zip(recorded, present) if size != now]
            differences.append(f"file size changed: {', '.join(changed)}")
        old = record["settings"]
        changed = [
            f"{key} (record {old.get(key)}, now {value})"
            for key, value in settings.items()
            if old.get(key) != value
        ]
        if changed:
            differences.append("settings differ: " + ", ".join(changed))
    except (KeyError, TypeError, AttributeError):
        differences.append("the record does not have the expected content")
    return differences


def prepare_participant(
    participant: str,
    raw_root: Path,
    prepared_root: Path,
    cache_root: Path,
    settings: dict,
    extensions: list[str],
    date_format: str,
) -> tuple[str, str]:
    """Prepare one participant. Returns the result (prepared, reused, kept) and a short text."""
    wav_path, record_path = output_paths(prepared_root, participant)
    part_path = wav_path.with_name(wav_path.name + ".part")

    # Prepared audio that exists is never replaced.
    if part_path.exists():
        raise ValueError(f"unfinished file {part_path} from an earlier run; nothing was changed")
    if record_path.exists() and not wav_path.exists():
        raise ValueError(
            f"{record_path} exists but {wav_path.name} is missing; nothing was changed"
        )
    if wav_path.exists() and not record_path.exists():
        return "kept", (
            "prepared audio exists without a preparation record; "
            "it is kept and was not checked against the raw recordings"
        )

    participant_dir = raw_root / participant
    date, wavs = select_first_day(participant_dir, extensions, date_format)
    print(f"{participant}: first day {date}, {len(wavs)} WAV file(s)")

    if wav_path.exists():
        differences = record_differences(read_record(record_path), date, wavs, settings)
        if differences:
            raise ValueError(
                f"prepared audio exists in {wav_path.parent} but its record does not match: "
                + "; ".join(differences)
                + ". Nothing was changed."
            )
        return "reused", "already prepared"

    for path in wavs:
        try:
            check_wav(path)
        except (RuntimeError, OSError, ValueError) as error:
            raise ValueError(
                f"WAV file of the first day cannot be used: {path} ({error}). "
                "Later days are not used."
            ) from error
    sources = [
        {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in wavs
    ]
    ignored = sorted(entry.name for entry in wavs[0].parent.iterdir() if entry not in wavs)

    # Temporary files are written to the cache folder and removed at the end.
    work_dir = cache_root / "audio-prepare" / participant
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        prepared = prepare_audio(participant, wavs, settings, work_dir)
        info = sf.info(str(prepared))
        new_output(wav_path)
        # The file gets its final name only when it is complete.
        shutil.move(str(prepared), str(part_path))
        part_path.rename(wav_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    record = {
        "participant": participant,
        "selected_date": date,
        "date_folders": date_folders(participant_dir, date_format),
        "source_files": sources,
        "ignored_files": ignored,
        "settings": settings,
        "output": {
            "file": wav_path.name,
            "duration_sec": float(info.duration),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
        },
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    new_output(record_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return "prepared", (
        f"{info.duration:.1f} s, {info.samplerate} Hz, {info.channels} channel(s)"
    )


# ---------------------------------------------------------------------------
# Check (read-only)
# ---------------------------------------------------------------------------


def check_participant(
    participant: str,
    raw_root: Path,
    raw_found: bool,
    prepared_root: Path,
    settings: dict,
    extensions: list[str],
    date_format: str,
) -> tuple[str, bool]:
    """Describe one participant in one line. Only folder lists and file headers are read."""
    wav_path, record_path = output_paths(prepared_root, participant)
    participant_dir = raw_root / participant

    date, wavs, raw_duration = None, [], None
    if not raw_found:
        raw_text = "raw folder not reachable"
    elif not participant_dir.is_dir():
        raw_text = "no raw folder"
    else:
        try:
            date, wavs = select_first_day(participant_dir, extensions, date_format)
            later = len(date_folders(participant_dir, date_format)) - 1
            raw_text = f"first day {date} | {len(wavs)} WAV file(s) | {later} later date(s)"
            raw_duration = header_duration(wavs)
        except ValueError as error:
            raw_text = f"first day cannot be used: {error}"

    has_wav = wav_path.is_file()
    prepared_duration = header_duration([wav_path]) if has_wav else None

    if not record_path.is_file():
        record_text = "none"
    elif not has_wav:
        record_text = "without WAV"
    elif date is None:
        record_text = "present, not compared"
    else:
        try:
            differences = record_differences(read_record(record_path), date, wavs, settings)
            record_text = "differs: " + "; ".join(differences) if differences else "matches"
        except ValueError:
            record_text = "not readable"

    flagged = (
        raw_duration is not None
        and prepared_duration is not None
        and abs(raw_duration - prepared_duration) > DURATION_TOLERANCE_SEC
    )
    line = (
        f"{participant}: {raw_text} | prepared WAV {'yes' if has_wav else 'no'} | "
        f"record {record_text} | "
        f"first day {_seconds(raw_duration)}, prepared {_seconds(prepared_duration)}"
    )
    if flagged:
        line += " | DURATION DIFFERS"
    return line, flagged


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} s"


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def _sub_folders(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def _choose(names: list[str], participant: str | None, limit: int | None) -> list[str]:
    """One participant, or the first N in name order."""
    if participant is not None:
        names = [name for name in names if name == participant]
    if limit is not None:
        names = names[:limit]
    return names


def run(config, params: dict, participant: str | None = None, limit: int | None = None,
        check: bool = False) -> int:
    """Prepare the audio of all participants (or check it). Returns the exit code."""
    raw_root = config.path("raw_audio")
    prepared_root = config.path("prepared_audio")
    settings = preparation_settings(params)
    extensions = list(params["raw_audio"]["extensions"])
    date_format = str(params["raw_audio"]["date_format"])
    raw_found = raw_root.is_dir()

    if check:
        if not raw_found:
            print(f"Raw audio folder is not reachable: {raw_root}. "
                  "Only the prepared files are reported.")
        found = set(_sub_folders(raw_root)) | set(_sub_folders(prepared_root))
        names = _choose(sorted(found), participant, limit)
        n_flagged = 0
        for name in names:
            line, flagged = check_participant(
                name, raw_root, raw_found, prepared_root, settings, extensions, date_format
            )
            n_flagged += flagged
            print(line)
        print(f"Checked {len(names)} participant(s); {n_flagged} with a duration difference "
              f"above {DURATION_TOLERANCE_SEC:g} s. Nothing was written.")
        return 0

    if not raw_found:
        print(f"Raw audio folder not found: {raw_root}. "
              f"Check data_dir and paths.raw_audio in {config.file}.")
        return 1
    names = _choose(_sub_folders(raw_root), participant, limit)
    if not names:
        if participant is not None:
            print(f"Participant {participant} was not found in {raw_root}.")
        else:
            print(f"No participant folder was found in {raw_root}.")
        return 1

    counts = {"prepared": 0, "reused": 0, "kept": 0, "error": 0}
    for name in names:
        try:
            result, text = prepare_participant(
                name, raw_root, prepared_root, config.path("cache"),
                settings, extensions, date_format,
            )
        except Exception as error:  # one participant with an error does not stop the others
            result, text = "error", str(error)
        counts[result] += 1
        print(f"{name}: {result} - {text}")

    print(f"Prepared audio folder: {prepared_root}")
    print("Summary: " + ", ".join(f"{number} {result}" for result, number in counts.items()))
    return 1 if counts["error"] else 0
