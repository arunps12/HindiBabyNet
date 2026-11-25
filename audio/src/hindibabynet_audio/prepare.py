"""Prepare the raw recordings: join, one channel, resample, normalise.

The raw recordings can be organised in different ways (raw_audio.structure):

    <raw_audio>/<participant>/<YYYYMMDD>/*.WAV   participant folders with recording days
    <raw_audio>/<participant>/*.WAV              participant folders without dates
    <raw_audio>/*.WAV                            single files, no participant folder
    <raw_audio> is one WAV file                  a single file

With joining (audio_preparation.join_multiple_files) the files of one participant
give one prepared file <participant>.wav. The first recording day or all days are
used (audio_preparation.recording_days). Files are joined in the order of the date
folders and then of the file names. Without joining, and for single files, every
WAV file is prepared as its own file and keeps its name and its folder.

Prepared audio:  <prepared_audio>/<participant>.wav or <prepared_audio>/<same path>
Record:          <name of the prepared file>_preparation.json, next to it

Recordings of different participants are never joined. Other files (for example
the .ASR files of the recorder) are not read. The raw recordings are only read.
Prepared audio that exists is never replaced.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from hbn_paths import new_output
from scipy.signal import resample_poly

from hindibabynet_audio.recordings import prepared_recordings, recording_id, select

# Settings of the audio_preparation part of params.yaml and their types.
SETTING_TYPES = {
    "join_multiple_files": bool,
    "recording_days": str,
    "combine_gap_sec": float,
    "resample": bool,
    "target_sr": int,
    "convert_to_mono": bool,
    "normalize": bool,
    "target_peak_dbfs": float,
}
STRUCTURES = ("auto", "participants", "files")
RECORDING_DAYS = ("first", "all")
# How the raw files of a prepared recording were found.
DAYS, FOLDER, SINGLE = "recording days", "participant folder", "single file"
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
# Settings
# ---------------------------------------------------------------------------


def preparation_settings(params: dict) -> dict:
    """The audio_preparation settings of params.yaml. Every setting must be present."""
    section = params.get("audio_preparation") or {}
    missing = [key for key in SETTING_TYPES if key not in section]
    if missing:
        raise ValueError(f"params.yaml: audio_preparation has no value for {', '.join(missing)}")
    settings = {key: kind(section[key]) for key, kind in SETTING_TYPES.items()}
    if settings["recording_days"] not in RECORDING_DAYS:
        raise ValueError(
            "params.yaml: audio_preparation.recording_days must be first or all, "
            f"not {settings['recording_days']}"
        )
    return settings


def raw_structure(params: dict) -> str:
    """The raw_audio.structure setting of params.yaml (auto if it is not written)."""
    structure = str((params.get("raw_audio") or {}).get("structure", "auto"))
    if structure not in STRUCTURES:
        raise ValueError(
            "params.yaml: raw_audio.structure must be auto, participants or files, "
            f"not {structure}"
        )
    return structure


# ---------------------------------------------------------------------------
# Raw recordings: which raw files give which prepared file
# ---------------------------------------------------------------------------


@dataclass
class Recording:
    """One planned prepared file and the raw WAV files it is made from."""

    recording_id: str  # for example P001, REC_0001 or P001/20250323/REC_0001
    output: Path | None = None  # prepared WAV file, relative to the prepared audio folder
    sources: list[Path] = field(default_factory=list)  # raw WAV files in join order
    base: Path | None = None  # raw audio folder; the source names are written from here
    participant: str | None = None  # name of the participant folder; None for a single file
    structure: str = ""  # DAYS, FOLDER or SINGLE
    joined: bool = False  # True: one file per participant, named <participant>.wav
    dates: list[str] = field(default_factory=list)  # date folders that are used
    all_dates: list[str] = field(default_factory=list)  # all date folders of the participant
    error: str | None = None  # the reason if this entry cannot be prepared


def _is_wav(path: Path, allowed: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in allowed


def _name_order(path: Path) -> tuple[str, str]:
    """Order of the WAV files of one folder: by file name, as in select_first_day."""
    return (path.stem, path.name)


def participant_files(
    folder: Path, extensions: list[str], date_format: str, structure: str, recording_days: str
) -> tuple[str, list[str], list[str], list[Path]]:
    """The WAV files of one participant folder in join order.

    Returns how the files are organised (DAYS or FOLDER), the date folders that are
    used, all date folders, and the files. A ValueError gives the reason if the
    files cannot be selected in a sure way.
    """
    allowed = {extension.lower() for extension in extensions}
    dates = date_folders(folder, date_format)
    wavs = [path for path in folder.rglob("*") if _is_wav(path, allowed)]
    outside = [
        path for path in wavs
        if not (path.parent.parent == folder and path.parent.name in dates)
    ]

    # All WAV files are directly inside date folders: recording days.
    if dates and not outside:
        if recording_days == "first":
            date, files = select_first_day(folder, extensions, date_format)
            return DAYS, [date], dates, files
        used, files = [], []
        for date in dates:
            day_files = sorted(
                (path for path in wavs if path.parent.name == date), key=_name_order
            )
            if day_files:
                used.append(date)
                files.extend(day_files)
        if not files:
            raise ValueError(f"the date folders of {folder} have no WAV file")
        return DAYS, used, dates, files

    if structure == "auto":
        if dates:
            raise ValueError(
                f"{len(outside)} WAV file(s) are not directly inside a date folder, for example "
                f"{outside[0].relative_to(folder)}; correct the folder names, or set "
                "raw_audio.structure in audio/params.yaml"
            )
        deeper = [path for path in wavs if path.parent != folder]
        if deeper:
            raise ValueError(
                "WAV files are in sub-folders that are not recording dates (8 digits, for "
                f"example 20250131), for example {deeper[0].relative_to(folder)}. It is not "
                "clear how the recordings are organised: set raw_audio.structure to "
                "participants or files in audio/params.yaml"
            )

    # A participant folder without recording days.
    if recording_days == "first":
        raise ValueError(
            f"no date folder (8 digits, for example 20250131) in {folder}, so the first "
            "recording day cannot be selected. To use all WAV files of this folder, set "
            "audio_preparation.recording_days to all in audio/params.yaml"
        )
    files = sorted(
        wavs, key=lambda path: (path.relative_to(folder).parts[:-1], *_name_order(path))
    )
    return FOLDER, [], dates, files


def plan_recordings(
    raw_root: Path, settings: dict, extensions: list[str], date_format: str, structure: str
) -> tuple[list[Recording], list[str]]:
    """Look at the raw audio folder and return the planned prepared files in name order.

    An entry that cannot be used is returned with its reason (error). The second
    result has notes about folders that are not used. Nothing is written.
    """
    allowed = {extension.lower() for extension in extensions}

    def single(path: Path, base: Path) -> Recording:
        relative = path.relative_to(base)
        return Recording(recording_id(relative), relative, [path], base, structure=SINGLE)

    if raw_root.is_file():
        if not _is_wav(raw_root, allowed):
            raise ValueError(f"raw_audio is a file, but not a WAV file: {raw_root}")
        return [single(raw_root, raw_root.parent)], []

    if structure == "files":
        files = sorted(
            (path for path in raw_root.rglob("*") if _is_wav(path, allowed)),
            key=lambda path: path.relative_to(raw_root).parts,
        )
        return [single(path, raw_root) for path in files], []

    items: list[Recording] = []
    notes: list[str] = []
    for entry in sorted(raw_root.iterdir(), key=lambda path: path.name):
        name = entry.name
        if _is_wav(entry, allowed):
            if structure == "participants":
                items.append(Recording(name, error=(
                    "the file is not inside a participant folder "
                    "(raw_audio.structure is participants)"
                )))
            else:
                items.append(single(entry, raw_root))
            continue
        if not entry.is_dir():
            continue
        has_wav = any(_is_wav(path, allowed) for path in entry.rglob("*"))
        if not has_wav and not date_folders(entry, date_format):
            notes.append(f"{name}: no WAV file in this folder, it is not used")
            continue
        try:
            if structure == "auto" and is_date(name, date_format):
                raise ValueError(
                    "the folder name is a date, so the participant is not known. Set "
                    "raw_audio.structure to participants or files in audio/params.yaml"
                )
            found, used, all_dates, files = participant_files(
                entry, extensions, date_format, structure, settings["recording_days"]
            )
        except ValueError as error:
            items.append(Recording(name, participant=name, error=str(error)))
            continue
        if settings["join_multiple_files"]:
            items.append(Recording(
                name, Path(f"{name}.wav"), files, raw_root, name, found, True, used, all_dates
            ))
        else:
            for path in files:
                relative = path.relative_to(raw_root)
                day = [path.parent.name] if found == DAYS else []
                items.append(Recording(
                    recording_id(relative), relative, [path], raw_root, name, found,
                    False, day, all_dates,
                ))
    return items, notes


def mark_conflicts(items: list[Recording]) -> None:
    """Two recordings must not give the same prepared file or the same recording ID.

    Names that differ only in capital letters count as the same, because they are
    the same file on Windows. All recordings of such a group get an error.
    """
    groups: dict[tuple[str, str], list[Recording]] = defaultdict(list)
    for item in items:
        if item.error is None:
            groups[("file", item.output.as_posix().lower())].append(item)
            groups[("name", item.recording_id.lower())].append(item)
    for group in groups.values():
        if len(group) > 1:
            origins = ", ".join(
                f"folder {item.participant}" if item.joined
                else item.sources[0].relative_to(item.base).as_posix()
                for item in group
            )
            for item in group:
                item.error = (
                    f"more than one recording would get the prepared file "
                    f"{group[0].output.as_posix()}: {origins}. Rename one of them. "
                    "Nothing was prepared for them"
                )


def describe(recording: Recording, recording_days: str) -> str:
    """Short text: how the raw files of this recording were found."""
    if recording.structure == SINGLE:
        return "single file"
    count = f"{len(recording.sources)} WAV file(s)"
    if recording.structure == DAYS and recording_days == "first":
        text = (
            f"first day {recording.dates[0]} | {count} | "
            f"{len(recording.all_dates) - 1} later date(s)"
        )
    elif recording.structure == DAYS:
        text = f"all days, here {', '.join(recording.dates)} | {count}"
    else:
        text = f"participant folder without dates | {count}"
    return text if recording.joined else text + " | not joined"


def order_note(recording: Recording) -> str | None:
    """A note if the join order by file name differs from the order by number."""

    def by_number(name: str) -> list:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]

    for folder in dict.fromkeys(path.parent for path in recording.sources):
        names = [path.name for path in recording.sources if path.parent == folder]
        if sorted(names, key=by_number) != names:
            shown = ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")
            return (
                f"the files are joined in the order of their names ({shown}). The order by "
                "number is different. Rename the files if this order is not right"
            )
    return None


# ---------------------------------------------------------------------------
# Prepared audio and its record
# ---------------------------------------------------------------------------


def record_file(wav_path: Path) -> Path:
    """Preparation record of a prepared WAV file: next to it, <name>_preparation.json."""
    return wav_path.with_name(f"{wav_path.stem}_preparation.json")


def source_names(recording: Recording) -> list[str]:
    """Paths of the raw files from the raw audio folder, as they are written in the record."""
    return [path.relative_to(recording.base).as_posix() for path in recording.sources]


def read_record(record_path: Path) -> dict:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"the preparation record cannot be read: {record_path}") from error
    if not isinstance(record, dict):
        raise ValueError(f"the preparation record cannot be read: {record_path}")
    return record


def record_differences(record: dict, recording: Recording, settings: dict) -> list[str]:
    """Compare a preparation record with the raw files and the settings of today."""
    differences = []
    try:
        if record.get("selected_dates") != recording.dates:
            differences.append(
                f"date folders in the record {record.get('selected_dates')}, "
                f"now {recording.dates}"
            )
        recorded = [(item["path"], item["size_bytes"]) for item in record["source_files"]]
        present = [
            (name, path.stat().st_size)
            for name, path in zip(source_names(recording), recording.sources)
        ]
        if [name for name, _ in recorded] != [name for name, _ in present]:
            differences.append(
                "source files in the record: "
                + ", ".join(name for name, _ in recorded)
                + "; WAV files that are selected now: "
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


def prepare_recording(
    recording: Recording,
    prepared_root: Path,
    existing: dict[str, Path],
    cache_root: Path,
    settings: dict,
) -> tuple[str, str]:
    """Prepare one recording. Returns the result (prepared, reused, kept) and a short text.

    existing: prepared files that are already there (recording ID in small letters -> file).
    A prepared file with the same ID is used where it is, also in another sub-folder.
    """
    name = recording.recording_id
    wav_path = existing.get(name.lower(), prepared_root / recording.output)
    record_path = record_file(wav_path)
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

    print(f"{name}: {describe(recording, settings['recording_days'])}")

    if wav_path.exists():
        differences = record_differences(read_record(record_path), recording, settings)
        if differences:
            raise ValueError(
                f"prepared audio exists in {wav_path.parent} but its record does not match: "
                + "; ".join(differences)
                + ". Nothing was changed."
            )
        return "reused", "already prepared"

    note = order_note(recording)
    if note:
        print(f"{name}: note - {note}")
    first_day = recording.structure == DAYS and settings["recording_days"] == "first"
    for path in recording.sources:
        try:
            check_wav(path)
        except (RuntimeError, OSError, ValueError) as error:
            raise ValueError(
                f"WAV file{' of the first day' if first_day else ''} cannot be used: "
                f"{path} ({error})." + (" Later days are not used." if first_day else "")
            ) from error
    sources = [
        {"path": source, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for source, path in zip(source_names(recording), recording.sources)
    ]
    ignored = []
    if recording.joined:
        # Other files in the folders of the joined WAV files, for example .ASR files.
        ignored = sorted(
            entry.relative_to(recording.base).as_posix()
            for folder in dict.fromkeys(path.parent for path in recording.sources)
            for entry in folder.iterdir()
            if entry.is_file() and entry not in recording.sources
        )

    # Temporary files are written to the cache folder and removed at the end.
    work_dir = cache_root / "audio-prepare" / name.replace("/", "__")
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        prepared = prepare_audio(work_dir.name, recording.sources, settings, work_dir)
        info = sf.info(str(prepared))
        new_output(wav_path)
        # The file gets its final name only when it is complete.
        shutil.move(str(prepared), str(part_path))
        part_path.rename(wav_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    record = {
        "recording_id": name,
        "participant": recording.participant,
        "structure": recording.structure,
        "joined": recording.joined,
        "selected_dates": recording.dates,
        "date_folders": recording.all_dates,
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


def check_recording(
    recording: Recording, prepared_root: Path, existing: dict[str, Path], settings: dict
) -> tuple[str, bool]:
    """Describe one planned recording in one line. Only file headers are read."""
    name = recording.recording_id
    wav_path = existing.get(name.lower(), prepared_root / recording.output)
    record_path = record_file(wav_path)
    raw_duration = header_duration(recording.sources)
    has_wav = wav_path.is_file()
    prepared_duration = header_duration([wav_path]) if has_wav else None

    if not record_path.is_file():
        record_text = "none"
    elif not has_wav:
        record_text = "without WAV"
    else:
        try:
            differences = record_differences(read_record(record_path), recording, settings)
            record_text = "differs: " + "; ".join(differences) if differences else "matches"
        except ValueError:
            record_text = "not readable"

    flagged = (
        raw_duration is not None
        and prepared_duration is not None
        and abs(raw_duration - prepared_duration) > DURATION_TOLERANCE_SEC
    )
    line = (
        f"{name}: {describe(recording, settings['recording_days'])} | "
        f"prepared WAV {'yes' if has_wav else 'no'} | record {record_text} | "
        f"raw {_seconds(raw_duration)}, prepared {_seconds(prepared_duration)}"
    )
    if flagged:
        line += " | DURATION DIFFERS"
    return line, flagged


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} s"


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def run(config, params: dict, participant: str | None = None, limit: int | None = None,
        check: bool = False) -> int:
    """Prepare the raw recordings (or check them). Returns the exit code."""
    raw_root = config.path("raw_audio")
    prepared_root = config.path("prepared_audio")
    settings = preparation_settings(params)
    structure = raw_structure(params)
    extensions = list(params["raw_audio"]["extensions"])
    date_format = str(params["raw_audio"]["date_format"])
    raw_found = raw_root.exists()

    prepared = prepared_recordings(prepared_root)
    existing = {name.lower(): path for name, path in prepared}
    items: list[Recording] = []
    notes: list[str] = []
    if raw_found:
        items, notes = plan_recordings(raw_root, settings, extensions, date_format, structure)
        mark_conflicts(items)
    chosen = select(items, participant, limit, key=lambda item: item.recording_id)

    if check:
        if not raw_found:
            print(f"Raw audio folder is not reachable: {raw_root}. "
                  "Only the prepared files are reported.")
        for note in notes:
            print(note)
        n_flagged = 0
        for item in chosen:
            if item.error:
                state = "yes" if item.recording_id.lower() in existing else "no"
                print(f"{item.recording_id}: cannot be prepared - {item.error} | "
                      f"prepared WAV {state}")
                continue
            note = order_note(item) if item.joined else None
            if note:
                print(f"{item.recording_id}: note - {note}")
            line, flagged = check_recording(item, prepared_root, existing, settings)
            n_flagged += flagged
            print(line)
        # Prepared files that have no raw recording.
        planned = {item.recording_id.lower() for item in items}
        others = [] if limit is not None else select(
            [entry for entry in prepared if entry[0].lower() not in planned], participant, None
        )
        reason = "no raw recording" if raw_found else "raw folder not reachable"
        for name, path in others:
            record_text = "present, not compared" if record_file(path).is_file() else "none"
            print(f"{name}: {reason} | prepared WAV yes | record {record_text} | "
                  f"raw -, prepared {_seconds(header_duration([path]))}")
        print(f"Checked {len(chosen) + len(others)} recording(s); {n_flagged} with a duration "
              f"difference above {DURATION_TOLERANCE_SEC:g} s. Nothing was written.")
        return 0

    if not raw_found:
        print(f"Raw audio folder not found: {raw_root}. "
              f"Check data_dir and paths.raw_audio in {config.file}.")
        return 1
    for note in notes:
        print(note)
    if not chosen:
        if participant is not None:
            print(f"{participant} was not found in {raw_root}.")
        else:
            print(f"No WAV file was found in {raw_root}.")
        return 1

    counts = {"prepared": 0, "reused": 0, "kept": 0, "error": 0}
    for item in chosen:
        try:
            if item.error:
                raise ValueError(item.error)
            result, text = prepare_recording(
                item, prepared_root, existing, config.path("cache"), settings
            )
        except Exception as error:  # one recording with an error does not stop the others
            result, text = "error", str(error)
        counts[result] += 1
        print(f"{item.recording_id}: {result} - {text}")

    print(f"Prepared audio folder: {prepared_root}")
    print("Summary: " + ", ".join(f"{number} {result}" for result, number in counts.items()))
    return 1 if counts["error"] else 0
