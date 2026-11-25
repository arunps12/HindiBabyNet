"""Tests of the audio preparation with small synthetic recordings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from hindibabynet_audio import prepare

TARGET_PEAK = 10 ** (-1.0 / 20)  # -1 dBFS


def day(config, participant: str, date: str) -> Path:
    """Create and return the raw folder of one recording day."""
    folder = config.path("raw_audio", participant, date)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def outputs(config, participant: str) -> tuple[Path, Path]:
    """Prepared WAV file and preparation record of one participant (joined audio)."""
    folder = config.path("prepared_audio")
    return folder / f"{participant}.wav", folder / f"{participant}_preparation.json"


def prepared_files(config) -> list[str]:
    """All files in the prepared audio folder, with their sub-folders."""
    root = config.path("prepared_audio")
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def record_of(wav: Path) -> dict:
    return json.loads(prepare.record_file(wav).read_text(encoding="utf-8"))


def listing(root: Path) -> list:
    """All files and folders under root, files with their modification time."""
    return sorted(
        (str(path.relative_to(root)), path.stat().st_mtime_ns if path.is_file() else None)
        for path in root.rglob("*")
    )


def line_of(output: str, participant: str) -> str:
    """The last console line of one participant."""
    return [line for line in output.splitlines() if line.startswith(f"{participant}:")][-1]


# --- selection of the first day ---------------------------------------------


def test_select_first_day_gives_earliest_date_and_wav_files_in_name_order(
    tmp_path, write_wav, write_asr
):
    participant = tmp_path / "P1"
    for name in ("b.WAV", "a.wav", "c.WaV"):
        write_wav(participant / "20250201" / name, seconds=0.1)
    write_asr(participant / "20250201" / "a.ASR")
    write_wav(participant / "20250301" / "later.WAV", seconds=0.1)
    # Folder names that are not a date of 8 digits are not recording days.
    for name in ("20250132", "2025011", "session_a"):
        (participant / name).mkdir()
        (participant / name / "notes.txt").write_text("no recording here", encoding="utf-8")

    date, wavs = prepare.select_first_day(participant, [".wav", ".WAV"], "%Y%m%d")

    assert date == "20250201"
    assert [path.name for path in wavs] == ["a.wav", "b.WAV", "c.WaV"]
    assert prepare.date_folders(participant, "%Y%m%d") == ["20250201", "20250301"]


def test_only_the_earliest_date_is_used_and_later_dates_are_never_read(
    make_config, params, write_wav, monkeypatch
):
    config = make_config()
    first = day(config, "P1", "20250201")
    write_wav(first / "a.WAV", seconds=0.5)
    write_wav(first / "b.WAV", seconds=0.5)
    later = day(config, "P1", "20250305")
    (later / "c.WAV").write_bytes(b"this file cannot be read as audio")

    opened = []
    real_info = sf.info

    def spy(path, *args, **kwargs):
        opened.append(Path(path))
        return real_info(path, *args, **kwargs)

    monkeypatch.setattr(prepare.sf, "info", spy)

    assert prepare.run(config, params) == 0

    raw_root = config.path("raw_audio")
    assert {path.parent for path in opened if raw_root in path.parents} == {first}
    wav, record_path = outputs(config, "P1")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["selected_dates"] == ["20250201"]
    assert record["date_folders"] == ["20250201", "20250305"]
    assert [item["path"] for item in record["source_files"]] == [
        "P1/20250201/a.WAV", "P1/20250201/b.WAV"
    ]
    assert sf.info(str(wav)).duration == pytest.approx(1.0, abs=0.01)


def test_sub_folders_that_are_not_dates_are_an_error(make_config, params, write_wav, capsys):
    config = make_config()
    write_wav(config.path("raw_audio", "P1", "session_a", "x.WAV"))

    with pytest.raises(ValueError, match="no date folder"):
        prepare.select_first_day(config.path("raw_audio", "P1"), [".wav"], "%Y%m%d")
    assert prepare.run(config, params) == 1

    line = line_of(capsys.readouterr().out, "P1")
    assert "error" in line and "not clear" in line and "raw_audio.structure" in line
    assert not config.path("prepared_audio").exists()


def test_wav_outside_the_date_folders_is_an_error(make_config, params, write_wav, capsys):
    config = make_config()
    # The first folder has a wrongly typed date (7 digits), so the first day is not sure.
    write_wav(config.path("raw_audio", "P1", "2025021", "a.WAV"))
    write_wav(config.path("raw_audio", "P1", "20250215", "b.WAV"))
    write_wav(config.path("raw_audio", "P2", "20250215", "b.WAV"))
    write_wav(config.path("raw_audio", "P2", "loose.wav"))

    assert prepare.run(config, params) == 1

    output = capsys.readouterr().out
    for participant in ("P1", "P2"):
        line = line_of(output, participant)
        assert "error" in line and "not directly inside a date folder" in line
    assert not config.path("prepared_audio").exists()


def test_empty_first_day_is_an_error_and_the_later_day_is_not_used(
    make_config, params, write_wav, write_asr, capsys
):
    config = make_config()
    first = day(config, "P1", "20250201")
    write_asr(first / "rec.ASR")
    write_wav(day(config, "P1", "20250202") / "later.WAV")

    with pytest.raises(ValueError, match="no WAV file"):
        prepare.select_first_day(config.path("raw_audio", "P1"), [".wav"], "%Y%m%d")
    assert prepare.run(config, params) == 1

    line = line_of(capsys.readouterr().out, "P1")
    assert "error" in line and str(first) in line
    assert not config.path("prepared_audio").exists()


@pytest.mark.parametrize("state", ["no_audio", "cut_off", "no_samples", "good_and_bad"])
def test_wav_file_of_the_first_day_that_cannot_be_read_is_an_error(
    make_config, params, write_wav, capsys, state
):
    config = make_config()
    first = day(config, "P1", "20250201")
    write_wav(day(config, "P1", "20250202") / "later.WAV")
    if state in ("no_audio", "good_and_bad"):
        (first / "broken.WAV").write_bytes(b"not a WAV file")
        if state == "good_and_bad":
            write_wav(first / "good.WAV")
    elif state == "cut_off":
        path = write_wav(first / "cut.WAV")
        path.write_bytes(path.read_bytes()[:-100])
    else:
        sf.write(str(first / "empty.WAV"), np.empty(0, dtype=np.float32), 16000,
                 format="WAV", subtype="PCM_16")

    assert prepare.run(config, params) == 1

    line = line_of(capsys.readouterr().out, "P1")
    assert "error" in line and "cannot be used" in line and str(first) in line
    assert not config.path("prepared_audio").exists()
    assert not config.path("cache", "audio-prepare", "P1").exists()


def test_one_participant_with_an_error_does_not_stop_the_others(
    make_config, params, write_wav, write_asr, capsys
):
    config = make_config()
    write_asr(day(config, "P1", "20250201") / "rec.ASR")
    write_wav(day(config, "P2", "20250201") / "rec.WAV")

    assert prepare.run(config, params) == 1

    output = capsys.readouterr().out
    assert "error" in line_of(output, "P1")
    assert "prepared" in line_of(output, "P2")
    assert "Summary: 1 prepared, 0 reused, 0 kept, 1 error" in output
    assert not outputs(config, "P1")[0].exists()
    assert outputs(config, "P2")[0].is_file()


# --- prepared audio -----------------------------------------------------------


def test_output_is_mono_16_khz_pcm16_with_the_target_peak(make_config, params, write_wav):
    config = make_config()
    first = day(config, "P1", "20250201")  # two stereo files of 44.1 kHz
    write_wav(first / "rec_1.WAV", seconds=0.5)
    write_wav(first / "rec_2.WAV", seconds=0.5)
    write_wav(day(config, "P2", "20250201") / "rec.WAV",  # one mono file of 16 kHz
              seconds=0.5, sr=16000, channels=1, amplitude=0.25)

    assert prepare.run(config, params) == 0

    for participant, seconds in (("P1", 1.0), ("P2", 0.5)):
        wav, _ = outputs(config, participant)
        info = sf.info(str(wav))
        assert (info.samplerate, info.channels) == (16000, 1)
        assert (info.format, info.subtype) == ("WAV", "PCM_16")
        assert info.duration == pytest.approx(seconds, abs=0.01)
        samples, _ = sf.read(str(wav), dtype="float32")
        assert float(np.max(np.abs(samples))) == pytest.approx(TARGET_PEAK, abs=1e-4)
        # Temporary files are removed.
        assert not config.path("cache", "audio-prepare", participant).exists()


def test_stereo_resampling_and_normalisation_give_the_expected_samples(
    make_config, params, tmp_path
):
    config = make_config()
    source = day(config, "P1", "20250201") / "stereo.WAV"
    x = np.linspace(-0.2, 0.3, 800, dtype=np.float32)
    sf.write(str(source), np.stack([x, x * 0.5], axis=1), 8000, format="WAV", subtype="PCM_16")

    assert prepare.run(config, params) == 0

    # The same steps written out: one channel, resampling, 16 bit, normalisation, 16 bit.
    samples, _ = sf.read(str(source), dtype="float32", always_2d=True)
    converted = resample_poly(samples.mean(axis=1, keepdims=True), 2, 1).astype(np.float32)
    reference = tmp_path / "reference.wav"
    sf.write(str(reference), np.clip(converted, -1, 1), 16000, subtype="PCM_16")
    quantized, _ = sf.read(str(reference), dtype="float32", always_2d=True)
    gain = (10 ** (-1 / 20)) / float(np.max(np.abs(quantized)))
    sf.write(str(reference), np.clip(quantized * gain, -1, 1), 16000, subtype="PCM_16")

    actual, sr = sf.read(str(outputs(config, "P1")[0]))
    expected, _ = sf.read(str(reference))
    np.testing.assert_array_equal(actual, expected)
    assert sr == 16000 and len(actual) == 1600


def test_files_are_joined_in_name_order_with_the_gap(make_config, params):
    config = make_config()
    first = day(config, "P1", "20250201")
    sf.write(str(first / "b.WAV"), np.full(800, 0.25), 8000, format="WAV", subtype="PCM_16")
    sf.write(str(first / "a.wav"), np.full(800, -0.5), 8000, format="WAV", subtype="PCM_16")
    params["audio_preparation"].update(
        resample=False, convert_to_mono=False, normalize=False, combine_gap_sec=0.025
    )

    assert prepare.run(config, params) == 0

    samples, sr = sf.read(str(outputs(config, "P1")[0]))
    expected = np.concatenate(
        [np.full(800, -0.5), np.zeros(200), np.full(800, 0.25), np.zeros(200)]
    )
    np.testing.assert_array_equal(samples, expected)
    assert sr == 8000


def test_without_joining_every_file_of_the_first_day_keeps_its_name_and_folder(
    make_config, params, write_wav, capsys
):
    config = make_config()
    first = day(config, "P1", "20250201")
    write_wav(first / "REC_0001.WAV", seconds=0.2)
    write_wav(first / "REC_0002.WAV", seconds=0.4)
    write_wav(day(config, "P1", "20250305") / "REC_0003.WAV")
    # Another participant has a file with the same name.
    write_wav(day(config, "P2", "20250201") / "REC_0001.WAV", seconds=0.3)
    params["audio_preparation"]["join_multiple_files"] = False

    assert prepare.run(config, params) == 0

    # No joined file, the later day is not used, the same name in two folders is kept apart.
    assert prepared_files(config) == [
        "P1/20250201/REC_0001.WAV", "P1/20250201/REC_0001_preparation.json",
        "P1/20250201/REC_0002.WAV", "P1/20250201/REC_0002_preparation.json",
        "P2/20250201/REC_0001.WAV", "P2/20250201/REC_0001_preparation.json",
    ]
    root = config.path("prepared_audio")
    durations = {
        name: sf.info(str(root / name)).duration
        for name in prepared_files(config) if name.endswith(".WAV")
    }
    assert durations == pytest.approx({
        "P1/20250201/REC_0001.WAV": 0.2, "P1/20250201/REC_0002.WAV": 0.4,
        "P2/20250201/REC_0001.WAV": 0.3,
    }, abs=0.01)
    record = record_of(root / "P1" / "20250201" / "REC_0002.WAV")
    assert record["recording_id"] == "P1/20250201/REC_0002"
    assert record["participant"] == "P1" and record["joined"] is False
    assert record["selected_dates"] == ["20250201"]
    assert [item["path"] for item in record["source_files"]] == ["P1/20250201/REC_0002.WAV"]
    output = capsys.readouterr().out
    assert "not joined" in output
    assert "Summary: 3 prepared, 0 reused, 0 kept, 0 error" in output

    # One participant can be selected by its name.
    assert prepare.run(config, params, participant="P2") == 0
    assert "Summary: 0 prepared, 1 reused, 0 kept, 0 error" in capsys.readouterr().out


def test_preparation_record(make_config, params, write_wav, write_asr):
    config = make_config()
    first = day(config, "P1", "20250201")
    write_wav(first / "b.WAV", seconds=0.5)
    write_wav(first / "a.WAV", seconds=0.25)
    write_asr(first / "a.ASR")
    write_asr(first / "b.ASR")
    write_wav(day(config, "P1", "20250307") / "c.WAV")
    raw_root = config.path("raw_audio")
    raw_before = {path: path.read_bytes() for path in raw_root.rglob("*") if path.is_file()}

    assert prepare.run(config, params) == 0

    wav, record_path = outputs(config, "P1")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["recording_id"] == "P1"
    assert record["participant"] == "P1"
    assert record["structure"] == "recording days" and record["joined"] is True
    assert record["selected_dates"] == ["20250201"]
    assert record["date_folders"] == ["20250201", "20250307"]
    assert [item["path"] for item in record["source_files"]] == [
        "P1/20250201/a.WAV", "P1/20250201/b.WAV"
    ]
    for item in record["source_files"]:
        content = (raw_root / item["path"]).read_bytes()
        assert item["size_bytes"] == len(content)
        assert item["sha256"] == hashlib.sha256(content).hexdigest()
    assert record["ignored_files"] == ["P1/20250201/a.ASR", "P1/20250201/b.ASR"]
    assert record["settings"] == params["audio_preparation"]
    assert record["settings"]["recording_days"] == "first"
    assert record["output"]["file"] == "P1.wav"
    assert record["output"]["duration_sec"] == pytest.approx(0.75, abs=0.01)
    assert record["output"]["sample_rate"] == 16000
    assert record["output"]["channels"] == 1
    assert record["prepared_at"]

    # The prepared audio folder holds the WAV file and the record only.
    assert sorted(path.name for path in wav.parent.iterdir()) == [
        "P1.wav", "P1_preparation.json"
    ]
    # The raw recordings are only read.
    raw_after = {path: path.read_bytes() for path in raw_root.rglob("*") if path.is_file()}
    assert raw_after == raw_before


# --- prepared audio that exists is never replaced ----------------------------


def test_second_run_reuses_the_prepared_audio(make_config, params, write_wav, capsys, monkeypatch):
    config = make_config()
    write_wav(day(config, "P1", "20250201") / "a.WAV")
    assert prepare.run(config, params) == 0
    wav, record_path = outputs(config, "P1")
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in (wav, record_path)]
    capsys.readouterr()

    def forbidden(*args, **kwargs):
        pytest.fail("prepared audio must not be made again")

    monkeypatch.setattr(prepare, "prepare_audio", forbidden)
    assert prepare.run(config, params) == 0

    output = capsys.readouterr().out
    assert "reused" in line_of(output, "P1")
    assert "Summary: 0 prepared, 1 reused, 0 kept, 0 error" in output
    assert before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in (wav, record_path)]


@pytest.mark.parametrize("change", ["new_file", "file_size", "earlier_date", "settings"])
def test_changed_recordings_or_settings_are_an_error_and_nothing_is_replaced(
    make_config, params, write_wav, capsys, change
):
    config = make_config()
    first = day(config, "P1", "20250201")
    write_wav(first / "a.WAV", seconds=0.5)
    assert prepare.run(config, params) == 0
    wav, record_path = outputs(config, "P1")
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in (wav, record_path)]
    capsys.readouterr()

    if change == "new_file":
        write_wav(first / "b.WAV")
    elif change == "file_size":
        write_wav(first / "a.WAV", seconds=0.7)
    elif change == "earlier_date":
        write_wav(day(config, "P1", "20250101") / "x.WAV")
    else:
        params["audio_preparation"]["target_peak_dbfs"] = -3.0

    assert prepare.run(config, params) == 1

    line = line_of(capsys.readouterr().out, "P1")
    assert "error" in line and "Nothing was changed" in line
    assert before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in (wav, record_path)]
    assert sorted(path.name for path in wav.parent.iterdir()) == [
        "P1.wav", "P1_preparation.json"
    ]


def test_prepared_audio_without_a_record_is_kept(make_config, params, write_wav, capsys):
    config = make_config()
    write_wav(day(config, "P1", "20250201") / "a.WAV")
    wav, record_path = outputs(config, "P1")
    write_wav(wav, seconds=0.3, sr=16000, channels=1)
    before = (wav.read_bytes(), wav.stat().st_mtime_ns)

    assert prepare.run(config, params) == 0

    line = line_of(capsys.readouterr().out, "P1")
    assert "kept" in line and "without a preparation record" in line
    assert (wav.read_bytes(), wav.stat().st_mtime_ns) == before
    assert not record_path.exists()


@pytest.mark.parametrize("left_over", ["record", "unfinished_file"])
def test_partial_state_is_an_error_and_nothing_is_changed(
    make_config, params, write_wav, capsys, left_over
):
    config = make_config()
    write_wav(day(config, "P1", "20250201") / "a.WAV")
    wav, record_path = outputs(config, "P1")
    wav.parent.mkdir(parents=True)
    existing = record_path if left_over == "record" else wav.with_name("P1.wav.part")
    existing.write_text("{}", encoding="utf-8")

    assert prepare.run(config, params) == 1

    assert "error" in line_of(capsys.readouterr().out, "P1")
    assert [path.name for path in wav.parent.iterdir()] == [existing.name]
    assert existing.read_text(encoding="utf-8") == "{}"


# --- check and selection of participants -------------------------------------


def test_check_writes_nothing_and_flags_a_duration_difference(
    make_config, params, write_wav, write_asr, tmp_path, capsys
):
    config = make_config()
    # P1: the prepared audio is much longer than the recordings of the first day.
    first = day(config, "P1", "20250201")
    write_wav(first / "a.WAV", seconds=0.5)
    write_wav(first / "b.WAV", seconds=0.5)
    write_asr(first / "a.ASR")
    write_wav(day(config, "P1", "20250302") / "c.WAV")
    write_wav(outputs(config, "P1")[0], seconds=3.0, sr=16000, channels=1)
    # P2: raw recordings only.
    write_wav(day(config, "P2", "20250201") / "a.WAV", seconds=0.5)
    # P3: prepared audio only.
    write_wav(outputs(config, "P3")[0], seconds=0.5, sr=16000, channels=1)
    before = listing(tmp_path)

    assert prepare.run(config, params, check=True) == 0

    assert listing(tmp_path) == before
    output = capsys.readouterr().out
    line = line_of(output, "P1")
    for text in ("first day 20250201", "2 WAV file(s)", "1 later date(s)", "prepared WAV yes",
                 "record none", "raw 1.0 s, prepared 3.0 s", "DURATION DIFFERS"):
        assert text in line
    line = line_of(output, "P2")
    assert "1 WAV file(s)" in line and "0 later date(s)" in line
    assert "prepared WAV no" in line and "DURATION DIFFERS" not in line
    line = line_of(output, "P3")
    assert "no raw recording" in line and "prepared WAV yes" in line
    assert "DURATION DIFFERS" not in line
    assert "Nothing was written" in output


def test_check_after_preparation_shows_a_matching_record(
    make_config, params, write_wav, tmp_path, capsys
):
    config = make_config()
    write_wav(day(config, "P1", "20250201") / "a.WAV", seconds=0.5)
    assert prepare.run(config, params) == 0
    before = listing(tmp_path)
    capsys.readouterr()

    assert prepare.run(config, params, check=True) == 0

    line = line_of(capsys.readouterr().out, "P1")
    assert "record matches" in line and "DURATION DIFFERS" not in line
    assert listing(tmp_path) == before


def test_without_raw_folder_the_check_reports_the_prepared_files(
    make_config, params, write_wav, tmp_path, capsys
):
    config = make_config()
    write_wav(outputs(config, "P1")[0], seconds=0.5, sr=16000, channels=1)
    before = listing(tmp_path)

    assert prepare.run(config, params, check=True) == 0

    output = capsys.readouterr().out
    assert "not reachable" in output
    assert "prepared WAV yes" in line_of(output, "P1")
    assert listing(tmp_path) == before

    # The preparation itself needs the raw folder.
    assert prepare.run(config, params) == 1
    assert "Raw audio folder not found" in capsys.readouterr().out
    assert listing(tmp_path) == before


def test_participant_and_limit_select_the_participants(make_config, params, write_wav, capsys):
    config = make_config()
    for name in ("P1", "P2", "P3"):
        write_wav(day(config, name, "20250201") / "a.WAV", seconds=0.2)

    assert prepare.run(config, params, limit=2) == 0
    prepared = [name for name in ("P1", "P2", "P3") if outputs(config, name)[0].is_file()]
    assert prepared == ["P1", "P2"]

    assert prepare.run(config, params, participant="P3") == 0
    assert outputs(config, "P3")[0].is_file()

    capsys.readouterr()
    assert prepare.run(config, params, participant="P9") == 1
    assert "P9" in capsys.readouterr().out


# --- other folder structures --------------------------------------------------


def constant(path: Path, value: float, frames: int = 800) -> Path:
    """Write a short WAV file (8 kHz, one channel) where every sample has the same value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.full(frames, value), 8000, format="WAV", subtype="PCM_16")
    return path


def keep_samples(params: dict) -> None:
    """Switch off resampling and normalisation, so that the joined samples can be compared."""
    params["audio_preparation"].update(resample=False, convert_to_mono=False, normalize=False)


def test_participant_folders_without_dates_need_all_days(make_config, params, capsys):
    config = make_config()
    constant(config.path("raw_audio", "P1", "REC_0002.WAV"), 0.5)
    constant(config.path("raw_audio", "P1", "REC_0001.WAV"), 0.25)
    constant(config.path("raw_audio", "P2", "REC_0003.WAV"), -0.25)
    keep_samples(params)

    # Default: first recording day. Without date folders nothing is selected.
    assert prepare.run(config, params) == 1
    output = capsys.readouterr().out
    for participant in ("P1", "P2"):
        line = line_of(output, participant)
        assert "error" in line and "no date folder" in line and "recording_days" in line
    assert not config.path("prepared_audio").exists()

    params["audio_preparation"]["recording_days"] = "all"
    assert prepare.run(config, params) == 0

    assert prepared_files(config) == [
        "P1.wav", "P1_preparation.json", "P2.wav", "P2_preparation.json"
    ]
    samples, _ = sf.read(str(outputs(config, "P1")[0]))
    np.testing.assert_array_equal(samples, np.concatenate([np.full(800, 0.25), np.full(800, 0.5)]))
    record = record_of(outputs(config, "P1")[0])
    assert record["structure"] == "participant folder" and record["joined"] is True
    assert record["selected_dates"] == [] and record["date_folders"] == []
    assert [item["path"] for item in record["source_files"]] == [
        "P1/REC_0001.WAV", "P1/REC_0002.WAV"
    ]
    assert "participant folder without dates" in capsys.readouterr().out


def test_files_directly_in_the_raw_folder_are_prepared_one_by_one(
    make_config, params, write_wav, capsys
):
    config = make_config()
    raw_root = config.path("raw_audio")
    write_wav(raw_root / "REC_0001.WAV", seconds=0.2)
    write_wav(raw_root / "REC_0002.WAV", seconds=0.4)
    write_wav(raw_root / "talk.wav", seconds=0.3)
    (raw_root / "notes.txt").write_text("not a recording", encoding="utf-8")

    # Joining is switched on, but single files have no participant: they are not joined.
    assert prepare.run(config, params) == 0

    assert prepared_files(config) == [
        "REC_0001.WAV", "REC_0001_preparation.json",
        "REC_0002.WAV", "REC_0002_preparation.json",
        "talk.wav", "talk_preparation.json",
    ]
    wav = config.path("prepared_audio", "REC_0002.WAV")
    info = sf.info(str(wav))
    assert (info.samplerate, info.channels) == (16000, 1)
    assert info.duration == pytest.approx(0.4, abs=0.01)
    record = record_of(wav)
    assert record["recording_id"] == "REC_0002"
    assert record["participant"] is None  # no participant name is made up
    assert record["structure"] == "single file" and record["joined"] is False
    assert [item["path"] for item in record["source_files"]] == ["REC_0002.WAV"]
    assert "Summary: 3 prepared, 0 reused, 0 kept, 0 error" in capsys.readouterr().out

    assert prepare.run(config, params, participant="talk") == 0
    assert "Summary: 0 prepared, 1 reused, 0 kept, 0 error" in capsys.readouterr().out


def test_raw_audio_can_be_one_wav_file(make_config, params, write_wav, tmp_path, capsys):
    source = write_wav(tmp_path / "my_recordings" / "REC_0001.WAV", seconds=0.3)
    write_wav(tmp_path / "my_recordings" / "REC_0002.WAV")  # not named in config.yaml
    config = make_config(paths={"raw_audio": source})

    assert prepare.run(config, params, check=True) == 0
    assert "REC_0001: single file" in capsys.readouterr().out
    assert not config.path("prepared_audio").exists()

    assert prepare.run(config, params) == 0
    assert prepared_files(config) == ["REC_0001.WAV", "REC_0001_preparation.json"]


def test_one_raw_folder_can_hold_different_structures(make_config, params, write_wav, capsys):
    config = make_config()
    raw_root = config.path("raw_audio")
    write_wav(day(config, "P1", "20250201") / "a.WAV", seconds=0.2)
    write_wav(raw_root / "REC_0001.WAV", seconds=0.2)
    (raw_root / "documents").mkdir()
    (raw_root / "documents" / "consent.txt").write_text("no recording", encoding="utf-8")

    assert prepare.run(config, params) == 0

    assert prepared_files(config) == [
        "P1.wav", "P1_preparation.json", "REC_0001.WAV", "REC_0001_preparation.json"
    ]
    output = capsys.readouterr().out
    assert "documents: no WAV file in this folder, it is not used" in output
    assert "Summary: 2 prepared, 0 reused, 0 kept, 0 error" in output


def test_all_recording_days_are_joined_in_date_order_and_then_name_order(make_config, params):
    config = make_config()
    constant(day(config, "P1", "20250305") / "a.WAV", 0.75)
    constant(day(config, "P1", "20250201") / "z.WAV", 0.5)
    constant(day(config, "P1", "20250201") / "b.WAV", 0.25)
    day(config, "P1", "20250210")  # a day without recordings
    keep_samples(params)
    params["audio_preparation"]["recording_days"] = "all"

    assert prepare.run(config, params) == 0

    wav = outputs(config, "P1")[0]
    samples, _ = sf.read(str(wav))
    np.testing.assert_array_equal(
        samples, np.concatenate([np.full(800, 0.25), np.full(800, 0.5), np.full(800, 0.75)])
    )
    record = record_of(wav)
    assert record["selected_dates"] == ["20250201", "20250305"]
    assert record["date_folders"] == ["20250201", "20250210", "20250305"]
    assert [item["path"] for item in record["source_files"]] == [
        "P1/20250201/b.WAV", "P1/20250201/z.WAV", "P1/20250305/a.WAV"
    ]

    # A record of all days does not match the setting "first day": nothing is replaced.
    params["audio_preparation"]["recording_days"] = "first"
    before = wav.read_bytes()
    assert prepare.run(config, params) == 1
    assert wav.read_bytes() == before


def test_without_joining_all_days_and_folders_without_dates(make_config, params, write_wav):
    config = make_config()
    write_wav(day(config, "P1", "20250201") / "a.WAV", seconds=0.2)
    write_wav(day(config, "P1", "20250305") / "a.WAV", seconds=0.2)
    write_wav(config.path("raw_audio", "P2", "a.WAV"), seconds=0.2)
    params["audio_preparation"].update(join_multiple_files=False, recording_days="all")

    assert prepare.run(config, params) == 0

    assert [name for name in prepared_files(config) if name.endswith(".WAV")] == [
        "P1/20250201/a.WAV", "P1/20250305/a.WAV", "P2/a.WAV"
    ]
    assert record_of(config.path("prepared_audio", "P2", "a.WAV"))["recording_id"] == "P2/a"


def test_two_recordings_for_one_prepared_file_are_an_error(
    make_config, params, write_wav, capsys
):
    config = make_config()
    # The joined audio of the folder P1 and the single file P1.WAV would be the same file
    # on Windows (P1.wav and P1.WAV).
    write_wav(day(config, "P1", "20250201") / "a.WAV", seconds=0.2)
    write_wav(config.path("raw_audio", "P1.WAV"), seconds=0.2)
    write_wav(day(config, "P2", "20250201") / "a.WAV", seconds=0.2)

    assert prepare.run(config, params) == 1

    output = capsys.readouterr().out
    assert output.count("more than one recording would get the prepared file") == 2
    assert "Summary: 1 prepared, 0 reused, 0 kept, 2 error" in output
    assert prepared_files(config) == ["P2.wav", "P2_preparation.json"]


@pytest.mark.parametrize("case", ["date_as_first_folder", "files_and_sub_folders"])
def test_structure_that_is_not_clear_is_an_error_until_it_is_set(
    make_config, params, write_wav, capsys, case
):
    config = make_config()
    raw_root = config.path("raw_audio")
    if case == "date_as_first_folder":
        # <date>/<file>: the first folder is a date, not a participant.
        files = ["20250201/REC_0001.WAV", "20250201/REC_0002.WAV"]
        folder, reason = "20250201", "the folder name is a date"
    else:
        files = ["P1/REC_0001.WAV", "P1/part_2/REC_0002.WAV"]
        folder, reason = "P1", "not clear"
    for name in files:
        write_wav(raw_root / name, seconds=0.2)
    params["audio_preparation"]["recording_days"] = "all"

    assert prepare.run(config, params) == 1
    line = line_of(capsys.readouterr().out, folder)
    assert "error" in line and reason in line and "raw_audio.structure" in line
    assert not config.path("prepared_audio").exists()

    # files: every WAV file is its own recording and keeps its place.
    params["raw_audio"]["structure"] = "files"
    assert prepare.run(config, params) == 0
    assert [name for name in prepared_files(config) if name.endswith(".WAV")] == files


def test_structure_participants_joins_all_files_below_each_folder(
    make_config, params, capsys
):
    config = make_config()
    raw_root = config.path("raw_audio")
    constant(raw_root / "P1" / "session_b" / "x.WAV", 0.75)
    constant(raw_root / "P1" / "session_a" / "y.WAV", 0.5)
    constant(raw_root / "P1" / "first.WAV", 0.25)
    constant(raw_root / "loose.WAV", -0.5)
    keep_samples(params)
    params["raw_audio"]["structure"] = "participants"
    params["audio_preparation"]["recording_days"] = "all"

    assert prepare.run(config, params) == 1  # the loose file has no participant folder

    output = capsys.readouterr().out
    assert "not inside a participant folder" in line_of(output, "loose.WAV")
    assert prepared_files(config) == ["P1.wav", "P1_preparation.json"]
    # Order: files of the folder first, then the sub-folders in name order.
    samples, _ = sf.read(str(outputs(config, "P1")[0]))
    np.testing.assert_array_equal(
        samples, np.concatenate([np.full(800, 0.25), np.full(800, 0.5), np.full(800, 0.75)])
    )


def test_join_order_is_the_name_order_and_a_note_tells_when_numbers_differ(
    make_config, params, capsys
):
    config = make_config()
    first = day(config, "P1", "20250201")
    constant(first / "REC_2.WAV", 0.5)
    constant(first / "REC_10.WAV", 0.25)
    keep_samples(params)

    assert prepare.run(config, params, check=True) == 0
    assert "P1: note - the files are joined in the order of their names" in capsys.readouterr().out

    assert prepare.run(config, params) == 0
    assert "REC_10.WAV, REC_2.WAV" in capsys.readouterr().out
    samples, _ = sf.read(str(outputs(config, "P1")[0]))
    np.testing.assert_array_equal(samples, np.concatenate([np.full(800, 0.25), np.full(800, 0.5)]))


def test_prepared_audio_in_its_own_folder_is_found_and_kept(
    make_config, params, write_wav, capsys
):
    config = make_config()
    write_wav(day(config, "P1", "20250201") / "a.WAV")
    # One folder per recording: <prepared_audio>/P1/P1.wav
    old = write_wav(config.path("prepared_audio", "P1", "P1.wav"), sr=16000, channels=1)
    before = (old.read_bytes(), old.stat().st_mtime_ns)

    assert prepare.run(config, params) == 0

    assert "kept" in line_of(capsys.readouterr().out, "P1")
    assert prepared_files(config) == ["P1/P1.wav"]
    assert (old.read_bytes(), old.stat().st_mtime_ns) == before


def test_check_shows_the_structures_and_writes_nothing(
    make_config, params, write_wav, tmp_path, capsys
):
    config = make_config()
    write_wav(day(config, "P1", "20250201") / "a.WAV", seconds=0.5)
    write_wav(config.path("raw_audio", "P2", "a.WAV"), seconds=0.5)
    write_wav(config.path("raw_audio", "REC_0001.WAV"), seconds=0.5)
    before = listing(tmp_path)

    assert prepare.run(config, params, check=True) == 0

    output = capsys.readouterr().out
    assert "first day 20250201" in line_of(output, "P1")
    line = line_of(output, "P2")
    assert "cannot be prepared" in line and "recording_days" in line
    assert "single file" in line_of(output, "REC_0001")
    assert "Checked 3 recording(s)" in output and "Nothing was written" in output
    assert listing(tmp_path) == before


@pytest.mark.parametrize(
    "section, key, value",
    [("audio_preparation", "recording_days", "second"), ("raw_audio", "structure", "mixed")],
)
def test_unknown_setting_values_are_refused(make_config, params, write_wav, section, key, value):
    config = make_config()
    write_wav(day(config, "P1", "20250201") / "a.WAV")
    params[section][key] = value

    with pytest.raises(ValueError, match=key):
        prepare.run(config, params)
    assert not config.path("prepared_audio").exists()
