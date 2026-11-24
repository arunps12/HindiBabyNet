from pathlib import Path

import pandas as pd

from hindibabynet_audio_stats.ids import (
    attach_participant_ids,
    create_participant_lookup,
    save_participant_lookup,
)
from hindibabynet_audio_stats.io import (
    AudioDiscoveryResult,
    EXPECTED_VTC_FILENAME,
    ensure_directory,
    find_audio_file,
    find_vtc_csv,
    load_vtc_csv,
    read_table,
    resolve_participant_match,
    write_dataset_build_report,
    write_validation_report,
)


def test_load_config_reads_the_study_settings(make_config) -> None:
    config = make_config()

    assert config.metadata_path.name == "metadata_cleaned.xlsx"
    assert config.lookup_path.name == "participant_lookup.csv"
    assert config.derived_data_dir.name == "statistical-data"
    assert config.metadata_id_column == "par_id"
    assert config.participant_id_digits == 3
    assert config.minimum_recording_hours_warning == 1.0
    assert config.manual_mapping_csv is None
    assert config.age_month_denominator == 30.44
    assert config.ses_source == "mother_education"
    assert config.recording_duration_source == "raw_audio_sessions"
    assert config.session_selection == "earliest"
    assert config.session_date_format == "%Y%m%d"
    assert config.audio_extensions == (".wav", ".WAV", ".flac", ".FLAC")
    assert config.audio_layout.participant_folder_name == "{par_id}"
    assert config.audio_layout.recursive is True
    assert config.audio_layout.prefer_largest_audio_file is True
    assert config.vtc_layout.participant_folder_name == "{par_id}"
    assert config.vtc_layout.rttm_csv_name == "rttm.csv"


def test_load_config_uses_metadata_file_name_from_params(make_config) -> None:
    config = make_config(params={"metadata_file": "metadata.csv"})

    assert config.metadata_path.name == "metadata.csv"
    assert config.metadata_path.parent == config.lookup_path.parent


def test_resolve_participant_match_uses_exact_basename() -> None:
    match = resolve_participant_match("  TEST000001  ")

    assert match.original_par_id == "TEST000001"
    assert match.source_id == "TEST000001"


def test_find_audio_file_prefers_recursive_participant_directory(tmp_path: Path) -> None:
    participant_dir = tmp_path / "TEST000001" / "2024-12-14"
    ensure_directory(participant_dir)
    audio_file = participant_dir / "joined.WAV"
    audio_file.write_bytes(b"fake")

    found = find_audio_file(tmp_path, "TEST000001", [".wav", ".WAV"])

    assert found.selected_path == audio_file
    assert found.warning_message is None


def test_find_audio_file_selects_largest_when_multiple_matches(tmp_path: Path) -> None:
    participant_dir = tmp_path / "TEST000001"
    ensure_directory(participant_dir)
    small_audio = participant_dir / "small.wav"
    large_audio = participant_dir / "large.wav"
    small_audio.write_bytes(b"123")
    large_audio.write_bytes(b"123456")

    found = find_audio_file(tmp_path, "TEST000001", [".wav"], prefer_largest_audio_file=True)

    assert isinstance(found, AudioDiscoveryResult)
    assert found.selected_path == large_audio
    assert found.warning_message is not None


def test_find_vtc_csv_requires_exact_participant_basename(tmp_path: Path) -> None:
    vtc_file = tmp_path / "TEST000001" / EXPECTED_VTC_FILENAME
    ensure_directory(vtc_file.parent)
    vtc_file.write_text("uid,start_time_s,duration_s,label\nTEST000001,0.0,1.0,FEM\n", encoding="utf-8")

    assert find_vtc_csv(tmp_path, "TEST000001") == vtc_file
    assert find_vtc_csv(tmp_path, "OTHER") is None


def test_find_vtc_csv_falls_back_to_recursive_search(tmp_path: Path) -> None:
    vtc_file = tmp_path / "TEST000001" / "nested" / EXPECTED_VTC_FILENAME
    ensure_directory(vtc_file.parent)
    vtc_file.write_text("uid,start_time_s,duration_s,label\nTEST000001,0.0,1.0,FEM\n", encoding="utf-8")

    assert find_vtc_csv(tmp_path, "TEST000001") == vtc_file


def test_load_vtc_csv_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / EXPECTED_VTC_FILENAME
    path.write_text("uid,start_time_s,label\nP001,0.0,FEM\n", encoding="utf-8")

    try:
        load_vtc_csv(path)
    except ValueError as exc:
        assert "duration_s" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation error")


def test_load_vtc_csv_infers_duration_from_end_time(tmp_path: Path) -> None:
    path = tmp_path / EXPECTED_VTC_FILENAME
    path.write_text("speaker,start,end\nFEM,1.0,2.5\n", encoding="utf-8")

    dataframe = load_vtc_csv(path)

    assert dataframe.loc[0, "label"] == "FEM"
    assert dataframe.loc[0, "start_time_s"] == 1.0
    assert dataframe.loc[0, "duration_s"] == 1.5


def test_read_table_supports_excel_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metadata.xlsx"
    expected = pd.DataFrame({"par_id": ["TEST000001"], "REC_date": ["2024-01-01"]})
    expected.to_excel(path, index=False)

    loaded = read_table(path)

    assert loaded.to_dict("records") == expected.to_dict("records")


def test_write_reports_are_deterministic(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation_report.csv"
    report_path = tmp_path / "dataset_build_report.txt"

    write_validation_report(
        [
            {"participant_id": "P002", "original_par_id": "BBB", "issue_type": "missing_audio", "message": "No audio"},
            {"participant_id": "P001", "original_par_id": "AAA", "issue_type": "missing_vtc", "message": "No VTC"},
        ],
        validation_path,
    )
    write_dataset_build_report(["line 2", "line 1"], report_path)

    validation = pd.read_csv(validation_path)
    assert validation["participant_id"].tolist() == ["P001", "P002"]
    assert report_path.read_text(encoding="utf-8") == "line 2\nline 1\n"


def test_participant_id_anonymization_is_stable_and_private(tmp_path: Path) -> None:
    metadata = pd.DataFrame({"par_id": [" BBB ", "AAA", "BBB", "CCC"]})

    lookup = create_participant_lookup(metadata, participant_id_digits=3)
    attached = attach_participant_ids(metadata, lookup)
    output_path = tmp_path / "participant_lookup.csv"
    save_participant_lookup(lookup, output_path)

    assert lookup.to_dict("records") == [
        {"original_par_id": "AAA", "participant_id": "P001"},
        {"original_par_id": "BBB", "participant_id": "P002"},
        {"original_par_id": "CCC", "participant_id": "P003"},
    ]
    assert attached["participant_id"].tolist() == ["P002", "P001", "P002", "P003"]
    persisted = pd.read_csv(output_path)
    assert list(persisted.columns) == ["original_par_id", "participant_id"]