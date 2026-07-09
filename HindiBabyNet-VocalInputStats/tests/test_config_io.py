from pathlib import Path

import pandas as pd

from hindibabynet_vocalinputstats.config import load_config, normalize_path
from hindibabynet_vocalinputstats.ids import (
    attach_participant_ids,
    create_participant_lookup,
    save_participant_lookup,
)
from hindibabynet_vocalinputstats.io import (
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


def test_load_config_resolves_repo_relative_paths() -> None:
    config = load_config()

    assert config.metadata_path.name == "metadata_cleaned.xlsx"
    assert config.derived_data_dir.name == "derived"
    assert config.metadata_id_column == "par_id"
    assert config.participant_id_digits == 3
    assert config.minimum_recording_hours_warning == 1.0
    assert config.manual_mapping_csv is None


def test_normalize_path_keeps_unc_like_absolute_path() -> None:
    repo_root = Path("C:/repo")

    path = normalize_path("//hypatia.uio.no/share/folder/file.xlsx", repo_root)

    assert path is not None
    assert path.is_absolute()
    assert path.as_posix().startswith("//hypatia.uio.no/")


def test_load_config_accepts_metadata_csv_alias(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "metadata_csv: data/raw/metadata.csv",
                "vtc_output_root: data/vtc",
                "audio_root: data/audio",
                "derived_data_dir: data/derived",
                "private_data_dir: data/private",
                "figures_dir: figures",
                "tables_dir: tables",
                "results_dir: results",
                "participant_id_digits: 3",
                "age_month_denominator: 30.44",
                "ses_source: mother_education",
                "minimum_recording_hours_warning: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.metadata_path.name == "metadata.csv"


def test_resolve_participant_match_uses_exact_basename() -> None:
    match = resolve_participant_match("  ABAN141223  ")

    assert match.original_par_id == "ABAN141223"
    assert match.source_id == "ABAN141223"


def test_find_audio_file_prefers_recursive_participant_directory(tmp_path: Path) -> None:
    participant_dir = tmp_path / "ABAN141223" / "2024-12-14"
    ensure_directory(participant_dir)
    audio_file = participant_dir / "joined.WAV"
    audio_file.write_bytes(b"fake")

    found = find_audio_file(tmp_path, "ABAN141223", [".wav", ".WAV"])

    assert found.selected_path == audio_file
    assert found.warning_message is None


def test_find_audio_file_selects_largest_when_multiple_matches(tmp_path: Path) -> None:
    participant_dir = tmp_path / "ABAN141223"
    ensure_directory(participant_dir)
    small_audio = participant_dir / "small.wav"
    large_audio = participant_dir / "large.wav"
    small_audio.write_bytes(b"123")
    large_audio.write_bytes(b"123456")

    found = find_audio_file(tmp_path, "ABAN141223", [".wav"], prefer_largest_audio_file=True)

    assert isinstance(found, AudioDiscoveryResult)
    assert found.selected_path == large_audio
    assert found.warning_message is not None


def test_find_vtc_csv_requires_exact_participant_basename(tmp_path: Path) -> None:
    vtc_file = tmp_path / "ABAN141223" / EXPECTED_VTC_FILENAME
    ensure_directory(vtc_file.parent)
    vtc_file.write_text("uid,start_time_s,duration_s,label\nABAN141223,0.0,1.0,FEM\n", encoding="utf-8")

    assert find_vtc_csv(tmp_path, "ABAN141223") == vtc_file
    assert find_vtc_csv(tmp_path, "OTHER") is None


def test_find_vtc_csv_falls_back_to_recursive_search(tmp_path: Path) -> None:
    vtc_file = tmp_path / "ABAN141223" / "nested" / EXPECTED_VTC_FILENAME
    ensure_directory(vtc_file.parent)
    vtc_file.write_text("uid,start_time_s,duration_s,label\nABAN141223,0.0,1.0,FEM\n", encoding="utf-8")

    assert find_vtc_csv(tmp_path, "ABAN141223") == vtc_file


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
    expected = pd.DataFrame({"par_id": ["ABAN141223"], "REC_date": ["2024-01-01"]})
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