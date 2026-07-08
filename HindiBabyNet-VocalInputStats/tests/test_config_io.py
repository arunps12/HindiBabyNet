from pathlib import Path

import pandas as pd

from hindibabynet_vocalinputstats.config import load_config
from hindibabynet_vocalinputstats.ids import (
    attach_participant_ids,
    create_participant_lookup,
    save_participant_lookup,
)
from hindibabynet_vocalinputstats.io import (
    EXPECTED_VTC_FILENAME,
    ensure_directory,
    find_audio_file,
    find_vtc_csv,
    load_vtc_csv,
    resolve_participant_match,
    write_dataset_build_report,
    write_validation_report,
)


def test_load_config_resolves_repo_relative_paths() -> None:
    config = load_config()

    assert config.metadata_csv.name == "metadata.csv"
    assert config.derived_data_dir.name == "derived"
    assert config.participant_id_digits == 3
    assert config.minimum_recording_hours_warning == 1.0
    assert config.manual_mapping_csv is None


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

    assert found == audio_file


def test_find_vtc_csv_requires_exact_participant_basename(tmp_path: Path) -> None:
    vtc_file = tmp_path / "ABAN141223" / EXPECTED_VTC_FILENAME
    ensure_directory(vtc_file.parent)
    vtc_file.write_text("uid,start_time_s,duration_s,label\nABAN141223,0.0,1.0,FEM\n", encoding="utf-8")

    assert find_vtc_csv(tmp_path, "ABAN141223") == vtc_file
    assert find_vtc_csv(tmp_path, "OTHER") is None


def test_load_vtc_csv_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / EXPECTED_VTC_FILENAME
    path.write_text("uid,start_time_s,label\nP001,0.0,FEM\n", encoding="utf-8")

    try:
        load_vtc_csv(path)
    except ValueError as exc:
        assert "duration_s" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation error")


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