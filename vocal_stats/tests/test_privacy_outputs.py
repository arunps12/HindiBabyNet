from pathlib import Path
import wave

import numpy as np
import pandas as pd

from hindibabynet_vocalinputstats.build_master_dataset import run_build_master
from hindibabynet_vocalinputstats.create_long_format import run_create_long
from hindibabynet_vocalinputstats.eda import run_eda


def _write_test_wav(path: Path, *, seconds: float, samplerate: int = 16000) -> None:
    frame_count = int(seconds * samplerate)
    samples = np.zeros(frame_count, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(samplerate)
        handle.writeframes(samples.tobytes())


def test_original_participant_id_never_appears_in_public_outputs(tmp_path: Path) -> None:
    metadata_path = tmp_path / "data" / "raw" / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    original_id = "SECRET123"
    pd.DataFrame(
        {
            "par_id": [original_id],
            "REC_date": ["2024-01-02"],
            "birthdate": ["2023-01-02"],
            "child_sex": ["F"],
            "mother_education": ["college"],
            "father_education": ["college"],
            "Location": ["Delhi"],
        }
    ).to_csv(metadata_path, index=False)

    vtc_root = tmp_path / "vtc"
    (vtc_root / original_id).mkdir(parents=True)
    pd.DataFrame(
        {
            "uid": [original_id],
            "start_time_s": [0.0],
            "duration_s": [1.0],
            "label": ["FEM"],
        }
    ).to_csv(vtc_root / original_id / "rttm.csv", index=False)

    audio_root = tmp_path / "audio"
    _write_test_wav(audio_root / original_id / "joined.wav", seconds=3600.0)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"metadata_csv: {metadata_path.as_posix()}",
                f"vtc_output_root: {vtc_root.as_posix()}",
                f"audio_root: {audio_root.as_posix()}",
                f"derived_data_dir: {(tmp_path / 'data' / 'derived').as_posix()}",
                f"private_data_dir: {(tmp_path / 'data' / 'private').as_posix()}",
                f"figures_dir: {(tmp_path / 'figures').as_posix()}",
                f"tables_dir: {(tmp_path / 'tables').as_posix()}",
                f"results_dir: {(tmp_path / 'results').as_posix()}",
                "audio_extensions: ['.wav', '.WAV']",
                "participant_id_digits: 3",
                "age_month_denominator: 30.44",
                "ses_source: mother_education",
                "minimum_recording_hours_warning: 1.0",
                "manual_mapping_csv:",
            ]
        ),
        encoding="utf-8",
    )

    run_build_master(config_path)
    run_create_long(config_path)
    run_eda(config_path)

    public_paths = [
        tmp_path / "data" / "derived" / "final_master.csv",
        tmp_path / "data" / "derived" / "input_long.csv",
        tmp_path / "data" / "derived" / "input_output_long.csv",
        tmp_path / "results" / "validation_report.csv",
        tmp_path / "results" / "dataset_build_report.txt",
    ]
    public_paths.extend(sorted((tmp_path / "tables").glob("*.csv")))

    for path in public_paths:
        content = path.read_text(encoding="utf-8")
        assert original_id not in content

    private_lookup = (tmp_path / "data" / "private" / "participant_lookup.csv").read_text(encoding="utf-8")
    assert original_id in private_lookup