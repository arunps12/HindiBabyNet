from pathlib import Path
import wave

import numpy as np
import pandas as pd

from hindibabynet_audio_stats.cli import prepare_dataset


def _write_test_wav(path: Path, *, seconds: float, samplerate: int = 16000) -> None:
    frame_count = int(seconds * samplerate)
    samples = np.zeros(frame_count, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(samplerate)
        handle.writeframes(samples.tobytes())


def test_original_participant_id_never_appears_in_public_outputs(make_config) -> None:
    config = make_config(params={"metadata_file": "metadata.csv", "recording_duration_source": "audio_file"})
    metadata_path = config.metadata_path
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

    vtc_root = config.vtc_output_root
    (vtc_root / original_id).mkdir(parents=True)
    pd.DataFrame(
        {
            "uid": [original_id],
            "start_time_s": [0.0],
            "duration_s": [1.0],
            "label": ["FEM"],
        }
    ).to_csv(vtc_root / original_id / "rttm.csv", index=False)

    _write_test_wav(config.audio_root / original_id / "joined.wav", seconds=3600.0)

    prepare_dataset(config)

    public_paths = [
        config.derived_data_dir / "final_master.csv",
        config.derived_data_dir / "input_long.csv",
        config.derived_data_dir / "input_output_long.csv",
        config.results_dir / "validation_report.csv",
        config.results_dir / "dataset_build_report.txt",
    ]

    for path in public_paths:
        content = path.read_text(encoding="utf-8")
        assert original_id not in content

    private_lookup = config.lookup_path.read_text(encoding="utf-8")
    assert original_id in private_lookup