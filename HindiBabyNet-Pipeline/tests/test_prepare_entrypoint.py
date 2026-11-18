"""Exercise the prepare CLI workflow with real configuration and synthetic audio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from hindibabynet_pipeline.config.configuration import ConfigurationManager
from hindibabynet_pipeline.workflow.audio_preparation import run_prepare_audio


@pytest.mark.parametrize("mode", ["wav", "manifest"])
@pytest.mark.parametrize("run_id", ["explicit-test-run", None, ""])
def test_run_prepare_audio_real_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    run_id: str | None,
) -> None:
    """Guard recording_id keywords and run-ID fallback in both entrypoint branches.

    Keep the configuration manager and preparation component real: mocking either
    would hide the original participant_id keyword / None run_id regressions.
    Only observe run-ID generation, delegating to the real implementation.
    """
    recording_id = "TEST000001"
    target_peak_dbfs = -6.0  # Nondefault value proves the native params sidecar loads.
    config_path = tmp_path / "config.yaml"
    params_path = tmp_path / "params.yaml"
    artifacts_root = tmp_path / "artifacts"
    prepared_root = tmp_path / "prepared"
    config_path.write_text(
        yaml.safe_dump(
            {
                "artifacts_root": str(artifacts_root),
                "logs_root": str(tmp_path / "logs"),
                "paths": {
                    "prepared_audio_root": str(prepared_root),
                    "raw_joined_audio_root": str(tmp_path / "raw_joined"),
                },
                "audio_preparation": {
                    "save_raw_joined_audio": False,
                    "save_prepared_audio": True,
                },
            }
        ),
        encoding="utf-8",
    )
    params_path.write_text(
        yaml.safe_dump(
            {
                "audio_preparation": {
                    "target_sr": 16000,
                    "convert_to_mono": True,
                    "resample": True,
                    "normalize": True,
                    "target_peak_dbfs": target_peak_dbfs,
                    "join_multiple_files": True,
                    "combine_gap_sec": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HBN_CONFIG_PATH", str(config_path))
    monkeypatch.chdir(tmp_path)

    wav = tmp_path / "synthetic_source.wav"
    samples = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
    sf.write(wav, samples, 16000, format="WAV", subtype="PCM_16")
    source_bytes = wav.read_bytes()
    source, source_sr = sf.read(wav, dtype="float32")

    generated_run_ids: list[str] = []
    original_make_run_id = ConfigurationManager.make_run_id

    def observe_make_run_id(self: ConfigurationManager) -> str:
        generated = original_make_run_id(self)
        generated_run_ids.append(generated)
        return generated

    monkeypatch.setattr(ConfigurationManager, "make_run_id", observe_make_run_id)

    if mode == "wav":
        result = run_prepare_audio(wav=wav, recording_id=recording_id, run_id=run_id)
    else:
        recordings_path = tmp_path / "recordings.parquet"
        pd.DataFrame(
            [
                {
                    "participant_id": recording_id,
                    "recording_id": "synthetic_source",
                    "path": str(wav),
                }
            ]
        ).to_parquet(recordings_path, index=False, engine="pyarrow")
        result = run_prepare_audio(recordings_parquet=recordings_path, run_id=run_id)

    assert result is None
    if run_id:
        assert generated_run_ids == []
        expected_run_id = run_id
    else:
        assert len(generated_run_ids) == 1
        expected_run_id = generated_run_ids[0]
        assert expected_run_id
        assert expected_run_id != "None"

    assert sorted(path.name for path in artifacts_root.iterdir()) == [expected_run_id]
    stage_dir = artifacts_root / expected_run_id / "audio_preparation"
    output = prepared_root / recording_id / f"{recording_id}.wav"
    assert output.is_file()
    assert output != wav
    info = sf.info(output)
    assert info.samplerate == source_sr == 16000
    assert info.channels == 1
    assert info.frames == 1600
    assert info.duration == pytest.approx(0.1)
    assert info.subtype == "PCM_16"
    prepared, prepared_sr = sf.read(output, dtype="float32")
    assert prepared_sr == source_sr
    assert prepared.shape == source.shape
    # Allow one PCM-16 quantization step, not a loose signal-level tolerance.
    assert float(np.max(np.abs(prepared))) == pytest.approx(
        10 ** (target_peak_dbfs / 20), abs=1 / 32768, rel=0
    )
    assert not np.array_equal(prepared, source)
    assert wav.read_bytes() == source_bytes

    manifest_path = stage_dir / f"{recording_id}_audio_manifest.parquet"
    assert manifest_path.is_file()
    manifest = pd.read_parquet(manifest_path)
    assert len(manifest) == 1
    row = manifest.iloc[0]
    assert row["participant_id"] == recording_id
    assert row["recording_id"] == recording_id
    assert Path(row["source_path"]) == wav
    assert row["source_index"] == 0
    assert row["sample_rate"] == 16000
    assert row["channels"] == 1
    assert row["source_duration_sec"] == pytest.approx(0.1)

    metadata_path = stage_dir / f"{recording_id}_analysis_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert Path(metadata["analysis_wav"]) == output
    assert metadata["save_prepared_audio"] is True
    assert metadata["normalize"]["target_peak"] == pytest.approx(
        10 ** (target_peak_dbfs / 20)
    )
    assert metadata["mode"] == (
        "single_wav" if mode == "wav" else "combine_by_participant_id"
    )