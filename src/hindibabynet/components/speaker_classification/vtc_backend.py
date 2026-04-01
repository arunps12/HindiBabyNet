"""
VTC 2.0 backend adapter for Stage 03.

Wraps the existing :class:`VTCInferenceRunner` behind the
:class:`ClassificationBackend` interface, managing temp inputs internally.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from src.hindibabynet.components.speaker_classification.base import ClassificationBackend
from src.hindibabynet.components.speaker_classification.metadata import (
    make_run_info,
    utcnow_iso,
    write_run_info,
)
from src.hindibabynet.components.speaker_classification.output_checks import is_vtc_complete
from src.hindibabynet.config.configuration import ConfigurationManager
from src.hindibabynet.logging.logger import get_logger

logger = get_logger(__name__)

# Expected VTC output structure
_EXPECTED_OUTPUTS = ("rttm", "raw_rttm", "rttm.csv", "raw_rttm.csv")


class VTCBackend(ClassificationBackend):
    """
    External VTC 2.0 voice-type classifier backend.

    VTC runs in its own virtual environment via subprocess.
    HindiBabyNet does NOT rename or transform VTC outputs.
    """

    def __init__(self, cfg: ConfigurationManager) -> None:
        self._cfg = cfg
        vtc = cfg.get_vtc_params()
        self._repo_path = Path(vtc.get("repo_path", "external_models/VTC"))
        self._device = str(vtc.get("device", "cuda"))
        self._keep_inputs = bool(vtc.get("keep_inputs", False))

    # -- protocol ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "vtc"

    def run_participant(
        self,
        wav_path: Path,
        participant_id: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        logger.info("=" * 60)
        logger.info(f"VTC Inference | participant_id={participant_id}")
        logger.info(f"  source_wav = {wav_path}")
        logger.info(f"  vtc_repo   = {self._repo_path}")
        logger.info(f"  device     = {self._device}")
        logger.info(f"  output_dir = {output_dir}")
        logger.info("=" * 60)

        if not wav_path.is_file():
            raise FileNotFoundError(f"Source WAV not found: {wav_path}")

        self._validate_repo()

        # Temporary input folder (hidden from user)
        tmp_input_dir = output_dir.parent / "_tmp_vtc_inputs" / participant_id
        try:
            tmp_input_dir.mkdir(parents=True, exist_ok=True)
            dest = tmp_input_dir / wav_path.name
            if not dest.exists():
                shutil.copy2(wav_path, dest)
                logger.info(f"Copied WAV to VTC temp input: {dest}")

            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "uv", "run", "scripts/infer.py",
                "--wavs", str(tmp_input_dir),
                "--output", str(output_dir),
                "--device", self._device,
            ]

            started = utcnow_iso()
            t0 = time.monotonic()
            status = "success"
            error_msg = ""

            try:
                self._run_subprocess(cmd)
            except RuntimeError as exc:
                status = "failed"
                error_msg = str(exc)
                raise
            finally:
                runtime = round(time.monotonic() - t0, 2)
                output_check = self._verify_outputs(output_dir)
                info = make_run_info(
                    participant_id=participant_id,
                    backend="vtc",
                    input_wav=wav_path,
                    output_dir=output_dir,
                    started_at=started,
                    finished_at=utcnow_iso(),
                    runtime_sec=runtime,
                    status=status,
                    command=" ".join(cmd),
                    device=self._device,
                    vtc_repo_path=str(self._repo_path),
                    outputs_present=output_check,
                    **({"error": error_msg} if error_msg else {}),
                )
                write_run_info(output_dir, info)
        finally:
            # Cleanup temp inputs
            if not self._keep_inputs and tmp_input_dir.exists():
                shutil.rmtree(tmp_input_dir, ignore_errors=True)
                logger.info(f"Cleaned up VTC temp input: {tmp_input_dir}")

        missing = [k for k, v in output_check.items() if not v]
        if missing:
            logger.warning(f"VTC completed but missing outputs: {missing}")

        logger.info(
            f"VTC done | participant_id={participant_id} "
            f"runtime={runtime:.1f}s status={status}"
        )
        return info

    def is_complete(self, participant_id: str, output_dir: Path) -> bool:
        return is_vtc_complete(participant_id, output_dir)

    def expected_outputs(self, participant_id: str) -> list[str]:
        return ["rttm/", "raw_rttm/", "rttm.csv", "raw_rttm.csv", "run_info.json"]

    # -- internal ----------------------------------------------------------

    def _validate_repo(self) -> None:
        if not self._repo_path.is_dir():
            raise FileNotFoundError(
                f"VTC repo not found at: {self._repo_path}. "
                f"Clone it with: git lfs install && "
                f"git clone --recurse-submodules "
                f"https://github.com/LAAC-LSCP/VTC.git {self._repo_path}"
            )
        infer_script = self._repo_path / "scripts" / "infer.py"
        if not infer_script.is_file():
            raise FileNotFoundError(
                f"VTC infer script not found: {infer_script}. "
                f"Is the repo cloned with --recurse-submodules?"
            )

    def _run_subprocess(self, cmd: list[str]) -> subprocess.CompletedProcess:
        logger.info(f"VTC command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=str(self._repo_path),
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.info(f"VTC stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"VTC stderr:\n{result.stderr}")
        if result.returncode != 0:
            raise RuntimeError(
                f"VTC inference failed (exit code {result.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {result.stderr}"
            )
        return result

    def _verify_outputs(self, output_dir: Path) -> dict[str, bool]:
        results = {}
        for name in _EXPECTED_OUTPUTS:
            path = output_dir / name
            results[name] = path.exists()
            if not path.exists():
                logger.warning(f"Expected VTC output missing: {path}")
        return results
