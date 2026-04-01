"""
VTC 2.0 (Voice Type Classifier) inference runner.

Orchestrates external VTC inference via subprocess.
VTC runs in its own virtual environment inside its own repo directory.
HindiBabyNet does NOT parse, rename, or transform VTC outputs.

VTC output classes (unchanged):
  FEM  = adult female speech
  MAL  = adult male speech
  KCHI = key-child speech
  OCH  = other child speech
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from src.hindibabynet.entity.config_entity import VTCConfig
from src.hindibabynet.entity.artifact_entity import VTCInferenceArtifact
from src.hindibabynet.logging.logger import get_logger

logger = get_logger(__name__)

# Expected VTC output structure
_EXPECTED_OUTPUTS = ("rttm", "raw_rttm", "rttm.csv", "raw_rttm.csv")


class VTCInferenceRunner:
    """Run VTC 2.0 inference on a single prepared WAV."""

    def __init__(self, vtc_config: VTCConfig) -> None:
        self.cfg = vtc_config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_repo(self) -> None:
        repo = self.cfg.repo_path
        if not repo.is_dir():
            raise FileNotFoundError(
                f"VTC repo not found at: {repo}. "
                f"Clone it with: git lfs install && "
                f"git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git {repo}"
            )
        infer_script = repo / "scripts" / "infer.py"
        if not infer_script.is_file():
            raise FileNotFoundError(
                f"VTC infer script not found at: {infer_script}. "
                f"Is the VTC repo cloned correctly (with --recurse-submodules)?"
            )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    def _prepare_input_folder(self, wav_path: Path, participant_id: str) -> Path:
        """Create a participant-specific input folder containing the WAV."""
        input_dir = self.cfg.input_root / participant_id
        input_dir.mkdir(parents=True, exist_ok=True)
        dest = input_dir / wav_path.name
        if not dest.exists():
            shutil.copy2(wav_path, dest)
            logger.info(f"Copied WAV to VTC input folder: {dest}")
        else:
            logger.info(f"WAV already in VTC input folder: {dest}")
        return input_dir

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _build_command(self, input_dir: Path, output_dir: Path) -> list[str]:
        return [
            "uv", "run", "scripts/infer.py",
            "--wavs", str(input_dir),
            "--output", str(output_dir),
            "--device", self.cfg.device,
        ]

    def _run_subprocess(self, cmd: list[str]) -> subprocess.CompletedProcess:
        logger.info(f"VTC command: {' '.join(cmd)}")
        logger.info(f"VTC cwd: {self.cfg.repo_path}")
        result = subprocess.run(
            cmd,
            cwd=str(self.cfg.repo_path),
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

    # ------------------------------------------------------------------
    # Output verification
    # ------------------------------------------------------------------
    def _verify_outputs(self, output_dir: Path) -> dict[str, bool]:
        results = {}
        for name in _EXPECTED_OUTPUTS:
            path = output_dir / name
            exists = path.exists()
            results[name] = exists
            if not exists:
                logger.warning(f"Expected VTC output missing: {path}")
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _write_run_info(
        self,
        output_dir: Path,
        participant_id: str,
        source_wav: Path,
        cmd: list[str],
        runtime_sec: float,
        status: str,
        output_check: dict[str, bool],
    ) -> Path:
        run_info_path = output_dir / "vtc_run_info.json"
        info = {
            "participant_id": participant_id,
            "source_wav": str(source_wav),
            "backend": "vtc",
            "vtc_repo_path": str(self.cfg.repo_path),
            "output_dir": str(output_dir),
            "command": " ".join(cmd),
            "device": self.cfg.device,
            "runtime_sec": round(runtime_sec, 2),
            "status": status,
            "outputs_present": output_check,
        }
        run_info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        logger.info(f"VTC run info saved: {run_info_path}")
        return run_info_path

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup_input(self, input_dir: Path) -> None:
        if not self.cfg.keep_inputs and input_dir.exists():
            shutil.rmtree(input_dir)
            logger.info(f"Cleaned up VTC input folder: {input_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, wav_path: Path, participant_id: str) -> VTCInferenceArtifact:
        """
        Run VTC inference on one prepared WAV.

        Parameters
        ----------
        wav_path : Path
            Path to the mono 16 kHz analysis WAV from Stage 02.
        participant_id : str
            Participant identifier.

        Returns
        -------
        VTCInferenceArtifact
        """
        logger.info("=" * 60)
        logger.info(f"VTC Inference | participant_id={participant_id}")
        logger.info(f"  source_wav   = {wav_path}")
        logger.info(f"  vtc_repo     = {self.cfg.repo_path}")
        logger.info(f"  device       = {self.cfg.device}")
        logger.info("=" * 60)

        if not wav_path.is_file():
            raise FileNotFoundError(f"Source WAV not found: {wav_path}")

        self._validate_repo()

        # Prepare input & output folders
        input_dir = self._prepare_input_folder(wav_path, participant_id)
        output_dir = self.cfg.output_root / participant_id
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(input_dir, output_dir)

        # Run inference
        t0 = time.monotonic()
        status = "success"
        try:
            self._run_subprocess(cmd)
        except RuntimeError:
            status = "failed"
            raise
        finally:
            runtime_sec = time.monotonic() - t0
            output_check = self._verify_outputs(output_dir)
            run_info_path = self._write_run_info(
                output_dir, participant_id, wav_path, cmd, runtime_sec, status, output_check
            )
            self._cleanup_input(input_dir)

        missing = [k for k, v in output_check.items() if not v]
        if missing:
            logger.warning(f"VTC completed but missing outputs: {missing}")

        logger.info(
            f"VTC inference done | participant_id={participant_id} "
            f"runtime={runtime_sec:.1f}s status={status}"
        )

        return VTCInferenceArtifact(
            participant_id=participant_id,
            output_dir=output_dir,
            rttm_dir=output_dir / "rttm",
            raw_rttm_dir=output_dir / "raw_rttm",
            rttm_csv_path=output_dir / "rttm.csv",
            raw_rttm_csv_path=output_dir / "raw_rttm.csv",
            run_info_json_path=run_info_path,
            runtime_sec=round(runtime_sec, 2),
            status=status,
        )
