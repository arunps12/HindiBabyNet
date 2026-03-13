"""Diagnostics service: system and environment checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hindibabynet_gui.utils import PROJECT_ROOT, CONFIG_PATH, resolve_path
from hindibabynet_gui.utils.yaml_utils import load_yaml


@dataclass
class CheckResult:
    name: str
    status: str       # "ok" | "warn" | "fail"
    message: str
    detail: str = ""


def _run_cmd(cmd: list[str]) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.returncode == 0, r.stdout.strip() or r.stderr.strip()
    except FileNotFoundError:
        return False, "not found"
    except Exception as e:
        return False, str(e)


def run_all_checks() -> list[CheckResult]:
    """Run all diagnostic checks and return results."""
    results: list[CheckResult] = []

    # 1. Python version
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 10)
    results.append(CheckResult(
        "Python Version", "ok" if ok else "fail",
        f"Python {ver}", "Requires ≥ 3.10"
    ))

    # 2. Project root
    has_pyproject = (PROJECT_ROOT / "pyproject.toml").exists()
    results.append(CheckResult(
        "Project Root", "ok" if has_pyproject else "fail",
        str(PROJECT_ROOT),
        "pyproject.toml found" if has_pyproject else "pyproject.toml NOT found"
    ))

    # 3. config.yaml
    cfg_exists = CONFIG_PATH.exists()
    results.append(CheckResult(
        "Config File", "ok" if cfg_exists else "fail",
        str(CONFIG_PATH) if cfg_exists else "NOT FOUND",
        "configs/config.yaml" if cfg_exists else "Create configs/config.yaml"
    ))

    # Load config for path checks
    cfg = load_yaml(CONFIG_PATH) if cfg_exists else {}

    # 4. Model file
    model_path_str = (cfg.get("speaker_classification") or {}).get("model_path", "")
    if model_path_str:
        mp = resolve_path(model_path_str)
        exists = mp.exists()
        results.append(CheckResult(
            "XGBoost Model", "ok" if exists else "fail",
            str(mp), "Model file found" if exists else "Model file missing"
        ))

    # 5. .env / HF_TOKEN
    env_path = PROJECT_ROOT / ".env"
    hf_token = os.environ.get("HF_TOKEN", "")
    if env_path.exists():
        results.append(CheckResult("Env File (.env)", "ok", str(env_path)))
    else:
        results.append(CheckResult("Env File (.env)", "warn", "NOT FOUND",
                                   "Optional but recommended for HF_TOKEN"))

    if hf_token:
        results.append(CheckResult("HF_TOKEN", "ok", "Set in environment"))
    else:
        results.append(CheckResult("HF_TOKEN", "warn", "NOT SET",
                                   "Needed for pyannote diarization model download"))

    # 6. raw_audio_root
    raw = (cfg.get("data_ingestion") or {}).get("raw_audio_root", "")
    if raw:
        rp = resolve_path(raw)
        exists = rp.exists()
        results.append(CheckResult(
            "Raw Audio Root", "ok" if exists else "fail",
            str(rp), "" if exists else "Directory does not exist"
        ))

    # 7. processed_audio_root
    proc = (cfg.get("audio_preparation") or {}).get("processed_audio_root", "")
    if proc:
        pp = resolve_path(proc)
        results.append(CheckResult(
            "Processed Audio Root", "ok" if pp.exists() else "warn",
            str(pp), "Exists" if pp.exists() else "Will be created on first run"
        ))

    # 8. output_audio_root
    out = (cfg.get("speaker_classification") or {}).get("output_audio_root", "")
    if out:
        op = resolve_path(out)
        results.append(CheckResult(
            "Output Audio Root", "ok" if op.exists() else "warn",
            str(op), "Exists" if op.exists() else "Will be created on first run"
        ))

    # 9. ffmpeg
    ok, ver = _run_cmd(["ffmpeg", "-version"])
    first_line = ver.split("\n")[0] if ver else ""
    results.append(CheckResult(
        "ffmpeg", "ok" if ok else "warn",
        first_line[:80] if ok else "NOT FOUND",
        "" if ok else "Some audio formats may not work without ffmpeg"
    ))

    # 10. git-lfs
    ok, ver = _run_cmd(["git", "lfs", "version"])
    results.append(CheckResult(
        "git-lfs", "ok" if ok else "warn",
        ver[:60] if ok else "NOT FOUND",
        "" if ok else "Needed if models are stored via LFS"
    ))

    # 11. uv
    ok, ver = _run_cmd(["uv", "version"])
    results.append(CheckResult(
        "uv", "ok" if ok else "warn",
        ver[:60] if ok else "NOT FOUND",
        "" if ok else "Recommended package manager"
    ))

    # 12. VTC repo
    backend = (cfg.get("speaker_classification") or {}).get("backend", "xgb")
    vtc_path_str = (cfg.get("vtc") or {}).get("repo_path", "")
    if vtc_path_str:
        vp = resolve_path(vtc_path_str)
        if backend == "vtc":
            results.append(CheckResult(
                "VTC Repository", "ok" if vp.exists() else "fail",
                str(vp), "Found" if vp.exists() else "Required when backend=vtc"
            ))
        else:
            results.append(CheckResult(
                "VTC Repository", "ok" if vp.exists() else "warn",
                str(vp), "Found" if vp.exists() else "Not needed for xgb backend"
            ))

    # 13. GPU / CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            results.append(CheckResult("GPU / CUDA", "ok", f"Available: {gpu_name}"))
        else:
            results.append(CheckResult("GPU / CUDA", "warn", "CUDA not available",
                                       "Diarization will be slow on CPU"))
    except ImportError:
        results.append(CheckResult("GPU / CUDA", "warn", "torch not importable"))

    # 14. Required scripts
    scripts = [
        "scripts/run_all.sh",
        "scripts/run_stage_01.sh",
        "scripts/run_stage_02_from_parquet.sh",
        "scripts/run_stage_02_single_wav.sh",
        "scripts/run_stage_03.sh",
        "scripts/annotate_ads_ids.py",
    ]
    missing = [s for s in scripts if not (PROJECT_ROOT / s).exists()]
    if missing:
        results.append(CheckResult(
            "Pipeline Scripts", "fail",
            f"{len(missing)} missing: {', '.join(missing)}",
        ))
    else:
        results.append(CheckResult("Pipeline Scripts", "ok", f"All {len(scripts)} scripts present"))

    return results
