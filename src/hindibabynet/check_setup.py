"""Environment/config diagnostics for HindiBabyNet."""
from __future__ import annotations

import shutil
from pathlib import Path

from src.hindibabynet.config.configuration import ConfigurationManager


def _status(label: str, ok: bool, msg: str) -> str:
    prefix = "[OK]" if ok else "[ERROR]"
    return f"{prefix} {label}: {msg}"


def _warn(label: str, msg: str) -> str:
    return f"[WARN] {label}: {msg}"


def main() -> None:
    lines: list[str] = []
    has_error = False

    try:
        cfg = ConfigurationManager()
        lines.append(_status("Config", True, f"loaded {cfg.config_path}"))
    except Exception as exc:
        print(_status("Config", False, str(exc)))
        raise SystemExit(1)

    # Common checks
    try:
        di_cfg = cfg.get_data_ingestion_config(run_id="setup_check")
        lines.append(_status("Raw audio root", di_cfg.raw_audio_root.exists(), str(di_cfg.raw_audio_root)))
        has_error |= not di_cfg.raw_audio_root.exists()
    except Exception as exc:
        lines.append(_status("Raw audio root", False, str(exc)))
        has_error = True

    try:
        processed_root = cfg.get_processed_audio_root()
        processed_root.mkdir(parents=True, exist_ok=True)
        lines.append(_status("Processed audio root", True, str(processed_root)))
    except Exception as exc:
        lines.append(_status("Processed audio root", False, str(exc)))
        has_error = True

    try:
        output_root = cfg.get_classification_output_root()
        output_root.mkdir(parents=True, exist_ok=True)
        lines.append(_status("Classification output root", True, str(output_root)))
    except Exception as exc:
        lines.append(_status("Classification output root", False, str(exc)))
        has_error = True

    # XGB checks
    xgb = cfg.get_xgb_params()
    model_path = Path(xgb.get("model_path", "models/xgb_egemaps.pkl"))
    lines.append(_status("XGB model", model_path.exists(), str(model_path)))
    has_error |= not model_path.exists()

    diar_model = str(xgb.get("diarization_model", "")).strip()
    if diar_model:
        lines.append(_status("Pyannote model", True, diar_model))
    else:
        lines.append(_status("Pyannote model", False, "speaker_classification.xgb.diarization_model missing"))
        has_error = True

    hf_token = (Path(".env").read_text(encoding="utf-8") if Path(".env").exists() else "")
    if "HF_TOKEN=" in hf_token:
        lines.append(_status("HF token", True, "Found in .env"))
    else:
        lines.append(_warn("HF token", "Not found in .env (needed for pyannote models)"))

    # VTC checks
    vtc = cfg.get_vtc_params()
    vtc_repo = Path(vtc.get("repo_path", "external_models/VTC"))
    infer_script = vtc_repo / "scripts" / "infer.py"
    lines.append(_status("VTC repo", vtc_repo.exists(), str(vtc_repo)))
    lines.append(_status("VTC infer script", infer_script.exists(), str(infer_script)))
    has_error |= not vtc_repo.exists() or not infer_script.exists()

    uv_bin = shutil.which("uv")
    ffmpeg_bin = shutil.which("ffmpeg")
    lines.append(_status("uv executable", uv_bin is not None, uv_bin or "not found in PATH"))
    lines.append(_status("ffmpeg executable", ffmpeg_bin is not None, ffmpeg_bin or "not found in PATH"))
    has_error |= uv_bin is None or ffmpeg_bin is None

    vtc_env = (vtc_repo / ".venv").exists()
    if vtc_env:
        lines.append(_status("VTC environment", True, str(vtc_repo / ".venv")))
    else:
        lines.append(_warn("VTC environment", f"Expected virtual env at {vtc_repo / '.venv'}"))

    print("\n".join(lines))
    raise SystemExit(1 if has_error else 0)


if __name__ == "__main__":
    main()
