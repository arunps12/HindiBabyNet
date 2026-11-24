"""Read config.yaml and give the data folders to all HindiBabyNet workflows.

The config file has one base folder (data_dir). Every data folder has a default
place under it. A folder stored somewhere else is listed under ``paths``.
Relative paths start from the folder where config.yaml is saved.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default place of every folder under data_dir.
DEFAULT_FOLDERS = {
    "raw_audio": "raw/audio",
    "raw_cdi": "raw/cdi",
    "metadata": "raw/metadata",
    "prepared_audio": "derived/prepared-audio",
    "vtc": "derived/vtc-classification",
    "xgb": "derived/xgb-classification",
    "prosody_features": "derived/prosody-features",
    "statistical_data": "derived/statistical-data",
    "textgrids": "derived/textgrids",
    "annotations": "derived/manual-annotations",
    "results": "results",
    "cache": "cache",
}
# Single files that can be stored apart from their default folder.
DEFAULT_FILES = {"word_mapping": ("raw_cdi", "word_mapping.csv")}


def local_path(value: str | Path, base: Path, wsl_distro: str = "Ubuntu") -> Path:
    """Return a path that works on this computer (Windows or Ubuntu/WSL)."""
    text = os.path.expanduser(str(value).strip()).replace("\\", "/")
    drive = re.match(r"^([A-Za-z]):/(.*)$", text)
    mounted = re.match(r"^/mnt/([A-Za-z])/(.*)$", text)
    if os.name == "nt":
        if mounted:
            text = f"{mounted.group(1).upper()}:/{mounted.group(2)}"
        elif text.startswith("/") and not text.startswith("//"):
            text = f"//wsl.localhost/{wsl_distro}{text}"
    elif drive:
        text = f"/mnt/{drive.group(1).lower()}/{drive.group(2)}"
    elif text.startswith("//"):
        raise ValueError(
            f"The network path {value} cannot be opened from Ubuntu directly. "
            "Mount it in Ubuntu first and write the mounted path in config.yaml."
        )
    absolute = text.startswith("/") or re.match(r"^[A-Za-z]:/", text) is not None
    return Path(text) if absolute else Path(os.path.normpath(Path(base) / text))


class Config:
    """Paths and local choices from config.yaml."""

    def __init__(self, config_path: str | Path | None = None):
        chosen = config_path or os.environ.get("HBN_CONFIG") or REPO_ROOT / "config.yaml"
        self.file = Path(chosen).resolve()
        if not self.file.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {self.file}. "
                "Copy config.example.yaml to config.yaml and change the paths."
            )
        with self.file.open(encoding="utf-8-sig") as handle:
            self.values = yaml.safe_load(handle) or {}
        if not isinstance(self.values, dict) or not self.values.get("data_dir"):
            raise ValueError(f"{self.file} must contain data_dir.")
        self.wsl_distro = str(self.values.get("wsl_distro") or "Ubuntu")
        self.data_dir = local_path(self.values["data_dir"], self.file.parent, self.wsl_distro)

    def path(self, key: str, *parts: str) -> Path:
        """Folder (or file) for one key, with optional sub-folders or a file name."""
        override = (self.values.get("paths") or {}).get(key)
        if override:
            base = local_path(override, self.file.parent, self.wsl_distro)
        elif key in DEFAULT_FILES:
            folder, name = DEFAULT_FILES[key]
            base = self.path(folder, name)
        elif key in DEFAULT_FOLDERS:
            base = self.data_dir / DEFAULT_FOLDERS[key]
        else:
            raise KeyError(f"Unknown path name in config: {key}")
        return base.joinpath(*parts)

    def input(self, key: str, *parts: str) -> Path:
        """Like path(), but the file or folder must exist."""
        found = self.path(key, *parts)
        if not found.exists():
            raise FileNotFoundError(
                f"Not found: {found}\nCheck data_dir and paths.{key} in {self.file}."
            )
        return found

    def results(self, workflow: str, *parts: str) -> Path:
        """Results folder of one workflow, for example results('cdi')."""
        return self.path("results", workflow, *parts)

    def setting(self, section: str, key: str, default=None):
        """One local choice, for example setting('audio', 'vtc_device', 'cpu')."""
        return (self.values.get(section) or {}).get(key, default)


def new_output(path: str | Path) -> Path:
    """Return the path for a new output file. Existing files are never replaced."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"{target} already exists and is not replaced. "
            "Move or rename the old file, or choose another output folder in config.yaml."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
