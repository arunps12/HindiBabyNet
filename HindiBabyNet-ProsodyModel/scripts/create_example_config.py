"""Write a default config file from the checked-in example."""

from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "config" / "config.example.yaml"
    target = repo_root / "config" / "config.yaml"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {target}")
    shutil.copyfile(source, target)
    print(target)


if __name__ == "__main__":
    main()
