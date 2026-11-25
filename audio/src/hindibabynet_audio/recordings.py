"""Names of the prepared recordings.

A prepared recording is one WAV file in the prepared audio folder. Its name
(the recording ID) is the place of the file without the file ending:

    P001.wav                      ->  P001
    REC_0001.WAV                  ->  REC_0001
    P001/20250323/REC_0001.WAV    ->  P001/20250323/REC_0001
    P001/P001.wav                 ->  P001   (one folder per recording)

The classification results of a recording are saved in a folder with this name.
"""

from __future__ import annotations

from pathlib import Path


def recording_id(relative: str | Path) -> str:
    """Recording ID of a prepared WAV file, from its path inside the prepared audio folder."""
    relative = Path(relative)
    parts = [*relative.parts[:-1], relative.stem]
    if len(parts) == 2 and parts[0] == parts[1]:
        parts = parts[:1]
    return "/".join(parts)


def prepared_recordings(prepared_root: str | Path) -> list[tuple[str, Path]]:
    """All prepared WAV files as (recording ID, file), in the order of the IDs.

    Two files with the same ID (also when only capital letters differ) are an error,
    because their results would be saved in the same folder.
    """
    root = Path(prepared_root)
    if not root.is_dir():
        return []
    found: dict[str, tuple[str, Path]] = {}
    for path in sorted(root.rglob("*")):
        if not (path.is_file() and path.suffix.lower() == ".wav"):
            continue
        name = recording_id(path.relative_to(root))
        if name.lower() in found:
            raise ValueError(
                f"Two prepared files have the same name {name}: "
                f"{found[name.lower()][1]} and {path}. Keep only one of them."
            )
        found[name.lower()] = (name, path)
    return sorted(found.values())


def select(items: list, name: str | None, limit: int | None, key=lambda item: item[0]) -> list:
    """One recording or participant (all recordings below this name), or the first N items."""
    if name is not None:
        items = [item for item in items if key(item) == name or key(item).startswith(name + "/")]
    if limit is not None:
        items = items[:limit]
    return items
