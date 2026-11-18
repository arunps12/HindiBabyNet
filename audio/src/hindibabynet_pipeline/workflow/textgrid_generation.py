from __future__ import annotations

from pathlib import Path

from hindibabynet_pipeline.components.textgrid.csv_to_textgrid import load_segment_table
from hindibabynet_pipeline.components.textgrid.textgrid_utils import write_textgrid
from hindibabynet_pipeline.config.configuration import ConfigurationManager


DEFAULT_TIER_MAP = {
    "adult_female": "adult_female",
    "adult_male": "adult_male",
    "key_child": "key_child",
    "other_child": "other_child",
    "child": "child",
    "background": "background",
}


def _discover_inputs(classification_root: Path, backend: str) -> list[tuple[str, Path]]:
    backend_root = classification_root / backend.lower()
    if backend.lower() == "vtc":
        return [(participant_dir.name, participant_dir / "rttm.csv") for participant_dir in sorted(backend_root.iterdir()) if participant_dir.is_dir() and (participant_dir / "rttm.csv").is_file()]
    return [
        (participant_dir.name, participant_dir / f"{participant_dir.name}_segments.parquet")
        for participant_dir in sorted(backend_root.iterdir())
        if participant_dir.is_dir() and (participant_dir / f"{participant_dir.name}_segments.parquet").is_file()
    ]


def generate_textgrids(
    *,
    backend: str,
    participant_id: str | None = None,
    input_file: str | Path | None = None,
    limit: int | None = None,
    child_label: str = "key_child",
) -> list[Path]:
    cfg = ConfigurationManager()
    output_root = cfg.get_textgrid_output_root()
    classification_root = cfg.get_classification_output_root()

    if input_file is not None:
        pid = participant_id or Path(input_file).parent.name or Path(input_file).stem
        inputs = [(pid, Path(input_file))]
    else:
        inputs = _discover_inputs(classification_root, backend)
        if participant_id is not None:
            inputs = [item for item in inputs if item[0] == participant_id]

    if limit is not None:
        inputs = inputs[:limit]

    textgrid_paths: list[Path] = []
    for pid, segment_path in inputs:
        df = load_segment_table(segment_path, backend=backend, child_label=child_label)
        duration_sec = float(df["end_sec"].max()) if not df.empty else 0.0
        out_dir = output_root / pid
        out_path = out_dir / f"{pid}.TextGrid"
        write_textgrid(df=df, duration_sec=duration_sec, out_path=out_path, tier_map=DEFAULT_TIER_MAP)
        textgrid_paths.append(out_path)

    return textgrid_paths