from __future__ import annotations

from pathlib import Path

from hindibabynet_pipeline.components.annotation.interactive_annotator import run_interactive_annotation
from hindibabynet_pipeline.components.annotation.segment_sampler import sample_segments
from hindibabynet_pipeline.components.textgrid.csv_to_textgrid import load_segment_table
from hindibabynet_pipeline.config.configuration import ConfigurationManager


def _discover_inputs(classification_root: Path, backend: str) -> list[tuple[str, Path]]:
    backend_root = classification_root / backend.lower()
    if backend.lower() == "vtc":
        return [(participant_dir.name, participant_dir / "rttm.csv") for participant_dir in sorted(backend_root.iterdir()) if participant_dir.is_dir() and (participant_dir / "rttm.csv").is_file()]
    return [
        (participant_dir.name, participant_dir / f"{participant_dir.name}_segments.parquet")
        for participant_dir in sorted(backend_root.iterdir())
        if participant_dir.is_dir() and (participant_dir / f"{participant_dir.name}_segments.parquet").is_file()
    ]


def annotate_segments(
    *,
    backend: str,
    participant_id: str | None = None,
    input_file: str | Path | None = None,
    n_segments: int | None = None,
    random_seed: int | None = None,
    child_label: str = "key_child",
) -> list[Path]:
    cfg = ConfigurationManager()
    annotation_params = cfg.get_annotation_params()
    sample_n = n_segments or int(annotation_params.get("n_segments_per_participant", 50))
    seed = random_seed if random_seed is not None else int(annotation_params.get("random_seed", 42))
    output_root = cfg.get_manual_annotation_root()
    prepared_audio_root = cfg.get_processed_audio_root()
    classification_root = cfg.get_classification_output_root()

    if input_file is not None:
        pid = participant_id or Path(input_file).parent.name or Path(input_file).stem
        inputs = [(pid, Path(input_file))]
    else:
        inputs = _discover_inputs(classification_root, backend)
        if participant_id is not None:
            inputs = [item for item in inputs if item[0] == participant_id]

    output_paths: list[Path] = []
    for pid, segment_path in inputs:
        audio_path = prepared_audio_root / pid / f"{pid}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Prepared audio not found for participant {pid}: {audio_path}")

        df = load_segment_table(segment_path, backend=backend, child_label=child_label).copy()
        df.insert(0, "segment_index", range(len(df)))
        df.insert(1, "participant_id", pid)
        sampled_df = sample_segments(df, n_segments=sample_n, random_seed=seed)
        output_csv = output_root / pid / "speaker_class_annotations.csv"
        output_paths.append(run_interactive_annotation(sampled_df, audio_path=audio_path, output_csv=output_csv))

    return output_paths