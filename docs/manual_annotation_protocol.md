# Manual Annotation Protocol

The active manual annotation workflow is speaker-class annotation for model evaluation.

Run it with:

```bash
uv run bash scripts/run_annotate_segments.sh --backend vtc --participant-id ABAN141223
```

The tool:

- loads normalized segment tables from VTC or XGB outputs
- samples segments per participant using the `annotation` settings in `configs/params.yaml`
- plays each segment from the prepared participant WAV
- supports replay, back, and quit during annotation
- writes resumable CSV outputs to `paths.manual_annotation_root`

Annotation labels:

- `adult_female`
- `adult_male`
- `key_child`
- `other_child`
- `unclear`
- `noise`

ADS/IDS annotation is not part of the supported pipeline workflow.