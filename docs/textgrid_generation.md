# TextGrid Generation

TextGrids are generated from normalized segment tables rather than from raw audio directly.

Supported inputs:

- VTC `rttm.csv`
- XGB segment parquet or CSV outputs

Run the workflow with:

```bash
uv run bash scripts/run_generate_textgrids.sh --backend vtc --participant-id ABAN141223
uv run bash scripts/run_generate_textgrids.sh --backend xgb --participant-id ABAN141223
```

Normalized label mapping:

- VTC: `FEM -> adult_female`, `MAL -> adult_male`, `KCHI -> key_child`, `OCH -> other_child`
- XGB: `adult_female -> adult_female`, `adult_male -> adult_male`, `child -> key_child` by default, `background -> background`

TextGrids are written to the external `paths.textgrid_output_root`.