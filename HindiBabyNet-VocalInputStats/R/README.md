# HindiBabyNet Vocal Input Stats R Workflow

This directory contains the preregistered mixed-effects analysis workflow for HindiBabyNet. It is intentionally separate from the Python package and reads only the derived public CSV outputs produced by the Python pipeline.

## Status

Phase 1 scaffold only. The files in this directory are parseable placeholders that define the analysis structure, required packages, report inputs, and execution order. Statistical implementation is added incrementally in later phases.

## Planned inputs

- `data/derived/final_master.csv`
- `data/derived/input_long.csv`
- `data/derived/input_output_long.csv`

## Planned execution order

```r
source("R/00_setup.R")
source("R/01_input_models.R")
source("R/02_input_output_models.R")
source("R/03_model_tables_and_plots.R")
source("R/04_final_report.R")
```

## Reproducibility

The intended R environment manager is `renv`. The scaffold includes `renv/activate.R`, `.Rprofile`, and `renv/settings.json`. Later phases will add fuller restore and verification guidance once the workflow logic is implemented.

## Output policy

Generated model objects, rendered reports, plots, Monte Carlo outputs, and other heavy artifacts should remain under `results/` and should not be committed unless explicitly versioned on purpose.
