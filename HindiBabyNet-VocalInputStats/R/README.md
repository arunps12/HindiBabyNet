# HindiBabyNet Vocal Input Stats R Workflow

This directory contains the preregistered mixed-effects analysis workflow for HindiBabyNet. It is intentionally separate from the Python package and reads only the derived public CSV outputs produced by the Python pipeline.

## Inputs

- `data/derived/final_master.csv`
- `data/derived/input_long.csv`
- `data/derived/input_output_long.csv`

## Installation

Use a local R installation plus the packages listed in `R/00_setup.R`. The workflow is designed to work with `renv`.

1. Open the repository root in VS Code.
2. Start an R terminal in the repository root.
3. Restore or install packages.

```r
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
renv::restore(prompt = FALSE)
```

If you are not using `renv`, install the required packages manually:

```r
install.packages(c(
	"tidyverse", "lme4", "lmerTest", "glmmTMB", "performance",
	"DHARMa", "emmeans", "ggeffects", "broom.mixed", "parameters",
	"boot", "ggplot2", "yaml", "knitr"
))
```

If `renv.lock` is not present, `renv::restore()` cannot fully reconstruct the package set by itself. In that case, install the required packages manually into your active library.

## Running In VS Code

Run the workflow from the repository root so that all repo-relative paths resolve correctly.

```r
source("R/00_setup.R")
source("R/01_input_models.R")
source("R/02_input_output_models.R")
source("R/03_model_tables_and_plots.R")
source("R/04_final_report.R")
```

This is the full end-to-end command block requested for routine execution.

## Execution Order

```r
source("R/00_setup.R")
source("R/01_input_models.R")
source("R/02_input_output_models.R")
source("R/03_model_tables_and_plots.R")
source("R/04_final_report.R")
```

## Reproducibility

The workflow includes `.Rprofile`, `renv/activate.R`, and `renv/settings.json` so the R session can bootstrap a local reproducible environment.

## Model And Plot Age Variables

Models use `age_z`.

Prediction grids and plots display `age_days`.

The helper functions in `R/functions/monte_carlo_ci.R` and `R/functions/prediction_plots.R` enforce this mapping by converting between `age_days` and `age_z` with the stored `age_days_mean` and `age_days_sd` values from `final_master.csv`.

## Changing `nsim`

Default Monte Carlo settings are defined in `analysis_options` inside `R/00_setup.R`.

- `analysis_options$nsim_default` is the standard run value.
- `analysis_options$nsim_testing` is the lighter development value.

Example:

```r
source("R/00_setup.R")
analysis_options$nsim_default <- analysis_options$nsim_testing
source("R/03_model_tables_and_plots.R")
```

## Output Locations

- `results/r_models/`: saved fitted-model bundles (`.rds`)
- `results/r_tables/`: dataset summary, selection rationale, fixed effects, evidence summaries, reporting bundle
- `results/r_diagnostics/`: residual, QQ, and diagnostics outputs
- `results/r_predictions/`: plot-ready prediction grids and interval tables
- `results/r_plots/`: publication-style PNG figures
- `results/final_report/`: rendered PDF and HTML reports

## Report Rendering

The workflow uses the Quarto CLI, not the R package `quarto`.

If Quarto is not on `PATH`, set `QUARTO_PATH` to the executable location before running `R/04_final_report.R`.

HTML rendering should work first.

PDF rendering additionally requires a LaTeX engine such as TinyTeX, MiKTeX, or TeX Live. If that dependency is missing, the workflow keeps PDF support but falls back to HTML-only output for smoke testing.

## Reporting Principles

The workflow evaluates hypotheses using effect direction, interval estimates, diagnostics, and uncertainty. It does not rely on p-values alone and avoids causal language. No formal a priori power analysis is required here because the preregistration is feasibility-based.

## Output policy

Generated model objects, rendered reports, plots, Monte Carlo outputs, and other heavy artifacts should remain under `results/` and should not be committed unless explicitly versioned on purpose.
