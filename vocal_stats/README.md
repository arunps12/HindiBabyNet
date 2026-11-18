# Vocal input and output analysis

This folder contains two related workflows: Python dataset preparation (`vocal-stats`) and R statistical modelling (`vocal-models`). They share configured data locations but remain separate launcher commands.

Python package: `hindibabynet-vocalinputstats`; import name: `hindibabynet_vocalinputstats`. Existing data, private mappings, native configurations, environment, and results remain at their original ignored asset locations, not in this source folder.

## Methods and inputs

Python combines participant metadata, VTC segment tables, and recording durations. It builds participant-level and long-format tables, validation audits, descriptive tables, and plots. VTC labels map to adult female (`FEM`), adult male (`MAL`), other child (`OCH`), and key child (`KCHI`). Adult/other-child speech provides input measures; key-child speech provides output measures. XGB is not an interchangeable input.

Rates use the **full recording duration**, never the sum of VTC speech segments. The approved configuration selects whether duration comes from a prepared recording or raw audio sessions; raw-session mode applies the configured date/session selection. Preserve that selection and its relationship to the classified recording.

- Count/hour = segment count divided by recording hours.
- Duration/hour = speaker seconds divided by recording hours (seconds/hour).
- Duration in minutes/hour = duration/hour divided by 60.
- Parallel count `*_min_hour` columns also divide the corresponding count/hour by 60; check the column's measure rather than assuming every similarly named column is a duration.

R fits the configured/preregistered input and input–output model families, with effective age terms, transforms, contrasts, random effects, model selection/fallbacks, diagnostics, prediction simulations, tables, and plots. These details are not identical to the separate prosody statistics analysis. Preserve approved `HBN_*` overrides as well as configuration values, and inspect saved requested/fitted formulas and warnings.

## Setup and configuration

Complete the [root installation](../README.md). Python needs its own uv environment with a compatible Python 3.10–3.12 interpreter. Do not merge its NumPy dependencies with audio/prosody.

In the [local configuration](../config.local.yaml):

- Set both vocal workflow `project` values to `vocal_stats`, enabling only the stages needed.
- For an existing study, keep the approved `vocal` section and original asset-based `project_root`; a new root-template copy uses `project_root: "."`.
- Review metadata, VTC, audio, and optional manual-mapping inputs. Preserve duration source, session selection, identity mapping, age/SES definitions, and inclusion behavior.
- Set fresh `derived_data_dir`, `private_data_dir`, `figures_dir`, `tables_dir`, and `results_dir`. R reads those derived datasets and writes under the shared results directory.

The full [root template](../config.example.yaml) includes the shared `vocal` mapping and current source paths. Edit paths and enable the intended commands in the single root local configuration; no duplicate Python/R path mappings are needed. Supply private inputs separately. Use paths valid for Windows or WSL; they are not automatically translated. See [scientific boundaries](../docs/README.md#scientific-and-privacy-boundaries) before changing settings.

## R packages

Install R separately and make `Rscript` available, or set `tools.rscript`. Manage the vocal analysis's R library independently from prosody statistics. Root `setup` does not install R packages. Use approved compatible versions; existing renv scaffolding is not a complete version lock or a reason to restore/upgrade packages blindly.

The required packages declared in [R/00_setup.R](R/00_setup.R) are shown below. In an R session, install missing compatible versions as appropriate for your environment:

```r
install.packages(c(
  "tidyverse", "lme4", "lmerTest", "glmmTMB", "performance", "DHARMa",
  "emmeans", "ggeffects", "broom.mixed", "parameters", "boot", "ggplot2",
  "yaml", "knitr", "car"
))
```

This command uses your configured package repository; it is not a version lock. Record the R version and installed package versions for the actual analysis. Quarto is an additional requirement **only** for the optional vocal report; it is not needed to fit the models. `testthat` is a separate dependency for R tests.

## Run from the repository root

After explicit enablement and fresh output allocation:

```text
uv run python run.py setup
uv run python run.py vocal-stats --dry-run
uv run python run.py vocal-stats
uv run python run.py vocal-models
```

Run R modelling only after the configured Python-derived inputs exist. Alternatively point R at approved existing derived data and leave Python preprocessing disabled. Root `all` respects enabled workflows and runs preprocessing before modelling.

The default Python sequence is `build-master` → `create-long` → `eda` → `plots`. Select one with `--stage`, for example:

```text
uv run python run.py vocal-stats --stage build-master
```

Do not use separate stages as an automatic in-place recipe: conservative ownership guards can refuse their shared outputs. Prefer `vocal-stats` for a new complete build.

The default R command sources [setup](R/00_setup.R), [input models](R/01_input_models.R), [input–output models](R/02_input_output_models.R), and [tables/plots](R/03_model_tables_and_plots.R) in one R process. For advanced stage selection, use an exact script basename; setup alone is not a completed scientific stage. Retain approved simulation counts/seeds and inspect model diagnostics rather than changing them to make a run pass.

### Optional HTML report

Once model tables, diagnostics, predictions, and the reporting bundle exist:

```text
uv run python run.py vocal-models --stage report
```

This invokes [R/04_final_report.R](R/04_final_report.R) with HTML output using the [Quarto template](R/report.qmd). Install Quarto separately; `QUARTO_PATH` can specify its executable. The root report stage accepts HTML only; it is not run by `all`. Destination and temporary report filenames must pass output guards. Legacy PDF templates are not a requirement for this command.

## Outputs and privacy

Configured destinations hold participant master/long tables, duration and validation audits, descriptive tables/figures, and a private participant lookup. Under `results_dir`, R uses `r_models`, `r_tables`, `r_diagnostics`, `r_predictions`, `r_plots`, and optional `final_report` subdirectories.

Pseudonymization is not publication approval. Original identifiers and their lookup must remain private, and all derived tables, aggregate results, plots, fitted models, reports, and notebooks remain local/ignored too. Review diagnostics and participant exclusions before interpreting model results.

## Testing

```text
uv run --project vocal_stats --frozen --no-sync pytest
```

R checks live alongside the workflow tests and need the relevant R testing/model packages. Run only the intended tests in an approved isolated environment; passing synthetic Python tests does not validate R fits or full-cohort equivalence. For stale/partial output errors, retain existing results and choose fresh destinations; see [safe recovery](../docs/README.md#outputs-and-recovery).
