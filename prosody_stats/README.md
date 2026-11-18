# Prosody statistical analysis

This R workflow analyses participant-by-speaker acoustic summaries. It reads prosody features and vocal input/output tables, validates their joins, fits two model families, and saves diagnostics, predictions, tables, and figures. It does not extract audio features or train speech models.

Code now lives here. Existing native configuration, data, fitted models, and outputs remain at their original ignored asset locations. The analysis reads upstream files; it does not source the other statistical project's R helpers or share its exact model specification.

## Inputs and scientific methods

Configure the intended versions of:

- Participant-by-speaker acoustic features from the [prosody workflow](../prosody/README.md), including pitch in Hz/semitones and wav2vec 2.0/HuBERT PCA scores.
- Participant master, input-long, and input–output-long tables from [vocal preprocessing](../vocal_stats/README.md).
- Participant lookup and cleaned metadata needed for the configured merges.

All files must describe the intended cohort and compatible participant/speaker identities. Acoustic tables include source and analysis identifiers; they are not public data. PCA and standardization depend on the extraction cohort. Preserve the approved pitch semitone reference, feature definitions, speaker selection, and lookup relationships.

Preparation checks merge keys, labels, repeated child outcomes, missingness, and complete-case counts. Family 1 models acoustic measures by speaker and configured age terms with covariates and participant effects. Family 2 relates key-child vocal outcomes to acoustic measures, speaker, vocal input, and configured covariates/effects. Exact formulas, age specifications, outcome units, primary/sensitivity measures, reference levels, and fallback rules come from the approved configuration, code, and effective `HBN_PROSODY_*` overrides—not a universal formula in this guide.

Preserve the configured ordered SES scoring and all analysis options. Do not copy vocal-model transforms, random effects, or robustness defaults into this workflow. Inspect saved requested/fitted formulas, merge audits, model manifests, and warnings before interpretation.

## Setup and configuration

Complete the [root installation](../README.md), including root uv dependencies for the launcher. R models themselves do not require a Python environment here. Manage this analysis's R library separately from vocal modelling; the shared launcher does not merge their scientific environments.

In the [local configuration](../config.local.yaml):

1. Set `workflows.prosody-stats.project` to `prosody_stats` and enable it only when ready.
2. For an existing study, preserve the approved `prosody_stats` section and original asset-based `project_root`; a new root-template copy uses `project_root: "."`.
3. Review all six input path mappings: `acoustic_features`, `final_master`, `input_long`, `input_output_long`, `participant_lookup`, and `metadata_cleaned`.
4. Allocate fresh `paths.data_derived` and every `paths.outputs_*` destination, including reports. Never redirect writes into existing models/tables.

The [root template](../config.example.yaml) includes the full native prosody-statistics mapping and current source paths. Edit input/output paths and enable the workflow in the single root local configuration; no separate native YAML is needed. Supply authorized upstream tables separately. The template is not approval to change scientific settings; see [scientific boundaries](../docs/README.md#scientific-and-privacy-boundaries).

## R packages

Install R and make `Rscript` available on `PATH`, or set `tools.rscript`. Keep a compatible known-working R/package combination and record its versions. No complete R lockfile establishes the environment; do not assume that installing the latest R or running an unconfigured restore reproduces prior models.

The exact required list in [R/00_setup.R](R/00_setup.R) is:

```r
install.packages(c(
  "dplyr", "tidyr", "readr", "stringr", "purrr", "tibble", "ggplot2",
  "yaml", "readxl", "lme4", "lmerTest", "glmmTMB", "performance", "DHARMa",
  "emmeans", "ggeffects", "broom.mixed", "parameters", "boot", "knitr",
  "rmarkdown", "car"
))
```

Run installation in an R session and select compatible versions/repositories as needed; this is a package list, not a version lock. Root `setup` does not install R libraries. `testthat` is additionally needed for tests. Pandoc is an extra system requirement for the optional HTML report, not for the model sequence; Quarto and LaTeX are not required for this report stage.

## Run from the repository root

With enabled workflow, available inputs/packages, and fresh destinations:

```text
uv run python run.py prosody-stats --dry-run
uv run python run.py prosody-stats
```

The default sequence sources [setup](R/00_setup.R), [preparation](R/01_prepare_prosody_data.R), [Family 1 models](R/02_fit_family1_models.R), [Family 2 models](R/03_fit_family2_models.R), and [tables/plots](R/04_create_tables_plots.R) in a single R process.

For advanced use, `--stage` takes an exact script basename, for example:

```text
uv run python run.py prosody-stats --stage 01_prepare_prosody_data.R
```

Setup alone is not a completed scientific stage. Separate stages may share occupied write roots and be refused; prefer the whole workflow command rather than chaining stages in place. To use approved existing upstream data, point to it explicitly and leave upstream workflows disabled.

### Optional HTML report

After successful model/table generation, with Pandoc available and a fresh report destination:

```text
uv run python run.py prosody-stats --stage report
```

The launcher renders the [R Markdown report](HindiBabyNet_prosody_stat_model_report.Rmd) as HTML with `run_workflow=FALSE`, reading existing model outputs rather than rerunning the analysis. It writes report/intermediate files to the configured reports directory and does not run as part of `all`. Manual knitting uses template defaults that can rerun analysis; do not use it as a safe substitute without reviewing those parameters and write locations.

## Outputs, testing, and safety

Configured destinations contain the prepared prosody-long dataset, merge/missingness audits, model bundles, model-family manifests, fixed effects/fit statistics, diagnostics, prediction intervals, figures, and optional reports. Keep them private and ignored, including pseudonymized or aggregate results.

The R test entry point is [tests/testthat.R](tests/testthat.R). With its packages installed, run from this source folder:

```text
Rscript tests/testthat.R
```

Synthetic tests do not establish full-cohort equivalence or validate every external feature source. Root resume skips only matching successful whole stages; stale, incomplete, or unowned occupied results require fresh paths, never an overwrite flag. See the [operating guide](../docs/README.md) for recovery and privacy limitations.



