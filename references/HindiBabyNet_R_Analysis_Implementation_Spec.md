# Implementation Specification: HindiBabyNet-VocalInputStats R Analysis Workflow

> This document specifies the R statistical analysis workflow to be
> added to the existing Python project. Do **not** modify the Python
> package. Add only R code, documentation, and supporting files.

## Objectives

-   Implement the preregistered statistical analyses in R.
-   Read the Python-generated datasets:
    -   `data/derived/final_master.csv`
    -   `data/derived/input_long.csv`
    -   `data/derived/input_output_long.csv`
-   Keep all paths relative.
-   Generate reproducible models, diagnostics, tables,
    publication-quality figures, Monte Carlo confidence intervals, and a
    final PDF report.

## Directory Structure

``` text
R/
  README.md
  research_questions.yaml
  00_setup.R
  functions/
    data_prep.R
    model_selection.R
    model_diagnostics.R
    monte_carlo_ci.R
    prediction_plots.R
    reporting.R
    final_report.R
  01_input_models.R
  02_input_output_models.R
  03_model_tables_and_plots.R
  04_final_report.R

results/
  r_models/
  r_tables/
  r_diagnostics/
  r_predictions/
  r_plots/
  final_report/
```

## Required Packages

-   tidyverse
-   lme4
-   lmerTest
-   glmmTMB
-   performance
-   DHARMa
-   emmeans
-   ggeffects
-   broom.mixed
-   parameters
-   merTools
-   boot
-   ggplot2
-   quarto (or rmarkdown)

## Data Preparation

-   Read all three derived CSV files.
-   Convert categorical variables to factors.
-   Keep `age_days` and `age_z`.
-   Models use `age_z`.
-   Figures display `age_days`.
-   Store:
    -   `age_days_mean`
    -   `age_days_sd`
-   Create:
    -   `input_duration_min_hour`
    -   `key_child_duration_min_hour`

## Model Selection

Automatically inspect each response variable.

Decision rules:

-   Approximately normal → LMM (`lmer`)
-   Skewed positive → log-transform + LMM
-   Strongly skewed positive → optional Gamma GLMM (`glmmTMB`)
-   Save justification for chosen model.

## Diagnostics

Produce:

-   Residual vs fitted
-   QQ plot
-   Heteroscedasticity checks
-   Singular fit checks
-   Convergence checks
-   DHARMa diagnostics for GLMMs

Save diagnostics to `results/r_diagnostics`.

## Monte Carlo Confidence Intervals

Default:

-   seed = 123
-   nsim = 1000

Testing:

-   nsim = 100

Optional robustness:

-   2000--5000 simulations

Use `bootMer()` where possible.

For each prediction:

-   mean
-   lower 2.5%
-   upper 97.5%

Save prediction grids and CI tables.

## Prediction Plots

Always:

-   fit using `age_z`
-   predict using `age_z`
-   plot using `age_days`

Create reusable plotting functions:

1.  Estimated marginal means by speaker.
2.  Age × speaker fitted lines with Monte Carlo CI ribbon.
3.  Input → output prediction plots with Monte Carlo CI ribbon.
4.  Age → key-child prediction plots.
5.  Forest plots of fixed effects.

## Preregistered Models

### Model Family 1

Count:

`input_count_hour ~ speaker * age_z + SES + child_sex + (1|participant_id) + (1|Location)`

Duration:

`input_duration_min_hour ~ speaker * age_z + SES + child_sex + (1|participant_id) + (1|Location)`

### Model Family 2

Count:

`key_child_count_hour ~ input_count_hour * speaker + age_z + SES + child_sex + (1|participant_id) + (1|Location)`

Duration:

`key_child_duration_min_hour ~ input_duration_min_hour * speaker + age_z + SES + child_sex + (1|participant_id) + (1|Location)`

## Research Questions

Store the preregistered research questions and hypotheses in:

`R/research_questions.yaml`

The final report should read this YAML automatically.

## Final Report

Generate:

-   PDF
-   optional HTML

Include:

-   Dataset summary
-   Model selection rationale
-   Diagnostics
-   Fixed effects
-   Monte Carlo confidence intervals
-   Publication-quality figures
-   Interpretation for each research question
-   Evidence-based hypothesis evaluation
-   Overall conclusions

## Evidence-Based Hypothesis Evaluation

Do **not** classify hypotheses using only p-values.

Use:

-   Effect size
-   Effect direction
-   95% Monte Carlo confidence interval
-   Statistical uncertainty
-   Diagnostics
-   Consistency with preregistered hypothesis

Evidence categories:

-   Strong evidence supporting the hypothesis
-   Moderate evidence supporting the hypothesis
-   Weak evidence supporting the hypothesis
-   Inconclusive evidence
-   Evidence inconsistent with the hypothesis

Avoid causal language and avoid overstating conclusions.

## README

Document:

-   Installation
-   Running in VS Code
-   Required packages
-   Execution order
-   Changing `nsim`
-   Output locations
-   Relationship between `age_z` (model) and `age_days` (plots)

State that no formal a priori power analysis is required because the
preregistration specifies a feasibility-based sample.

## Git Workflow

After implementation:

``` bash
git add .
git commit -m "Add preregistered R mixed-effects workflow with Monte Carlo CI reporting"
git push
```
