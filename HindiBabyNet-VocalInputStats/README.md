# HindiBabyNet-VocalInputStats

This repository builds privacy-preserving HindiBabyNet vocal input and vocal output datasets, generates descriptive EDA tables and figures, and runs the preregistered R mixed-effects analysis workflow through a final Quarto report.

## Full workflow overview

The project has two main stages.

### Python and `uv` stage

The Python workflow:

1. Reads participant metadata, VTC outputs, and full-audio durations.
2. Builds the participant-level master dataset.
3. Creates long-format analysis datasets.
4. Produces descriptive EDA tables.
5. Produces descriptive EDA figures.

Main Python outputs:

- `data/derived/final_master.csv`
- `data/derived/input_long.csv`
- `data/derived/input_output_long.csv`
- descriptive tables in `tables/`
- descriptive figures in `figures/`

### R stage

The R workflow:

1. Reads the Python-derived public datasets.
2. Fits the preregistered mixed-effects models.
3. Runs diagnostics.
4. Computes Monte Carlo confidence intervals for fitted predictions.
5. Produces fitted model plots and summary tables.
6. Renders the final Quarto statistical report.

Main R outputs:

- fitted model bundles in `results/r_models/`
- model tables in `results/r_tables/`
- diagnostic outputs in `results/r_diagnostics/`
- prediction grids and interval tables in `results/r_predictions/`
- fitted model figures in `results/r_plots/`
- final report files in `results/final_report/`

## Purpose

The workflow is designed for count-per-hour and duration-per-hour analyses outside the main HindiBabyNet pipeline repository. It uses full recording duration as the denominator because the entire recording was processed by VTC.

Input speakers:

- `adult_female`
- `adult_male`
- `other_child`

Output speaker:

- `key_child`

VTC labels are normalized as follows:

- `FEM -> adult_female`
- `MAL -> adult_male`
- `KCHI -> key_child`
- `OCH -> other_child`

## Input requirements

The workflow expects:

1. A metadata table in `.csv`, `.xlsx`, or `.xls` format with at least `par_id`, `REC_date`, `birthdate`, `child_sex`, `mother_education`, `father_education`, and `Location`.
2. A VTC output root containing one participant folder per child with `rttm.csv` files.
3. An audio root containing the full recording processed by VTC.

## Privacy model

- Public outputs use only anonymized `participant_id` values such as `P001`.
- Original participant IDs are stored only in `data/private/participant_lookup.csv`.
- The public derived datasets and R outputs should not contain original participant identifiers.

## Python setup and run commands

Install `uv` first:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a terminal in the project folder:

```powershell
Set-Location "C:/Users/arunps/OneDrive/Projects/HindiBabyNet/HindiBabyNet-VocalInputStats"
```

Create or sync the local Python environment:

```bash
uv sync
```

Confirm that the package CLI is available:

```bash
uv run hindibabynet-vocalinputstats --help
```

Run the complete Python workflow:

```bash
uv run hindibabynet-vocalinputstats all --config configs/config.yaml
```

Run individual Python steps if needed:

```bash
uv run hindibabynet-vocalinputstats build-master --config configs/config.yaml
uv run hindibabynet-vocalinputstats create-long --config configs/config.yaml
uv run hindibabynet-vocalinputstats eda --config configs/config.yaml
uv run hindibabynet-vocalinputstats plots --config configs/config.yaml
```

## Configuration

Edit [configs/config.yaml](c:/Users/arunps/OneDrive/Projects/HindiBabyNet/HindiBabyNet-VocalInputStats/configs/config.yaml) to point at your local inputs.

Key settings include:

- `metadata_path` or `metadata_csv`
- `metadata_id_column`
- `vtc_output_root`
- `audio_root`
- `derived_data_dir`
- `private_data_dir`
- `figures_dir`
- `tables_dir`
- `results_dir`
- `audio_layout`
- `vtc_layout`
- `participant_id_digits`
- `age_month_denominator`
- `ses_source`
- `minimum_recording_hours_warning`

## R setup in VS Code

To run the R analysis workflow in VS Code:

1. Install R on your machine.
2. Install Quarto.
3. Install the VS Code R extension.
4. Open the project root folder in VS Code.
5. Use the R terminal from the project root so all paths stay repo-relative.

If `Rscript` is not on `PATH`, you can still run the workflow by calling the full executable path directly.

If Quarto is not on `PATH`, set `QUARTO_PATH` to the Quarto executable location before rendering the report.

## R package setup

If you are using `renv`, run:

```r
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
renv::restore()
```

If `renv.lock` is not present or `renv::restore()` cannot fully reproduce the environment, install the packages listed in [R/README.md](c:/Users/arunps/OneDrive/Projects/HindiBabyNet/HindiBabyNet-VocalInputStats/R/README.md).

Manual installation example:

```r
install.packages(c(
  "tidyverse", "lme4", "lmerTest", "glmmTMB", "performance",
  "DHARMa", "emmeans", "ggeffects", "broom.mixed", "parameters",
  "boot", "ggplot2", "yaml", "knitr"
))
```

## R analysis run order

Run the R workflow from the project root in this order:

```r
source("R/00_setup.R")
source("R/01_input_models.R")
source("R/02_input_output_models.R")
source("R/03_model_tables_and_plots.R")
source("R/04_final_report.R")
```

This sequence:

1. Loads the shared runtime setup.
2. Fits the Model Family 1 input models.
3. Fits the Model Family 2 input-output models.
4. Produces diagnostics, prediction intervals, fitted plots, and reporting tables.
5. Renders the final report.

## Monte Carlo simulation settings

Use the following `nsim` conventions:

- `nsim = 100` for smoke testing and quick runtime validation.
- `nsim = 1000` for main reported results.
- `nsim = 2000–5000` as an optional final robustness check.

The fitted model plot confidence ribbons come from Monte Carlo prediction simulation.

The workflow can be smoke tested by setting the runtime environment variable before running the R scripts. For example in PowerShell:

```powershell
$env:HBN_NSIM = "100"
```

For main results, use:

```powershell
$env:HBN_NSIM = "1000"
```

## Age variable explanation

- Models use `age_z`.
- Plots display `age_days`.
- For fitted predictions, `age_days` is converted internally to `age_z` before model prediction.

This keeps the model specification numerically consistent while making figures easier to interpret.

## Output locations

Python outputs:

- `data/derived/`: final public analysis datasets
- `tables/`: Python descriptive EDA tables
- `figures/`: Python descriptive EDA figures

R outputs:

- `results/r_models/`: saved fitted-model bundles (`.rds`)
- `results/r_tables/`: model selection, fixed effects, evidence summaries, reporting bundle
- `results/r_diagnostics/`: residual, QQ, convergence, singularity, and related diagnostics outputs
- `results/r_predictions/`: prediction grids and Monte Carlo interval tables
- `results/r_plots/`: fitted model plots and forest plots
- `results/final_report/`: rendered Quarto report files

## Final report

Generate the report with:

```r
source("R/04_final_report.R")
```

The report is rendered from the Quarto template in `R/report.qmd` and writes to `results/final_report/`.

Main report outputs:

- `results/final_report/report.pdf`
- `results/final_report/report.html`

If you want the PDF named as `HindiBabyNet_Statistical_Analysis_Report.pdf` for sharing, rename the rendered PDF after generation or adjust the Quarto output settings later. The current workflow renders `report.pdf` and optional `report.html`.

## Using network and UNC paths on Windows

Prefer forward-slash UNC paths in [configs/config.yaml](c:/Users/arunps/OneDrive/Projects/HindiBabyNet/HindiBabyNet-VocalInputStats/configs/config.yaml), for example `//server/share/folder`, rather than raw backslash strings. This avoids YAML escaping issues and works with Windows UNC shares.

Example HindiNet structure:

- Metadata Excel: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Audio_metadata/metadata_cleaned.xlsx`
- Audio root: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Audio_data_processing`
- Audio participant folders: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Audio_data_processing/ABAN141223/*.wav`
- VTC root: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Classification_outputs/VTC`
- VTC participant folders: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Classification_outputs/VTC/ABAN141223/rttm.csv`

## Troubleshooting

### UNC and network path issues

- Use forward-slash UNC paths in `configs/config.yaml`.
- If a path fails, check VPN access, network drive permissions, and whether the UNC share is mounted and reachable.
- The Python `build-master` command prints existence checks for metadata, audio, and VTC roots at startup.

### Missing `Rscript`

- Confirm that R is installed.
- If `Rscript` is not on `PATH`, use the full executable path.
- In VS Code, make sure the R extension is pointing to a valid R installation.

### Missing Quarto

- Install Quarto separately or use a bundled Quarto CLI if your R environment provides one.
- If Quarto is installed but not on `PATH`, set `QUARTO_PATH` before running `source("R/04_final_report.R")`.

### Missing LaTeX or TinyTeX for PDF

- HTML report rendering should work first.
- PDF rendering additionally requires a LaTeX engine such as TinyTeX, MiKTeX, or TeX Live.
- If PDF fails, check the Quarto and TeX installation, but keep PDF support enabled.

### Package installation problems

- If package installation fails in a user library, try a local writable library.
- If `renv::restore()` is incomplete because there is no `renv.lock`, install the packages listed in [R/README.md](c:/Users/arunps/OneDrive/Projects/HindiBabyNet/HindiBabyNet-VocalInputStats/R/README.md) manually.
- On older R versions, some package versions may be constrained by compiled dependencies.

### Model convergence warnings

- Convergence warnings do not automatically invalidate a smoke test.
- Check `results/r_diagnostics/` and `results/r_tables/diagnostics_summary.csv` to inspect which models showed warnings.
- Keep the scientific model specification fixed unless a true runtime or identifiability error requires a justified change.

### Singular random effects

- Singular fit warnings can happen when the random-effects structure is weakly identified in a small sample.
- Review the saved diagnostics before deciding whether a modeling adjustment is necessary.

### Where to check diagnostics

- `results/r_diagnostics/` contains residual-vs-fitted plots, QQ plots, and per-model diagnostics summaries.
- `results/r_tables/diagnostics_summary.csv` provides a compact overview.

## Scientific notes

- Recording duration is never computed from summed VTC segment durations.
- `count_hour = count / recording_duration_hours`
- `duration_hour = total_speaker_duration_sec / recording_duration_hours`
- Participants with missing data are retained where possible and flagged in the validation report.

## Git and data privacy reminder

- Do not commit `data/private/`.
- Do not commit raw audio.
- Do not commit `participant_lookup.csv`.
- Generated heavy results such as model objects, diagnostic plots, prediction files, and rendered reports should stay ignored unless you explicitly intend to version them.

## Status

The repository includes reproducible Python dataset builders, EDA tables, EDA plots, tests, and an R workflow for preregistered mixed-effects analysis and final reporting.
