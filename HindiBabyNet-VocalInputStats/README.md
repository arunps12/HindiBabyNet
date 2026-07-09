# HindiBabyNet-VocalInputStats

This repository builds the final privacy-preserving vocal input and vocal
output analysis datasets for the HindiBabyNet study. It combines participant
metadata, VTC speaker-classification outputs, and full recording durations to
produce participant-level and long-format analysis files, Python-based EDA
tables, and publication-style plots before mixed-effects modelling in R.

## Purpose

The workflow is designed for count/hour and duration/hour analyses outside the
main HindiBabyNet pipeline repository. It uses the full recording duration as
the denominator because the whole recording was processed by VTC.

Input speakers:

- `adult_female`
- `adult_male`
- `other_child`

Output speaker:

- `key_child`

## Input requirements

The workflow expects:

1. A metadata table in `.csv`, `.xlsx`, or `.xls` format with at least `par_id`, `REC_date`, `birthdate`,
	 `child_sex`, `mother_education`, `father_education`, and `Location`.
2. A VTC output root containing one folder per participant with
	 `rttm.csv` files.
3. An audio root containing the full recording processed by VTC.

VTC labels are normalized as follows:

- `FEM -> adult_female`
- `MAL -> adult_male`
- `KCHI -> key_child`
- `OCH -> other_child`

## Privacy model

- Public outputs use only anonymized `participant_id` values such as `P001`.
- Original participant IDs are stored only in
	`data/private/participant_lookup.csv`.
- `data/private/`, `data/raw/`, and local audio files are excluded from Git.

## Setup with uv

This project is packaged and run with `uv`.

Install `uv` first:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a terminal in the project folder:

```powershell
Set-Location "C:/Users/arunps/OneDrive/Projects/HindiBabyNet/HindiBabyNet-VocalInputStats"
```

Then create or sync the local environment and install the package dependencies:

```bash
uv sync
```

You can confirm the package CLI is available with:

```bash
uv run hindibabynet-vocalinputstats --help
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

## Using network/UNC paths on Windows

Prefer forward-slash UNC paths in [configs/config.yaml](c:/Users/arunps/OneDrive/Projects/HindiBabyNet/HindiBabyNet-VocalInputStats/configs/config.yaml), for example `//server/share/folder`, rather than raw backslash strings. This avoids YAML escaping issues and works with Windows UNC shares.

Example HindiNet structure:

- Metadata Excel: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Audio_metadata/metadata_cleaned.xlsx`
- Audio root: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Audio_data_processing`
- Audio participant folders: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Audio_data_processing/ABAN141223/*.wav`
- VTC root: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Classification_outputs/VTC`
- VTC participant folders: `//hypatia.uio.no/lh-hf-iln-sociocognitivelab/Research/HindiNet/Classification_outputs/VTC/ABAN141223/rttm.csv`

The `build-master` command prints the metadata path, audio root, and VTC root with existence checks at startup. If a UNC path is unavailable, first check VPN or network-share access.

```bash
uv run hindibabynet-vocalinputstats build-master --config configs/config.yaml
```

## Commands

```bash
uv run hindibabynet-vocalinputstats build-master --config configs/config.yaml
uv run hindibabynet-vocalinputstats create-long --config configs/config.yaml
uv run hindibabynet-vocalinputstats eda --config configs/config.yaml
uv run hindibabynet-vocalinputstats plots --config configs/config.yaml
uv run hindibabynet-vocalinputstats all --config configs/config.yaml
```

## Outputs

Main derived datasets:

- `data/derived/final_master.csv`
- `data/derived/input_long.csv`
- `data/derived/input_output_long.csv`

Private reproducibility file:

- `data/private/participant_lookup.csv`

Validation and build reports:

- `results/validation_report.csv`
- `results/dataset_build_report.txt`

EDA tables:

- `tables/participant_summary.csv`
- `tables/missing_values_summary.csv`
- `tables/recording_duration_summary.csv`
- `tables/speaker_count_summary.csv`
- `tables/speaker_duration_summary.csv`
- `tables/age_summary.csv`
- `tables/sex_distribution.csv`
- `tables/location_distribution.csv`
- `tables/education_distribution.csv`

Figures:

- Age and recording-duration distributions
- Child sex, location, and education distributions
- Input count/hour and duration/hour boxplots
- Age and input/output scatterplots
- Correlation heatmap
- Participant stacked composition plot
- Mean input summaries with confidence intervals

## Downstream R use

The long-format datasets are intended for mixed-effects models such as:

- `key_child_count_hour ~ input_count_hour * speaker + age_z + SES + child_sex + random effects`
- `key_child_duration_hour ~ input_duration_hour * speaker + age_z + SES + child_sex + random effects`

## Scientific notes

- Recording duration is never computed from summed VTC segment durations.
- `count_hour = count / recording_duration_hours`
- `duration_hour = total_speaker_duration_sec / recording_duration_hours`
- Participants with missing data are retained where possible and flagged in the
	validation report.

## Status

The package includes reproducible dataset builders, EDA tables, plots, tests,
and a starter notebook for exploratory work.
