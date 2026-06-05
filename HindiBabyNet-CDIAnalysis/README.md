# HindiBabyNet-CDIAnalysis

Hindi CDI questionnaire analysis for the broader HindiBabyNet project.

This repository processes Hindi CDI Nettskjema exports into privacy-safe linkage tables, word-level CDI rows, participant-level vocabulary scores, category summaries, and first-pass EDA outputs. The separate contact or lottery form is intentionally kept out of the analysis pipeline.

## Included Form Types

- consent form
- eligibility form
- language background form
- CDI 8-18 months form
- CDI 19-36 months form
- separate contact or lottery form excluded from analysis merges

## Workflow

```text
raw Nettskjema exports
  -> link consent, eligibility, background, and CDI submissions
  -> derive participant tracking and exclusions
  -> match CDI item columns to metadata/word_mapping.csv
  -> create word-level long CDI dataset
  -> compute participant and category vocabulary scores
  -> write summary tables, figures, and markdown reports
```

## Setup

```bash
uv sync
uv run python -c "import hindibabynet_cdi"
```

## Run Order

```bash
uv run python scripts/check_form_links.py
uv run python scripts/create_participant_tracking.py
uv run python scripts/create_word_level_long.py
uv run python scripts/compute_vocabulary_scores.py
uv run python scripts/run_eda.py
```

## Main Outputs

- `outputs/reports/form_link_check.md`
- `outputs/tables/form_link_summary.csv`
- `data/interim/participant_tracking.csv`
- `data/interim/word_level_long.csv`
- `data/processed/participant_info.csv`
- `data/processed/vocabulary_scores.csv`
- `data/processed/category_scores.csv`
- `data/processed/master_dataset.csv`
- `outputs/reports/eda_report.md`

## Repository Layout

- `configs/` stores runtime paths and form IDs.
- `data/` contains ignored raw, interim, processed, and personal-data directories.
- `metadata/` stores safe committed metadata such as `word_mapping.csv`.
- `notebooks/` contains runnable workflow notebooks.
- `scripts/` contains the main entry points for linkage, scoring, and EDA.
- `src/hindibabynet_cdi/` contains the reusable pipeline code.
- `outputs/` stores generated figures, tables, and reports.

## Privacy Boundary

Do not merge the separate contact or lottery form into analysis datasets. Keep personal contact details outside this repository and outside all generated tables, notebooks, and reports.
