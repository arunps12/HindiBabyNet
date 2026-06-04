# HindiBabyNet-CDIAnalysis

Hindi CDI questionnaire analysis for the broader HindiBabyNet project.

This repository is intended for analysis of Hindi CDI questionnaire data collected through Nettskjema forms. It is structured to keep analysis code, metadata, documentation, and placeholder outputs separate from any sensitive participant data.

## Included Form Types

- consent form
- eligibility form
- language background/family information form
- CDI 8-18 months form
- CDI 19-36 months form
- separate contact/lottery form kept outside analysis


## Planned Workflow

```text
raw Nettskjema exports
	↓
link forms using submission/reference IDs
	↓
filter by consent and eligibility
	↓
calculate age from date of birth and created timestamp
	↓
create participant tracking file
	↓
create word-level long dataset
	↓
compute comprehension and production scores
	↓
compute category-level scores using word_mapping.csv
	↓
run exploratory data analysis
```

## Setup

```bash
uv sync
uv run python -c "import hindibabynet_cdi"
```

## Repository Layout

- `configs/` stores analysis configuration placeholders.
- `data/` contains ignored raw, interim, processed, and personal-data directories with tracked `.gitkeep` files only.
- `metadata/` stores safe committed metadata such as `word_mapping.csv` and blank templates.
- `notebooks/` contains planned analysis notebooks.
- `scripts/` contains placeholder entry points for data checks and dataset creation.
- `src/hindibabynet_cdi/` contains the Python package scaffold.
- `outputs/` contains ignored generated figures, tables, and reports with tracked `.gitkeep` files only.
- `docs/` contains analysis planning and data-management notes.

## Status

This is an initial scaffold only. Analysis logic, data ingestion, form linking, scoring, and plotting are not implemented yet.
