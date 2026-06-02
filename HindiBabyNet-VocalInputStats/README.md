# HindiBabyNet-VocalInputStats

Vocal input statistics for the HindiBabyNet project: count-per-hour and
duration-per-hour analyses of adult and child speech directed at target children.

## Setup

```bash
uv sync
uv run python -c "import hindibabynet_vocalinputstats"
```

## Data

Expects VTC or XGB classification outputs (produced by `HindiBabyNet-Pipeline`)
under `data/Classification_outputs/`.

## Usage

See `notebooks/` for exploratory analyses and `scripts/` for batch runs.
