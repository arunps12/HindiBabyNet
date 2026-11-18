# HindiBabyNet

HindiBabyNet studies language development in Hindi-speaking children. This repository contains audio processing, vocal input/output analysis, prosodic feature extraction, prosodic statistics, and Hindi CDI questionnaire analysis.

## Find the right workflow

| Source folder | Purpose | Launcher command |
| --- | --- | --- |
| [audio](audio/README.md) | Prepare recordings and classify speaker types | `audio` |
| [vocal_stats](vocal_stats/README.md) | Build vocal datasets and descriptive summaries | `vocal-stats` |
| [prosody](prosody/README.md) | Extract pitch and speech-model features; not model training | `prosody` |
| [vocal_stats](vocal_stats/README.md) | Fit vocal input/output statistical models in R | `vocal-models` |
| [prosody_stats](prosody_stats/README.md) | Fit prosodic statistical models in R | `prosody-stats` |
| [cdi](cdi/README.md) | Link and score CDI questionnaires | `cdi` |
| [docs](docs/README.md) | Configuration, safety, and troubleshooting | — |

Source code now lives in these folders, with launcher tests in [tests/test_run.py](tests/test_run.py). Python package names are unchanged. Existing research data, caches, outputs, weights, native configurations, and old environments remain in the ignored original `HindiBabyNet-*` asset directories; they were not moved into the new source folders. A fresh clone does not include them.

## 1. Install the software

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Use a compatible Python 3.10–3.12 interpreter for the projects that require it; uv manages each project's environment independently. The locked dependencies may further constrain the usable interpreter.
- Install FFmpeg and make `ffmpeg` available on `PATH` for audio processing.
- For statistical models, install R and the packages listed in the [vocal guide](vocal_stats/README.md#r-packages) or [prosody statistics guide](prosody_stats/README.md#r-packages). `Rscript` must be available on `PATH`, or configured under `tools.rscript`.
- Quarto is needed only for the optional vocal HTML report. Pandoc and R Markdown support are needed for the optional prosody HTML report. PDF/LaTeX is not required by the launcher.
- Audio VTC requires a separately installed external model repository, weights, and environment. Use WSL/Linux for audio: the current VTC integration has Unix-oriented paths and is not established as a native Windows workflow.

Keep scientific environments independent. The root environment is only for the launcher; each Python project has its own environment and lockfile, including incompatible NumPy requirements. Manage each R analysis's packages separately and retain its known-working versions rather than merging or upgrading the scientific environments. Root setup does not provision R libraries.

## 2. Create or review your local configuration

Open the repository root. Example PowerShell location:

```powershell
Set-Location C:/research/HindiBabyNet
if (-not (Test-Path config.local.yaml)) { Copy-Item config.example.yaml config.local.yaml }
```

WSL/Linux equivalent:

```bash
cd /path/to/HindiBabyNet
cp -n config.example.yaml config.local.yaml
```

**If [config.local.yaml](config.local.yaml) already exists, keep it; do not overwrite it with the example.** The migrated local file imports the existing native mappings and starts with all workflows disabled. Preserve all original scientific settings, including settings that differ from examples.

The [root example](config.example.yaml) contains full native mappings, including CDI and prosody statistics, with current source folders and all workflows disabled. Its six `project_root: "."` values resolve paths from the root YAML's directory; inputs use private `Data` locations and outputs use dedicated `results` locations, with local caches separate. You edit **one root local configuration**—no duplicate per-workflow mappings are needed.

1. For a new study, copy the example as above, then edit input/model paths and choose **fresh, dedicated output, cache, log, artifact, private-table, and report locations**. Connect downstream inputs to the intended outputs; example destinations are not guaranteed empty.
2. For an existing study, keep the existing configuration and scientific settings. Retain each section's original asset-based `project_root` when preserving relative asset paths; `workflows.<name>.project` locates source code, not data.
3. Set `enabled: true` only for workflows you intend to run. No dependencies are enabled automatically.

Alternatively, `configure` creates a new local configuration from native configs in the new source folders, then falls back to the original asset-folder configs, then the versioned full root template for missing sections. It disables all workflows and refuses to overwrite a destination. See [configuration and migration](docs/README.md#configuration-and-migration).

**The template is not approval for scientific changes.** Preserve approved settings: active prosody and the root template use a **100 Hz** semitone reference; the older [per-workflow example](prosody/config/config.example.yaml) uses **1 Hz**. These references are not interchangeable.

Use paths valid in the operating system where commands run; Windows paths do not automatically become WSL paths. See the [configuration guide](docs/README.md#configuration-and-migration).

## 3. Install the enabled environments

From the repository root, after editing your local configuration:

```text
uv sync
uv run python run.py setup
```

Root sync installs launcher dependencies. `setup` runs frozen uv sync in each enabled Python source project. It does not install R packages, FFmpeg, external VTC, or model weights. Existing environments in old asset folders are not moved or reused automatically. If no workflows are enabled, setup has nothing to install.

## 4. Check, then run

```text
uv run python run.py check
uv run python run.py all --dry-run
uv run python run.py all
```

`all` runs only explicitly enabled workflows, sequentially in this dependency order:

Execution order: audio → vocal-stats → prosody → vocal-models → prosody-stats → cdi.

Workflows remain independent: you can use approved existing inputs instead of enabling upstream processing. CDI does not require audio. VTC-dependent consumers cannot use XGB output as a substitute. Reports and interactive annotation are not included in `all`.

Run just one enabled workflow, for example:

```text
uv run python run.py vocal-stats
uv run python run.py prosody
uv run python run.py cdi
```

The [launcher](run.py) also supports `--config`, `--stage`, `--resume` (the default), `--no-resume`, and `--dry-run`. Put supported native stage options after `--`. See the [command and safety reference](docs/README.md#launcher-reference) before using individual stages.

## Safe reuse and private data

The launcher skips only a completed **whole stage** whose local ownership record and checked fingerprints match. Stale, partial, altered, or unowned occupied outputs are refused. `--no-resume` does not permit overwrites; there is no `--force`. Prefer a full workflow command over chaining stages that share write locations. A stopped run may require fresh output paths rather than resuming participants.

Keep all participant-derived material private, including pseudonymized tables, classification outputs, weights, reports, and executed notebooks. [.gitignore](.gitignore) is not publication approval. See the [operating guide](docs/README.md#outputs-and-recovery) for fingerprint/recovery limits and [scientific and privacy boundaries](docs/README.md#scientific-and-privacy-boundaries) before sharing or changing an analysis.

## Testing

Synthetic/unit tests cover launcher and workflow behavior; external-model and original-cohort checks are separate. See [testing guidance](docs/README.md#testing-and-validation), including CDI's opt-in original-data tests and validation limits.
