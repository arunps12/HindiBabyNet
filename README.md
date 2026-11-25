# HindiBabyNet

A computational toolkit for automatic speaker classification, language input analysis, acoustic feature extraction, and early language development in Hindi-learning infants. Developed under the Hindi-BabyNet project, funded by the EU Horizon Europe Marie Skłodowska-Curie Postdoctoral Fellowship (MSCA-PF), Grant No. 101107938.

This repository has the code for five workflows.

| Workflow | What it does |
| --- | --- |
| `audio` | Prepares the day-long audio recordings and classifies the speakers with VTC or with our XGBoost model |
| `audio-stats` | Statistical analysis of vocal input and child vocalizations |
| `prosody` | Extracts acoustic and prosodic features of the speakers |
| `prosody-stats` | Statistical analysis of the prosodic features |
| `cdi` | Processes the Hindi CDI questionnaires |

The repository has only code. The research data is not part of it. You can keep the data in any folder, also outside this project.

Most users need only the `audio` workflow: it prepares audio recordings and classifies the speakers. It works with your own recordings. The other four workflows are mainly for reproducing the results of the HindiBabyNet study.

## Folders of the project

```
HindiBabyNet/
├── README.md
├── config.example.yaml     example of the configuration file
├── shared/                 small helper that reads the configuration file
├── audio/                  audio preparation and speaker classification
├── audio-stats/            audio-stats.Rmd and its R functions
├── prosody/                prosodic feature extraction
├── prosody-stats/          prosody-stats.Rmd and its R functions
├── cdi/                    CDI questionnaire processing
└── references/             scientific documents of the project
```

Each workflow folder has a file `params.yaml` with the settings of the method (for example sample rate or pitch range). Do not change these values for an existing study. The data locations are not in these files. They are in one file, `config.yaml`, which is explained below.

## Folders of the data

Your audio recordings can stay where they are. Everything that the workflows make is saved under one main data folder (`data_dir` in `config.yaml`). This is its structure:

```
HindiBabyNet-data/
├── derived/
│   ├── prepared-audio/           prepared WAV files
│   ├── vtc-classification/       VTC results
│   ├── xgb-classification/       XGBoost results
│   ├── prosody-features/         prosodic features                      (research)
│   └── statistical-data/         datasets for audio-stats               (research)
├── raw/
│   ├── cdi/                      CDI questionnaires, word_mapping.csv   (research)
│   └── metadata/                 participant information                (research)
├── results/                      tables, figures and reports            (research)
└── cache/                        temporary files
```

The commands make the output folders themselves. For audio preparation and speaker classification you need only the first three folders of `derived`. The folders marked "(research)" are used by the CDI, prosody and statistical workflows of the HindiBabyNet study.

## 1. Install the software

You need:

- **uv** for all Python workflows. Install it as described on <https://docs.astral.sh/uv/getting-started/installation/>.
  - Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - Ubuntu: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **R and RStudio** for `audio-stats` and `prosody-stats`.
- **Ubuntu (WSL)** on a Windows computer for the `audio` workflow. See [audio/README.md](audio/README.md).

You do not need to install Python packages yourself. The first time you run a command, uv installs the packages of that workflow.

For the two statistical analyses, open RStudio and run this once in the Console:

```r
install.packages(c("tidyverse", "lme4", "lmerTest", "glmmTMB", "performance", "DHARMa",
                   "emmeans", "ggeffects", "broom.mixed", "parameters", "boot", "yaml",
                   "knitr", "rmarkdown", "car", "readxl"))
```

## 2. Set your folders

1. Copy the file `config.example.yaml` and rename the copy to `config.yaml`. Keep it in the project folder.
2. Open `config.yaml` and change `data_dir` to the folder where the results should be saved.
3. Change `raw_audio` to the folder with your audio recordings.

```yaml
data_dir: /path/to/your/data-folder

paths:
  raw_audio: /path/to/your/audio-recordings
```

| Path | Windows example | Ubuntu example |
| --- | --- | --- |
| `data_dir` | `D:/HindiBabyNet-data` | `/home/yourname/HindiBabyNet-data` |
| `raw_audio` | `D:/my_recordings` | `/home/yourname/my_recordings` |

This is all you need for audio preparation and speaker classification. The prepared audio and the classification results are saved in `data_dir`. This folder is made if it does not exist.

**Another place for one folder.** All results are saved inside `data_dir`. To save only one folder in another place, remove the `#` in front of its line in `config.yaml` and write your path. For example:

```yaml
data_dir: /path/to/your/data-folder

paths:
  raw_audio: /path/to/your/audio-recordings
  prepared_audio: D:/my_data/prepared-audio
  vtc: D:/my_data/vtc-classification
```

**Reproducing the HindiBabyNet results.** The CDI, prosody and statistical workflows also need the questionnaire files in `raw/cdi/` and the participant files in `raw/metadata/` of the data folder. Their paths are in the last part of `config.yaml`. You do not need them for audio preparation and speaker classification.

Some rules for the paths:

- Use `/` in the paths, also on Windows.
- A path without a drive letter starts from the folder of `config.yaml`. For example `../HindiBabyNet-data` is a folder next to the project folder.
- A Windows path like `C:/data` also works in Ubuntu. An Ubuntu path like `/home/yourname/data` also works in Windows and RStudio. If your Ubuntu has another name than "Ubuntu", add a line `wsl_distro: Ubuntu-22.04` (with your name) to `config.yaml`.
- A network folder (`//server/share/...`) works on Windows. In Ubuntu you must mount it first.

You never need to change a Python or R file to change a data location.

## 3. Run a workflow

Run all commands from the project folder. Existing output files are never replaced. If an output file already exists, the command stops and tells you which file it is.

### audio

This workflow runs in Ubuntu. All steps are explained in [audio/README.md](audio/README.md).

| Step | Command (Ubuntu terminal) | Output |
| --- | --- | --- |
| Prepare the audio | `uv run --project audio hbn-audio prepare` | `derived/prepared-audio/` |
| Classify speakers with VTC | `uv run --project audio hbn-audio classify-vtc` | `derived/vtc-classification/` |
| Classify speakers with XGBoost | `uv run --project audio --extra xgb hbn-audio classify-xgb` | `derived/xgb-classification/` |

If the prepared audio files are already available, you do not need to prepare them again. If VTC results are already available, you do not need to run VTC again. Set `prepared_audio` and `vtc` in `config.yaml` to these folders and go to the next workflow.

The raw recordings can be organised in different ways: participant folders with date folders (as in HindiBabyNet), participant folders without dates, WAV files in one folder, or one WAV file. You do not need to change your folders. Joining of the files and the selection of the first recording day are settings in `audio/params.yaml`. See section 4 of [audio/README.md](audio/README.md).

### cdi

- Where: Windows PowerShell (or Ubuntu), in the project folder.
- Input: the questionnaire exports from Nettskjema and the file `word_mapping.csv` in `raw/cdi/`.
- Command:

```
uv run --project cdi hbn-cdi
```

- Output: four files in `results/cdi/`.

| File | Content |
| --- | --- |
| `cdi1_items.csv` | item responses of CDI-1 (8 to 18 months), one row per questionnaire and word |
| `cdi2_items.csv` | item responses of CDI-2 (19 to 36 months) |
| `cdi1_summary.csv` | one row per CDI-1 questionnaire: age, gender, birthdate, education of the parents, completion date, comprehension and production scores |
| `cdi2_summary.csv` | one row per CDI-2 questionnaire, with the production score |

The item files and the summary files are joined with `child_id` and `submission_id`. A questionnaire is included when the parent gave consent, the forms are linked, the child fulfils the eligibility criteria, the Hindi exposure is above 75 %, and the age fits the questionnaire. If a child has more than one questionnaire, every questionnaire is kept as a separate row. The command prints how many questionnaires were included and why the others were not.

### prosody

- Where: Windows PowerShell (or Ubuntu), in the project folder.
- Input: the prepared audio, the VTC results and `participant_lookup.csv`.
- First check the input files. This does not write anything:

```
uv run --project prosody hbn-prosody --check
```

- Then run the extraction. It takes a long time. A GPU makes it faster.

```
uv run --project prosody hbn-prosody
```

- Output: `derived/prosody-features/participant_speaker_acoustic_features.csv` (one row per participant and speaker), with quality-check tables in the folder `qc`.

If the features file is already available, you do not need to run this step. The first run downloads the two speech models from Hugging Face.

### audio-stats

There are two steps.

**Step 1: prepare the dataset.** Skip this step if the files `final_master.csv`, `input_long.csv` and `input_output_long.csv` are already in `derived/statistical-data/`.

- Where: Windows PowerShell, in the project folder.
- Input: the VTC results, the metadata file, and the raw recordings (the command reads the duration of the first recording day). If your raw recordings have no date folders, set `recording_duration_source: audio_file` in `audio-stats/params.yaml`. The duration is then read from the prepared audio.

```
uv run --project audio-stats hbn-audio-stats
```

**Step 2: run the analysis.**

- Open the file `audio-stats/audio-stats.Rmd` in RStudio and click **Knit**.
- It loads the dataset, makes the descriptive tables and figures, fits the models and makes the report.
- Output: `results/audio-stats/audio-stats.html`, with the folders `tables`, `figures`, `models`, `predictions` and `diagnostics`.

The settings of the models are in the `params` block at the top of the Rmd file. They are explained in the first section of the report.

### prosody-stats

- Input: the features file from `prosody` and the dataset of `audio-stats` (Step 1).
- Open the file `prosody-stats/prosody-stats.Rmd` in RStudio and click **Knit**. It takes about 10 minutes.
- Output: `results/prosody-stats/prosody-stats.html`, with the folders `tables`, `figures`, `models`, `predictions` and `diagnostics`.

## Common problems

| Problem | What to do |
| --- | --- |
| `Configuration file not found` | Copy `config.example.yaml` to `config.yaml` in the project folder. |
| `Not found: ...` with the name of a path | The folder or file is not at this place. Check `data_dir` and the line under `paths` that the message names. |
| `... already exists and is not replaced` | The output is already there. Use it, or move the old file to another folder and run the command again. |
| `uv` is not found | Close and open the terminal again after you installed uv. |
| Knit stops with `there is no package called ...` | Install the R packages with the command in section 1. |
| The Knit button makes no HTML file in the Rmd folder | This is correct. The report is saved in the `results` folder of your data folder. |
| A network folder does not work in Ubuntu | Mount the folder in Ubuntu, or copy the files to a local folder, and write this path in `config.yaml`. |

## Data protection

The recordings, the questionnaires, the participant information and all results are research data. Keep them on your computer or on the secured project storage. Do not add them to this repository. The file `.gitignore` is written so that only the code folders listed above can be uploaded.
