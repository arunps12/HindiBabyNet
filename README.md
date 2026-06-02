# HindiBabyNet-Pipeline

This repository is the technical pipeline component of the broader HindiBabyNet project. It focuses on audio preparation, speaker classification, TextGrid generation, manual speaker-class annotation, and model evaluation. It does not cover CDI analysis, vocal input statistics, prosodic statistics, prosodic model training, or ADS/IDS annotation.

## HindiBabyNet Project Repositories

| Repository | Purpose |
|---|---|
| HindiBabyNet | Main umbrella/project homepage |
| HindiBabyNet-Pipeline | Audio preparation, speaker classification, TextGrid generation, annotation, model evaluation |
| HindiBabyNet-VocalInputStats | Count/hour and duration/hour analyses |
| HindiBabyNet-CDIAnalysis | CDI questionnaire analyses |
| HindiBabyNet-ProsodyStats | Prosodic prominence statistical analyses |
| HindiBabyNet-ProsodyModel | Prosodic prominence model training |
| HindiBabyNet-Docs | Protocols, manuals, collaborator documentation |
| VocalBaby | Infant vocalization classifier/tool |

## Scope

The pipeline repository currently provides:

- Flexible audio preparation with optional joining, resampling, mono conversion, normalization, and configurable output persistence.
- Speaker classification through both the XGB backend and the external VTC backend.
- TextGrid generation from normalized XGB and VTC segment tables.
- Manual speaker-class annotation for evaluation.
- Model evaluation against manual labels.

The repository intentionally does not present ADS/IDS annotation as an active workflow. Older ADS/IDS assets have been archived.

## Data Layout

Large generated data must live outside the repository. Use `configs/config.yaml` to point at external roots such as:

```text
/home/arunps/hindinet_data/
├── RawAudioData/
├── RawJoinedAudio/
├── PreparedAudio/
├── Classification_outputs/
├── TextGrids/
├── ManualAnnotations/
└── EvaluationOutputs/
```

Keep repository `artifacts/` limited to small reproducibility files such as run metadata, summaries, config snapshots, and small tables. Do not commit large WAV, MP3, FLAC, parquet, or external-model output files.

## Setup

```bash
git clone https://github.com/arunps12/HindiBabyNet.git
cd HindiBabyNet
uv sync
uv sync --extra xgb
uv run hindibabynet-pipeline-check
```

Use `uv` throughout the repository:

- `uv sync` for environment creation
- `uv sync --extra xgb` for the XGB/TextGrid stack
- `uv run pytest` for tests
- `uv run python -c "import hindibabynet_pipeline"` for import smoke checks

## Configuration

The pipeline reads two files:

- `configs/config.yaml` for runtime paths, output locations, and backend selection
- `configs/params.yaml` for audio-processing, model, annotation, and TextGrid parameters

Key runtime roots live under `paths` in `config.yaml`. Tunable audio settings such as `join_multiple_files`, `target_sr`, `convert_to_mono`, and `normalize` live in `params.yaml`.

## Commands

Primary task-based scripts:

```bash
uv run bash scripts/run_pipeline.sh --limit 2 --backend vtc
uv run bash scripts/run_prepare_audio.sh --recordings-parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
uv run bash scripts/run_prepare_audio.sh --wav /path/to/input.wav --recording-id REC001
uv run bash scripts/run_classify_xgb.sh --recordings-parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
uv run bash scripts/run_classify_vtc.sh --recordings-parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet
uv run bash scripts/run_generate_textgrids.sh --backend vtc --participant-id ABAN141223
uv run bash scripts/run_annotate_segments.sh --backend vtc --participant-id ABAN141223
uv run bash scripts/run_evaluate_models.sh --input-csv /path/to/ManualAnnotations/ABAN141223/ABAN141223_vtc_annotations.csv
```

Legacy stage scripts remain as deprecation shims for backward compatibility.

## Backends

XGB backend:

- Uses VAD, diarization, eGeMAPS features, and XGBoost classification.
- Produces class streams, segment tables, summaries, and TextGrids.
- Requires `uv sync --extra xgb`.

VTC backend:

- Uses the external VTC repository configured in `configs/config.yaml`.
- Produces `rttm.csv` segment tables with `FEM`, `MAL`, `KCHI`, and `OCH` labels.
- TextGrid generation normalizes these labels to `adult_female`, `adult_male`, `key_child`, and `other_child`.

## Evaluation Workflow

1. Prepare audio from raw recordings.
2. Run either `run_classify_xgb.sh` or `run_classify_vtc.sh`.
3. Generate TextGrids if needed for inspection.
4. Run `run_annotate_segments.sh` to sample and label segments.
5. Run `run_evaluate_models.sh` to produce accuracy, confusion matrix, and classwise metrics.

## Repository Layout

```text
configs/
docs/
external_models/
models/
scripts/
src/hindibabynet_pipeline/
tests/
```

See the docs directory for protocol- and workflow-specific details.

speaker_classification:
  output_root: /path/to/your/classification_outputs     # ← CHANGE THIS
```

### Backend: XGB (default)

**Setup:** Install the XGB extra first: `uv sync --extra xgb`. (Core Stage 01/02 dependencies like `scipy` are already included in `uv sync`.) The pre-trained XGBoost model ships with the repository at `models/xgb_egemaps.pkl`.

Set `backend: xgb` in `config.yaml` (this is the default):

```yaml
speaker_classification:
  backend: xgb
  output_root: /path/to/scratch/classification_outputs

  xgb:
    model_path: models/xgb_egemaps.pkl
    class_names: ["adult_male", "adult_female", "child", "background"]
    egemaps_dim: 88
    vad_aggressiveness: 2
    vad_frame_ms: 30
    vad_min_region_ms: 300
    diarization_model: "pyannote/speaker-diarization-3.1"
    chunk_sec: 900.0
    overlap_sec: 10.0
    min_speakers: 2
    max_speakers: 4
    merge_gap_sec: 0.7
    min_segment_sec: 0.2
    classify_win_sec: 1.0
    classify_hop_sec: 0.5
```

Exports: caregiver WAVs, child/background WAVs, TextGrid, segments parquet, and summary JSON. ADS/IDS annotation is designed for XGB outputs.

### Backend: VTC (optional) [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC)

**Setup required.** The VTC model is **not** included in this repository (only `external_models/README.md` is committed). You must clone and install VTC yourself:

```bash
# 1. Install system dependencies (if not already present)
#    - ffmpeg: sudo apt install ffmpeg
#    - git-lfs: sudo apt install git-lfs

# 2. Clone VTC into external_models/ (from HindiBabyNet root)
git lfs install
git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git external_models/VTC

# 3. Install VTC dependencies (creates its own separate venv)
cd external_models/VTC
uv sync
cd ../..
```

> **Note:** VTC runs in its own virtual environment, completely isolated from HindiBabyNet.

Set `backend: vtc` in `config.yaml`:

```yaml
speaker_classification:
  backend: vtc
  output_root: /path/to/your/classification_outputs

  vtc:
    repo_path: external_models/VTC    # path to cloned VTC repo
    device: cuda                      # cpu / cuda / gpu / mps
    keep_inputs: false                # keep temp VTC input folders after inference
```

Exports: `FEM/MAL/KCHI/OCH` RTTM/CSV outputs unchanged from [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC). Does **not** produce `main_female.wav`, `TextGrid`, etc.

### Full Configuration Reference

All parameters in **`configs/config.yaml`**:

```yaml
# ─── Paths ───
artifacts_root: artifacts/runs           # where run artifacts are saved
logs_root: logs                          # where log files are saved

# ─── Stage 01: Data Ingestion ───
data_ingestion:
  raw_audio_root: /path/to/raw/audio     # root directory to scan for WAV files
  allowed_ext: [".wav", ".WAV"]          # accepted file extensions
  recordings_filename: recordings.parquet

# ─── Stage 02: Audio Preparation ───
audio_preparation:
  processed_audio_root: /path/to/scratch/audio_processed
  target_sr: 16000          # target sample rate (Hz)
  to_mono: true             # convert to mono
  target_peak_dbfs: -1.0    # peak normalization target (dBFS)
  combine_gap_sec: 0.0      # silence gap when concatenating files (seconds)

# ─── Stage 03: Speaker Classification ───
speaker_classification:
  backend: xgb              # "xgb" (default) or "vtc"
  output_root: /path/to/scratch/classification_outputs

  xgb:                      # see "Backend: XGB" above for key descriptions
    model_path: models/xgb_egemaps.pkl
    class_names: ["adult_male", "adult_female", "child", "background"]
    egemaps_dim: 88
    vad_aggressiveness: 2
    vad_frame_ms: 30
    vad_min_region_ms: 300
    diarization_model: "pyannote/speaker-diarization-3.1"
    chunk_sec: 900.0
    overlap_sec: 10.0
    min_speakers: 2
    max_speakers: 4
    merge_gap_sec: 0.7
    min_segment_sec: 0.2
    classify_win_sec: 1.0
    classify_hop_sec: 0.5

  vtc:                      # see "Backend: VTC" above for key descriptions
    repo_path: external_models/VTC
    device: cuda             # cpu / cuda / gpu / mps
    keep_inputs: false
```

---

## Usage

> **Tip:** You can either set `backend:` in `config.yaml` or pass `--backend xgb|vtc` on the command line. The CLI flag overrides whatever is in the config file.

### Quick Start

Run the entire pipeline (Stage 01 → 02 → 03) end-to-end with a single command:

```bash
# Optional setup diagnostics (recommended once per machine):
uv run hindibabynet-check

# ─── Using XGB (default, reads backend from config.yaml) ───
uv run bash scripts/run_all.sh

# ─── Using VTC (override backend on CLI) ───
uv run bash scripts/run_all.sh --backend vtc

# Process only the first N participants (useful for testing):
uv run bash scripts/run_all.sh --limit 3
```

This runs all three stages sequentially:
1. **Stage 01** scans `raw_audio_root` and catalogues every WAV file
2. **Stage 02** combines all WAVs per participant into a single analysis-ready WAV
3. **Stage 03** runs speaker classification on each participant using the selected backend

If a run stops midway, run the same command again — Stage 02 and Stage 03 automatically skip participants whose outputs are already complete.

### Process a Single Raw WAV

```bash
# Step 1: Prepare the audio (mono, 16 kHz, normalized)
uv run bash scripts/run_stage_02_single_wav.sh /path/to/your/recording.wav

# Step 2: Classify with XGB (default)
uv run bash scripts/run_stage_03.sh --wav /path/to/audio_processed/recording/recording.wav

# Step 2 (alternative): Classify with VTC
uv run bash scripts/run_stage_03.sh --wav /path/to/audio_processed/recording/recording.wav --backend vtc
```

### Process a Directory of Raw WAVs

If you have a directory tree of recordings organised by participant:

```
your_audio_directory/
  ├── ParticipantA/
  │     ├── session_2024-01-15/
  │     │     ├── file1.WAV
  │     │     └── file2.WAV
  │     └── session_2024-02-20/
  │           └── file3.WAV
  ├── ParticipantB/
  │     └── session_2024-03-10/
  │           └── file1.WAV
  └── ...
```

Use `run_all.sh` as shown in [Quick Start](#quick-start), or run stages individually below.

### Running Individual Stages

Stage 01 and Stage 02 are shared for both backends. Only Stage 03 differs.

```bash
# ─── Stage 01: Scan raw audio directory ───
uv run bash scripts/run_stage_01.sh
# Output: artifacts/runs/<run_id>/data_ingestion/recordings.parquet

# ─── Stage 02: Prepare audio (batch, from Stage 01 output) ───
uv run bash scripts/run_stage_02_from_parquet.sh \
    artifacts/runs/<run_id>/data_ingestion/recordings.parquet
# Optional: limit to first N participants
uv run bash scripts/run_stage_02_from_parquet.sh \
    artifacts/runs/<run_id>/data_ingestion/recordings.parquet 5

# ─── Stage 02: Prepare audio (single WAV) ───
uv run bash scripts/run_stage_02_single_wav.sh /path/to/input.wav
# Optional: provide a custom recording_id
uv run bash scripts/run_stage_02_single_wav.sh /path/to/input.wav my_custom_id

# ─── Stage 03: Classify speakers (single prepared WAV) ───
uv run bash scripts/run_stage_03.sh --wav /path/to/audio_processed/<pid>/<pid>.wav

# ─── Stage 03: Classify all prepared WAVs in a directory ───
uv run bash scripts/run_stage_03.sh \
    --analysis_dir /path/to/audio_processed

# ─── Stage 03: Classify from recordings parquet (needs Stage 02 done) ───
uv run bash scripts/run_stage_03.sh \
    --recordings_parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet

# ─── Stage 03: Override backend or limit participants ───
uv run bash scripts/run_stage_03.sh --analysis_dir /path/to/audio_processed --backend vtc --limit 5
```

---

## Output Files

### XGB Outputs

For each participant:

```
<output_root>/xgb/<participant_id>/
  ├── <pid>_main_female.wav       # Main female caregiver audio (16 kHz mono)
  ├── <pid>_main_male.wav         # Main male caregiver audio (16 kHz mono)
  ├── <pid>_child.wav             # Child vocalisations (16 kHz mono)
  ├── <pid>_background.wav        # Background / non-speech (16 kHz mono)
  ├── <pid>_segments.parquet      # All classified segments
  ├── <pid>_summary.json          # Per-class duration statistics
  └── <pid>.TextGrid              # Praat-compatible annotation
```

**Four output classes:**

| Class | Description |
|-------|-------------|
| `adult_female` | Adult female speech (typically the mother) |
| `adult_male` | Adult male speech (typically the father) |
| `child` | Child / infant vocalisations |
| `background` | Non-speech or ambient noise |

### VTC Outputs

For each participant (unchanged VTC 2.0 output):

```
<output_root>/vtc/<participant_id>/
  ├── rttm/
  ├── raw_rttm/
  ├── rttm.csv
  ├── raw_rttm.csv
  └── run_info.json              # HindiBabyNet metadata (participant, command, runtime)
```

**VTC output classes:**

| Class | Description |
|-------|-------------|
| `FEM` | Adult female speech |
| `MAL` | Adult male speech |
| `KCHI` | Key-child speech |
| `OCH` | Other child speech |

> **Note:** When backend is `vtc`, HindiBabyNet does **not** produce its own speaker-classification artifacts (`main_female.wav`, `TextGrid`, `segments.parquet`, etc.). Only VTC outputs are generated.

### Segments Parquet Schema

(XGB only)

| Column | Type | Description |
|--------|------|-------------|
| `start_sec` | float | Segment start time (seconds from WAV beginning) |
| `end_sec` | float | Segment end time |
| `duration_sec` | float | Segment duration |
| `chunk_id` | int | Diarization chunk this segment came from |
| `speaker_id_local` | str | Local speaker label from diarization |
| `n_merged` | int | How many raw segments were merged into this one |
| `n_windows` | int | Number of eGeMAPS windows used for classification |
| `probs_adult_male` | float | Probability of adult_male |
| `probs_adult_female` | float | Probability of adult_female |
| `probs_child` | float | Probability of child |
| `probs_background` | float | Probability of background |
| `predicted_class` | str | Winning class label |
| `predicted_confidence` | float | Probability of the winning class |

### Summary JSON

(XGB only)

```json
{
  "participant_id": "ABAN141223",
  "duration_sec": 28800.0,
  "n_classified_segments": 4523,
  "total_speech_sec": 12456.7,
  "class_durations": {
    "adult_male": 2345.6,
    "adult_female": 5678.9,
    "child": 3210.1,
    "background": 1222.1
  }
}
```

---

## ADS / IDS Annotation

> **Applies to XGB backend only.** After Stage 03 (XGB) produces `main_female.wav` and `main_male.wav` for each participant, you can manually annotate each speech segment as **ADS** (Adult-Directed Speech), **IDS** (Infant-Directed Speech), or **Other**.

### How Segmentation Works

The annotation tool runs **fresh energy-based silence detection** on the exported WAV files (rather than reusing the pipeline's intermediate segments). This is more robust because:

- The concatenated WAVs have definite ~0.15 s silence gaps between speech chunks
- Energy-based detection reliably recovers those boundaries
- No dependency on intermediate parquet files or pipeline state
- Very long segments are automatically split at natural pauses for comfortable listening

### Method 1: Jupyter Notebook (Recommended for Remote Servers)

Open **`notebooks/02_annotation_player.ipynb`** — everything happens in one notebook:

1. **Cell 2** — Set `PARTICIPANT_ID` and `SPEAKER` (`"female"` or `"male"`)
2. **Cell 3** — Loads the WAV and detects segments (run once)
3. **Cell 4** — Annotation loop: each segment **auto-plays in the browser**, type `0`/`1`/`2` in the input box and press Enter
4. **Cell 5** — Export labeled ADS / IDS / Other WAV files

Press `q` to save & stop at any time — re-run Cell 4 to resume exactly where you left off.

### Method 2: Terminal Script (For Machines with Audio Output)

```bash
# Install playback dependency (one time)
uv pip install sounddevice

# Check annotation status for all participants
python scripts/annotate_ads_ids.py --status

# Annotate one participant (both speakers)
python scripts/annotate_ads_ids.py -p ABAN141223

# Only female speaker
python scripts/annotate_ads_ids.py -p ABAN141223 -s female

# Export WAVs from existing annotations without re-annotating
python scripts/annotate_ads_ids.py -p ABAN141223 --export-only
```

### Controls During Annotation

| Input | Action |
|-------|--------|
| `0` | Label as **Other** |
| `1` | Label as **ADS** (Adult-Directed Speech) |
| `2` | Label as **IDS** (Infant-Directed Speech) |
| `r` | Replay current segment |
| `b` | Go back one segment |
| `q` | Save progress & quit (auto-resumes next time) |

### Annotation Output

```
<annotations_root>/<participant_id>/
  ├── <pid>_female_annotations.csv     # Segment-level labels
  ├── <pid>_male_annotations.csv
  ├── <pid>_female_ADS.wav             # All female ADS segments concatenated
  ├── <pid>_female_IDS.wav             # All female IDS segments concatenated
  ├── <pid>_female_Other.wav
  ├── <pid>_male_ADS.wav
  ├── <pid>_male_IDS.wav
  └── <pid>_male_Other.wav
```

The CSV contains per-segment timestamps and labels:

| Column | Description |
|--------|-------------|
| `segment_index` | Segment number (0-based) |
| `start_sec` | Start time within the source WAV |
| `end_sec` | End time within the source WAV |
| `duration_sec` | Segment duration |
| `label` | Numeric label (0, 1, or 2) |
| `label_name` | Human-readable label (Other, ADS, IDS) |

> **Note:** The annotation tools (`scripts/annotate_ads_ids.py` and `notebooks/02_annotation_player.ipynb`) are standalone utilities and are **not** part of the `hindibabynet` package.

---

## Pipeline Details (XGB, 9 Steps)

Stage 03 with the XGB backend internally executes these sub-steps per participant:

| Step | What Happens | Method |
|------|-------------|--------|
| **1** | Audio preprocessing | Mono, 16 kHz resample, peak normalize to −1 dBFS (done in Stage 02) |
| **2** | Voice Activity Detection | WebRTC-VAD (aggressiveness=2, 30 ms frames, min 300 ms regions) |
| **3** | Speaker Diarization | pyannote/speaker-diarization-3.1, chunked (15 min chunks, 10 s overlap) |
| **4** | Speech-only segments | Intersect VAD ∩ diarization turns, drop < 0.2 s, merge gaps ≤ 0.7 s |
| **5** | Speech-type classification | eGeMAPSv02 (88-dim) + XGBoost → 4 classes |
| **6** | Stream aggregation | Concatenate all `adult_female` segments → female stream, etc. |
| **7** | Secondary diarization | Diarize each gender stream to find the *dominant* speaker |
| **8** | Main caregiver export | Write `<pid>_main_female.wav`, `<pid>_main_male.wav`, `<pid>_child.wav`, `<pid>_background.wav` |
| **9** | TextGrid generation | One `.TextGrid` per participant with FEM / MAL / CHILD / BACKGROUND tiers |

### Classification Model

The XGBoost model (`models/xgb_egemaps.pkl`) classifies 1-second audio windows using **88-dimensional eGeMAPSv02 Functionals** features:

| Class Index | Label | Description |
|-------------|-------|-------------|
| 0 | `adult_male` | Adult male speech |
| 1 | `adult_female` | Adult female speech |
| 2 | `child` | Child / infant vocalisations |
| 3 | `background` | Non-speech, environmental noise |

**Windowing strategy** (for segments longer than 1 s):
- 1.0 s windows with 0.5 s hop
- End-anchored last window to cover the full segment
- Per-window probability aggregation via **weighted mean**
- Short segments (< 1 s) → single zero-padded window

---

## Project Structure

```
HindiBabyNet/
├── configs/
│   └── config.yaml                    # All pipeline parameters
├── external_models/
│   └── VTC/                           # 🔧 Cloned VTC repo (separate venv, optional)
├── models/
│   └── xgb_egemaps.pkl                # Pre-trained 4-class XGBoost classifier
├── scripts/
│   ├── run_all.sh                     # ⭐ Full end-to-end pipeline
│   ├── run_stage_01.sh                # Stage 01 only
│   ├── run_stage_02_from_parquet.sh   # Stage 02 batch
│   ├── run_stage_02_single_wav.sh     # Stage 02 single WAV
│   ├── run_stage_03.sh               # Stage 03 (single / batch / parquet, --backend xgb|vtc)
│   └── annotate_ads_ids.py            # ⭐ ADS/IDS annotation tool (standalone)
├── src/hindibabynet/
│   ├── check_setup.py                 # Environment/config diagnostics
│   ├── cli/
│   │   ├── run_all.py
│   │   └── run_stage_03.py
│   ├── components/
│   │   ├── data_ingestion.py          # Stage 01: scan & catalogue WAVs
│   │   ├── audio_preparation.py       # Stage 02: combine, resample, normalize
│   │   └── speaker_classification/
│   │       ├── base.py
│   │       ├── dispatcher.py
│   │       ├── xgb_backend.py
│   │       ├── vtc_backend.py
│   │       ├── output_checks.py
│   │       └── metadata.py
│   ├── config/
│   │   └── configuration.py           # ConfigurationManager (reads config.yaml)
│   ├── entity/
│   │   ├── config_entity.py           # Frozen config dataclasses
│   │   └── artifact_entity.py         # Frozen artifact dataclasses
│   ├── exception/
│   │   └── exception.py               # Custom exception with context
│   ├── logging/
│   │   └── logger.py                  # Console + file logging
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_audio_preparation_from_parquet.py
│   │   ├── stage_02_audio_preparation_single_wav.py
│   │   └── stage_03_speaker_classification.py  # Dispatches to xgb or vtc backend
│   └── utils/
│       ├── audio_utils.py             # Streaming resample, normalize, concatenate
│       └── io_utils.py                # YAML, JSON, Parquet, run_id helpers
├── tests/
│   ├── test_smoke.py                  # Import smoke tests
│   └── test_vtc_integration.py        # VTC backend unit tests
├── notebooks/
│   ├── 00_research.ipynb              # Research notebook (source of truth for ML logic)
│   └── 02_annotation_player.ipynb     # ⭐ ADS/IDS annotation notebook (listen + label)
├── docs/
│   └── pipeline_specification.md      # Formal pipeline specification
├── artifacts/                         # Auto-created: run artifacts & metadata
├── logs/                              # Auto-created: per-run log files
├── pyproject.toml                     # Dependencies & project metadata
└── .env                               # HF_TOKEN (not committed to git)
```

---

## Migration Notes

If you are upgrading from an older HindiBabyNet layout, update:

1. `speaker_classification.output_audio_root` → `speaker_classification.output_root`
2. Flat Stage 03 XGB keys under `speaker_classification.*` → `speaker_classification.xgb.*`
3. Top-level `vtc.*` block → nested `speaker_classification.vtc.*`

Backward-compatibility fallbacks are implemented for old keys, but new projects should use the nested schema.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `HF_TOKEN not loaded` | Create a `.env` file in the project root: `echo "HF_TOKEN=hf_..." > .env` |
| `Failed to build webrtcvad` (Windows) | Install Microsoft C++ Build Tools, or run VTC-only setup (`uv sync`) without XGB extra. |
| `webrtcvad needs sr in (8k,16k,32k,48k)` | Your input WAV has an unusual sample rate. Run Stage 02 first to resample to 16 kHz. |
| `No <pid>/<pid>.wav files found` | Stage 02 hasn't been run yet, or the `processed_audio_root` path in config.yaml is wrong. |
| Out of GPU memory | Reduce `chunk_sec` in config.yaml (e.g., from 900 to 300). Smaller chunks use less VRAM. |
| Very slow (no GPU) | The pipeline works on CPU but diarization is 10-50× slower. Use a CUDA-capable GPU. |
| `Model not found: models/xgb_egemaps.pkl` | Ensure the pre-trained model file is present in the `models/` directory. |
| `recordings.parquet not found` | Run Stage 01 first (`uv run bash scripts/run_stage_01.sh`) to scan your audio directory. |

### Logs

Every pipeline run creates timestamped log files:

```
logs/<run_id>/
  ├── stage_01_data_ingestion.log
  ├── stage_02_audio_preparation_batch.log
  └── stage_03_speaker_classification_<backend>.log
```

Check these for detailed progress, warnings, and error tracebacks.

---

## License

See [LICENSE](LICENSE).
