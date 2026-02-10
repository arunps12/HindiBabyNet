# HindiBabyNet

> Automatic speaker classification pipeline for long-form Hindi child–caregiver audio recordings.

Given **raw WAV recordings** (one file or an entire directory tree), this pipeline automatically finds and separates **female adult speech**, **male adult speech**, **child vocalisations**, and **background noise** — then exports the main female and male caregiver audio as standalone files plus a Praat-compatible TextGrid for annotation review.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
   - [Process a Single Raw WAV](#option-a--process-a-single-raw-wav-file)
   - [Process a Directory of Raw WAVs (All Participants)](#option-b--process-a-directory-of-raw-wavs-all-participants)
   - [Run Individual Stages Manually](#running-individual-stages-manually)
6. [Pipeline Details](#pipeline-details-9-steps)
7. [Output Files](#output-files)
8. [Configuration Reference](#configuration-reference)
9. [Project Structure](#project-structure)
10. [Troubleshooting](#troubleshooting)

---

## What It Does

```
Raw WAV recordings
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 01 — Data Ingestion                              │
│  Scan directory tree, catalogue every WAV file          │
├─────────────────────────────────────────────────────────┤
│  Stage 02 — Audio Preparation                           │
│  Combine per participant → mono → 16 kHz → peak norm    │
├─────────────────────────────────────────────────────────┤
│  Stage 03 — Speaker Classification                      │
│  VAD → diarization → eGeMAPS features → XGBoost         │
│  → separate class streams → secondary diarization       │
│  → export main_female, main_male, child, background WAV │
│  → export TextGrid                                      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   <pid>_main_female.wav  <pid>_main_male.wav  <pid>_child.wav
   <pid>_background.wav   <pid>.TextGrid       <pid>_summary.json
```

**Four output classes:**

| Class | Description |
|-------|-------------|
| `adult_female` | Adult female speech (typically the mother) |
| `adult_male` | Adult male speech (typically the father) |
| `child` | Child / infant vocalisations |
| `background` | Non-speech or ambient noise |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 – 3.12 |
| **GPU** | NVIDIA GPU with CUDA (strongly recommended for pyannote diarization; CPU works but is very slow) |
| **[uv](https://docs.astral.sh/uv/)** | Fast Python package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| **HuggingFace token** | Required by the pyannote diarization model — get one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| **Disk space** | Intermediate files can be large for long recordings; ensure sufficient scratch space |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd HindiBabyNet

# 2. Install all dependencies (creates a virtual environment automatically)
uv sync

# 3. Create a .env file with your HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env
```

That's it — `uv sync` handles Python, virtual environment creation, and all packages (including PyTorch with CUDA).

---

## Configuration

Before running, edit **`configs/config.yaml`** to set the paths for your system:

```yaml
# ─── Where raw WAV files live ───
data_ingestion:
  raw_audio_root: /path/to/your/raw/audio     # ← CHANGE THIS

# ─── Where intermediate analysis WAVs are written ───
audio_preparation:
  processed_audio_root: /path/to/scratch/audio_processed   # ← CHANGE THIS

# ─── Where final classified outputs are written ───
speaker_classification:
  output_audio_root: /path/to/scratch/audio_classified     # ← CHANGE THIS
```

All other parameters (sample rate, VAD aggressiveness, diarization settings, etc.) have sensible defaults and usually don't need changing. See [Configuration Reference](#configuration-reference) below for the full list.

---

## Usage

### Option A — Process a Single Raw WAV File

If you have **one WAV file** and want to classify speakers in it:

```bash
# Step 1: Prepare the audio (mono, 16 kHz, normalized)
uv run bash scripts/run_stage_02_single_wav.sh /path/to/your/recording.wav

# Step 2: Find the prepared analysis WAV (printed by Step 1), then classify
uv run bash scripts/run_stage_03.sh --wav /path/to/scratch/audio_processed/recording/recording.wav
```

**What you get:**
- `recording_main_female.wav` — isolated main female speaker
- `recording_main_male.wav` — isolated main male speaker
- `recording_child.wav` — child vocalisations
- `recording_background.wav` — background / non-speech
- `recording.TextGrid` — Praat-compatible annotation with FEM / MAL / CHILD / BACKGROUND tiers
- `recording_segments.parquet` — every classified segment with timestamps and probabilities
- `recording_summary.json` — per-class duration statistics

### Option B — Process a Directory of Raw WAVs (All Participants)

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

**Run the entire pipeline end-to-end with a single command:**

```bash
# Process ALL participants automatically
uv run bash scripts/run_all.sh

# Process only the first N participants (useful for testing)
uv run bash scripts/run_all.sh 3
```

This runs all three stages sequentially:
1. **Stage 01** scans `raw_audio_root` (from config.yaml) and catalogues every WAV file
2. **Stage 02** combines all WAVs per participant into a single analysis-ready WAV
3. **Stage 03** runs the full classification pipeline on each participant

**All participants are processed automatically — no manual per-participant runs needed.**

### Running Individual Stages Manually

You can also run each stage independently:

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
    --analysis_dir /path/to/scratch/audio_processed

# ─── Stage 03: Classify from recordings parquet (needs Stage 02 done) ───
uv run bash scripts/run_stage_03.sh \
    --recordings_parquet artifacts/runs/<run_id>/data_ingestion/recordings.parquet

# ─── Stage 03: Limit to first N participants ───
uv run bash scripts/run_stage_03.sh \
    --analysis_dir /path/to/scratch/audio_processed --limit 5
```

---

## Pipeline Details (9 Steps)

Stage 03 internally executes the following sub-steps per participant:

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

## Output Files

For each participant, the pipeline produces:

```
<output_audio_root>/<participant_id>/
  ├── <participant_id>_main_female.wav   # Main female caregiver audio (16 kHz mono)
  ├── <participant_id>_main_male.wav     # Main male caregiver audio (16 kHz mono)
  ├── <participant_id>_child.wav         # Child vocalisations (16 kHz mono)
  └── <participant_id>_background.wav    # Background / non-speech (16 kHz mono)

artifacts/runs/<run_id>/speaker_classification/
  ├── <participant_id>_segments.parquet   # All classified segments
  ├── <participant_id>_summary.json      # Per-class duration statistics
  └── <participant_id>.TextGrid          # Praat-compatible annotation
```

### Segments Parquet Schema

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

## Configuration Reference

All parameters live in **`configs/config.yaml`**:

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
  model_path: models/xgb_egemaps.pkl
  class_names: ["adult_male", "adult_female", "child", "background"]
  egemaps_dim: 88
  output_audio_root: /path/to/scratch/audio_classified

  # VAD
  vad_aggressiveness: 2     # 0 (least aggressive) to 3 (most aggressive)
  vad_frame_ms: 30          # VAD frame size in milliseconds
  vad_min_region_ms: 300    # minimum speech region to keep (ms)

  # Diarization
  diarization_model: "pyannote/speaker-diarization-3.1"
  chunk_sec: 900.0          # chunk size for diarization (seconds)
  overlap_sec: 10.0         # overlap between chunks (seconds)
  min_speakers: 2           # minimum expected speakers per chunk
  max_speakers: 4           # maximum expected speakers per chunk

  # Merge & Classify
  merge_gap_sec: 0.7        # merge same-speaker segments with gaps ≤ this (seconds)
  min_segment_sec: 0.2      # discard segments shorter than this (seconds)
  classify_win_sec: 1.0     # eGeMAPS extraction window (seconds)
  classify_hop_sec: 0.5     # eGeMAPS window hop (seconds)
```

---

## Project Structure

```
HindiBabyNet/
├── configs/
│   └── config.yaml                    # All pipeline parameters
├── models/
│   └── xgb_egemaps.pkl                # Pre-trained 4-class XGBoost classifier
├── scripts/
│   ├── run_all.sh                     # ⭐ Full end-to-end pipeline
│   ├── run_stage_01.sh                # Stage 01 only
│   ├── run_stage_02_from_parquet.sh   # Stage 02 batch
│   ├── run_stage_02_single_wav.sh     # Stage 02 single WAV
│   └── run_stage_03.sh               # Stage 03 (single / batch / parquet)
├── src/hindibabynet/
│   ├── components/
│   │   ├── data_ingestion.py          # Stage 01: scan & catalogue WAVs
│   │   ├── audio_preparation.py       # Stage 02: combine, resample, normalize
│   │   └── speaker_classification.py  # Stage 03: VAD → diar → classify → export
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
│   │   └── stage_03_speaker_classification.py
│   └── utils/
│       ├── audio_utils.py             # Streaming resample, normalize, concatenate
│       └── io_utils.py                # YAML, JSON, Parquet, run_id helpers
├── tests/
│   └── test_smoke.py                  # Import smoke tests
├── notebooks/
│   └── 00_research.ipynb              # Research notebook (source of truth for ML logic)
├── docs/
│   └── pipeline_specification.md      # Formal pipeline specification
├── artifacts/                         # Auto-created: run artifacts & metadata
├── logs/                              # Auto-created: per-run log files
├── pyproject.toml                     # Dependencies & project metadata
└── .env                               # HF_TOKEN (not committed to git)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `HF_TOKEN not loaded` | Create a `.env` file in the project root: `echo "HF_TOKEN=hf_..." > .env` |
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
  └── stage_03_speaker_classification.log
```

Check these for detailed progress, warnings, and error tracebacks.

---

## License

See [LICENSE](LICENSE).
