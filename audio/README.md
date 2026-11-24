# audio: audio preparation and speaker classification

This workflow has two parts:

1. **Audio preparation.** The recordings of the first recording day of each child are joined into one audio file (mono, 16 kHz).
2. **Speaker classification.** The prepared audio is divided into speech segments of different speakers. There are two methods. Choose one of them, they are not mixed:
   - **VTC** (Voice Type Classifier 2.0). It gives four classes: adult female (FEM), adult male (MAL), key child (KCHI) and other child (OCH). The statistical analyses and the prosody extraction of HindiBabyNet use the VTC results.
   - **XGBoost**. Our own model, trained on eGeMAPS features. It gives the classes adult male, adult female, child and background.

All commands of this workflow are run in **Ubuntu**. On a Windows computer, Ubuntu runs inside Windows (WSL). VTC does not run on Windows directly.

If you already have prepared audio and VTC results, you do not need this workflow. Write their folders in `config.yaml` (`prepared_audio` and `vtc`) and use the other workflows.

## 1. Install Ubuntu on Windows

Do this once.

1. Open **Windows PowerShell** as administrator.
2. Run:

```
wsl --install -d Ubuntu
```

3. Restart the computer. Ubuntu opens and asks for a user name and a password. Choose them and remember the password.

Later, open Ubuntu from the Windows Start menu (type "Ubuntu"). All commands below are typed in this Ubuntu window.

## 2. Install the required programs in Ubuntu

```bash
sudo apt update
sudo apt install ffmpeg git git-lfs
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close Ubuntu and open it again. Check that the programs are found:

```bash
uv --version
ffmpeg -version
git lfs version
```

## 3. Open the project folder in Ubuntu

The Windows drives are under `/mnt` in Ubuntu. The folder `C:\Users\yourname\Projects\HindiBabyNet` is `/mnt/c/Users/yourname/Projects/HindiBabyNet`.

```bash
cd /mnt/c/Users/yourname/Projects/HindiBabyNet
```

Run all `hbn-audio` commands from this folder. They read the data folders from `config.yaml` (see the main [README](../README.md)). You can write Windows paths (`D:/...`) or Ubuntu paths (`/home/yourname/...`) in `config.yaml`. Both work.

Recordings in the Ubuntu home folder are read much faster than recordings on a Windows drive. If you copy the raw recordings to Ubuntu, for example to `/home/yourname/hindinet_data/RawAudioData`, write this path as `raw_audio` in `config.yaml`.

## 4. Prepare the audio

The raw recordings must have this structure. The date folder has 8 digits (year, month, day):

```
raw/audio/<participant>/<YYYYMMDD>/*.WAV
```

The recorder also saves an `.ASR` file next to each `.WAV` file. The preparation does not use the `.ASR` files.

First look at what is there. This command only reads and does not write anything:

```bash
uv run --project audio hbn-audio prepare --check
```

For each participant it shows the first recording day, the number of WAV files of this day, and if a prepared file already exists. It also compares the duration of the first-day recordings with the duration of the prepared file.

Then prepare the audio:

```bash
uv run --project audio hbn-audio prepare
```

What it does for each participant:

- It takes the earliest date folder. Only the WAV files of this day are used. Recordings of later days are never used.
- If there is more than one WAV file, the files are joined in the order of their names.
- The audio is converted to mono, 16 kHz, and the volume is normalised to a peak of -1 dBFS.
- The original recordings are only read. They are not changed.

Output for each participant, in `derived/prepared-audio/<participant>/`:

- `<participant>.wav`: the prepared audio.
- `<participant>_preparation.json`: the selected date, the names of the source files, and the settings.

Important rules:

- If the first day folder is empty or a WAV file of this day is damaged, the command reports an error for this participant. It does not take another day. Correct the folder and run the command again.
- If a prepared file already exists, it is kept and not prepared again.
- To prepare only one participant, add `--participant <participant>`. To try the first two participants, add `--limit 2`.

## 5. Speaker classification with VTC

### What VTC is

VTC is a program of the LAAC-LSCP group: <https://github.com/LAAC-LSCP/VTC>. It is not part of this repository. It is installed once in its own folder and has its own Python environment. HindiBabyNet only calls it.

### Install VTC

We used VTC version 2.0.2. Install it in your Ubuntu home folder:

```bash
cd ~
git lfs install
git clone --recurse-submodules https://github.com/LAAC-LSCP/VTC.git
cd VTC
git checkout dcf5b4f03b68e1800a322f9075eabdb451d4e289
git submodule update --init --recursive
uv sync
```

`git checkout` selects the same version that we used. The model file (about 1 GB) is downloaded by `git lfs` into the folder `~/VTC/VTC-2.0/model`. You do not need to configure the model.

Check the installation:

```bash
cd ~/VTC
./check_sys_dependencies.sh
ls -lh VTC-2.0/model
```

The first command must say that all system dependencies are installed. The second must show `best.ckpt` with a size of about 1 GB. If the file is very small, run `git lfs pull` inside `~/VTC/VTC-2.0`.

If VTC is installed in another folder, write the folder in `config.yaml`:

```yaml
audio:
  vtc_repo: ~/VTC
  vtc_device: cpu      # use cuda if the computer has an NVIDIA GPU
```

### Run VTC

```bash
cd /mnt/c/Users/yourname/Projects/HindiBabyNet
uv run --project audio hbn-audio classify-vtc
```

- Input: the prepared audio in `derived/prepared-audio/`.
- Output for each participant, in `derived/vtc-classification/<participant>/`:
  - `rttm.csv`: one row per speech segment (start time, duration, speaker class).
  - `rttm/<participant>.rttm`: the same segments in RTTM format.
  - `run_info.json`: the command, the VTC version and the run time.
- Batch size is 32. It is set in `audio/params.yaml`.
- VTC is slow on a CPU. The VTC documentation gives about 4 minutes for 1 hour of audio on a CPU. Start with one participant: `uv run --project audio hbn-audio classify-vtc --limit 1`.

If the results of a participant are already available (`rttm.csv` and the folder `rttm`), VTC is not run again for this participant. Existing results are never changed.

### How the results are used

`audio-stats` counts the segments and their durations per speaker from `rttm.csv`. `prosody` uses the segments of `rttm.csv` to cut the speech of each speaker from the prepared audio.

### If VTC stops with an error

| Problem | What to do |
| --- | --- |
| `VTC runs only in Ubuntu` | You started the command in Windows PowerShell. Open Ubuntu and run it there. |
| `uv` or `scripts/infer.py` is not found | Check section 2 and the folder in `vtc_repo`. |
| Error about `torchcodec` or FFmpeg libraries | The `torchcodec` package of VTC could not load the FFmpeg libraries. On our computer (Ubuntu 20.04) we installed FFmpeg libraries in a separate micromamba environment named `ffmpeg-libs` and added its `lib` folder to `LD_LIBRARY_PATH` in the file `~/.bashrc`: `export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg-libs/lib:$LD_LIBRARY_PATH"`. With another Ubuntu version this may not be needed. |
| `incomplete results` for a participant | An earlier run stopped in the middle. Move this participant's folder out of `derived/vtc-classification/` and run again. |

## 6. Speaker classification with XGBoost

The trained model is in this repository: `audio/models/xgb_egemaps.pkl`. It is not trained again.

The method has these steps: voice activity detection (WebRTC VAD), speaker diarization (pyannote), eGeMAPS features (openSMILE) for each speech segment, and classification of each segment with the XGBoost model.

Before the first run:

1. The diarization uses the model `pyannote/speaker-diarization-3.1` from Hugging Face. Make a free account on <https://huggingface.co>, accept the conditions on the page of this model, and create an access token.
2. Give the token to the program in the Ubuntu window:

```bash
export HF_TOKEN=your_token
```

Run:

```bash
uv run --project audio --extra xgb hbn-audio classify-xgb
```

The first run installs the XGBoost packages. This is a large download. If the installation stops at the package `webrtcvad`, a C compiler is missing. Install it with `sudo apt install build-essential` and run the command again.

Output for each participant, in `derived/xgb-classification/<participant>/`: the segments table (`<participant>_segments.parquet`), a Praat TextGrid, a summary (`<participant>_summary.json`), `run_info.json`, and audio files with the speech of the main female speaker, the main male speaker, the child and the background.

The XGBoost results and the VTC results are saved in different folders. One method is never used in place of the other.

## 7. Optional tools

These tools are for checking the automatic classification. They are not needed for the main workflow. Use `--method vtc` or `--method xgb`.

| Command | What it does | Output |
| --- | --- | --- |
| `uv run --project audio hbn-audio textgrid --method vtc` | makes Praat TextGrid files from the classification results | `derived/textgrids/` |
| `uv run --project audio hbn-audio annotate --method vtc --participant <participant>` | plays sampled segments and asks you for the speaker label | `derived/manual-annotations/` |
| `uv run --project audio hbn-audio evaluate --method vtc` | compares the automatic labels with your labels (accuracy, confusion matrix, precision, recall, F1) | `results/classification-evaluation/` |

The protocols for manual annotation and evaluation are in the folder [references](../references).
