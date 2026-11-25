# audio: audio preparation and speaker classification

This workflow has two parts:

1. **Audio preparation.** The raw recordings are changed to one channel (mono) and 16 kHz, and the volume is normalised. For the HindiBabyNet data, the recordings of the first recording day of each child are joined into one audio file. Other folder structures are also possible, see section 3.
2. **Speaker classification.** The prepared audio is divided into speech segments of different speakers. There are two methods. Choose one of them, they are not mixed:
   - **VTC** (Voice Type Classifier 2.0). It gives four classes: adult female (FEM), adult male (MAL), key child (KCHI) and other child (OCH). The statistical analyses and the prosody extraction of HindiBabyNet use the VTC results.
   - **XGBoost**. Our own model, trained on eGeMAPS features. It gives the classes adult male, adult female, child and background.

If you already have prepared audio and VTC results, you do not need this workflow. Write their folders in `config.yaml` (`prepared_audio` and `vtc`) and use the other workflows.

## 1. Where to run the commands

There are two places to type the commands:

- **Windows PowerShell**: the terminal of Windows. Open it from the Start menu (type "PowerShell").
- **Ubuntu**: a Linux terminal. On a Windows computer, Ubuntu runs inside Windows (WSL).

| Step | Windows PowerShell | Ubuntu |
| --- | --- | --- |
| Prepare the audio (`hbn-audio prepare`) | yes | yes |
| Speaker classification with VTC (`hbn-audio classify-vtc`) | no | yes |
| Speaker classification with XGBoost (`hbn-audio classify-xgb`) | yes | yes |
| Optional tools (`hbn-audio textgrid`, `annotate`, `evaluate`) | yes | yes |

Choose one place and use it for all commands of this workflow:

- **You want to use VTC:** use Ubuntu. VTC does not run on Windows directly.
- **You want only the audio preparation and XGBoost:** use Windows PowerShell or Ubuntu. With Windows PowerShell you do not need to install Ubuntu.

Do not change between the two places. Each place makes its own Python environment in the folder `audio/.venv`, and the other place cannot use it. If a command stops with an error about `audio/.venv`, delete this folder and run the command again. The folder has only the Python packages, no data.

| Where | Command to delete the folder |
| --- | --- |
| Windows PowerShell | `cmd /c "rmdir /s /q audio\.venv"` |
| Ubuntu | `rm -rf audio/.venv` |

The `hbn-audio` commands are the same in both places. Where something is different, this guide shows it for **Windows PowerShell** and for **Ubuntu**.

## 2. Set up

Do this once, for the place that you have chosen.

### Windows PowerShell

1. Install uv with the Windows command in section 1 of the main [README](../README.md#1-install-the-software).
2. Open Windows PowerShell and go to the project folder:

```
cd C:\Users\yourname\Projects\HindiBabyNet
```

### Ubuntu

1. Install Ubuntu. Open **Windows PowerShell** as administrator and run:

```
wsl --install -d Ubuntu
```

Restart the computer. Ubuntu opens and asks for a user name and a password. Choose them and remember the password. Later, open Ubuntu from the Windows Start menu (type "Ubuntu"). All commands of the next steps are typed in this Ubuntu window.

2. Install uv with the Ubuntu command in section 1 of the main [README](../README.md#1-install-the-software). Then install the programs that VTC needs:

```bash
sudo apt update
sudo apt install ffmpeg git git-lfs
```

Close Ubuntu and open it again. Check that the programs are found:

```bash
uv --version
ffmpeg -version
git lfs version
```

3. Go to the project folder. The Windows drives are under `/mnt` in Ubuntu. The folder `C:\Users\yourname\Projects\HindiBabyNet` is `/mnt/c/Users/yourname/Projects/HindiBabyNet`.

```bash
cd /mnt/c/Users/yourname/Projects/HindiBabyNet
```

### Both places

- Run all `hbn-audio` commands from the project folder.
- Set your folders in `config.yaml` as described in section 2 of the main [README](../README.md#2-set-your-folders). The same `config.yaml` works in Windows PowerShell and in Ubuntu.
- The first run of a command installs the Python packages of this workflow. In Ubuntu this takes some minutes when the project folder is on a Windows drive.
- In Ubuntu, recordings in the Ubuntu home folder are read much faster than recordings on a Windows drive. If you copy the raw recordings to Ubuntu, for example to `/home/yourname/my_recordings`, write this path as `raw_audio`.

## 3. Prepare the audio

The preparation runs in Windows PowerShell or in Ubuntu.

The preparation reads the raw WAV files and writes prepared audio: one channel (mono), 16 kHz, volume normalised to a peak of -1 dBFS. The original recordings are only read. They are not changed.

### Folder structures of the raw recordings

You do not need to change the folders of your recordings. The command understands these structures:

| Structure | Example | What is prepared |
| --- | --- | --- |
| Participant folders with recording days | `P001/20250323/REC_0001.WAV` | one file per participant: the WAV files of the first recording day are joined |
| Participant folders without dates | `P001/REC_0001.WAV` | one file per participant: all WAV files of the folder are joined (needs `recording_days: all`) |
| WAV files directly in the folder | `REC_0001.WAV` | every WAV file alone, with the same name |
| One WAV file | `raw_audio` is the file itself | this file, with the same name |

A date folder has 8 digits (year, month, day), for example `20250323`. One raw folder can have participant folders and single files together.

The HindiBabyNet recordings have the first structure:

```
raw/audio/
├── P001/
│   ├── 20250323/
│   │   ├── REC_0001.WAV
│   │   └── REC_0002.WAV
│   └── 20250324/
│       └── REC_0003.WAV
└── P002/
    └── 20250325/
        └── REC_0004.WAV
```

The recorder also saves an `.ASR` file next to each `.WAV` file. The preparation does not use the `.ASR` files.

### Settings

The settings are in the file `audio/params.yaml`. The default values are the values of the HindiBabyNet study. Do not change them for an existing study.

| Setting | Values | Meaning |
| --- | --- | --- |
| `audio_preparation.join_multiple_files` | `true` (default), `false` | `true`: the WAV files of one participant are joined to one file. `false`: every WAV file is prepared alone. |
| `audio_preparation.recording_days` | `first` (default), `all` | `first`: only the earliest date folder of a participant is used. `all`: all date folders, or all files of a participant folder without dates. |
| `raw_audio.structure` | `auto` (default), `participants`, `files` | `auto`: the structure is found from the folders, as in the table above. Use the other values only when `auto` reports that the structure is not clear. |

- `structure: participants`: every folder directly inside the raw folder is one participant. All WAV files below it belong to this participant, also in sub-folders.
- `structure: files`: every WAV file is its own recording, also in sub-folders. Nothing is joined.

### Examples

**HindiBabyNet data (participant and date folders).** Nothing to change. Only the first recording day is used.

```
Input:   P001/20250323/REC_0001.WAV, P001/20250323/REC_0002.WAV, P001/20250324/REC_0003.WAV
Output:  derived/prepared-audio/P001.wav          (REC_0001 and REC_0002 joined)
```

**All recording days of a participant.** Set `recording_days: all`.

```
Output:  derived/prepared-audio/P001.wav          (REC_0001, REC_0002 and REC_0003 joined)
```

**Participant folders without dates.** Set `recording_days: all`. With `first` the command stops for this participant, because it cannot know the first day.

```
Input:   P001/REC_0001.WAV, P001/REC_0002.WAV, P002/REC_0003.WAV
Output:  derived/prepared-audio/P001.wav, derived/prepared-audio/P002.wav
```

**WAV files in one folder.** Nothing to change in `params.yaml`. Write the folder in `config.yaml`:

```yaml
paths:
  raw_audio: D:/my_recordings
```

```
Input:   D:/my_recordings/REC_0001.WAV, D:/my_recordings/REC_0002.WAV
Output:  derived/prepared-audio/REC_0001.WAV, derived/prepared-audio/REC_0002.WAV
```

**One WAV file.** Write the file in `config.yaml`:

```yaml
paths:
  raw_audio: D:/my_recordings/REC_0001.WAV
```

**Participant folders, but no joining.** Set `join_multiple_files: false`. Every file keeps its name and its folder, so files with the same name in different folders stay apart.

```
Input:   P001/20250323/REC_0001.WAV, P002/20250325/REC_0001.WAV
Output:  derived/prepared-audio/P001/20250323/REC_0001.WAV
         derived/prepared-audio/P002/20250325/REC_0001.WAV
```

### Run the preparation

First look at what is there. This command only reads and does not write anything:

```
uv run --project audio hbn-audio prepare --check
```

For each prepared file that is planned it shows the structure that was found, the recording days, the number of WAV files, and if a prepared file already exists. It also compares the duration of the raw files with the duration of the prepared file.

Then prepare the audio:

```
uv run --project audio hbn-audio prepare
```

Output in `derived/prepared-audio/`:

- the prepared WAV file, for example `P001.wav`.
- a record next to it, for example `P001_preparation.json`: the structure that was found, the selected date folders, the names of the source files in the order they were joined, and the settings.

The name of a prepared recording is the place of the file without the ending: `P001`, `REC_0001` or `P001/20250323/REC_0001`. The classification results are saved in a folder with this name. To work on one participant or one recording only, add `--participant P001`. To try the first two recordings, add `--limit 2`.

### Rules

- Recordings of different participants are never joined.
- Files are joined in this order: date folders from the earliest to the latest, and inside a folder by file name. Names are compared as text, so `REC_10.WAV` comes before `REC_2.WAV`. The command prints a note when this happens. Use names like `REC_02.WAV` and `REC_10.WAV` to avoid it.
- Files that are joined must have the same sample rate and the same number of channels.
- If the first day folder is empty or a WAV file of this day is damaged, the command reports an error for this participant. It does not take another day. Correct the folder and run the command again.
- If the structure of a folder is not clear, the command reports an error for this folder and prepares nothing for it. Examples: WAV files in the participant folder and also in sub-folders, sub-folders that are not dates, or a first folder whose name is a date. Set `raw_audio.structure` in this case.
- If two recordings would get the same prepared file, the command reports an error for both and prepares nothing for them. Names that differ only in capital letters count as the same.
- A folder without WAV files is not used. The command prints a note.
- If a prepared file already exists, it is kept and not prepared again. An error in one recording does not stop the other recordings.

### Your own prepared files

If your audio is already prepared, you can skip this step. Put the WAV files (mono, 16 kHz) into `derived/prepared-audio/`, as `<name>.wav` or as `<name>/<name>.wav`, or write their folder as `prepared_audio` in `config.yaml`.

### What the other workflows need

The classification (VTC, XGBoost) and the optional tools work with every prepared file. The statistics (`audio-stats`) and the prosody extraction (`prosody`) need one prepared recording per participant, and its name must be the participant ID in the metadata files. This is the case when the audio is joined, and also for single files that are named after the participant.

If your raw recordings do not have date folders, change one setting of `audio-stats`. It is explained in the audio-stats part of the main [README](../README.md#audio-stats).

## 4. Speaker classification with VTC

VTC runs only in Ubuntu. Type all commands of this section in the Ubuntu window.

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

In the Ubuntu window, in the project folder:

```bash
uv run --project audio hbn-audio classify-vtc
```

- Input: every prepared WAV file in `derived/prepared-audio/`. The files must be mono and 16 kHz, as the preparation writes them.
- Output for each prepared recording, in `derived/vtc-classification/<recording>/` (for example `derived/vtc-classification/P001/` for `P001.wav`):
  - `rttm.csv`: one row per speech segment (start time, duration, speaker class).
  - `rttm/<file name>.rttm`: the same segments in RTTM format.
  - `run_info.json`: the command, the VTC version and the run time.
- Batch size is 32. It is set in `audio/params.yaml`.
- VTC is slow on a CPU. The VTC documentation gives about 4 minutes for 1 hour of audio on a CPU. Start with one recording: `uv run --project audio hbn-audio classify-vtc --limit 1`.

If the results of a recording are already available (`rttm.csv` and the folder `rttm`), VTC is not run again for this recording. Existing results are never changed.

VTC always classifies all WAV files of one folder. For this reason the command gives each prepared file to VTC through a temporary folder in Ubuntu that has only a link to this file. The prepared file is not copied or changed, and the temporary folder is removed after the run.

### How the results are used

`audio-stats` counts the segments and their durations per speaker from `rttm.csv`. `prosody` uses the segments of `rttm.csv` to cut the speech of each speaker from the prepared audio.

### If VTC stops with an error

| Problem | What to do |
| --- | --- |
| `VTC runs only in Ubuntu` | You started the command in Windows PowerShell. Open Ubuntu and run it there. |
| `uv` or `scripts/infer.py` is not found | Check section 2 and the folder in `vtc_repo`. |
| Error about `torchcodec` or FFmpeg libraries | The `torchcodec` package of VTC could not load the FFmpeg libraries. On our computer (Ubuntu 20.04) we installed FFmpeg libraries in a separate micromamba environment named `ffmpeg-libs` and added its `lib` folder to `LD_LIBRARY_PATH` in the file `~/.bashrc`: `export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg-libs/lib:$LD_LIBRARY_PATH"`. With another Ubuntu version this may not be needed. |
| `incomplete results` for a recording | An earlier run stopped in the middle. Move the folder of this recording out of `derived/vtc-classification/` and run again. |
| `Two prepared files have the same name` | The prepared audio folder has for example `P001.wav` and `P001/P001.wav`. Keep only one of them. |

## 5. Speaker classification with XGBoost

XGBoost runs in Windows PowerShell or in Ubuntu.

The trained model is in this repository: `audio/models/xgb_egemaps.pkl`. It is not trained again.

### Steps of the method

1. **Speech chunks.** Voice activity detection (WebRTC VAD) cuts the recording into chunks of speech. The parts between the chunks are not used.
2. **Classification.** Every chunk is classified alone: eGeMAPS features (openSMILE) are calculated for the chunk and the XGBoost model gives the class.
3. **Merging.** After the classification, chunks that follow each other are merged into one segment when they have the same class and the pause between them is not longer than 0.3 seconds. Chunks are not merged across a chunk of another class. The classified chunks are kept as they are, and the merged segments are saved in a second table.

### Run

The command is the same in Windows PowerShell and in Ubuntu:

```
uv run --project audio --extra xgb hbn-audio classify-xgb
```

The first run installs the XGBoost packages (about 0.5 GB). You do not need an account, an access token or a C compiler.

If the results of a recording are already available, the recording is not run again. Existing results are never changed. If a run was stopped in the middle, the command reports an incomplete folder. Move this folder out of `derived/xgb-classification/` and run again.

### Settings

The settings are in the part `xgb` of `audio/params.yaml`.

| Setting | Value | Meaning |
| --- | --- | --- |
| `vad.aggressiveness` | 1 | 0 to 3. A higher value removes more sound that is not clear speech. |
| `vad.frame_ms` | 30 | The VAD decides for every frame of this length if it is speech (10, 20 or 30 ms). |
| `vad.min_region_ms` | 90 | Chunks shorter than this are not used. |
| `vad.min_silence_ms` | 60 | A pause of this length or longer ends a chunk. Shorter pauses stay inside the chunk. With frames of 30 ms, the value 60 means that one frame without speech stays inside the chunk, and two or more such frames after each other end the chunk. |
| `segmentation.merge_before_classification` | `false` | `true`: chunks with a pause up to `merge_gap_sec` are joined before the classification. The class is not known at this point, so chunks of different speakers can be joined. |
| `segmentation.merge_after_classification` | `true` | `true`: step 3 is done. `false`: only the table of the classified chunks is written. |
| `segmentation.merge_gap_sec` | 0.3 | The longest pause, in seconds, between two chunks that are merged. |

The features are calculated in windows of 1 second (`classification.win_sec`) with a step of 0.5 seconds (`classification.hop_sec`). A chunk shorter than 1 second is filled with silence up to 1 second for the features.

### Output

For each prepared recording, in `derived/xgb-classification/<recording>/`. The two tables are CSV files, like the `rttm.csv` of VTC:

| File | Content |
| --- | --- |
| `<name>_segments.csv` | the classified chunks, one row per chunk: `start_sec`, `end_sec`, `duration_sec`, `predicted_class`, the probability of each class, and `merged_segment` (the number of the merged segment that the chunk belongs to) |
| `<name>_merged_segments.csv` | the merged segments, one row per segment: `start_sec`, `end_sec`, `predicted_class`, `n_chunks`, `speech_duration_sec` and `span_sec` |
| `<name>.TextGrid` | the classified chunks as a Praat TextGrid |
| `<name>_summary.json` | the settings, the number and the length of the chunks, and the speech duration per class |
| `run_info.json` | the start time, the run time and the status of the run |

Two columns of the merged segments give a duration:

- `speech_duration_sec` is the sum of the chunks of the segment. The pauses between the chunks are **not** counted. Use this column for the amount of speech.
- `span_sec` is the time from the start of the first chunk to the end of the last chunk, with the pauses.

The speech durations in the summary and the TextGrid are made from the chunks, so a pause is never counted as speech. Audio files of the classes are not written.

The XGBoost results and the VTC results are saved in different folders. One method is never used in place of the other.

### Speaker diarization (switched off)

An earlier version of this method used speaker diarization (pyannote) to divide the speech by speaker before the classification, and a second diarization to find the main female and the main male speaker. These steps are switched off, because the VAD chunks are classified directly. The code, the settings and the packages are kept as comments. To use them again, remove the `#` signs at the places marked "Not used at present" in `audio/src/hindibabynet_audio/xgb.py`, `audio/params.yaml` and `audio/pyproject.toml`, and run `uv lock --project audio`.

## 6. Optional tools

These tools are for checking the automatic classification. They are not needed for the main workflow. They run in Windows PowerShell or in Ubuntu. Use `--method vtc` or `--method xgb`.

| Command | What it does | Output |
| --- | --- | --- |
| `uv run --project audio hbn-audio textgrid --method vtc` | makes Praat TextGrid files from the classification results | `derived/textgrids/` |
| `uv run --project audio hbn-audio annotate --method vtc --participant <participant>` | plays sampled segments and asks you for the speaker label | `derived/manual-annotations/` |
| `uv run --project audio hbn-audio evaluate --method vtc` | compares the automatic labels with your labels (accuracy, confusion matrix, precision, recall, F1) | `results/classification-evaluation/` |

The protocols for manual annotation and evaluation are in the folder [references](../references).
