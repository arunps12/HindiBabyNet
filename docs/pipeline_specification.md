# HindiBabyNet Long-Form Audio Processing Pipeline

## Dataset Layout (Input)

**Raw audio root directory:**
```
/scratch/users/arunps/hindibabynet/audio_raw/RawAudioData/
```

**Directory structure:**
```
RawAudioData/
  ├── <participant_id>/
  │     ├── <recording_date>/
  │     │     ├── <wav_file_1>.WAV
  │     │     ├── <wav_file_2>.WAV
  │     │     └── ...
  │     ├── <recording_date_2>/
  │     │     └── ...
  │     └── ...
  └── ...
```

**Rule:**  
All WAV files under the same `participant_id` belong to the same participant,
regardless of recording date.

---

## High-Level Objective

For each `participant_id`, process all WAV recordings and produce:

- `main_female.wav`
- `main_male.wav`
- `<participant_id>.TextGrid` aligned to original timeline with tiers:
  - `FEM` (main female)
  - `MAL` (main male)
  - `CHILD`
  - `BACKGROUND`

---

## Processing Pipeline (Exact Order)

### Step 1 — Audio Preprocessing
- Convert to mono
- Resample to 16 kHz
- Peak normalize to ≈ −1 dBFS
- **No noise reduction, compression, or filtering**

### Step 2 — Voice Activity Detection (VAD)
- Apply WebRTC-VAD
- Output `(start_sec, end_sec)` speech intervals

### Step 3 — Speaker Diarization
- Apply diarization per WAV (chunked if needed)
- Output speaker turns with local speaker IDs

### Step 4 — Speech-Only Segments
- Intersect VAD regions with diarization turns
- **Drop very short segments (e.g., < 0.2 seconds)**
- **Merge temporally close segments from same speaker**

#### Segment merging logic (reference implementation)

```python
def merge_close_segments(df, gap_thresh=0.5):
    ...
```
(Same speaker segments with gaps ≤ `gap_thresh` seconds are merged.)

### Step 5 — Speech-Type Classification
- Extract acoustic features
- Apply trained model from `models/`
- Labels:
  - adult_female
  - adult_male
  - child
  - background

### Step 6 — Global Aggregation (Per Participant)
- Combine all segments across recordings
- Create listen-only audio:
  - adult_female → female stream
  - adult_male → male stream

### Step 7 — Secondary Diarization
- Diarize female-only audio → find main female
- Diarize male-only audio → find main male

### Step 8 — Main Caregiver Audio Export
- Export:
  - `main_female.wav`
  - `main_male.wav`

### Step 9 — TextGrid Generation
- Create one TextGrid per participant
- Timeline aligned to original recordings
- Tiers: FEM, MAL, CHILD, BACKGROUND

---

## Constraints
- Group strictly by `participant_id`
- Preserve infant vocalizations
- All stages must save metadata
- Fully automatic (no manual per-participant runs)
