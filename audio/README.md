# Audio preparation and speaker classification

This workflow catalogues recordings, selects the configured sessions, joins/prepares audio, and classifies speaker types. TextGrid export, manual speaker-class annotation, and evaluation are separate optional stages. It does not perform CDI analysis, prosodic model training, or an active ADS/IDS annotation workflow.

Package: `hindibabynet-pipeline`; import name: `hindibabynet_pipeline`. Source is now in this folder. Existing recordings, model weights, native configurations, environments, and results remain at their original ignored asset locations.

## Setup and configuration

Complete the [root installation](../README.md) first. Use WSL/Linux for audio: the current external VTC integration assumes Unix-oriented repository/interpreter paths; native Windows audio VTC is not established as supported.

Required software is uv, a compatible Python 3.10–3.12 environment, and FFmpeg on `PATH`. Root `setup` installs the enabled audio project's frozen Python dependencies, not the external VTC environment or weights.

In the [local configuration](../config.local.yaml):

- Set `workflows.audio.project` to `audio`, and enable it only when ready.
- The full [root template](../config.example.yaml) already includes `audio` and `audio_params`; edit paths in the single root local configuration, without duplicate workflow YAMLs. The [runtime example](configs/config.example.yaml) and [parameter file](configs/params.yaml) remain native schema references.
- For an existing study, preserve both approved mappings and their original asset-based `project_root` values. A new root-template copy uses `project_root: "."`; review `audio.paths.raw_audio_root` and other input/model paths, using absolute paths where working-directory semantics would be ambiguous.
- Review session selection, joining, mono conversion, sample rate, peak normalization, and backend settings without changing their approved values. The workflow supports earliest-valid-date session selection; selecting all sessions is a methodological change, not a path repair.
- Assign fresh prepared/joined audio, classification, logs, artifacts, TextGrid, annotation, and evaluation write locations. Match downstream audio and RTTM paths to the same recording timebase.

### VTC

The [LAAC-LSCP Voice Type Classifier](https://github.com/LAAC-LSCP/VTC) is an external dependency, not included model weights. Obtain its approved checkout, submodules/LFS content, weights, lockfile, and separate working environment using its instructions. Configure `speaker_classification.vtc.repo_path` and the approved device/settings. Root setup does not clone, install, or download VTC. The launcher checks its inference script, lockfile, and local interpreter, but those checks are not a successful inference test.

VTC produces `FEM` (adult female), `MAL` (adult male), `KCHI` (key child), and `OCH` (other child) segments. It supplies the VTC inputs used by the vocal and prosody workflows.

### XGB

XGB is a distinct standalone route: voice activity detection and diarization, 88-dimensional eGeMAPSv02 features, and XGBoost classification into adult male, adult female, child, and background. It also exports speaker streams, segment tables, summaries, and TextGrids. Obtain the approved trained weights separately and configure their path; do not assume weights ship in a fresh clone. Any gated diarization model access must also be arranged privately.

For this optional stack, explicitly install the declared extra from the repository root:

```text
uv sync --project audio --frozen --extra xgb
```

Select `xgb` in the configuration, rather than passing an unsupported backend override through the root launcher. **Never substitute XGB outputs for VTC outputs in dependent analyses.** TextGrid/annotation dependencies may also require the relevant optional packages; minimal audio setup is not a guarantee that every manual tool is installed.

## Run from the repository root

After enabling audio, completing external setup, and choosing fresh outputs:

```text
uv run python run.py setup
uv run python run.py audio --dry-run
uv run python run.py audio
```

The default `pipeline` stage performs ingestion, preparation, and the configured classification backend. A limited run uses:

```text
uv run python run.py audio --stage pipeline -- --limit 1
```

Use a separate fresh configuration/output allocation for a limited test, not the destination intended for a subsequent complete run.

### Individual and manual stages

These are alternatives, not an in-place sequence guaranteed to pass output guards:

```text
uv run python run.py audio --stage prepare-audio -- --wav /path/to/input.wav
uv run python run.py audio --stage classify-vtc -- --wav /path/to/prepared.wav
uv run python run.py audio --stage generate-textgrids
uv run python run.py audio --stage annotate-segments
uv run python run.py audio --stage evaluate-models
```

Replace placeholder paths before use. `prepare-audio` requires either `--wav` or `--recordings-parquet`; only `--wav` accepts `--recording-id`. Classification accepts either source form too, with optional `--participant-id` only alongside `--wav`. `classify-vtc` or `classify-xgb` must match the configured backend. Batch manifest stages are not automatically resumed by the launcher.

TextGrid generation and annotation use the configured backend and can accept `--input-file` or `--participant-id`; annotation is interactive. Evaluation reads the configured manual annotation root and optionally accepts `--participant-id`. It does **not** accept `--input-csv`. Shared bookkeeping roots can make separate stages refuse existing files: review all write roots first and prefer the full workflow for routine processing. Annotation/evaluation are never automatically included in root `all`.

## Outputs and interpretation

Outputs go to configured locations, not automatically to this source folder:

- Prepared participant recordings and ingestion/preparation manifests.
- VTC: per-recording processed/raw RTTM directories and CSV segment tables, plus run metadata. Automatic VTC classification does not itself export caregiver streams or TextGrids.
- XGB: segment tables, class/dominant-speaker WAV streams, summary metadata, and TextGrids.
- Optional TextGrids normalize VTC labels to `adult_female`, `adult_male`, `key_child`, and `other_child`.
- Evaluation: accuracy/report, confusion matrix, classwise metrics, and evaluated annotations.

Segment boundaries/durations are in seconds on the corresponding recording. Joining, trimming, or replacing audio after classification can invalidate that timebase. Keep speaker mappings, processing settings, and recording identity with results.

The root launcher only reuses matching successful whole stages. Existing native caches, partial participant results, or logs are not automatically adopted. See [safe recovery](../docs/README.md#outputs-and-recovery).

## Tests and privacy

```text
uv run --project audio --frozen --no-sync pytest
```

Unit/synthetic checks do not verify external VTC installation, trained weights, native Windows support, or scientific equivalence on the research corpus. Keep recordings, classification outputs, annotations, logs, and weights private and ignored; review staged content before sharing. Respect the external model licenses and the local [audio license](LICENSE).
