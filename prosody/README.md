# Prosodic and speech-model feature extraction

This workflow extracts participant-by-speaker acoustic summaries from VTC segment tables and the corresponding recordings. Despite the former folder name, it is **feature extraction, not model training**.

Package: `hindibabynet-prosody`; import name: `hindibabynet_prosody`. Code now lives here. Existing recordings, mappings, caches, outputs, native configuration, and old environment remain in their original ignored asset locations.

## Methods

1. Read RTTM CSV segments and map source identifiers to analysis identifiers.
2. Resolve the matching recording and verify segment boundaries against its timebase.
3. Extract pitch with Parselmouth/Praat using speaker-specific initial ranges and the configured adaptive second pass/fallback.
4. Extract and pool representations from the configured wav2vec 2.0 and HuBERT self-supervised speech models.
5. Aggregate features by participant and speaker, then fit separate configured standardization/PCA transformations for the two model representations.

The supplied example selects adult female, adult male, and other-child speech and uses three PCA components per representation. Preserve the active speaker selection, model identities/layers, pooling, pitch settings, seeds, and PCA configuration. PCA depends on the configured cohort; changing the participant subset changes the fitted representation, not just runtime.

**Pitch reference:** active prosody and the [root template](../config.example.yaml) use `pitch.semitone_reference_hz: 100.0`; the older [per-workflow example](config/config.example.yaml) uses `1.0`. Preserve the approved reference; these semitone origins are not interchangeable. See [scientific boundaries](../docs/README.md#scientific-and-privacy-boundaries).

## Setup and configuration

Complete the [root installation](../README.md). Keep this uv environment separate from CDI/vocal preprocessing. A compatible Python 3.10–3.12 interpreter suits the common project range; the prosody package itself declares a wider range. CPU execution is available; GPU use requires a compatible installed PyTorch/device environment. Real SSL extraction needs access to the configured model weights, with internet/authentication for first downloads where applicable. Cached remote weights are not pinned or fully verified by the launcher.

In the [local configuration](../config.local.yaml):

- Set `workflows.prosody.project` to `prosody` and enable it only when ready.
- For an existing study, preserve the approved `prosody` section with its original asset-based `project_root`; a new root-template copy uses `project_root: "."`.
- Check `paths.rttm_root`, `paths.audio_root`, and `paths.participant_lookup`. The lookup and RTTM column mappings must match the intended source data.
- Assign **fresh** `paths.output_dir` and `paths.cache_dir`. Do not reuse the existing research cache as a new launcher's output root.
- Keep scientific settings unchanged. For example, CPU participant workers and SSL batch size are separate controls, not reasons to alter pitch extraction or the cohort.

The full [root template](../config.example.yaml) already uses the new source folders and includes the prosody settings. Edit paths and enable extraction in the single root local configuration; no duplicate workflow YAML is needed. Private inputs and model access must be supplied separately.

## Run from the repository root

```text
uv run python run.py setup
uv run python run.py prosody --dry-run
uv run python run.py prosody
```

These assume explicit enablement and fresh configured destinations. To select CPU at the command line:

```text
uv run python run.py prosody --stage extract -- --device cpu
```

Optional diagnostic stages:

```text
uv run python run.py prosody --stage validate
uv run python run.py prosody --stage inspect -- --participant SYNTHETIC --max-segments 10
```

`SYNTHETIC` is a placeholder for an authorized local identifier, not supplied research data. Inspect requires `--participant`. Both diagnostics can write native logs into the configured output location, so do not treat them as read-only launcher checks or run them into the destination reserved for extraction. Use `--dry-run` for the read-only launcher plan and separate fresh diagnostic outputs when needed.

For extraction filters, pass `--participants` once per selected identifier and `--speakers` once per selected speaker after `--`. `--no-cache` is allowlisted but does not bypass output guards. Native `--resume`/`--overwrite` pass-through is not supported by the root launcher. Prefer the whole workflow command rather than chaining stages sharing output roots.

## Outputs and interpretation

The main table contains one row per participant-by-speaker combination: pitch summaries in Hz/semitones and separate wav2vec 2.0/HuBERT PCA scores. Configured outputs also include PCA/scaler artifacts, explained variance, skipped-segment and mapping/audio-resolution QC, processing summaries, logs, and run metadata.

It does not export a valid segment-level feature table or a raw high-dimensional embedding CSV. PCA scores are cohort-dependent summaries, not direct physical units or newly trained speech models. Review skipped segments, voiced-frame failures, and recording/RTTM alignment before interpreting results. Do not silently relax pitch or duration criteria to remove warnings.

Root resume verifies successful whole stages only; it does not promise participant-level recovery or adoption of old caches. See [recovery limitations](../docs/README.md#outputs-and-recovery).

## Tests and privacy

```text
uv run --project prosody --frozen --no-sync pytest
```

Real model-download integration checks are separate opt-ins; default checks do not prove full extraction on the cohort. Preserve attribution and licensing for Praat/Parselmouth, wav2vec 2.0, HuBERT, and the configured model providers. Keep all participant summaries, identifiers, QC, embeddings/caches, models, and reports private and ignored.
