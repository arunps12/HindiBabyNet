# Configuration and operating guide

Start with the [main README](../README.md) for installation and workflow selection. This guide explains the current launcher's boundaries; it does not change scientific methods or supply missing research data.

## Configuration and migration

Edit the ignored [local configuration](../config.local.yaml), not the checked-in template. Configuration selection is: explicit `--config`, then `HBN_CONFIG_PATH`, then the local configuration beside [run.py](../run.py). Relative `--config` paths are resolved from your current directory.

Two roots have different jobs:

| Setting | Meaning |
| --- | --- |
| `workflows.<name>.project` | Source directory; relative to the selected YAML's parent |
| `<section>.project_root` | Base for native section-relative asset paths; relative to that YAML's parent |

The source directories are `audio`, `vocal_stats`, `prosody`, `prosody_stats`, and `cdi`. Both `vocal-stats` and `vocal-models` use the `vocal` section and `vocal_stats` source directory. Other section names are `audio`, `audio_params`, `prosody`, `prosody_stats`, and `cdi`.

For a new study, copy the [root template](../config.example.yaml) beside itself only if the local configuration does not exist. It already contains current source paths and full `audio`, `audio_params`, `vocal`, `prosody`, `prosody_stats`, and `cdi` mappings, preserving native scientific settings, including the full CDI and prosody-statistics options. All workflows start disabled. The six `project_root: "."` values base paths at the root YAML's directory. Review the private `Data` inputs, `results` outputs, model locations, and local caches; select unoccupied write destinations and enable only the intended workflows.

For an existing study, keep the local configuration and its original asset-based `project_root` values to preserve relative input meaning. Source moves are complete; original asset directories remain in place. Changing a code path is not a reason to relocate assets or replace scientific settings. Use absolute paths where native working-directory semantics or Windows/WSL differences would be ambiguous.

Maintain **one root user-edited configuration**; there is no need to copy mappings into separate workflow YAMLs. A workflow's optional legacy `config` pointer selects a separate native YAML instead of the shared section; avoid stale pointers when using the root mappings. The template supplies settings, not scientific approval; see [scientific boundaries](#scientific-and-privacy-boundaries).

### Optional import of existing native configurations

`configure` creates a new root local configuration, selecting each native mapping in this order:

1. Native config in the new source project.
2. Native config in the original asset directory.
3. The corresponding full mapping in the versioned [root template](../config.example.yaml) if neither native config exists.

It is an alternative to copying the example, not an installation step. Run only with a destination that does not exist:

```text
uv run python run.py configure
```

The command refuses to overwrite an existing destination, copies mappings rather than datasets, and disables every workflow. Native imports retain their settings and path bases; template fallbacks use the template's example locations. Review the resulting single local configuration before enabling a run; do not manually duplicate its sections elsewhere.

## Launcher reference

Run commands from the repository root after root `uv sync` and enabled-project `setup`.

| Command or option | Current behavior |
| --- | --- |
| `setup` | Frozen sync of enabled Python environments; R installation remains manual |
| `check` | Read-only filesystem/tool and execution-plan checks; requires installed workflow environments |
| `all` | Runs enabled workflows in order: audio, vocal-stats, prosody, vocal-models, prosody-stats, cdi |
| Workflow name | Runs only that enabled workflow; does not launch dependencies |
| `--dry-run` | Shows a read-only plan without launching subprocesses or creating run state |
| `--config PATH` | Selects a different local configuration |
| `--stage NAME` | Selects an allowed native stage for one workflow, not for `all` |
| `-- OPTION VALUE` | Passes only the launcher's allowlisted native options |
| `--resume` | Default: skip a verified, completed whole stage |
| `--no-resume` | Disable skipping; still refuses occupied write locations |

Read `BLOCKED for execution` messages: `check` and dry-run can return successfully while reporting an occupied-root block. They do not prove R packages, model downloads, GPU execution, or scientific outputs work. Enabled workflows with missing prerequisites can fail before a plan is displayed. All-disabled commands do no work.

### Native stages

| Workflow | Default | Other stages |
| --- | --- | --- |
| `audio` | `pipeline` | `prepare-audio`, `classify-vtc`, `classify-xgb`, `generate-textgrids`, `annotate-segments`, `evaluate-models` |
| `vocal-stats` | `all` | `build-master`, `create-long`, `eda`, `plots` |
| `prosody` | `extract` | `validate`, `inspect` |
| `vocal-models` | Full model sequence | Exact model script basename or `report` |
| `prosody-stats` | Full preparation/model sequence | Exact preparation/model script basename or `report` |
| `cdi` | Full questionnaire pipeline | None |

R setup scripts are bootstraps, not standalone completed scientific stages. The exact model/preparation script names are in the [vocal](../vocal_stats/README.md) and [prosody statistics](../prosody_stats/README.md) guides.

Examples, each subject to enabled-workflow and output guards:

```text
uv run python run.py audio --dry-run
uv run python run.py audio --stage pipeline -- --limit 1
uv run python run.py prosody --stage extract -- --device cpu
uv run python run.py vocal-models --stage report
uv run python run.py prosody-stats --stage report
```

Do not chain these as one recipe. Stage commands often share logs, output roots, or cache locations. A later stage can therefore refuse an earlier stage's occupied root. Prefer the full workflow command; for diagnostic or manual stages, first review all write locations and choose separate fresh destinations where needed. Even prosody validation/inspection can write native logs.

## Outputs and recovery

Before a new run, allocate fresh dedicated write locations and explicitly connect downstream inputs to them. Include every workflow's logs, artifacts, cache, processed/private tables, figures, model results, and report destinations. Never use an existing result directory as scratch space. Do not change original input paths or scientific settings as a workaround for an output guard.

Local ownership records are stored under the selected YAML's parent in an ignored `.local/runs/` directory. They compare configuration, command, checked inputs/source/environment metadata, and output fingerprints. Non-audio files are hashed; audio uses size and modification time only. Remote Hugging Face model revisions, unavailable LFS weights, installed R libraries, and system tool versions are not fully pinned or verified.

- Matching successful whole stage: rerunning may skip it.
- Partial, stale, changed, or unowned occupied stage: choose fresh write locations; no overwrite bypass is provided.
- Audio supplied through `--recordings-parquet`: not automatically resumed, because referenced membership/audio paths are not decoded by the launcher.
- Crash with a leftover execution lock: verify no process is active and inspect the failed run before manually removing the lock. Removing it does not make partial outputs reusable.

There is no participant-level recovery guarantee, cross-configuration concurrency guarantee, or filesystem sandbox. Do not run overlapping configurations concurrently. Native participant caches are not automatically adopted by the root launcher.

## Scientific and privacy boundaries

The full template is not approval to change an existing analysis. Keep session selection, audio preparation/timebase, mappings, exclusions, pitch references, SSL models, cohort composition, PCA, model formulas, transforms, random effects, seeds, and runtime overrides unchanged unless a scientific change is approved. Active prosody and the [root template](../config.example.yaml) use a **100 Hz** semitone reference; the older [per-workflow example](../prosody/config/config.example.yaml) uses **1 Hz**. These are different semitone origins, not interchangeable defaults. Record the approved effective value.

Keep the Python projects' locked environments independent and manage the two R analyses' libraries separately. A common launcher and configuration do not imply shared scientific dependencies, formulas, or package upgrades. Root setup installs enabled Python environments only; R libraries remain an explicit per-analysis setup.

VTC and XGB use different methods and labels; XGB is never a fallback for VTC-dependent statistics. Prosody and vocal R analyses also have different formulas and fallback rules. Inspect recorded formulas and diagnostics rather than assuming the two are interchangeable.

All research-derived material remains private, including pseudonymized/aggregate tables, audio, metadata, mappings, classification outputs, caches, model weights, reports, and notebooks. CDI can produce a contact export as well as analysis tables; protect both. Ignore rules are not a confidentiality audit. Review staged content, notebook source/output/metadata, configuration values, and embedded paths before publication. Earlier tracked classifier outputs, weights, or notebooks may remain in history even after exclusion; do not rewrite history or force-push as part of routine documentation or migration work.

## Testing and validation

After installing the relevant environments, these commands run the Python suites from the repository root:

```text
uv run pytest
uv run --project audio --frozen --no-sync pytest
uv run --project cdi --frozen --no-sync pytest
uv run --project prosody --frozen --no-sync pytest
uv run --project vocal_stats --frozen --no-sync pytest
```

These are not full research runs. Keep `HBN_TEST_LOCAL_DATA` unset: CDI's original-cohort tests require explicit `HBN_TEST_LOCAL_DATA=1`, can execute notebooks or write figures, and are excluded by default. Real SSL/model integration checks are separate opt-ins and can download weights. Arrange approved data, resource limits, and isolated destinations before enabling either kind of integration test.

For R checks, see [prosody statistics tests](../prosody_stats/tests/testthat.R) and the [vocal tests](../vocal_stats/README.md#testing). Synthetic checks do not establish original-versus-migrated full-cohort equivalence. This documentation update does not report new test results or platform certification.

## Common problems

| Symptom | First action |
| --- | --- |
| Nothing runs | Set `enabled: true` only for the intended workflows |
| Missing source project | Check `workflows.<name>.project` against the current source-folder table in the main README |
| Missing section or input | Check the root mapping and authorized input paths; use the full template for a new study, not invented defaults |
| Missing environment or lockfile | Run explicit setup; restore the approved lockfile rather than regenerating scientific dependencies |
| Occupied output refused | Keep existing results and choose fresh write locations |
| VTC fails on Windows | Run audio in WSL/Linux with paths and the external environment valid there |
| Missing R package | Install the declared compatible packages; root setup does not install them |
| Report renderer missing | Install Quarto for vocal reports or Pandoc for prosody reports; models do not require rendering |
| Audio/RTTM mismatch | Confirm both refer to the same prepared recording and timebase |
| Model warning or changed results | Review QC, fitted formulas, effective overrides, and environment versions; do not silently alter methods |
