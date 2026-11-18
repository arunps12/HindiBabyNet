# Hindi CDI questionnaire analysis

This independent workflow processes Nettskjema exports for consent, eligibility, language background, CDI-I (8–18 months), and CDI-II (19–36 months). It links submissions, applies the configured inclusion rules, matches questionnaire items, calculates vocabulary/category scores, and produces quality checks, descriptive tables, figures, and configured reporting analyses. It does not require audio processing.

Package: `hindibabynet-cdi`; import name: `hindibabynet_cdi`. Code now lives here; existing questionnaires, metadata, notebooks, native configuration, environment, and outputs remain at their original ignored asset locations.

## Prepare inputs and configuration

Follow the [root installation](../README.md). uv creates a separate compatible Python 3.10–3.12 environment for CDI; do not merge it with the audio/prosody dependencies.

In the [local configuration](../config.local.yaml):

- Set `workflows.cdi.project` to `cdi`; enable it only when ready.
- For an existing study, retain the approved `cdi` mapping and its original asset-based `project_root`; a new root-template copy uses `project_root: "."`.
- Check raw export filenames, form detection/linking fields, metadata/item dictionary locations, eligibility rules, age ranges, scoring definitions, and reporting settings.
- Set `paths.processed_data` and **all** configured output table/figure destinations to fresh dedicated locations. Defaults must not lead new writes into existing results.

The [root example](../config.example.yaml) contains the full CDI scientific mapping, with private `Data` inputs and `results` destinations. Edit paths and enable CDI in the single root local configuration; no separate native YAML is needed. Obtain authorized inputs separately and review study-specific fields without treating the template as approval to change eligibility, scoring, or reporting methods.

## Run from the repository root

After configuration and explicit enablement:

```text
uv run python run.py setup
uv run python run.py cdi --dry-run
uv run python run.py cdi
```

The launcher invokes the existing complete pipeline. CDI has no root `--stage` or pass-through options. Do not run old lists of separate scripts as a replacement for the complete pipeline.

## Scientific processing and outputs

The pipeline loads detected forms, builds participant linkage and eligibility, constructs the item dictionary, scores item responses, and produces reporting outputs. Already-built linkage, eligibility, dictionaries, and loaded forms are reused within the run.

Key configured outputs include:

- A master item mapping and item-matching validation.
- Participant linkage/crosswalk and included/excluded participant audits.
- Combined and form-specific participant analysis tables.
- CDI-I, CDI-II, and combined item-level long tables.
- Unknown-response, completeness, merge, form-detection, and participant-flow checks.
- Vocabulary/category summaries, sample characteristics, reporting tables, figures, and model metadata.

CDI-I supports comprehension and production scores. CDI-II comprehension is missing/not applicable, **not zero**. Preserve approved response coding, age/eligibility definitions, form selection, and norms; changing these changes the analysis.

Results are not automatically public. The current pipeline can also write an **included-participant contact export** into the processed-data destination. Contact/lottery information is separate from scientific scoring, but that does not make the destination safe to publish. Treat linkage, crosswalks, contact exports, scores, figures, and notebooks as private research material. Never merge contact fields into analysis tables or commit them.

## Testing and safe reuse

Default tests, from the repository root:

```text
uv run --project cdi --frozen --no-sync pytest
```

The original-cohort acceptance tests are excluded unless `HBN_TEST_LOCAL_DATA=1` is explicitly set. Leave it unset for normal checks: those tests require private data, may execute notebooks, and can write figures/outputs. Enable them only with authorization and isolated destinations. Synthetic reuse/configuration checks are not full-cohort validation.

Rerunning can skip only a matching completed whole stage owned by the root launcher. Partial, stale, or previously generated unowned results require fresh write locations; there is no overwrite flag. See the [operating guide](../docs/README.md).
