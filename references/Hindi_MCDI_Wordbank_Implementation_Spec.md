# Hindi M-CDI: Wordbank-style processing, tables, and plots

## Purpose

This document is an implementation specification for a coding agent that will modify the existing Hindi M-CDI project so that it:

1. links all survey forms through their reference-number chain;
2. retains only participants who gave consent;
3. retains only participants with **more than 75% Hindi exposure**;
4. validates the age range and submitted CDI form;
5. scores CDI-I comprehension and production separately;
6. scores CDI-II production only;
7. improves `word_mapping.csv` into a master item dictionary;
8. creates clean participant-level and item-level datasets;
9. generates Wordbank-style tables and plots;
10. produces one separate, protected file containing the mobile number and email address of all included CDI-I and CDI-II participants;
11. uses **mother's education only** in education analyses.

The raw Excel exports must never be edited in place.

---

# 1. Source files and form-linking chain

The project currently contains six Excel exports:

| Logical form | Current file | Main ID | Reference to previous form |
|---|---|---|---|
| Consent | `data-539547-2026-06-04-1053.xlsx` | `$submission_id` | none |
| Language background and family information | `data-539576-2026-06-04-1113.xlsx` | `$submission_id` | `SUBMISSION_REFERENCE` |
| Eligibility | `data-539591-2026-06-04-1058.xlsx` | `$submission_id` | `Reference ID` |
| CDI-I, 8–18 months | `data-539642-2026-06-04-1113.xlsx` | `$submission_id` | `SUBMISSION_REFERENCE` |
| CDI-II, 19–36 months | `data-539644-2026-06-04-1114.xlsx` | `$submission_id` | `SUBMISSION_REFERENCE` |
| Contact information | `data-540031-2026-06-04-1114.xlsx` | `$submission_id` | `Reference ID` |

The expected reference chain is:

```text
Consent.$submission_id
    ↓
Eligibility.Reference ID

Eligibility.$submission_id
    ↓
LanguageBackground.SUBMISSION_REFERENCE

LanguageBackground.$submission_id
    ↓
CDI-I.SUBMISSION_REFERENCE
or
CDI-II.SUBMISSION_REFERENCE

CDI-I/CDI-II.$submission_id
    ↓
Contact.Reference ID
```

Do not merge files by row order. Every merge must be based on these IDs.

---

# 2. Recommended project structure

```text
Hindi_MCDI/
├── data/
│   ├── raw/
│   │   ├── consent.xlsx
│   │   ├── eligibility.xlsx
│   │   ├── language_background.xlsx
│   │   ├── cdi1.xlsx
│   │   ├── cdi2.xlsx
│   │   └── contact.xlsx
│   ├── metadata/
│   │   ├── word_mapping.csv
│   │   └── word_mapping_master.csv
│   └── processed/
│       ├── participant_linkage.csv
│       ├── participant_analysis_all.csv
│       ├── participant_analysis_cdi1.csv
│       ├── participant_analysis_cdi2.csv
│       ├── cdi1_items_long.csv
│       ├── cdi2_items_long.csv
│       ├── cdi_all_items_long.csv
│       ├── exclusions.csv
│       ├── item_mapping_validation.csv
│       └── included_participant_contacts.csv
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── qc/
├── notebooks/
│   ├── 01_CDI1_EDA_and_Norms.ipynb
│   └── 02_CDI2_EDA_and_Norms.ipynb
├── src/
│   └── hindi_mcdi/
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       ├── columns.py
│       ├── linkage.py
│       ├── eligibility.py
│       ├── exposure.py
│       ├── age.py
│       ├── item_dictionary.py
│       ├── scoring.py
│       ├── reshape.py
│       ├── qc.py
│       ├── norms.py
│       ├── trajectories.py
│       └── plotting.py
├── scripts/
│   ├── 01_build_master_item_dictionary.py
│   ├── 02_link_and_filter_participants.py
│   ├── 03_score_cdi1.py
│   ├── 04_score_cdi2.py
│   └── 05_generate_outputs.py
├── tests/
└── README.md
```

The coding agent may adapt this structure to the existing repository, but the separation between raw, metadata, processed, outputs, notebooks, and source code should be preserved.

---

# 3. Improve `word_mapping.csv`

The existing file contains:

```text
word
word_english
category
category_english
questionnaire
```

Convert it into `word_mapping_master.csv` with at least:

| item_id | word | word_english | category | category_english | questionnaire | cdi1 | cdi2 | cdi1_order | cdi2_order | active |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | पानी | water | भोजन | Food | 8_18 | 1 | 0 | 45 | NA | 1 |
| 2 | गेंद | ball | खिलौने | Toys | 8_18;19_36 | 1 | 1 | 73 | 112 | 1 |
| 3 | केला | banana | भोजन | Food | 19_36 | 0 | 1 | NA | 88 | 1 |

Rules:

```text
questionnaire == "8_18"       → cdi1=1, cdi2=0
questionnaire == "19_36"      → cdi1=0, cdi2=1
questionnaire == "8_18;19_36" → cdi1=1, cdi2=1
```

Use `1` and `0`, not check marks.

## 3.1 Required dictionary validation

The implementation must report:

- duplicate `item_id`;
- duplicate Hindi item labels within the same form;
- item columns present in CDI-I but absent from mapping;
- items marked `cdi1=1` but absent from the CDI-I file;
- item columns present in CDI-II but absent from mapping;
- items marked `cdi2=1` but absent from the CDI-II file;
- differences caused only by whitespace, Unicode normalization, punctuation, or HTML encoding.

Normalize Hindi strings for matching, but retain the original Hindi form in final outputs.

Suggested normalization:

```python
import html
import re
import unicodedata

def normalize_hindi_label(x):
    x = "" if x is None else str(x)
    x = html.unescape(x)
    x = unicodedata.normalize("NFC", x)
    x = x.replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x
```

Do not silently discard unmatched items. Save all mismatches to `item_mapping_validation.csv`.

---

# 4. Standardize identifiers and columns

Rename IDs immediately after loading:

```text
consent_submission_id
eligibility_submission_id
background_submission_id
cdi_submission_id
contact_submission_id

eligibility_to_consent_id
background_to_eligibility_id
cdi_to_background_id
contact_to_cdi_id
```

Create an anonymous analysis ID such as:

```text
P000001
P000002
...
```

Store the crosswalk between `participant_id` and source submission IDs in a restricted linkage file. Do not expose raw submission IDs in public analysis tables.

---

# 5. Participant inclusion criteria

A participant is included only when all required conditions are true:

```text
consent_given
AND successfully_linked
AND eligibility_passed
AND hindi_percentage > 75
AND age_is_valid
AND age_form_match
AND cdi_is_complete_enough
```

Create separate columns for every criterion and a single final column:

```text
included_final
```

Also create:

```text
primary_exclusion_reason
all_exclusion_reasons
```

Do not delete excluded rows until the exclusion report has been generated.

## 5.1 Consent

Use the actual consent response:

```text
मैं सहमति देता/देती हूँ कि मेरी गुमनाम जानकारी ऊपर वर्णित परियोजना में उपयोग की जाएगी।
```

Do not infer consent from the presence of a submitted form.

## 5.2 Eligibility

At minimum, preserve and evaluate:

- Hindi as mother tongue;
- pre-term status;
- speech/hearing/vision problem;
- age 8–36 months.

The exact eligibility rule must be stored in configuration so it can be changed without rewriting the pipeline.

---

# 6. Hindi exposure calculation

The background form contains second- and third-language exposure percentages.

```text
non_hindi_percentage =
    second_language_percentage
    + third_language_percentage

hindi_percentage =
    100 - non_hindi_percentage
```

The inclusion rule is strictly:

```text
hindi_percentage > 75
```

Therefore, exactly 75% Hindi is excluded.

## 6.1 Missing-value rules

| Situation | Treatment |
|---|---|
| No second language and percentage blank | second-language exposure = 0 |
| No third language and percentage blank | third-language exposure = 0 |
| A language is named but percentage is blank | Hindi exposure = unknown; manual review |
| Hindi numerals such as `२५` | convert to 25 |
| Percent symbols or text such as `20%` | parse numeric value |
| Total non-Hindi exposure >100 | invalid; manual review |
| Negative exposure | invalid; manual review |

Do not use percentage of time with mother or father as language exposure.

---

# 7. Age calculation and form validation

Calculate age from birthdate and the CDI completion date:

```text
age_days = cdi_date - birthdate
age_months_exact = age_days / 30.4375
age_month = floor(age_months_exact)
```

Expected age ranges:

```text
CDI-I:  8 <= age_months_exact < 19
CDI-II: 19 <= age_months_exact <= 36.999...
```

Retain:

```text
reported_age_months
calculated_age_months
age_difference_months
submitted_form
expected_form
age_form_match
```

Flag large discrepancies between reported and calculated age.

---

# 8. CDI response scoring

Keep raw Hindi responses unchanged in the raw data. Convert them programmatically.

## 8.1 CDI-I: comprehension and production

| Raw Hindi response | understand | produce |
|---|---:|---:|
| blank/unselected item | 0 | 0 |
| `केवल समझता/समझती है` | 1 | 0 |
| `समझता/समझती है और कहता/कहती है` | 1 | 1 |

Participant scores:

```text
cdi1_comprehension_total = sum(understand)
cdi1_production_total = sum(produce)
```

Unknown nonblank response labels must be reported, not coded as zero.

## 8.2 CDI-II: production only

| Raw Hindi response | produce |
|---|---:|
| blank/unselected item | 0 |
| `कहता/कहती है` | 1 |

Participant score:

```text
cdi2_production_total = sum(produce)
```

For CDI-II:

```text
comprehension_total = NA
```

It must not be zero, because comprehension was not measured.

## 8.3 Completeness

Distinguish:

- a normally completed form where unselected item cells mean 0;
- an interrupted or technically incomplete form where a large block of items was never presented.

Define and document a completeness rule. Save:

```text
n_items_expected
n_items_present
n_nonblank_responses
completion_status
```

---

# 9. Required processed datasets

## 9.1 Participant analysis table

One row per participant:

| participant_id | form | age_months_exact | age_month | sex | mother_education | hindi_percentage | comprehension_total | production_total | included_final |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| P0001 | CDI-I | 13.2 | 13 | Female | Graduate | 92 | 180 | 105 | 1 |
| P0002 | CDI-II | 25.4 | 25 | Male | College | 88 | NA | 312 | 1 |

Use **mother's education only**. Do not average mother and father education, do not use father's education as a substitute, and do not label the variable merely `caregiver_education`.

Recommended variable:

```text
mother_education
```

## 9.2 Item-level long data

One row per participant × item:

| participant_id | form | item_id | word | word_english | category_english | age_months_exact | understand | produce |
|---|---|---:|---|---|---|---:|---:|---:|
| P0001 | CDI-I | 1 | पानी | water | Food | 13.2 | 1 | 1 |
| P0001 | CDI-I | 2 | गेंद | ball | Toys | 13.2 | 1 | 0 |
| P0002 | CDI-II | 2 | गेंद | ball | Toys | 25.4 | NA | 1 |

## 9.3 Included participant contacts

Create a single protected file:

```text
data/processed/included_participant_contacts.csv
```

It must combine included CDI-I and CDI-II participants and contain:

| participant_id | form | age_months_exact | child_sex | mother_education | mobile_number | email |
|---|---|---:|---|---|---|---|
| P0001 | CDI-I | 13.2 | Female | Graduate | ... | ... |
| P0002 | CDI-II | 25.4 | Male | College | ... | ... |

Requirements:

- include only `included_final == 1`;
- include both CDI-I and CDI-II;
- one row per included participant;
- normalize mobile number as text, never numeric;
- trim whitespace from email;
- preserve leading `+` and leading zeroes in phone numbers;
- identify duplicate phone numbers and duplicate emails;
- save duplicate-contact warnings separately;
- do not merge phone/email into public analysis data;
- treat this file as sensitive and exclude it from Git using `.gitignore`.

Suggested `.gitignore` entry:

```text
data/processed/included_participant_contacts.csv
data/processed/*contact*
```

---

# 10. Required quality-control tables

## Table A: participant flow

| Stage | CDI-I | CDI-II | Total |
|---|---:|---:|---:|
| Raw CDI submissions |  |  |  |
| Successfully linked to background |  |  |  |
| Successfully linked to eligibility |  |  |  |
| Successfully linked to consent |  |  |  |
| Consent provided |  |  |  |
| Eligibility passed |  |  |  |
| Hindi exposure >75% |  |  |  |
| Age matches form |  |  |  |
| Passed completeness check |  |  |  |
| Final included participants |  |  |  |

## Table B: exclusions

| Exclusion reason | CDI-I | CDI-II | Total |
|---|---:|---:|---:|
| No consent |  |  |  |
| Broken reference chain |  |  |  |
| Failed eligibility |  |  |  |
| Hindi exposure ≤75% |  |  |  |
| Hindi exposure unknown |  |  |  |
| Age outside form range |  |  |  |
| Incomplete CDI |  |  |  |

## Table C: sample characteristics

Report separately for CDI-I and CDI-II:

| Variable | Value |
|---|---|
| N |  |
| Age, mean (SD) |  |
| Age, median [IQR] |  |
| Age range |  |
| Female, n (%) |  |
| Male, n (%) |  |
| Hindi exposure, mean (SD) |  |
| Mother's education: Primary, n (%) |  |
| Mother's education: Secondary, n (%) |  |
| Mother's education: College, n (%) |  |
| Mother's education: Graduate, n (%) |  |

---

# 11. Wordbank-style normative tables

Create separate normative tables for:

1. CDI-I comprehension;
2. CDI-I production;
3. CDI-II production.

Never combine CDI-I and CDI-II raw totals into a single norm table.

## 11.1 Full quantile table by month

| age_month | n | mean | sd | min | p10 | p20 | p25 | p30 | p40 | p50 | p60 | p70 | p75 | p80 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Use empirical quantiles for tables and smooth quantile regression for curves.

Store quantile values as vocabulary counts, with appropriate rounding and bounding:

```text
0 <= predicted vocabulary <= maximum number of items in that form
```

## 11.2 Mother-education table

| age_month | mother_education | n | mean | median | p25 | p50 | p75 |
|---:|---|---:|---:|---:|---:|---:|---:|

Only use mother's education.

## 11.3 Sex-specific table

| age_month | sex | n | p10 | p25 | p50 | p75 | p90 |
|---:|---|---:|---:|---:|---:|---:|---:|

---

# 12. Required Wordbank-style plots

All plots must be generated separately for:

```text
CDI-I comprehension
CDI-I production
CDI-II production
```

where applicable.

The word “none” below means **no percentile/quantile stratification**: one overall fitted trajectory.

## Plot 1: Overall trajectory — no quantiles

Purpose: show individual participant scores and one overall smooth fitted curve.

```text
X-axis: age in months
Y-axis: vocabulary score
Grey points: individual participants
One fitted curve: all included participants
```

Output examples:

```text
cdi1_comprehension_overall.png
cdi1_production_overall.png
cdi2_production_overall.png
```

## Plot 2: Quartile curves

Show:

```text
25th percentile
50th percentile
75th percentile
```

This corresponds to the Wordbank-style 25/50/75 figure.

Output examples:

```text
cdi1_comprehension_quartiles.png
cdi1_production_quartiles.png
cdi2_production_quartiles.png
```

## Plot 3: Decile curves

Show all deciles:

```text
10, 20, 30, 40, 50, 60, 70, 80, 90
```

Output examples:

```text
cdi1_comprehension_deciles.png
cdi1_production_deciles.png
cdi2_production_deciles.png
```

## Plot 4: Selected quantile curves

Show the standard selected quantiles:

```text
10, 25, 50, 75, 90
```

This is useful for concise normative reporting.

Output examples:

```text
cdi1_comprehension_selected_quantiles.png
cdi1_production_selected_quantiles.png
cdi2_production_selected_quantiles.png
```

## Plot 5: Sex-specific selected quantiles

Two panels:

```text
Female
Male
```

Each panel shows:

```text
10, 25, 50, 75, 90
```

## Plot 6: Mother's education trajectories

One fitted overall/median trajectory for each mother's education group:

```text
Primary
Secondary
College
Graduate
Other/unknown only when analytically justified
```

Use mother's education only. Include group sample size in the legend.

Do not fit unstable curves for groups with very small `n`; either combine defensible categories, omit the curve, or label it exploratory.

## Plot 7: Sample size by age

Bar plot:

```text
X-axis: age month
Y-axis: included participant count
```

This must be reviewed before interpreting normative curves.

## Plot 8: Score distribution by age

Boxplot, violin plot, or jittered boxplot:

```text
X-axis: age month
Y-axis: vocabulary score
```

## Plot 9: Item production trajectories

For selected words:

```text
X-axis: age in months
Y-axis: estimated proportion producing
One curve per word
```

CDI-I and CDI-II production trajectories must be separate.

## Plot 10: CDI-I item comprehension trajectories

For selected CDI-I words:

```text
X-axis: age in months
Y-axis: estimated proportion understanding
One curve per word
```

There is no CDI-II comprehension trajectory.

## Plot 11: Category trajectories

Use the proportion of category items understood/produced rather than raw category totals:

```text
category proportion =
number of category items endorsed / number of category items available
```

---

# 13. Statistical modelling recommendations

## 13.1 Overall trajectory

Use a bounded smooth model appropriate for vocabulary counts. Acceptable implementation options include:

- GAM/GAMLSS;
- spline regression with predictions clipped to valid range;
- monotonic smoother where justified.

The coding agent must document the selected method.

## 13.2 Quantile curves

Use quantile regression with smooth age terms. Requirements:

- curves must not be allowed to cross without warning;
- predictions must be bounded between 0 and form maximum;
- include empirical month-level quantiles in tables;
- include smooth fitted quantiles in figures;
- record model formula and software version.

If the sample is too small for stable smooth quantile regression, first create empirical quantile plots and report the limitation.

## 13.3 Item trajectories

Use binomial logistic regression or a binomial GAM:

```text
response: produce or understand (0/1)
predictor: smooth age
```

Do not fit item trajectories for words with insufficient variation or too few observations without flagging them.

---

# 14. Example plot references

The following Wordbank-style reference images are included with this specification and should be used only as visual guidance:

```text
wordbank_item_trajectories(1).png
wordbank_vocab_norms (4)(1).png
wordbank_vocab_norms (3)(1).png
wordbank_vocab_norms (2)(1).png
wordbank_vocab_norms (1)(1).png
wordbank_vocab_norms(1).png
```

Interpretation:

| Reference figure | Target Hindi M-CDI output |
|---|---|
| Individual item trajectories | Selected Hindi item comprehension/production trajectories |
| 25/50/75 curves | Quartile plot |
| 10–90 curves | Decile plot |
| Female/Male panels | Sex-specific selected quantile plot |
| Education curves | Mother's education trajectories |
| One fitted curve | Overall trajectory with no quantiles |

Example Markdown image embedding:

```markdown
![Wordbank item trajectory example](wordbank_item_trajectories(1).png)

![Wordbank quartile example](wordbank_vocab_norms (4)(1).png)

![Wordbank decile example](wordbank_vocab_norms (3)(1).png)
```

---

# 15. Example output tables

## 15.1 Example CDI-I production norms

| age_month | n | mean | sd | p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 42 | 4.8 | 6.1 | 0 | 1 | 3 | 7 | 13 |
| 9 | 51 | 7.2 | 8.5 | 0 | 2 | 5 | 10 | 18 |
| 10 | 49 | 11.9 | 12.8 | 1 | 4 | 8 | 16 | 29 |

Values above are illustrative only.

## 15.2 Example CDI-I comprehension norms

| age_month | n | mean | sd | p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 42 | 24.6 | 18.0 | 5 | 11 | 20 | 34 | 52 |
| 9 | 51 | 35.2 | 22.7 | 8 | 18 | 31 | 48 | 67 |

Values above are illustrative only.

## 15.3 Example CDI-II production norms

| age_month | n | mean | sd | p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 61 | 92 | 76 | 15 | 35 | 71 | 127 | 198 |
| 20 | 75 | 128 | 91 | 22 | 52 | 104 | 180 | 266 |

Values above are illustrative only.

---

# 16. Notebook requirements

Create two notebooks:

```text
notebooks/01_CDI1_EDA_and_Norms.ipynb
notebooks/02_CDI2_EDA_and_Norms.ipynb
```

## CDI-I notebook must include

1. configuration and paths;
2. loading processed included CDI-I participants;
3. sample flow and exclusion table;
4. participant characteristics;
5. age distribution and sample size by age;
6. missingness report;
7. comprehension and production score summaries;
8. comprehension–production relationship;
9. overall curves with no quantiles;
10. quartile curves;
11. decile curves;
12. selected quantile curves;
13. sex-specific selected quantiles;
14. mother's education trajectories;
15. item production trajectories;
16. item comprehension trajectories;
17. category-level comprehension and production;
18. export of all displayed tables and figures.

## CDI-II notebook must include

1. configuration and paths;
2. loading processed included CDI-II participants;
3. sample flow and exclusion table;
4. participant characteristics;
5. age distribution and sample size by age;
6. missingness report;
7. production score summaries;
8. overall production curve with no quantiles;
9. quartile curves;
10. decile curves;
11. selected quantile curves;
12. sex-specific selected quantiles;
13. mother's education trajectories;
14. item production trajectories;
15. category-level production;
16. export of all displayed tables and figures.

Both notebooks must make it clear that CDI-II has no comprehension score.

---

# 17. Reproducibility and testing

Add tests for:

- ID chain integrity;
- one-to-one or expected one-to-many merge relationships;
- duplicated submissions;
- consent coding;
- Hindi-exposure parsing;
- strict `>75` threshold;
- age-boundary cases at 8, 18, 19, and 36 months;
- CDI-I response coding;
- CDI-II response coding;
- item dictionary completeness;
- maximum possible scores;
- combined contact file contains only included participants;
- mother education is used in education plots;
- CDI-II comprehension remains missing.

Store package versions and random seeds.

---

# 18. Final deliverables

The implementation is complete only when it creates:

```text
data/metadata/word_mapping_master.csv
data/processed/participant_analysis_cdi1.csv
data/processed/participant_analysis_cdi2.csv
data/processed/cdi1_items_long.csv
data/processed/cdi2_items_long.csv
data/processed/included_participant_contacts.csv
outputs/tables/cdi1_comprehension_norms.csv
outputs/tables/cdi1_production_norms.csv
outputs/tables/cdi2_production_norms.csv
outputs/tables/participant_flow.csv
outputs/tables/exclusions.csv
outputs/tables/item_trajectory_summary.csv
outputs/figures/*.png
notebooks/01_CDI1_EDA_and_Norms.ipynb
notebooks/02_CDI2_EDA_and_Norms.ipynb
```

The analysis datasets must be de-identified. The combined contact file must remain separate and protected.
