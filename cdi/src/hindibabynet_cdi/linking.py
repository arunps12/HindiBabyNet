"""Link every CDI submission to its background, eligibility and consent form."""

from __future__ import annotations

import pandas as pd

from .columns import (
    CDI1_FORM,
    CDI2_FORM,
    CHILD_SEX_COLUMN,
    CREATED_AT,
    REFERENCE_ID,
    SUBMISSION_ID,
    SUBMISSION_REFERENCE,
)
from .config import ProjectConfig
from .io import LoadedForm


def _take(form: LoadedForm, columns: dict[str, str], optional: dict[str, str] | None = None) -> pd.DataFrame:
    """Take the needed columns of one form and give them short names.

    columns: name in the export -> short name. These columns must be there.
    optional: the same, but a column that is not in the export stays empty.
    """
    missing = [name for name in columns if name not in form.data.columns]
    if missing:
        raise ValueError(
            f"The file {form.path.name} ({form.role} form) has no column: {'; '.join(missing)}. "
            "Check the file and the column names in cdi/params.yaml."
        )
    taken = form.data[list(columns)].rename(columns=columns)
    for name, short_name in (optional or {}).items():
        taken[short_name] = form.data[name] if name in form.data.columns else pd.NA
    return taken


def build_participant_linkage(config: ProjectConfig, forms: dict[str, LoadedForm]) -> pd.DataFrame:
    """One row per CDI submission with the answers of the linked forms.

    The child ID (participant_id) is also given here: one child is one
    background form.
    """
    consent = _take(
        forms["consent"],
        {
            SUBMISSION_ID: "consent_submission_id",
            config.eligibility.consent_column: "consent_response",
        },
    )
    eligibility = _take(
        forms["eligibility"],
        {
            SUBMISSION_ID: "eligibility_submission_id",
            REFERENCE_ID: "eligibility_to_consent_id",
            config.eligibility.mother_tongue_column: "mother_tongue_response",
            config.eligibility.preterm_column: "preterm_response",
            config.eligibility.impairment_column: "impairment_response",
            config.eligibility.age_range_column: "age_range_response",
        },
    )
    background = _take(
        forms["background"],
        {
            SUBMISSION_ID: "background_submission_id",
            SUBMISSION_REFERENCE: "background_to_eligibility_id",
            config.age.birthdate_column: "birthdate",
            config.age.reported_age_column: "reported_age_months",
            CHILD_SEX_COLUMN: "child_sex",
            config.education.mother_education_column: "mother_education",
            config.exposure.second_language_column: "second_language",
            config.exposure.second_language_percent_column: "second_language_percent",
            config.exposure.third_language_column: "third_language",
            config.exposure.third_language_percent_column: "third_language_percent",
        },
        optional={config.education.father_education_column: "father_education"},
    )

    cdi_forms = []
    for role, questionnaire, form_label in (
        ("cdi_8_18", config.analysis.younger_questionnaire, CDI1_FORM),
        ("cdi_19_36", config.analysis.older_questionnaire, CDI2_FORM),
    ):
        cdi_form = _take(
            forms[role],
            {
                SUBMISSION_ID: "cdi_submission_id",
                SUBMISSION_REFERENCE: "cdi_to_background_id",
                CREATED_AT: "cdi_created",
            },
        )
        cdi_form["questionnaire"] = questionnaire
        cdi_form["submitted_form"] = form_label
        cdi_forms.append(cdi_form)
    cdi = pd.concat(cdi_forms, ignore_index=True)
    if cdi["cdi_submission_id"].duplicated().any():
        raise ValueError(
            "A submission ID occurs more than once in the CDI exports. "
            "Every CDI submission must have its own ID."
        )

    linkage = cdi.merge(
        background,
        left_on="cdi_to_background_id",
        right_on="background_submission_id",
        how="left",
        validate="m:1",
    )
    linkage = linkage.merge(
        eligibility,
        left_on="background_to_eligibility_id",
        right_on="eligibility_submission_id",
        how="left",
        validate="m:1",
    )
    linkage = linkage.merge(
        consent,
        left_on="eligibility_to_consent_id",
        right_on="consent_submission_id",
        how="left",
        validate="m:1",
    )

    # Child ID: "P" and six digits, numbered in this sort order. Do not change
    # the sort or the numbering: the IDs of an existing study would change.
    linkage = linkage.sort_values(
        ["cdi_to_background_id", "cdi_created", "questionnaire"], na_position="last"
    ).reset_index(drop=True)
    unique_participants = pd.factorize(linkage["background_submission_id"].fillna(linkage["cdi_submission_id"]))[0] + 1
    linkage.insert(0, "participant_id", [f"P{value:06d}" for value in unique_participants])

    linkage["has_background_link"] = linkage["background_submission_id"].notna()
    linkage["has_eligibility_link"] = linkage["eligibility_submission_id"].notna()
    linkage["has_consent_link"] = linkage["consent_submission_id"].notna()
    return linkage
