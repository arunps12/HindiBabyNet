from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cleaning import normalize_text
from .columns import (
    ANSWER_TIME_MS,
    CHILD_AGE_LABEL_COLUMN,
    CHILD_SEX_COLUMN,
    CONTACT_EMAIL_COLUMN,
    CONTACT_MOBILE_COLUMN,
    CREATED_AT,
    FORWARDED_FORM_ID,
    REFERENCE_ID,
    SUBMISSION_ID,
    SUBMISSION_REFERENCE,
)
from .config import ProjectConfig
from .io import LoadedForm, load_detected_forms


@dataclass(frozen=True)
class LinkageOutputs:
    participant_linkage: pd.DataFrame
    linkage_qc: pd.DataFrame
    participant_crosswalk: pd.DataFrame
    contact_warnings: pd.DataFrame
    merge_validation: pd.DataFrame


def _get_form_by_role(forms: list[LoadedForm], role: str) -> LoadedForm:
    for form in forms:
        if form.role == role:
            return form
    raise KeyError(f"Missing loaded form for role: {role}")


def _copy_frame(data: pd.DataFrame) -> pd.DataFrame:
    copied = data.copy()
    copied.columns = [normalize_text(column) for column in copied.columns]
    return copied


def _select_available_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available_columns = [column for column in columns if column in data.columns]
    return data[available_columns].copy()


def _prepare_reference_tables(forms: list[LoadedForm]) -> dict[str, pd.DataFrame]:
    consent = _copy_frame(_get_form_by_role(forms, "consent").data)
    eligibility = _copy_frame(_get_form_by_role(forms, "eligibility").data)
    background = _copy_frame(_get_form_by_role(forms, "background").data)
    cdi_8_18 = _copy_frame(_get_form_by_role(forms, "cdi_8_18").data)
    cdi_19_36 = _copy_frame(_get_form_by_role(forms, "cdi_19_36").data)
    contact = _copy_frame(_get_form_by_role(forms, "contact").data)

    consent = consent.rename(columns={SUBMISSION_ID: "consent_submission_id", CREATED_AT: "consent_created"})
    eligibility = eligibility.rename(
        columns={
            SUBMISSION_ID: "eligibility_submission_id",
            REFERENCE_ID: "eligibility_to_consent_id",
            CREATED_AT: "eligibility_created",
        }
    )
    background = background.rename(
        columns={
            SUBMISSION_ID: "background_submission_id",
            SUBMISSION_REFERENCE: "background_to_eligibility_id",
            FORWARDED_FORM_ID: "forwarded_form_id",
            CREATED_AT: "background_created",
        }
    )
    cdi_8_18 = cdi_8_18.rename(
        columns={
            SUBMISSION_ID: "cdi_submission_id",
            SUBMISSION_REFERENCE: "cdi_to_background_id",
            CREATED_AT: "cdi_created",
        }
    )
    cdi_19_36 = cdi_19_36.rename(
        columns={
            SUBMISSION_ID: "cdi_submission_id",
            SUBMISSION_REFERENCE: "cdi_to_background_id",
            CREATED_AT: "cdi_created",
        }
    )
    contact = contact.rename(
        columns={
            SUBMISSION_ID: "contact_submission_id",
            REFERENCE_ID: "contact_to_cdi_id",
            CONTACT_MOBILE_COLUMN: "mobile_number",
            CONTACT_EMAIL_COLUMN: "email",
            CREATED_AT: "contact_created",
        }
    )

    cdi_8_18["questionnaire"] = "8_18"
    cdi_19_36["questionnaire"] = "19_36"
    cdi_8_18["submitted_form"] = "CDI-I"
    cdi_19_36["submitted_form"] = "CDI-II"

    return {
        "consent": consent,
        "eligibility": eligibility,
        "background": background,
        "cdi": pd.concat([cdi_8_18, cdi_19_36], ignore_index=True, sort=False),
        "contact": contact,
    }


def _aggregate_contact_rows(contact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contact = contact.copy()
    contact["mobile_number"] = contact["mobile_number"].map(normalize_text)
    contact["email"] = contact["email"].map(normalize_text)

    warnings = contact.groupby("contact_to_cdi_id", dropna=False).agg(
        n_contact_rows=("contact_submission_id", "size"),
        n_unique_mobile_numbers=("mobile_number", lambda values: values.replace("", pd.NA).dropna().nunique()),
        n_unique_emails=("email", lambda values: values.replace("", pd.NA).dropna().nunique()),
    )
    warnings = warnings.reset_index().rename(columns={"contact_to_cdi_id": "cdi_submission_id"})
    warnings["has_contact_duplicates"] = warnings["n_contact_rows"] > 1
    warnings["has_conflicting_mobile_numbers"] = warnings["n_unique_mobile_numbers"] > 1
    warnings["has_conflicting_emails"] = warnings["n_unique_emails"] > 1

    aggregated = contact.groupby("contact_to_cdi_id", dropna=False).agg(
        contact_submission_ids=("contact_submission_id", lambda values: " | ".join(sorted({value for value in values if normalize_text(value)}))),
        contact_submission_id=("contact_submission_id", "first"),
        contact_created=("contact_created", "min"),
        mobile_number=("mobile_number", "first"),
        email=("email", "first"),
        n_contact_rows=("contact_submission_id", "size"),
    )
    aggregated = aggregated.reset_index().rename(columns={"contact_to_cdi_id": "contact_to_cdi_id"})
    return aggregated, warnings


def _build_merge_validation(tables: dict[str, pd.DataFrame], linkage: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "duplicate_consent_submission_ids",
                "value": int(tables["consent"]["consent_submission_id"].duplicated(keep=False).sum()),
            },
            {
                "metric": "duplicate_eligibility_submission_ids",
                "value": int(tables["eligibility"]["eligibility_submission_id"].duplicated(keep=False).sum()),
            },
            {
                "metric": "duplicate_background_submission_ids",
                "value": int(tables["background"]["background_submission_id"].duplicated(keep=False).sum()),
            },
            {
                "metric": "duplicate_cdi_submission_ids",
                "value": int(tables["cdi"]["cdi_submission_id"].duplicated(keep=False).sum()),
            },
            {
                "metric": "duplicate_contact_submission_ids",
                "value": int(tables["contact"]["contact_submission_id"].duplicated(keep=False).sum()),
            },
            {
                "metric": "multiple_cdi_per_background",
                "value": int(linkage["background_submission_id"].duplicated(keep=False).sum()),
            },
        ]
    )


def build_participant_linkage(config: ProjectConfig) -> LinkageOutputs:
    forms = load_detected_forms(config)
    tables = _prepare_reference_tables(forms)

    consent = _select_available_columns(
        tables["consent"],
        ["consent_submission_id", "consent_created", config.eligibility.consent_column, "fill_date"],
    )
    consent = consent.rename(columns={config.eligibility.consent_column: "consent_response"})

    eligibility = _select_available_columns(
        tables["eligibility"],
        [
            "eligibility_submission_id",
            "eligibility_to_consent_id",
            "eligibility_created",
            config.eligibility.mother_tongue_column,
            config.eligibility.preterm_column,
            config.eligibility.impairment_column,
            config.eligibility.age_range_column,
        ],
    )
    eligibility = eligibility.rename(
        columns={
            config.eligibility.mother_tongue_column: "mother_tongue_response",
            config.eligibility.preterm_column: "preterm_response",
            config.eligibility.impairment_column: "impairment_response",
            config.eligibility.age_range_column: "age_range_response",
        }
    )

    background = _select_available_columns(
        tables["background"],
        [
            "background_submission_id",
            "background_to_eligibility_id",
            "background_created",
            config.age.birthdate_column,
            config.age.reported_age_column,
            config.age.child_age_label_column,
            CHILD_SEX_COLUMN,
            config.education.mother_education_column,
            config.education.father_education_column,
            config.exposure.second_language_column,
            config.exposure.second_language_percent_column,
            config.exposure.third_language_column,
            config.exposure.third_language_percent_column,
            "forwarded_form_id",
        ],
    )
    background = background.rename(
        columns={
            config.age.birthdate_column: "birthdate",
            config.age.reported_age_column: "reported_age_months",
            config.age.child_age_label_column: "reported_age_label",
            CHILD_SEX_COLUMN: "child_sex",
            config.education.mother_education_column: "mother_education",
            config.education.father_education_column: "father_education",
            config.exposure.second_language_column: "second_language",
            config.exposure.second_language_percent_column: "second_language_percent",
            config.exposure.third_language_column: "third_language",
            config.exposure.third_language_percent_column: "third_language_percent",
        }
    )

    cdi = _select_available_columns(
        tables["cdi"],
        [
            "cdi_submission_id",
            "cdi_to_background_id",
            "cdi_created",
            "questionnaire",
            "submitted_form",
            ANSWER_TIME_MS,
        ],
    ).rename(columns={ANSWER_TIME_MS: "cdi_answer_time_ms"})

    contact, contact_warnings = _aggregate_contact_rows(
        _select_available_columns(
            tables["contact"],
            [
                "contact_submission_id",
                "contact_to_cdi_id",
                "contact_created",
                "mobile_number",
                "email",
            ],
        )
    )

    linkage = cdi.merge(background, left_on="cdi_to_background_id", right_on="background_submission_id", how="left", validate="m:1")
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
    linkage = linkage.merge(contact, left_on="cdi_submission_id", right_on="contact_to_cdi_id", how="left", validate="1:1")

    linkage = linkage.sort_values(["cdi_to_background_id", "cdi_created", "questionnaire"], na_position="last").reset_index(drop=True)
    unique_participants = pd.factorize(linkage["background_submission_id"].fillna(linkage["cdi_submission_id"]))[0] + 1
    linkage.insert(0, "participant_id", [f"P{value:06d}" for value in unique_participants])

    linkage["has_background_link"] = linkage["background_submission_id"].notna()
    linkage["has_eligibility_link"] = linkage["eligibility_submission_id"].notna()
    linkage["has_consent_link"] = linkage["consent_submission_id"].notna()
    linkage["has_contact_link"] = linkage["n_contact_rows"].fillna(0).gt(0)

    participant_crosswalk = linkage[
        [
            "participant_id",
            "submitted_form",
            "questionnaire",
            "consent_submission_id",
            "eligibility_submission_id",
            "background_submission_id",
            "cdi_submission_id",
            "contact_submission_ids",
        ]
    ].copy()

    merge_validation = _build_merge_validation(tables, linkage)

    qc = pd.DataFrame(
        [
            {"metric": "n_cdi_rows", "value": int(len(cdi))},
            {"metric": "n_linked_background", "value": int(linkage["has_background_link"].sum())},
            {"metric": "n_linked_eligibility", "value": int(linkage["has_eligibility_link"].sum())},
            {"metric": "n_linked_consent", "value": int(linkage["has_consent_link"].sum())},
            {"metric": "n_linked_contact", "value": int(linkage["has_contact_link"].sum())},
            {"metric": "n_unique_participants", "value": int(linkage["participant_id"].nunique())},
            {
                "metric": "n_multiple_contact_rows",
                "value": int(linkage["n_contact_rows"].fillna(0).gt(1).sum()),
            },
        ]
    )

    return LinkageOutputs(
        participant_linkage=linkage,
        linkage_qc=qc,
        participant_crosswalk=participant_crosswalk,
        contact_warnings=contact_warnings,
        merge_validation=merge_validation,
    )