"""Inclusion rules of the CDI questionnaires."""

from __future__ import annotations

import pandas as pd

from .age import add_age_columns
from .cleaning import normalize_text, parse_bool_yn, parse_percent
from .config import ProjectConfig


# Reasons why a questionnaire is not included, with the text for the summary on
# the screen. The first reason that applies is the primary reason.
EXCLUSION_REASONS = {
    "broken_reference_chain": "not linked to a background, eligibility and consent form",
    "no_consent": "no consent",
    "failed_eligibility": "eligibility questions not passed",
    "hindi_exposure_unknown": "Hindi exposure not known",
    "hindi_exposure_invalid": "Hindi exposure not valid",
    "hindi_exposure_leq_75": "Hindi exposure {limit} % or lower",
    "age_outside_form_range": "age not known or outside the age range",
    "age_form_mismatch": "age does not fit the questionnaire",
}


def _derive_hindi_exposure(data: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    derived = data.copy()

    second_language_named = derived["second_language"].map(normalize_text).replace("", pd.NA).notna()
    third_language_named = derived["third_language"].map(normalize_text).replace("", pd.NA).notna()

    second_language_percent = derived["second_language_percent"].map(parse_percent)
    third_language_percent = derived["third_language_percent"].map(parse_percent)

    second_unknown = second_language_named & second_language_percent.isna()
    third_unknown = third_language_named & third_language_percent.isna()
    second_invalid = second_language_percent.fillna(0).lt(0)
    third_invalid = third_language_percent.fillna(0).lt(0)

    second_language_percent = second_language_percent.where(~(~second_language_named & second_language_percent.isna()), 0)
    third_language_percent = third_language_percent.where(~(~third_language_named & third_language_percent.isna()), 0)
    second_language_percent = second_language_percent.fillna(0)
    third_language_percent = third_language_percent.fillna(0)

    derived["second_language_percent_parsed"] = second_language_percent
    derived["third_language_percent_parsed"] = third_language_percent
    derived["non_hindi_percentage"] = second_language_percent + third_language_percent
    derived["hindi_percentage"] = 100 - derived["non_hindi_percentage"]
    derived["hindi_exposure_unknown"] = second_unknown | third_unknown
    derived["hindi_exposure_invalid"] = (
        second_invalid
        | third_invalid
        | derived["non_hindi_percentage"].gt(100)
        | derived["non_hindi_percentage"].lt(0)
    )
    derived["hindi_exposure_passed"] = (
        derived["hindi_percentage"].gt(config.analysis.strict_hindi_threshold)
        & ~derived["hindi_exposure_unknown"]
        & ~derived["hindi_exposure_invalid"]
    )
    return derived


def evaluate_participant_criteria(linkage: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Apply the inclusion rules to every CDI submission.

    A submission is included (included_final) when consent is given, the forms
    are linked, the eligibility questions are passed, the Hindi exposure is
    known, valid and above the limit, and the age is valid and fits the form.
    """
    data = add_age_columns(linkage, config)
    data = _derive_hindi_exposure(data, config)

    data["consent_given"] = data["consent_response"].map(parse_bool_yn).eq(True)
    data["mother_tongue_is_hindi"] = data["mother_tongue_response"].map(parse_bool_yn).eq(
        config.eligibility.required_mother_tongue_response
    )
    data["preterm_allowed"] = data["preterm_response"].map(parse_bool_yn).ne(True) if not config.eligibility.allow_preterm else True
    data["impairment_allowed"] = data["impairment_response"].map(parse_bool_yn).ne(True) if not config.eligibility.allow_impairment else True
    data["eligibility_age_range_passed"] = data["age_range_response"].map(parse_bool_yn).eq(True)
    data["eligibility_passed"] = (
        data["mother_tongue_is_hindi"]
        & data["preterm_allowed"]
        & data["impairment_allowed"]
        & data["eligibility_age_range_passed"]
    )

    data["successfully_linked"] = (
        data["has_background_link"] & data["has_eligibility_link"] & data["has_consent_link"]
    )

    data["included_final"] = (
        data["consent_given"]
        & data["successfully_linked"]
        & data["eligibility_passed"]
        & data["hindi_exposure_passed"]
        & data["age_is_valid"].fillna(False)
        & data["age_form_match"].fillna(False)
    )

    reasons = pd.DataFrame(
        {
            "broken_reference_chain": ~data["successfully_linked"],
            "no_consent": ~data["consent_given"],
            "failed_eligibility": ~data["eligibility_passed"],
            "hindi_exposure_unknown": data["hindi_exposure_unknown"],
            "hindi_exposure_invalid": data["hindi_exposure_invalid"],
            "hindi_exposure_leq_75": ~data["hindi_exposure_passed"] & ~data["hindi_exposure_unknown"] & ~data["hindi_exposure_invalid"],
            "age_outside_form_range": ~data["age_is_valid"].fillna(False),
            "age_form_mismatch": ~data["age_form_match"].fillna(False),
        }
    )[list(EXCLUSION_REASONS)]
    found = [
        [reason for reason, applies in zip(reasons.columns, row) if applies]
        for row in reasons.itertuples(index=False, name=None)
    ]
    data["all_exclusion_reasons"] = ["; ".join(row_reasons) for row_reasons in found]
    data["primary_exclusion_reason"] = [row_reasons[0] if row_reasons else "" for row_reasons in found]
    return data
