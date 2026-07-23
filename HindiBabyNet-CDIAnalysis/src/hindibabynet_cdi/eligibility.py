from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .age import add_age_columns
from .cleaning import normalize_email, normalize_phone_number, parse_bool_yn, parse_percent, normalize_text
from .config import ProjectConfig
from .linking import LinkageOutputs, build_participant_linkage


@dataclass(frozen=True)
class EligibilityOutputs:
    participant_criteria: pd.DataFrame
    participant_flow: pd.DataFrame
    exclusions: pd.DataFrame


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
    data = add_age_columns(linkage, config)
    data = _derive_hindi_exposure(data, config)

    data["consent_given"] = data["consent_response"].map(parse_bool_yn).astype("boolean").fillna(False)
    data["mother_tongue_is_hindi"] = data["mother_tongue_response"].map(parse_bool_yn).eq(
        config.eligibility.required_mother_tongue_response
    )
    data["preterm_allowed"] = data["preterm_response"].map(parse_bool_yn).ne(True) if not config.eligibility.allow_preterm else True
    data["impairment_allowed"] = data["impairment_response"].map(parse_bool_yn).ne(True) if not config.eligibility.allow_impairment else True
    data["eligibility_age_range_passed"] = data["age_range_response"].map(parse_bool_yn).fillna(False)
    data["eligibility_passed"] = (
        data["mother_tongue_is_hindi"]
        & data["preterm_allowed"]
        & data["impairment_allowed"]
        & data["eligibility_age_range_passed"]
    )

    data["successfully_linked"] = (
        data["has_background_link"] & data["has_eligibility_link"] & data["has_consent_link"]
    )
    data["cdi_is_complete_enough"] = True

    data["included_final"] = (
        data["consent_given"]
        & data["successfully_linked"]
        & data["eligibility_passed"]
        & data["hindi_exposure_passed"]
        & data["age_is_valid"].fillna(False)
        & data["age_form_match"].fillna(False)
        & data["cdi_is_complete_enough"]
    )

    exclusion_reason_columns = {
        "no_consent": ~data["consent_given"],
        "broken_reference_chain": ~data["successfully_linked"],
        "failed_eligibility": ~data["eligibility_passed"],
        "hindi_exposure_unknown": data["hindi_exposure_unknown"],
        "hindi_exposure_invalid": data["hindi_exposure_invalid"],
        "hindi_exposure_leq_75": ~data["hindi_exposure_passed"] & ~data["hindi_exposure_unknown"] & ~data["hindi_exposure_invalid"],
        "age_outside_form_range": ~data["age_is_valid"].fillna(False),
        "age_form_mismatch": ~data["age_form_match"].fillna(False),
        "incomplete_cdi": ~data["cdi_is_complete_enough"],
    }

    data["all_exclusion_reasons"] = data.apply(
        lambda row: "; ".join(
            reason for reason, mask in exclusion_reason_columns.items() if bool(mask.loc[row.name])
        ),
        axis=1,
    )
    data["primary_exclusion_reason"] = data["all_exclusion_reasons"].str.split("; ").str[0].fillna("")
    data["mobile_number"] = data["mobile_number"].map(normalize_phone_number)
    data["email"] = data["email"].map(normalize_email)
    return data


def _build_participant_flow(criteria: pd.DataFrame) -> pd.DataFrame:
    stages = [
        ("Raw CDI submissions", criteria["cdi_submission_id"].notna()),
        ("Successfully linked to background", criteria["has_background_link"]),
        ("Successfully linked to eligibility", criteria["has_eligibility_link"]),
        ("Successfully linked to consent", criteria["has_consent_link"]),
        ("Consent provided", criteria["consent_given"]),
        ("Eligibility passed", criteria["eligibility_passed"]),
        ("Hindi exposure >75%", criteria["hindi_exposure_passed"]),
        ("Age matches form", criteria["age_form_match"].fillna(False)),
        ("Passed completeness check", criteria["cdi_is_complete_enough"]),
        ("Final included participants", criteria["included_final"]),
    ]
    rows: list[dict[str, object]] = []
    for stage, mask in stages:
        cdi1 = int((mask & criteria["submitted_form"].eq("CDI-I")).sum())
        cdi2 = int((mask & criteria["submitted_form"].eq("CDI-II")).sum())
        rows.append({"Stage": stage, "CDI-I": cdi1, "CDI-II": cdi2, "Total": cdi1 + cdi2})
    return pd.DataFrame(rows)


def _build_exclusions(criteria: pd.DataFrame) -> pd.DataFrame:
    exclusion_definitions = [
        ("No consent", criteria["primary_exclusion_reason"].eq("no_consent")),
        ("Broken reference chain", criteria["primary_exclusion_reason"].eq("broken_reference_chain")),
        ("Failed eligibility", criteria["primary_exclusion_reason"].eq("failed_eligibility")),
        ("Hindi exposure ≤75%", criteria["primary_exclusion_reason"].eq("hindi_exposure_leq_75")),
        ("Hindi exposure unknown", criteria["primary_exclusion_reason"].eq("hindi_exposure_unknown")),
        ("Age outside form range", criteria["primary_exclusion_reason"].isin(["age_outside_form_range", "age_form_mismatch"])),
        ("Incomplete CDI", criteria["primary_exclusion_reason"].eq("incomplete_cdi")),
    ]
    rows: list[dict[str, object]] = []
    for reason, mask in exclusion_definitions:
        cdi1 = int((mask & criteria["submitted_form"].eq("CDI-I")).sum())
        cdi2 = int((mask & criteria["submitted_form"].eq("CDI-II")).sum())
        rows.append({"Exclusion reason": reason, "CDI-I": cdi1, "CDI-II": cdi2, "Total": cdi1 + cdi2})
    return pd.DataFrame(rows)


def build_eligibility_outputs(config: ProjectConfig) -> EligibilityOutputs:
    linkage_outputs: LinkageOutputs = build_participant_linkage(config)
    criteria = evaluate_participant_criteria(linkage_outputs.participant_linkage, config)
    return EligibilityOutputs(
        participant_criteria=criteria,
        participant_flow=_build_participant_flow(criteria),
        exclusions=_build_exclusions(criteria),
    )