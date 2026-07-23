from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cleaning import normalize_response_value, score_cdi8_response, score_cdi19_response
from .columns import CREATED_AT, SUBMISSION_ID
from .config import ProjectConfig
from .eligibility import EligibilityOutputs, build_eligibility_outputs
from .io import LoadedForm, load_detected_forms
from .item_dictionary import ItemDictionaryOutputs, build_master_item_dictionary, normalize_hindi_label


@dataclass(frozen=True)
class ScoringOutputs:
    participant_analysis_all: pd.DataFrame
    participant_analysis_cdi1: pd.DataFrame
    participant_analysis_cdi2: pd.DataFrame
    cdi1_items_long: pd.DataFrame
    cdi2_items_long: pd.DataFrame
    included_participant_contacts: pd.DataFrame
    unknown_responses: pd.DataFrame
    completeness_report: pd.DataFrame


def _get_form(forms: list[LoadedForm], role: str) -> LoadedForm:
    for form in forms:
        if form.role == role:
            return form
    raise KeyError(f"Missing form role: {role}")


def _prepare_form_frame(form: LoadedForm) -> pd.DataFrame:
    data = form.data.copy()
    return data.rename(columns={SUBMISSION_ID: "cdi_submission_id", CREATED_AT: "cdi_created_raw"})


def _participant_columns(criteria: pd.DataFrame) -> list[str]:
    return [
        "participant_id",
        "submitted_form",
        "questionnaire",
        "cdi_submission_id",
        "child_sex",
        "mother_education",
        "reported_age_months",
        "calculated_age_months_exact",
        "age_month",
        "expected_form",
        "age_form_match",
        "hindi_percentage",
        "included_final",
        "primary_exclusion_reason",
        "all_exclusion_reasons",
        "mobile_number",
        "email",
    ]


def _evaluate_completeness(
    participant_frame: pd.DataFrame,
    item_map: pd.DataFrame,
    max_trailing_blank_run: int,
) -> pd.DataFrame:
    ordered_item_map = item_map.sort_values("item_order")
    ordered_columns = ordered_item_map["raw_column"].tolist()
    rows: list[dict[str, object]] = []

    for _, participant in participant_frame.iterrows():
        normalized_values = [normalize_response_value(participant.get(column, "")) for column in ordered_columns]
        answered_positions = [index for index, value in enumerate(normalized_values, start=1) if value]
        n_items_expected = len(ordered_columns)
        n_items_present = sum(column in participant_frame.columns for column in ordered_columns)
        n_nonblank_responses = len(answered_positions)

        if n_items_present < n_items_expected:
            completion_status = "missing_item_columns"
            cdi_is_complete_enough = False
            trailing_blank_run = pd.NA
        elif not answered_positions:
            completion_status = "all_blank"
            cdi_is_complete_enough = False
            trailing_blank_run = n_items_expected
        else:
            last_answered_position = max(answered_positions)
            trailing_blank_run = n_items_expected - last_answered_position
            completion_status = "complete" if trailing_blank_run <= max_trailing_blank_run else "trailing_blank_block"
            cdi_is_complete_enough = completion_status == "complete"

        rows.append(
            {
                "participant_id": participant["participant_id"],
                "cdi_submission_id": participant["cdi_submission_id"],
                "n_items_expected": n_items_expected,
                "n_items_present": n_items_present,
                "n_nonblank_responses": n_nonblank_responses,
                "trailing_blank_run": trailing_blank_run,
                "completion_status": completion_status,
                "cdi_is_complete_enough": cdi_is_complete_enough,
            }
        )

    return pd.DataFrame(rows)


def _score_form_items(
    criteria: pd.DataFrame,
    form_data: pd.DataFrame,
    item_dictionary: pd.DataFrame,
    form_label: str,
    raw_column_field: str,
    item_order_field: str,
    max_trailing_blank_run: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    form_participants = criteria.loc[criteria["submitted_form"].eq(form_label), _participant_columns(criteria)].copy()
    merged = form_participants.merge(form_data, on="cdi_submission_id", how="left", validate="1:1")

    item_map = item_dictionary.loc[
        item_dictionary[raw_column_field].ne(""),
        ["item_id", "word", "word_english", "category", "category_english", raw_column_field, item_order_field],
    ].copy()
    item_map = item_map.rename(columns={raw_column_field: "raw_column", item_order_field: "item_order"})
    item_map["raw_column"] = item_map["raw_column"].map(normalize_hindi_label)
    item_map = item_map.sort_values(["item_order", "item_id"]).drop_duplicates(subset=["raw_column"], keep="first")

    completeness = _evaluate_completeness(merged, item_map, max_trailing_blank_run)
    merged = merged.merge(completeness, on=["participant_id", "cdi_submission_id"], how="left", validate="1:1")

    long = merged.melt(
        id_vars=_participant_columns(criteria) + [
            "n_items_expected",
            "n_items_present",
            "n_nonblank_responses",
            "trailing_blank_run",
            "completion_status",
            "cdi_is_complete_enough",
        ],
        value_vars=item_map["raw_column"].tolist(),
        var_name="raw_column",
        value_name="raw_response",
    )
    long = long.merge(item_map, on="raw_column", how="left", validate="m:1")
    long["form"] = form_label
    long["raw_response"] = long["raw_response"].map(normalize_response_value)

    if form_label == "CDI-I":
        scored = long["raw_response"].map(score_cdi8_response)
        long[["understand", "produce", "unknown_response"]] = pd.DataFrame(scored.tolist(), index=long.index)
    else:
        scored = long["raw_response"].map(score_cdi19_response)
        scored_frame = pd.DataFrame(scored.tolist(), columns=["produce", "unknown_response"], index=long.index)
        long["understand"] = pd.NA
        long[["produce", "unknown_response"]] = scored_frame

    unknown_responses = long.loc[long["unknown_response"].fillna(False)].copy()
    long = long[
        [
            "participant_id",
            "form",
            "item_id",
            "word",
            "word_english",
            "category",
            "category_english",
            "calculated_age_months_exact",
            "age_month",
            "understand",
            "produce",
            "raw_response",
            "cdi_submission_id",
            "included_final",
            "completion_status",
            "cdi_is_complete_enough",
        ]
    ].rename(columns={"calculated_age_months_exact": "age_months_exact"})

    return long, unknown_responses, completeness


def _build_participant_analysis(
    criteria: pd.DataFrame,
    cdi1_items_long: pd.DataFrame,
    cdi2_items_long: pd.DataFrame,
    cdi1_completeness: pd.DataFrame,
    cdi2_completeness: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cdi1_totals = (
        cdi1_items_long.groupby("participant_id", dropna=False)
        .agg(
            comprehension_total=("understand", "sum"),
            production_total=("produce", "sum"),
            age_months_exact=("age_months_exact", "first"),
            age_month=("age_month", "first"),
        )
        .reset_index()
    )
    cdi2_totals = (
        cdi2_items_long.groupby("participant_id", dropna=False)
        .agg(
            production_total=("produce", "sum"),
            age_months_exact=("age_months_exact", "first"),
            age_month=("age_month", "first"),
        )
        .reset_index()
    )
    cdi2_totals["comprehension_total"] = pd.NA

    base_columns = [
        "participant_id",
        "submitted_form",
        "child_sex",
        "mother_education",
        "hindi_percentage",
        "included_final",
        "expected_form",
        "age_form_match",
        "primary_exclusion_reason",
        "all_exclusion_reasons",
    ]
    base = criteria[base_columns].copy().rename(columns={"submitted_form": "form", "child_sex": "sex"})

    cdi1_analysis = base.loc[base["form"].eq("CDI-I")].merge(cdi1_totals, on="participant_id", how="left", validate="1:1")
    cdi1_analysis = cdi1_analysis.merge(cdi1_completeness, on="participant_id", how="left", validate="1:1")

    cdi2_analysis = base.loc[base["form"].eq("CDI-II")].merge(cdi2_totals, on="participant_id", how="left", validate="1:1")
    cdi2_analysis = cdi2_analysis.merge(cdi2_completeness, on="participant_id", how="left", validate="1:1")

    participant_analysis_all = pd.concat([cdi1_analysis, cdi2_analysis], ignore_index=True, sort=False)
    return participant_analysis_all, cdi1_analysis, cdi2_analysis


def build_scoring_outputs(config: ProjectConfig) -> ScoringOutputs:
    eligibility_outputs: EligibilityOutputs = build_eligibility_outputs(config)
    dictionary_outputs: ItemDictionaryOutputs = build_master_item_dictionary(config)
    forms = load_detected_forms(config)

    cdi1_form = _prepare_form_frame(_get_form(forms, "cdi_8_18"))
    cdi2_form = _prepare_form_frame(_get_form(forms, "cdi_19_36"))

    cdi1_items_long, cdi1_unknown, cdi1_completeness = _score_form_items(
        eligibility_outputs.participant_criteria,
        cdi1_form,
        dictionary_outputs.master_dictionary.loc[dictionary_outputs.master_dictionary["cdi1"].eq(1)].copy(),
        "CDI-I",
        "raw_column_8_18",
        "cdi1_order",
        config.completeness.maximum_trailing_blank_run,
    )
    cdi2_items_long, cdi2_unknown, cdi2_completeness = _score_form_items(
        eligibility_outputs.participant_criteria,
        cdi2_form,
        dictionary_outputs.master_dictionary.loc[dictionary_outputs.master_dictionary["cdi2"].eq(1)].copy(),
        "CDI-II",
        "raw_column_19_36",
        "cdi2_order",
        config.completeness.maximum_trailing_blank_run,
    )

    participant_analysis_all, participant_analysis_cdi1, participant_analysis_cdi2 = _build_participant_analysis(
        eligibility_outputs.participant_criteria,
        cdi1_items_long,
        cdi2_items_long,
        cdi1_completeness,
        cdi2_completeness,
    )

    included_participant_contacts = eligibility_outputs.participant_criteria.loc[
        eligibility_outputs.participant_criteria["included_final"],
        [
            "participant_id",
            "submitted_form",
            "calculated_age_months_exact",
            "child_sex",
            "mother_education",
            "mobile_number",
            "email",
        ],
    ].copy().rename(columns={"submitted_form": "form", "calculated_age_months_exact": "age_months_exact"})

    unknown_responses = pd.concat([cdi1_unknown, cdi2_unknown], ignore_index=True, sort=False)
    completeness_report = pd.concat(
        [
            cdi1_completeness.assign(form="CDI-I"),
            cdi2_completeness.assign(form="CDI-II"),
        ],
        ignore_index=True,
        sort=False,
    )

    return ScoringOutputs(
        participant_analysis_all=participant_analysis_all,
        participant_analysis_cdi1=participant_analysis_cdi1,
        participant_analysis_cdi2=participant_analysis_cdi2,
        cdi1_items_long=cdi1_items_long,
        cdi2_items_long=cdi2_items_long,
        included_participant_contacts=included_participant_contacts,
        unknown_responses=unknown_responses,
        completeness_report=completeness_report,
    )