"""Score the answers of the included questionnaires."""

from __future__ import annotations

import pandas as pd

from .cleaning import normalize_response_value, score_cdi8_response, score_cdi19_response
from .columns import CDI1_FORM, CDI2_FORM, SUBMISSION_ID
from .io import LoadedForm
from .item_dictionary import normalize_hindi_label


# For each form: its export, and the columns of the item list with the items of
# the form, their column name in the export and their position in the export.
FORM_FIELDS = {
    CDI1_FORM: ("cdi_8_18", "cdi1", "raw_column_8_18", "cdi1_order"),
    CDI2_FORM: ("cdi_19_36", "cdi2", "raw_column_19_36", "cdi2_order"),
}

_ITEM_INFO = [
    "child_id",
    "submission_id",
    "item_id",
    "item_type",
    "category",
    "category_hindi",
    "item",
    "item_english",
    "response",
]
# Columns of the item tables. Only CDI-1 has a score for understanding.
ITEM_COLUMNS = {
    CDI1_FORM: _ITEM_INFO + ["understands", "produces"],
    CDI2_FORM: _ITEM_INFO + ["produces"],
}

_CHILD_INFO = [
    "child_id",
    "submission_id",
    "completion_date",
    "birthdate",
    "age_months",
    "age_months_exact",
    "gender",
    "mother_education",
    "father_education",
    "hindi_exposure_percent",
]
_TOTALS = [
    "production_total",
    "n_items",
    "n_answered",
    "age_discrepancy_flag",
    "n_submissions_for_child",
    "exact_duplicate_of",
]
# Columns of the summary tables.
SUMMARY_COLUMNS = {
    CDI1_FORM: _CHILD_INFO + ["comprehension_total"] + _TOTALS,
    CDI2_FORM: _CHILD_INFO + _TOTALS,
}


def _form_items(item_list: pd.DataFrame, form_label: str, form: LoadedForm) -> pd.DataFrame:
    """The items of one form, in the order of the export."""
    _, flag, raw_field, order_field = FORM_FIELDS[form_label]
    items = item_list.loc[
        item_list[flag].eq(1) & item_list[raw_field].ne(""),
        ["item_id", "word", "word_english", "category", "category_english", raw_field, order_field],
    ].copy()
    items = items.rename(columns={raw_field: "raw_column", order_field: "item_order"})
    items["raw_column"] = items["raw_column"].map(normalize_hindi_label)
    items = items.sort_values(["item_order", "item_id"]).drop_duplicates(subset=["raw_column"], keep="first")
    if items.empty:
        raise ValueError(f"The item list has no items for the form in {form.path.name}.")
    missing = items.loc[~items["raw_column"].isin(form.data.columns), "item_id"].tolist()
    if missing:
        raise ValueError(
            f"{len(missing)} items of the item list are not columns of {form.path.name} "
            f"(item IDs: {', '.join(str(item_id) for item_id in missing[:10])}). "
            "Check the item names in word_mapping.csv."
        )
    return items


def _iso_date(value: object) -> str:
    """Date as YYYY-MM-DD, empty when there is no date."""
    return "" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def _exact_duplicates(
    children: list[str],
    submissions: list[str],
    created: list[pd.Timestamp],
    answers_of: dict[str, tuple[str, ...]],
) -> list[str]:
    """For each questionnaire: the earliest one of the same child with the same answers.

    Empty for the earliest questionnaire itself and when no earlier one is the same.
    """
    earliest_with_answers: dict[tuple, str] = {}
    duplicate_of = [""] * len(submissions)
    for row in sorted(range(len(submissions)), key=lambda row: created[row]):
        key = (children[row], answers_of[submissions[row]])
        if key in earliest_with_answers:
            duplicate_of[row] = earliest_with_answers[key]
        else:
            earliest_with_answers[key] = submissions[row]
    return duplicate_of


def score_form(
    form_label: str,
    criteria: pd.DataFrame,
    forms: dict[str, LoadedForm],
    item_list: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Score the included questionnaires of one form (CDI1_FORM or CDI2_FORM).

    Returns the item table (one row per questionnaire and item), the summary
    table (one row per questionnaire) and the number of answers that were not
    recognised. An item without an answer has the score 0. An answer that is
    not recognised has no score.
    """
    form = forms[FORM_FIELDS[form_label][0]]
    form_items = _form_items(item_list, form_label, form)
    item_columns = form_items["raw_column"].tolist()
    score_columns = ITEM_COLUMNS[form_label][len(_ITEM_INFO):]

    included = criteria.loc[
        criteria["submitted_form"].eq(form_label) & criteria["included_final"]
    ].reset_index(drop=True)
    if included.empty:
        return (
            pd.DataFrame(columns=ITEM_COLUMNS[form_label]),
            pd.DataFrame(columns=SUMMARY_COLUMNS[form_label]),
            0,
        )

    # One row per questionnaire and item.
    answers = included[["participant_id", "cdi_submission_id"]].merge(
        form.data[[SUBMISSION_ID] + item_columns],
        left_on="cdi_submission_id",
        right_on=SUBMISSION_ID,
        how="left",
        validate="1:1",
    )
    long = answers.melt(
        id_vars=["participant_id", "cdi_submission_id"],
        value_vars=item_columns,
        var_name="raw_column",
        value_name="response",
    )
    long["response"] = long["response"].map(normalize_response_value)
    long = long.merge(form_items, on="raw_column", how="left", validate="m:1")

    # Row order: one questionnaire after the other, the items as in the export.
    questionnaire_position = {
        submission: position for position, submission in enumerate(included["cdi_submission_id"])
    }
    item_position = {column: position for position, column in enumerate(item_columns)}
    long["questionnaire_position"] = long["cdi_submission_id"].map(questionnaire_position)
    long["item_position"] = long["raw_column"].map(item_position)
    long = long.sort_values(["questionnaire_position", "item_position"]).reset_index(drop=True)

    score_answer = score_cdi8_response if form_label == CDI1_FORM else score_cdi19_response
    scores = pd.DataFrame(
        long["response"].map(score_answer).tolist(),
        index=long.index,
        columns=score_columns + ["not_recognised"],
    )

    items = pd.DataFrame(
        {
            "child_id": long["participant_id"],
            "submission_id": long["cdi_submission_id"],
            "item_id": long["item_id"],
            "item_type": "word",
            "category": long["category_english"],
            "category_hindi": long["category"],
            "item": long["word"],
            "item_english": long["word_english"],
            "response": long["response"],
        }
    )
    for column in score_columns:
        items[column] = scores[column].astype("Int64")

    summary = pd.DataFrame(
        {
            "child_id": included["participant_id"],
            "submission_id": included["cdi_submission_id"],
            "completion_date": included["cdi_completion_date"].map(_iso_date),
            "birthdate": included["birthdate_parsed"].map(_iso_date),
            "age_months": included["age_month"].astype("Int64"),
            "age_months_exact": included["calculated_age_months_exact"],
            "gender": included["child_sex"],
            "mother_education": included["mother_education"],
            "father_education": included["father_education"],
            "hindi_exposure_percent": included["hindi_percentage"].astype(float),
            # Empty when the age given by the parent is not known.
            "age_discrepancy_flag": included["age_discrepancy_flag"]
            .astype("boolean")
            .where(included["reported_age_months"].notna()),
        }
    )

    totals = {
        "production_total": ("produces", "sum"),
        "n_items": ("item_id", "size"),
        "n_answered": ("answered", "sum"),
    }
    if "understands" in score_columns:
        totals = {"comprehension_total": ("understands", "sum"), **totals}
    per_questionnaire = (
        items.assign(answered=items["response"].ne(""))
        .groupby("submission_id", sort=False)
        .agg(**totals)
        .reset_index()
    )
    summary = summary.merge(per_questionnaire, on="submission_id", how="left", validate="1:1")

    summary["n_submissions_for_child"] = summary.groupby("child_id")["submission_id"].transform("size")
    answers_of = {
        submission: tuple(responses)
        for submission, responses in items.groupby("submission_id", sort=False)["response"]
    }
    summary["exact_duplicate_of"] = _exact_duplicates(
        summary["child_id"].tolist(),
        summary["submission_id"].tolist(),
        included["cdi_created_timestamp"].tolist(),
        answers_of,
    )

    n_not_recognised = int(scores["not_recognised"].sum())
    return items[ITEM_COLUMNS[form_label]], summary[SUMMARY_COLUMNS[form_label]], n_not_recognised
