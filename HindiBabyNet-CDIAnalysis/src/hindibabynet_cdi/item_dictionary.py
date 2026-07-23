from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from .columns import FORM_METADATA_COLUMNS
from .config import ProjectConfig
from .io import LoadedForm, load_detected_forms


WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ItemDictionaryOutputs:
    master_dictionary: pd.DataFrame
    validation_report: pd.DataFrame


def normalize_hindi_label(value: object) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _get_form(forms: list[LoadedForm], role: str) -> LoadedForm:
    for form in forms:
        if form.role == role:
            return form
    raise KeyError(f"Missing form role: {role}")


def _item_columns(form: LoadedForm) -> list[str]:
    return [column for column in form.data.columns if column not in FORM_METADATA_COLUMNS]


def _normalized_index(columns: list[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for column in columns:
        index.setdefault(normalize_hindi_label(column), []).append(column)
    return index


def _parse_item_id(value: object, fallback_index: int) -> int:
    text = "" if value is None else str(value)
    digits = "".join(character for character in text if character.isdigit())
    return int(digits) if digits else fallback_index


def _form_membership_flags(questionnaire: str) -> tuple[int, int]:
    questionnaire = questionnaire or ""
    return (int("8_18" in questionnaire), int("19_36" in questionnaire))


def _build_validation_rows(
    mapping: pd.DataFrame,
    observed_columns: list[str],
    raw_column_field: str,
    form_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    observed_index = _normalized_index(observed_columns)

    mapped_columns = mapping[raw_column_field].fillna("").tolist()
    mapped_index = _normalized_index([column for column in mapped_columns if normalize_hindi_label(column)])

    for item_id, group in mapping.groupby("item_id", dropna=False):
        if len(group) > 1:
            rows.append(
                {
                    "issue_type": "duplicate_item_id",
                    "form": form_label,
                    "item_id": item_id,
                    "observed_column": "",
                    "mapped_column": "",
                    "detail": f"item_id occurs {len(group)} times",
                }
            )

    duplicated_words = (
        mapping.assign(word_normalized=mapping["word"].map(normalize_hindi_label))
        .groupby("word_normalized", dropna=False)
        .filter(lambda frame: len(frame) > 1)
    )
    for _, row in duplicated_words.iterrows():
        rows.append(
            {
                "issue_type": "duplicate_hindi_label_within_form",
                "form": form_label,
                "item_id": row["item_id"],
                "observed_column": "",
                "mapped_column": row["word"],
                "detail": "Duplicate normalized Hindi item label in mapping",
            }
        )

    for observed_normalized, originals in observed_index.items():
        if observed_normalized not in mapped_index:
            for observed_column in originals:
                rows.append(
                    {
                        "issue_type": "observed_column_absent_from_mapping",
                        "form": form_label,
                        "item_id": "",
                        "observed_column": observed_column,
                        "mapped_column": "",
                        "detail": "Observed CDI item column is not present in the source mapping",
                    }
                )

    for mapped_normalized, originals in mapped_index.items():
        if mapped_normalized not in observed_index:
            for mapped_column in originals:
                mapped_rows = mapping.loc[mapping[raw_column_field].eq(mapped_column), "item_id"].tolist()
                rows.append(
                    {
                        "issue_type": "mapped_item_absent_from_form",
                        "form": form_label,
                        "item_id": " | ".join(str(value) for value in mapped_rows),
                        "observed_column": "",
                        "mapped_column": mapped_column,
                        "detail": "Mapped item is flagged for this form but no matching CDI column was observed",
                    }
                )

    for _, row in mapping.iterrows():
        mapped_column = row[raw_column_field]
        mapped_normalized = normalize_hindi_label(mapped_column)
        if mapped_normalized and mapped_normalized in observed_index:
            exact_match = mapped_column in observed_index[mapped_normalized]
            if not exact_match:
                rows.append(
                    {
                        "issue_type": "normalization_only_match",
                        "form": form_label,
                        "item_id": row["item_id"],
                        "observed_column": " | ".join(observed_index[mapped_normalized]),
                        "mapped_column": mapped_column,
                        "detail": "Matched only after whitespace, Unicode, punctuation, or HTML normalization",
                    }
                )

    return rows


def build_master_item_dictionary(config: ProjectConfig) -> ItemDictionaryOutputs:
    forms = load_detected_forms(config)
    cdi1_columns = _item_columns(_get_form(forms, "cdi_8_18"))
    cdi2_columns = _item_columns(_get_form(forms, "cdi_19_36"))
    cdi1_index = {normalize_hindi_label(column): position for position, column in enumerate(cdi1_columns, start=1)}
    cdi2_index = {normalize_hindi_label(column): position for position, column in enumerate(cdi2_columns, start=1)}

    mapping = pd.read_csv(config.paths.metadata / "word_mapping.csv", dtype=str).fillna("")
    mapping.insert(0, "source_item_id", mapping["item_id"])
    mapping["item_id"] = [
        _parse_item_id(value, fallback_index=index)
        for index, value in enumerate(mapping["source_item_id"], start=1)
    ]

    cdi1_flags, cdi2_flags = zip(*mapping["questionnaire"].map(_form_membership_flags))
    mapping["cdi1"] = list(cdi1_flags)
    mapping["cdi2"] = list(cdi2_flags)
    mapping["cdi1_order"] = mapping["raw_column_8_18"].map(lambda value: cdi1_index.get(normalize_hindi_label(value), pd.NA))
    mapping["cdi2_order"] = mapping["raw_column_19_36"].map(lambda value: cdi2_index.get(normalize_hindi_label(value), pd.NA))
    mapping["active"] = 1

    master_dictionary = mapping[
        [
            "item_id",
            "source_item_id",
            "word",
            "word_english",
            "category",
            "category_english",
            "questionnaire",
            "cdi1",
            "cdi2",
            "cdi1_order",
            "cdi2_order",
            "active",
            "raw_column_8_18",
            "raw_column_19_36",
            "notes",
        ]
    ].copy()

    validation_rows = _build_validation_rows(
        master_dictionary.loc[master_dictionary["cdi1"].eq(1)].copy(),
        cdi1_columns,
        "raw_column_8_18",
        "CDI-I",
    )
    validation_rows.extend(
        _build_validation_rows(
            master_dictionary.loc[master_dictionary["cdi2"].eq(1)].copy(),
            cdi2_columns,
            "raw_column_19_36",
            "CDI-II",
        )
    )

    validation_report = pd.DataFrame(validation_rows)
    if validation_report.empty:
        validation_report = pd.DataFrame(
            columns=["issue_type", "form", "item_id", "observed_column", "mapped_column", "detail"]
        )

    return ItemDictionaryOutputs(
        master_dictionary=master_dictionary.sort_values("item_id").reset_index(drop=True),
        validation_report=validation_report.reset_index(drop=True),
    )