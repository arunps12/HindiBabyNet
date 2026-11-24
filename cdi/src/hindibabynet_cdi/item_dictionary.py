"""The item list of the two questionnaires (word_mapping.csv)."""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

from .columns import FORM_METADATA_COLUMNS
from .config import ProjectConfig
from .io import LoadedForm


WHITESPACE_RE = re.compile(r"\s+")

# Columns that word_mapping.csv must have.
ITEM_LIST_COLUMNS = [
    "item_id",
    "word",
    "word_english",
    "category",
    "category_english",
    "questionnaire",
    "raw_column_8_18",
    "raw_column_19_36",
]


def normalize_hindi_label(value: object) -> str:
    """Clean an item name so that the item list and the export can be matched."""
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _parse_item_id(value: object, fallback_index: int) -> int:
    """Number in the item ID of the item list, or the row number when there is none."""
    text = "" if value is None else str(value)
    digits = "".join(character for character in text if character.isdigit())
    return int(digits) if digits else fallback_index


def build_item_list(config: ProjectConfig, forms: dict[str, LoadedForm]) -> pd.DataFrame:
    """Read the item list and match it with the item columns of the two CDI exports.

    Added columns: cdi1 and cdi2 (1 when the item belongs to the form),
    cdi1_order and cdi2_order (position of the item column in the export).
    """
    items = pd.read_csv(config.word_mapping_path, dtype=str, encoding="utf-8-sig").fillna("")
    missing = [column for column in ITEM_LIST_COLUMNS if column not in items.columns]
    if missing:
        raise ValueError(f"The item list {config.word_mapping_path} has no column: {', '.join(missing)}.")
    items = items[ITEM_LIST_COLUMNS].copy()
    items["item_id"] = [
        _parse_item_id(value, fallback_index=index)
        for index, value in enumerate(items["item_id"], start=1)
    ]

    for role, questionnaire, flag, raw_column, order in (
        ("cdi_8_18", config.analysis.younger_questionnaire, "cdi1", "raw_column_8_18", "cdi1_order"),
        ("cdi_19_36", config.analysis.older_questionnaire, "cdi2", "raw_column_19_36", "cdi2_order"),
    ):
        item_columns = [column for column in forms[role].data.columns if column not in FORM_METADATA_COLUMNS]
        position = {normalize_hindi_label(column): index for index, column in enumerate(item_columns, start=1)}
        items[flag] = [int(questionnaire in text) for text in items["questionnaire"]]
        items[order] = [position.get(normalize_hindi_label(value), pd.NA) for value in items[raw_column]]

    return items.sort_values("item_id").reset_index(drop=True)
