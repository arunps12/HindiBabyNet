from __future__ import annotations

import html
import re
import unicodedata
from datetime import date

import pandas as pd


HINDI_DIGIT_TRANSLATION = str.maketrans("०१२३४५६७८९", "0123456789")
WHITESPACE_RE = re.compile(r"\s+")
PERCENT_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*से\s*(\d+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
YEAR_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?:year|years|yr|yrs)", re.IGNORECASE)
MONTH_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?:month|months|mo|mos)", re.IGNORECASE)

YES_VALUES = {"हाँ", "हां", "yes", "true"}
NO_VALUES = {"नहीं", "no", "false"}
NA_VALUES = {"लागू नहीं", "na", "n/a"}

CDI8_COMPREHENSION_ONLY = "केवल समझता/समझती है"
CDI8_COMPREHENSION_PRODUCTION = "समझता/समझती है और कहता/कहती है"
CDI19_PRODUCTION = "कहता/कहती है"


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = text.translate(HINDI_DIGIT_TRANSLATION)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_column_name(value: object) -> str:
    return normalize_text(value)


def deduplicate_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    deduplicated: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        deduplicated_name = name if count == 0 else f"{name}.{count}"
        deduplicated.append(deduplicated_name)
        seen[name] = count + 1
    return deduplicated


def parse_bool_yn(value: object) -> bool | None:
    text = normalize_text(value).lower()
    if not text:
        return None
    if text in YES_VALUES:
        return True
    if text in NO_VALUES:
        return False
    return None


def parse_percent(value: object) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    if text.lower() in NA_VALUES:
        return None
    range_match = PERCENT_RANGE_RE.search(text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2
    number_match = NUMBER_RE.search(text.replace("%", ""))
    if number_match:
        return float(number_match.group(0))
    return None


def parse_date_value(value: object) -> date | None:
    text = normalize_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def parse_timestamp(value: object) -> pd.Timestamp | None:
    text = normalize_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def parse_completed_months(value: object) -> int | None:
    text = normalize_text(value)
    if not text:
        return None

    year_match = YEAR_RE.search(text)
    month_match = MONTH_RE.search(text)
    if year_match:
        total_months = float(year_match.group("value")) * 12
        if month_match:
            total_months += float(month_match.group("value"))
        return int(total_months)

    number_match = NUMBER_RE.search(text)
    if number_match:
        return int(float(number_match.group(0)))
    return None


def normalize_phone_number(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return text.replace(" ", "")


def normalize_email(value: object) -> str:
    return normalize_text(value).lower()


def normalize_response_value(value: object) -> str:
    return normalize_text(value)


def score_cdi8_response(value: object) -> tuple[int | None, int | None, bool]:
    text = normalize_response_value(value)
    if not text:
        return (0, 0, False)
    if text == CDI8_COMPREHENSION_ONLY:
        return (1, 0, False)
    if text == CDI8_COMPREHENSION_PRODUCTION:
        return (1, 1, False)
    return (None, None, True)


def score_cdi19_response(value: object) -> tuple[int | None, bool]:
    text = normalize_response_value(value)
    if not text:
        return (0, False)
    if text == CDI19_PRODUCTION:
        return (1, False)
    return (None, True)