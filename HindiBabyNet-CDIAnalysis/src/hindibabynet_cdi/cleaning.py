"""Normalization and parsing helpers for Hindi CDI analysis."""

from __future__ import annotations

from datetime import datetime
from html import unescape
from hashlib import sha1
import math
import re
from typing import Any

from dateutil import parser as date_parser

WHITESPACE_RE = re.compile(r"[\s\u00A0\u2007\u202F]+")
AGE_HINT_RE = re.compile(r"\d+(?:\.\d+)?")
YEAR_PATTERN_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?:years?|yrs?|year|saal|sal|साल)", re.IGNORECASE)
MONTH_PATTERN_RE = re.compile(
	r"(?P<value>\d+(?:\.\d+)?)\s*(?:months?|month|mos?|mahine|mahina|महीने|महिने|माह)",
	re.IGNORECASE,
)

HINDI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

UNDERSTANDS_ONLY = "केवल समझता/समझती है"
UNDERSTANDS_AND_SAYS = "समझता/समझती है और कहता/कहती है"
SAYS_WORD = "कहता/कहती है"

PRIMARY_SCHOOL = 1
HIGH_SCHOOL = 2
SOME_COLLEGE = 3
BACHELOR = 4
MASTER = 5
OTHER_QUALIFICATIONS = 6
UNKNOWN_CODE = 99

TRUE_VALUES = {
	"1",
	"true",
	"t",
	"yes",
	"y",
	"eligible",
	"complete",
	"completed",
	"done",
	"haan",
	"haan ji",
	"हां",
	"हाँ",
	"जी हाँ",
}
FALSE_VALUES = {
	"0",
	"false",
	"f",
	"no",
	"n",
	"ineligible",
	"incomplete",
	"not completed",
	"nahin",
	"nahi",
	"नहीं",
	"ना",
}

EDUCATION_CODE_PATTERNS: tuple[tuple[int, tuple[str, ...]], ...] = (
	(PRIMARY_SCHOOL, ("primary", "प्राथमिक")),
	(HIGH_SCHOOL, ("high school", "secondary", "स्कूल", "बारहवीं", "12वीं", "12th", "10th")),
	(SOME_COLLEGE, ("some college", "diploma", "college", "डिप्लोमा")),
	(MASTER, ("master", "postgraduate", "post-graduate", "ma", "msc", "m.a", "m.sc", "परास्नातक")),
	(BACHELOR, ("bachelor", "graduate", "ba", "bsc", "b.a", "b.sc", "स्नातक")),
	(OTHER_QUALIFICATIONS, ("other", "अन्य")),
)

SEX_CODE_PATTERNS: tuple[tuple[int, tuple[str, ...]], ...] = (
	(2, ("female", "girl", "लड़की")),
	(1, ("male", "boy", "लड़का")),
)

LANGUAGE_TRANSLATIONS = {
	"अंग्रेज़ी": "English",
	"हिंदी": "Hindi",
	"हिंदी + अन्य": "Hindi + other",
	"अन्य कोई भाषा": "Other language",
	"अन्य भारतीय भाषा": "Other Indian language",
	"लागू नहीं": "Not applicable",
	"भारत": "India",
	"उत्तर नहीं देना चाहता/चाहती": "Prefer not to answer",
}

RESIDENCE_TRANSLATIONS = {
	"शहर": "City",
	"कस्बा": "Town",
	"गाँव": "Village",
	"गांव": "Village",
	"मेट्रो शहर/राजधानी": "Metro/capital city",
}

RESPONSE_NORMALIZATION_MAP = {
	UNDERSTANDS_ONLY: UNDERSTANDS_ONLY,
	UNDERSTANDS_AND_SAYS: UNDERSTANDS_AND_SAYS,
	SAYS_WORD: SAYS_WORD,
}


def is_missing(value: Any) -> bool:
	if value is None:
		return True
	if isinstance(value, float) and math.isnan(value):
		return True
	if isinstance(value, str):
		text = value.replace("\ufeff", "")
		return WHITESPACE_RE.sub(" ", text).strip() == ""
	return False


def convert_hindi_digits(value: Any) -> str:
	if value is None:
		return ""
	return str(value).translate(HINDI_DIGITS)


def clean_html_entities(value: Any) -> str:
	if value is None:
		return ""
	return unescape(str(value))


def normalize_text(value: Any) -> str:
	if is_missing(value):
		return ""
	text = convert_hindi_digits(clean_html_entities(value)).replace("\ufeff", "")
	return WHITESPACE_RE.sub(" ", text).strip()


def normalize_column_name(value: Any) -> str:
	return normalize_text(value)


def normalize_bool(value: Any) -> bool | None:
	normalized = normalize_text(value).casefold()
	if normalized == "":
		return None
	if normalized in TRUE_VALUES:
		return True
	if normalized in FALSE_VALUES:
		return False
	return None


def normalize_response(value: Any) -> str:
	normalized = normalize_text(value)
	return RESPONSE_NORMALIZATION_MAP.get(normalized, normalized)


def parse_datetime_safe(value: Any, *, dayfirst: bool = True) -> datetime | None:
	normalized = normalize_text(value)
	if normalized == "":
		return None
	try:
		return date_parser.parse(normalized, dayfirst=dayfirst)
	except (ValueError, OverflowError, TypeError):
		return None


def parse_age_months(value: Any) -> float | None:
	normalized = normalize_text(value)
	if normalized == "":
		return None
	normalized_casefold = normalized.casefold()
	matches = [float(match) for match in AGE_HINT_RE.findall(normalized)]
	if len(matches) > 1 and any(token in normalized_casefold for token in ("से", "between", "-", "to")):
		return sum(matches[:2]) / 2
	year_matches = [float(match.group("value")) for match in YEAR_PATTERN_RE.finditer(normalized_casefold)]
	month_matches = [float(match.group("value")) for match in MONTH_PATTERN_RE.finditer(normalized_casefold)]
	if year_matches or month_matches:
		years = year_matches[0] if year_matches else 0.0
		months = month_matches[0] if month_matches else 0.0
		return round((years * 12) + months, 2)
	if not matches:
		return None
	if len(matches) == 1:
		return matches[0]
	return matches[0]


def compute_age_days(
	birthdate: datetime | None,
	reference_datetime: datetime | None,
) -> int | None:
	if birthdate is None or reference_datetime is None:
		return None
	if birthdate.tzinfo is not None and reference_datetime.tzinfo is None:
		birthdate = birthdate.replace(tzinfo=None)
	elif birthdate.tzinfo is None and reference_datetime.tzinfo is not None:
		reference_datetime = reference_datetime.replace(tzinfo=None)
	delta_days = (reference_datetime - birthdate).total_seconds() / 86400
	if delta_days < 0:
		return None
	return int(round(delta_days))


def compute_age_months(
	birthdate: datetime | None,
	reference_datetime: datetime | None,
	*,
	age_month_divisor: float = 30.44,
) -> float | None:
	age_days = compute_age_days(birthdate, reference_datetime)
	if age_days is None:
		return None
	return round(age_days / age_month_divisor, 2)


def assess_age_quality(
	*,
	birthdate: datetime | None,
	reference_datetime: datetime | None,
	raw_age_months: float | None,
	age_month_divisor: float = 30.4375,
	discrepancy_threshold_months: float = 2.0,
) -> dict[str, Any]:
	age_days = compute_age_days(birthdate, reference_datetime)
	age_months = None if age_days is None else round(age_days / age_month_divisor, 2)
	flags: list[str] = []
	if birthdate is None:
		flags.append("missing_birthdate")
	if reference_datetime is None:
		flags.append("missing_reference_date")
	if birthdate is not None and reference_datetime is not None:
		if birthdate.tzinfo is not None and reference_datetime.tzinfo is None:
			birthdate = birthdate.replace(tzinfo=None)
		elif birthdate.tzinfo is None and reference_datetime.tzinfo is not None:
			reference_datetime = reference_datetime.replace(tzinfo=None)
	if birthdate is not None and reference_datetime is not None and birthdate > reference_datetime:
		flags.append("birthdate_after_created")
	if age_months is not None and age_months < 8:
		flags.append("age_below_range")
	if age_months is not None and age_months > 36:
		flags.append("age_above_range")
	age_discrepancy_flag = False
	if age_months is not None and raw_age_months is not None:
		age_discrepancy_flag = abs(age_months - raw_age_months) > discrepancy_threshold_months
		if age_discrepancy_flag:
			flags.append("raw_age_mismatch")
	return {
		"age_days": age_days,
		"age_months": age_months,
		"age_months_from_raw": raw_age_months,
		"age_discrepancy_flag": age_discrepancy_flag,
		"age_quality_flag": "; ".join(flags) if flags else "ok",
	}


def code_education(value: Any) -> int:
	normalized = normalize_text(value).casefold()
	if normalized == "":
		return UNKNOWN_CODE
	for code, tokens in EDUCATION_CODE_PATTERNS:
		if any(token in normalized for token in tokens):
			return code
	return UNKNOWN_CODE


def code_sex(value: Any) -> int:
	normalized = normalize_text(value).casefold()
	if normalized == "":
		return UNKNOWN_CODE
	for code, tokens in SEX_CODE_PATTERNS:
		if any(token in normalized for token in tokens):
			return code
	return UNKNOWN_CODE


def translate_language_value(value: Any) -> str:
	normalized = normalize_text(value)
	return LANGUAGE_TRANSLATIONS.get(normalized, normalized)


def translate_residence_value(value: Any) -> str:
	normalized = normalize_text(value)
	return RESIDENCE_TRANSLATIONS.get(normalized, normalized)


def create_participant_id(*parts: Any, prefix: str = "participant") -> str:
	normalized_parts = [normalize_text(part) for part in parts if normalize_text(part)]
	if not normalized_parts:
		raise ValueError("Cannot create participant ID from empty parts")
	digest = sha1("|".join(normalized_parts).encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"