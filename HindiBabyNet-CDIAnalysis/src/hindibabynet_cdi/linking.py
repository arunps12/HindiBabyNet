"""Participant linkage for Hindi CDI raw Nettskjema exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hindibabynet_cdi.cleaning import assess_age_quality, normalize_bool, normalize_text, parse_age_months, parse_datetime_safe
from hindibabynet_cdi.config import ProjectConfig, load_config
from hindibabynet_cdi.io import load_form_by_id

CONSENT_INFO_COLUMN = "मुझे ऊपर वर्णित परियोजना के संबंध में जानकारी प्राप्त हो गई है।"
CONSENT_GIVEN_COLUMN = "मैं सहमति देता/देती हूँ कि मेरी गुमनाम जानकारी ऊपर वर्णित परियोजना में उपयोग की जाएगी।"
CONSENT_INDIA_COLUMN = "सहमति देने के लिए धन्यवाद। क्या आपके बच्चे का जन्म भारत में हुआ है और क्या वह जन्म से अब तक भारत में ही रह रहा है?"
ELIGIBILITY_REFERENCE_COLUMN = "Reference ID"
BACKGROUND_REFERENCE_COLUMN = "SUBMISSION_REFERENCE"
FORWARDED_FORM_COLUMN = "$forwarded_to_form"
CHILD_AGE_MONTHS_COLUMN = "बच्चे की उम्र कितने महीनों की है?"
CHILD_AGE_GROUP_COLUMN = "बच्चे की आयु"
BIRTHDATE_COLUMN = "birthdate"

ELIGIBILITY_COLUMNS = {
	"mother_tongue_hindi": "क्या आपके बच्चे की मातृभाषा हिंदी है?",
	"preterm": "क्या आपका बच्चा प्री-टर्म जन्मा है?",
	"sensory": "क्या आपके बच्चे को बोलने, सुनने या देखने से संबंधित कोई समस्या है?",
	"age_range": "क्या आपके बच्चे की आयु 8 से 36 महीने के बीच है?",
}

PARTICIPANT_LINKAGE_COLUMNS = [
	"participant_id",
	"consent_submission_id",
	"eligibility_submission_id",
	"background_submission_id",
	"cdi_submission_id",
	"cdi_form_id",
	"questionnaire",
	"consent_completed",
	"consent_given",
	"consent_status",
	"eligibility_completed",
	"eligible",
	"eligibility_status",
	"background_completed",
	"cdi_completed",
	"included_analysis",
	"exclusion_reason",
	"linkage_quality_flag",
	"forwarded_form_mismatch",
	"age_group_raw_vs_calculated_mismatch",
	"questionnaire_age_range_mismatch",
]


def load_pipeline_forms(config: ProjectConfig | None = None) -> dict[str, pd.DataFrame]:
	project_config = config or load_config()
	return {
		"consent": load_form_by_id(project_config.forms.consent_form_id, config=project_config),
		"eligibility": load_form_by_id(project_config.forms.eligibility_form_id, config=project_config),
		"background": load_form_by_id(project_config.forms.background_form_id, config=project_config),
		"cdi_8_18": load_form_by_id(project_config.forms.cdi_8_18_form_id, config=project_config),
		"cdi_19_36": load_form_by_id(project_config.forms.cdi_19_36_form_id, config=project_config),
	}


def _prepare_form(dataframe: pd.DataFrame, *, reference_column: str | None = None, form_id: int | None = None) -> pd.DataFrame:
	prepared = dataframe.copy()
	if "$submission_id" not in prepared.columns:
		prepared["$submission_id"] = ""
	if "$created" not in prepared.columns:
		prepared["$created"] = ""
	prepared["_submission_id_norm"] = prepared["$submission_id"].map(normalize_text)
	prepared["_created_dt"] = prepared["$created"].map(parse_datetime_safe)
	prepared["_reference_norm"] = ""
	if reference_column is not None and reference_column in prepared.columns:
		prepared["_reference_norm"] = prepared[reference_column].map(normalize_text)
	prepared["_form_id"] = "" if form_id is None else str(form_id)
	return prepared


def _select_latest(dataframe: pd.DataFrame) -> pd.Series | None:
	if dataframe.empty:
		return None
	sorted_frame = dataframe.assign(
		_created_sort=dataframe["_created_dt"].map(lambda value: value.timestamp() if value is not None else float("-inf"))
	).sort_values(by=["_created_sort", "$submission_id"], ascending=[False, False])
	return sorted_frame.iloc[0]


def _lookup_latest(dataframe: pd.DataFrame, key_column: str) -> dict[str, pd.Series]:
	lookup: dict[str, pd.Series] = {}
	for key, frame in dataframe.groupby(key_column, dropna=False):
		normalized_key = normalize_text(key)
		if normalized_key == "":
			continue
		row = _select_latest(frame)
		if row is not None:
			lookup[normalized_key] = row
	return lookup


def _consent_status(consent_row: pd.Series | None) -> tuple[bool, bool | None, str]:
	if consent_row is None:
		return False, None, "missing_link"
	consent_given = normalize_bool(consent_row.get(CONSENT_GIVEN_COLUMN, ""))
	info_received = normalize_bool(consent_row.get(CONSENT_INFO_COLUMN, ""))
	india_resident = normalize_bool(consent_row.get(CONSENT_INDIA_COLUMN, ""))
	if consent_given is False:
		return True, False, "not_given"
	if consent_given is True and info_received is True and india_resident is True:
		return True, True, "confirmed"
	if consent_given is True:
		return True, True, "unclear"
	return True, None, "unclear"


def _eligibility_status(eligibility_row: pd.Series | None) -> tuple[bool, bool | None, str]:
	if eligibility_row is None:
		return False, None, "missing"
	observed = {
		name: normalize_bool(eligibility_row.get(column, ""))
		for name, column in ELIGIBILITY_COLUMNS.items()
	}
	if observed["mother_tongue_hindi"] is False or observed["preterm"] is True or observed["sensory"] is True or observed["age_range"] is False:
		return True, False, "not_eligible"
	if (
		observed["mother_tongue_hindi"] is True
		and observed["preterm"] is False
		and observed["sensory"] is False
		and observed["age_range"] is True
	):
		return True, True, "eligible"
	return True, None, "unclear"


def _questionnaire_from_form_id(form_id: str) -> str:
	if form_id == "539642":
		return "8_18"
	if form_id == "539644":
		return "19_36"
	return ""


def _questionnaire_from_age_months(age_months: float | None) -> str:
	if age_months is None:
		return ""
	if 8 <= age_months <= 18.999:
		return "8_18"
	if 19 <= age_months <= 36.999:
		return "19_36"
	return ""


def _raw_age_group_questionnaire(raw_value: object) -> str:
	normalized = normalize_text(raw_value)
	if normalized == "":
		return ""
	if "19" in normalized or "36" in normalized:
		return "19_36"
	if "8" in normalized or "18" in normalized:
		return "8_18"
	return _questionnaire_from_age_months(parse_age_months(normalized))


def _derive_participant_id(consent_row: pd.Series | None, eligibility_row: pd.Series | None, background_row: pd.Series | None, cdi_row: pd.Series | None) -> str:
	background_reference = normalize_text(background_row.get(BACKGROUND_REFERENCE_COLUMN, "")) if background_row is not None else ""
	cdi_reference = normalize_text(cdi_row.get(BACKGROUND_REFERENCE_COLUMN, "")) if cdi_row is not None else ""
	eligibility_reference = normalize_text(eligibility_row.get(ELIGIBILITY_REFERENCE_COLUMN, "")) if eligibility_row is not None else ""
	consent_submission_id = normalize_text(consent_row.get("$submission_id", "")) if consent_row is not None else ""
	anchor = background_reference or cdi_reference or eligibility_reference or consent_submission_id
	if anchor:
		return f"HBN_{anchor}"
	return "HBN_orphan"


def _build_flags(background_row: pd.Series | None, cdi_row: pd.Series | None, questionnaire: str, cdi_form_id: str, project_config: ProjectConfig) -> dict[str, object]:
	forwarded_form = normalize_text(background_row.get(FORWARDED_FORM_COLUMN, "")) if background_row is not None else ""
	forwarded_form_mismatch = bool(forwarded_form and cdi_form_id and forwarded_form != cdi_form_id)
	birthdate = parse_datetime_safe(background_row.get(BIRTHDATE_COLUMN, "")) if background_row is not None else None
	background_created = parse_datetime_safe(background_row.get("$created", "")) if background_row is not None else None
	cdi_created = parse_datetime_safe(cdi_row.get("$created", "")) if cdi_row is not None else None
	reference_datetime = cdi_created or background_created
	raw_age_months = parse_age_months(background_row.get(CHILD_AGE_MONTHS_COLUMN, "")) if background_row is not None else None
	age_quality = assess_age_quality(
		birthdate=birthdate,
		reference_datetime=reference_datetime,
		raw_age_months=raw_age_months,
		age_month_divisor=project_config.analysis.age_month_divisor,
	)
	calculated_questionnaire = _questionnaire_from_age_months(age_quality["age_months"])
	raw_questionnaire = _raw_age_group_questionnaire(background_row.get(CHILD_AGE_GROUP_COLUMN, "")) if background_row is not None else ""
	age_group_raw_vs_calculated_mismatch = bool(raw_questionnaire and calculated_questionnaire and raw_questionnaire != calculated_questionnaire)
	questionnaire_age_range_mismatch = bool(questionnaire and calculated_questionnaire and questionnaire != calculated_questionnaire)
	quality_flags = []
	if forwarded_form_mismatch:
		quality_flags.append("forwarded_form_mismatch")
	if age_group_raw_vs_calculated_mismatch:
		quality_flags.append("age_group_raw_vs_calculated_mismatch")
	if questionnaire_age_range_mismatch:
		quality_flags.append("questionnaire_age_range_mismatch")
	if age_quality["age_quality_flag"] != "ok":
		quality_flags.append(str(age_quality["age_quality_flag"]))
	return {
		"forwarded_form_mismatch": forwarded_form_mismatch,
		"age_group_raw_vs_calculated_mismatch": age_group_raw_vs_calculated_mismatch,
		"questionnaire_age_range_mismatch": questionnaire_age_range_mismatch,
		"linkage_quality_flag": "; ".join(quality_flags) if quality_flags else "ok",
	}


def _age_usable_or_flaggable(background_row: pd.Series | None, cdi_row: pd.Series | None, project_config: ProjectConfig) -> bool:
	if background_row is None:
		return False
	birthdate = parse_datetime_safe(background_row.get(BIRTHDATE_COLUMN, ""))
	background_created = parse_datetime_safe(background_row.get("$created", ""))
	cdi_created = parse_datetime_safe(cdi_row.get("$created", "")) if cdi_row is not None else None
	raw_age_months = parse_age_months(background_row.get(CHILD_AGE_MONTHS_COLUMN, ""))
	raw_age_group = normalize_text(background_row.get(CHILD_AGE_GROUP_COLUMN, ""))
	age_quality = assess_age_quality(
		birthdate=birthdate,
		reference_datetime=cdi_created or background_created,
		raw_age_months=raw_age_months,
		age_month_divisor=project_config.analysis.age_month_divisor,
	)
	return bool(age_quality["age_months"] is not None or raw_age_months is not None or raw_age_group)


def _build_row(*, consent_row: pd.Series | None, eligibility_row: pd.Series | None, background_row: pd.Series | None, cdi_row: pd.Series | None, project_config: ProjectConfig) -> dict[str, object]:
	cdi_form_id = normalize_text(cdi_row.get("_form_id", "")) if cdi_row is not None else ""
	questionnaire = _questionnaire_from_form_id(cdi_form_id)
	consent_completed, consent_given, consent_status = _consent_status(consent_row)
	eligibility_completed, eligible, eligibility_status = _eligibility_status(eligibility_row)
	background_completed = background_row is not None
	cdi_completed = cdi_row is not None
	age_usable_or_flaggable = _age_usable_or_flaggable(background_row, cdi_row, project_config)
	flags = _build_flags(background_row, cdi_row, questionnaire, cdi_form_id, project_config)
	reasons: list[str] = []
	if not cdi_completed:
		reasons.append("missing_cdi")
	if not background_completed:
		reasons.append("missing_background")
	if consent_status == "not_given":
		reasons.append("consent_not_given")
	if eligibility_status == "not_eligible":
		reasons.append("not_eligible")
	if background_completed and not age_usable_or_flaggable:
		reasons.append("age_unusable")
	consent_allows_inclusion = consent_given is True or consent_status in {"missing_link", "unclear"}
	eligibility_allows_inclusion = eligible is True or eligibility_status in {"unclear", "missing"}
	included_analysis = bool(
		cdi_completed
		and background_completed
		and consent_allows_inclusion
		and eligibility_allows_inclusion
		and age_usable_or_flaggable
	)
	return {
		"participant_id": _derive_participant_id(consent_row, eligibility_row, background_row, cdi_row),
		"consent_submission_id": normalize_text(consent_row.get("$submission_id", "")) if consent_row is not None else "",
		"eligibility_submission_id": normalize_text(eligibility_row.get("$submission_id", "")) if eligibility_row is not None else "",
		"background_submission_id": normalize_text(background_row.get("$submission_id", "")) if background_row is not None else "",
		"cdi_submission_id": normalize_text(cdi_row.get("$submission_id", "")) if cdi_row is not None else "",
		"cdi_form_id": cdi_form_id,
		"questionnaire": questionnaire,
		"consent_completed": consent_completed,
		"consent_given": consent_given,
		"consent_status": consent_status,
		"eligibility_completed": eligibility_completed,
		"eligible": eligible,
		"eligibility_status": eligibility_status,
		"background_completed": background_completed,
		"cdi_completed": cdi_completed,
		"included_analysis": included_analysis,
		"exclusion_reason": "; ".join(reasons),
		**flags,
	}


def build_participant_linkage(forms: dict[str, pd.DataFrame] | None = None, *, config: ProjectConfig | None = None) -> pd.DataFrame:
	project_config = config or load_config()
	loaded_forms = forms or load_pipeline_forms(project_config)

	consent_df = _prepare_form(loaded_forms["consent"])
	eligibility_df = _prepare_form(loaded_forms["eligibility"], reference_column=ELIGIBILITY_REFERENCE_COLUMN)
	background_df = _prepare_form(loaded_forms["background"], reference_column=BACKGROUND_REFERENCE_COLUMN)
	cdi_8_18_df = _prepare_form(loaded_forms["cdi_8_18"], reference_column=BACKGROUND_REFERENCE_COLUMN, form_id=project_config.forms.cdi_8_18_form_id)
	cdi_19_36_df = _prepare_form(loaded_forms["cdi_19_36"], reference_column=BACKGROUND_REFERENCE_COLUMN, form_id=project_config.forms.cdi_19_36_form_id)
	all_cdi_df = pd.concat([cdi_8_18_df, cdi_19_36_df], ignore_index=True)

	consent_by_submission = _lookup_latest(consent_df, "_submission_id_norm")
	eligibility_by_submission = _lookup_latest(eligibility_df, "_submission_id_norm")
	eligibility_by_reference = _lookup_latest(eligibility_df, "_reference_norm")
	background_by_submission = _lookup_latest(background_df, "_submission_id_norm")
	background_by_reference = _lookup_latest(background_df, "_reference_norm")
	cdi_by_reference = _lookup_latest(all_cdi_df, "_reference_norm")

	rows: list[dict[str, object]] = []
	seen_participants: set[str] = set()

	for _, background_row in background_df.iterrows():
		eligibility_row = eligibility_by_submission.get(normalize_text(background_row.get("_reference_norm", "")))
		consent_row = None if eligibility_row is None else consent_by_submission.get(normalize_text(eligibility_row.get("_reference_norm", "")))
		cdi_row = cdi_by_reference.get(normalize_text(background_row.get("_submission_id_norm", "")))
		row = _build_row(consent_row=consent_row, eligibility_row=eligibility_row, background_row=background_row, cdi_row=cdi_row, project_config=project_config)
		if row["participant_id"] not in seen_participants:
			seen_participants.add(str(row["participant_id"]))
			rows.append(row)

	for _, eligibility_row in eligibility_df.iterrows():
		consent_row = consent_by_submission.get(normalize_text(eligibility_row.get("_reference_norm", "")))
		background_row = background_by_reference.get(normalize_text(eligibility_row.get("_submission_id_norm", "")))
		cdi_row = None if background_row is None else cdi_by_reference.get(normalize_text(background_row.get("_submission_id_norm", "")))
		row = _build_row(consent_row=consent_row, eligibility_row=eligibility_row, background_row=background_row, cdi_row=cdi_row, project_config=project_config)
		if row["participant_id"] not in seen_participants:
			seen_participants.add(str(row["participant_id"]))
			rows.append(row)

	for _, consent_row in consent_df.iterrows():
		eligibility_row = eligibility_by_reference.get(normalize_text(consent_row.get("_submission_id_norm", "")))
		background_row = None if eligibility_row is None else background_by_reference.get(normalize_text(eligibility_row.get("_submission_id_norm", "")))
		cdi_row = None if background_row is None else cdi_by_reference.get(normalize_text(background_row.get("_submission_id_norm", "")))
		row = _build_row(consent_row=consent_row, eligibility_row=eligibility_row, background_row=background_row, cdi_row=cdi_row, project_config=project_config)
		if row["participant_id"] not in seen_participants:
			seen_participants.add(str(row["participant_id"]))
			rows.append(row)

	for _, cdi_row in all_cdi_df.iterrows():
		background_row = background_by_submission.get(normalize_text(cdi_row.get("_reference_norm", "")))
		eligibility_row = None if background_row is None else eligibility_by_submission.get(normalize_text(background_row.get("_reference_norm", "")))
		consent_row = None if eligibility_row is None else consent_by_submission.get(normalize_text(eligibility_row.get("_reference_norm", "")))
		row = _build_row(consent_row=consent_row, eligibility_row=eligibility_row, background_row=background_row, cdi_row=cdi_row, project_config=project_config)
		if row["participant_id"] not in seen_participants:
			seen_participants.add(str(row["participant_id"]))
			rows.append(row)

	linkage = pd.DataFrame(rows, columns=PARTICIPANT_LINKAGE_COLUMNS)
	if linkage.empty:
		return pd.DataFrame(columns=PARTICIPANT_LINKAGE_COLUMNS)
	return linkage.sort_values(by=["included_analysis", "participant_id"], ascending=[False, True]).reset_index(drop=True)


def build_participant_tracking(forms: dict[str, pd.DataFrame] | None = None, *, config: ProjectConfig | None = None) -> pd.DataFrame:
	return build_participant_linkage(forms=forms, config=config)


def summarize_participant_linkage(linkage: pd.DataFrame) -> pd.DataFrame:
	metrics = {
		"participants_total": int(len(linkage)),
		"included_analysis": int(linkage["included_analysis"].fillna(False).sum()),
		"excluded_analysis": int((~linkage["included_analysis"].fillna(False)).sum()),
		"consent_confirmed": int((linkage["consent_status"] == "confirmed").sum()),
		"consent_missing_link": int((linkage["consent_status"] == "missing_link").sum()),
		"consent_not_given": int((linkage["consent_status"] == "not_given").sum()),
		"eligibility_eligible": int((linkage["eligibility_status"] == "eligible").sum()),
		"eligibility_unclear": int((linkage["eligibility_status"] == "unclear").sum()),
		"eligibility_not_eligible": int((linkage["eligibility_status"] == "not_eligible").sum()),
		"forwarded_form_mismatch": int(linkage["forwarded_form_mismatch"].fillna(False).sum()),
		"age_group_raw_vs_calculated_mismatch": int(linkage["age_group_raw_vs_calculated_mismatch"].fillna(False).sum()),
		"questionnaire_age_range_mismatch": int(linkage["questionnaire_age_range_mismatch"].fillna(False).sum()),
	}
	return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})


def summarize_form_links(tracking: pd.DataFrame) -> pd.DataFrame:
	return summarize_participant_linkage(tracking)


def write_data_linking_report(linkage: pd.DataFrame, summary: pd.DataFrame, path: str | Path) -> Path:
	report_path = Path(path)
	report_path.parent.mkdir(parents=True, exist_ok=True)
	exclusion_counts = linkage.loc[linkage["exclusion_reason"] != "", "exclusion_reason"].value_counts()
	lines = ["# Data Linking Report", "", "## Summary", ""]
	for _, row in summary.iterrows():
		lines.append(f"- {row['metric']}: {row['value']}")
	lines.extend(["", "## Exclusion Reasons", ""])
	if exclusion_counts.empty:
		lines.append("- None")
	else:
		for reason, count in exclusion_counts.items():
			lines.append(f"- {reason}: {count}")
	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return report_path


def write_form_link_report(tracking: pd.DataFrame, summary: pd.DataFrame, path: str | Path) -> Path:
	return write_data_linking_report(tracking, summary, path)