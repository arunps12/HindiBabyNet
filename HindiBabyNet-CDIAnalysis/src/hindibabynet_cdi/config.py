from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    raw_data: Path
    interim_data: Path
    processed_data: Path
    metadata: Path
    outputs: Path
    personal_info: Path


@dataclass(frozen=True)
class FormIds:
    consent: str
    eligibility: str
    background: str
    cdi_8_18: str
    cdi_19_36: str
    contact: str


@dataclass(frozen=True)
class AnalysisSettings:
    age_month_divisor: float
    younger_questionnaire: str
    older_questionnaire: str
    younger_age_bins: tuple[str, ...]
    older_age_bins: tuple[str, ...]
    strict_hindi_threshold: float
    age_discrepancy_flag_months: float


@dataclass(frozen=True)
class EligibilitySettings:
    consent_column: str
    mother_tongue_column: str
    preterm_column: str
    impairment_column: str
    age_range_column: str
    required_mother_tongue_response: bool
    allow_preterm: bool
    allow_impairment: bool


@dataclass(frozen=True)
class ExposureSettings:
    second_language_column: str
    second_language_percent_column: str
    third_language_column: str
    third_language_percent_column: str


@dataclass(frozen=True)
class AgeSettings:
    reported_age_column: str
    birthdate_column: str
    child_age_label_column: str
    completion_date_column: str
    cdi1_min_month: int
    cdi1_max_month: int
    cdi2_min_month: int
    cdi2_max_month: int


@dataclass(frozen=True)
class ContactSettings:
    mobile_number_column: str
    email_column: str


@dataclass(frozen=True)
class EducationSettings:
    mother_education_column: str
    father_education_column: str
    analysis_groups: tuple[str, ...]


@dataclass(frozen=True)
class CompletenessSettings:
    minimum_present_ratio: float
    maximum_trailing_blank_run: int


@dataclass(frozen=True)
class OutputSettings:
    qc_dir: Path
    cdi1_tables_dir: Path
    cdi2_tables_dir: Path
    combined_tables_dir: Path
    cdi1_comprehension_figures_dir: Path
    cdi1_production_figures_dir: Path
    cdi1_items_figures_dir: Path
    cdi1_categories_figures_dir: Path
    cdi2_production_figures_dir: Path
    cdi2_items_figures_dir: Path
    cdi2_categories_figures_dir: Path
    combined_figures_dir: Path
    qc_figures_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    paths: ProjectPaths
    form_ids: FormIds
    analysis: AnalysisSettings
    eligibility: EligibilitySettings
    exposure: ExposureSettings
    age: AgeSettings
    contact: ContactSettings
    education: EducationSettings
    completeness: CompletenessSettings
    outputs: OutputSettings
    form_detection_rules: dict[str, tuple[str, ...]]


DEFAULT_FORM_DETECTION_RULES: dict[str, tuple[str, ...]] = {
    "consent": (
        "मैं सहमति देता/देती हूँ कि मेरी गुमनाम जानकारी ऊपर वर्णित परियोजना में उपयोग की जाएगी।",
        "fill_date",
    ),
    "eligibility": (
        "क्या आपके बच्चे की मातृभाषा हिंदी है?",
        "क्या आपका बच्चा प्री-टर्म जन्मा है?",
        "Reference ID",
    ),
    "background": (
        "बच्चे की उम्र कितने महीनों की है?",
        "birthdate",
        "SUBMISSION_REFERENCE",
        "$forwarded_to_form",
    ),
    "cdi_8_18": (
        "केवल समझता/समझती है",
        "समझता/समझती है और कहता/कहती है",
        "इश/आह/उह (दर्द)",
    ),
    "cdi_19_36": (
        "कहता/कहती है",
        "इश/आह/उह (दर्द)",
        "क्रिंग क्रिंग (साइकिल का शोर घंटी)",
    ),
    "contact": (
        "<b>कृपया अपना व्हाट्सएप नंबर प्रदान करें, यदि आपके पास है, अन्यथा सामान्य फोन नंबर प्रदान करें।</b>",
        "<b>Email address (ईमेल)</b>",
        "Reference ID",
    ),
}


def _repo_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    return (repo_root / relative_path).resolve()


def _load_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at top level of {config_path}")
    return data


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    resolved_path = Path(config_path or "configs/config.yaml").resolve()
    raw_config = _load_yaml(resolved_path)
    repo_root = _repo_root_from_config(resolved_path)

    paths_config = raw_config.get("paths", {})
    paths = ProjectPaths(
        repo_root=repo_root,
        raw_data=_resolve_repo_path(repo_root, paths_config.get("raw_data", "data/raw")),
        interim_data=_resolve_repo_path(repo_root, paths_config.get("interim_data", "data/interim")),
        processed_data=_resolve_repo_path(repo_root, paths_config.get("processed_data", "data/processed")),
        metadata=_resolve_repo_path(repo_root, paths_config.get("metadata", "metadata")),
        outputs=_resolve_repo_path(repo_root, paths_config.get("outputs", "outputs")),
        personal_info=_resolve_repo_path(repo_root, paths_config.get("personal_info", "data/personal_info")),
    )

    forms_config = raw_config.get("forms", {})
    form_ids = FormIds(
        consent=str(forms_config.get("consent_form_id", "")),
        eligibility=str(forms_config.get("eligibility_form_id", "")),
        background=str(forms_config.get("background_form_id", "")),
        cdi_8_18=str(forms_config.get("cdi_8_18_form_id", "")),
        cdi_19_36=str(forms_config.get("cdi_19_36_form_id", "")),
        contact=str(forms_config.get("contact_form_id", "")),
    )

    analysis_config = raw_config.get("analysis", {})
    analysis = AnalysisSettings(
        age_month_divisor=float(analysis_config.get("age_month_divisor", 30.4375)),
        younger_questionnaire=str(analysis_config.get("younger_questionnaire", "8_18")),
        older_questionnaire=str(analysis_config.get("older_questionnaire", "19_36")),
        younger_age_bins=tuple(analysis_config.get("younger_age_bins", [])),
        older_age_bins=tuple(analysis_config.get("older_age_bins", [])),
        strict_hindi_threshold=float(analysis_config.get("strict_hindi_threshold", 75.0)),
        age_discrepancy_flag_months=float(analysis_config.get("age_discrepancy_flag_months", 2.0)),
    )

    eligibility_config = raw_config.get("eligibility", {})
    eligibility = EligibilitySettings(
        consent_column=str(
            eligibility_config.get(
                "consent_column",
                "मैं सहमति देता/देती हूँ कि मेरी गुमनाम जानकारी ऊपर वर्णित परियोजना में उपयोग की जाएगी।",
            )
        ),
        mother_tongue_column=str(
            eligibility_config.get("mother_tongue_column", "क्या आपके बच्चे की मातृभाषा हिंदी है?")
        ),
        preterm_column=str(eligibility_config.get("preterm_column", "क्या आपका बच्चा प्री-टर्म जन्मा है?")),
        impairment_column=str(
            eligibility_config.get(
                "impairment_column",
                "क्या आपके बच्चे को बोलने, सुनने या देखने से संबंधित कोई समस्या है?",
            )
        ),
        age_range_column=str(
            eligibility_config.get("age_range_column", "क्या आपके बच्चे की आयु 8 से 36 महीने के बीच है?")
        ),
        required_mother_tongue_response=bool(eligibility_config.get("required_mother_tongue_response", True)),
        allow_preterm=bool(eligibility_config.get("allow_preterm", False)),
        allow_impairment=bool(eligibility_config.get("allow_impairment", False)),
    )

    exposure_config = raw_config.get("exposure", {})
    exposure = ExposureSettings(
        second_language_column=str(
            exposure_config.get("second_language_column", "यदि लागू हो, तो आपके बच्चे की दूसरी भाषा क्या है?")
        ),
        second_language_percent_column=str(
            exposure_config.get(
                "second_language_percent_column",
                "आपका बच्चा कितने प्रतिशत समय दूसरी भाषा सुनता है?",
            )
        ),
        third_language_column=str(
            exposure_config.get("third_language_column", "यदि लागू हो, तो आपके बच्चे की तीसरी भाषा क्या है?")
        ),
        third_language_percent_column=str(
            exposure_config.get(
                "third_language_percent_column",
                "आपका बच्चा कितने प्रतिशत समय तीसरी भाषा सुनता है?",
            )
        ),
    )

    age_config = raw_config.get("age", {})
    age = AgeSettings(
        reported_age_column=str(age_config.get("reported_age_column", "बच्चे की उम्र कितने महीनों की है?")),
        birthdate_column=str(age_config.get("birthdate_column", "birthdate")),
        child_age_label_column=str(age_config.get("child_age_label_column", "बच्चे की आयु")),
        completion_date_column=str(age_config.get("completion_date_column", "$created")),
        cdi1_min_month=int(age_config.get("cdi1_min_month", 8)),
        cdi1_max_month=int(age_config.get("cdi1_max_month", 18)),
        cdi2_min_month=int(age_config.get("cdi2_min_month", 19)),
        cdi2_max_month=int(age_config.get("cdi2_max_month", 36)),
    )

    contact_config = raw_config.get("contact", {})
    contact = ContactSettings(
        mobile_number_column=str(
            contact_config.get(
                "mobile_number_column",
                "<b>कृपया अपना व्हाट्सएप नंबर प्रदान करें, यदि आपके पास है, अन्यथा सामान्य फोन नंबर प्रदान करें।</b>",
            )
        ),
        email_column=str(contact_config.get("email_column", "<b>Email address (ईमेल)</b>")),
    )

    education_config = raw_config.get("education", {})
    education = EducationSettings(
        mother_education_column=str(
            education_config.get("mother_education_column", "माता की वर्तमान शिक्षा स्तर:")
        ),
        father_education_column=str(
            education_config.get("father_education_column", "पिता की वर्तमान शिक्षा स्तर:")
        ),
        analysis_groups=tuple(
            education_config.get(
                "analysis_groups",
                ["Primary", "Secondary", "College", "Graduate", "Other/unknown"],
            )
        ),
    )

    completeness_config = raw_config.get("completeness", {})
    completeness = CompletenessSettings(
        minimum_present_ratio=float(completeness_config.get("minimum_present_ratio", 0.8)),
        maximum_trailing_blank_run=int(completeness_config.get("maximum_trailing_blank_run", 25)),
    )

    outputs_config = raw_config.get("outputs", {})
    outputs = OutputSettings(
        qc_dir=_resolve_repo_path(repo_root, outputs_config.get("qc_dir", "outputs/tables/qc")),
        cdi1_tables_dir=_resolve_repo_path(repo_root, outputs_config.get("cdi1_tables_dir", "outputs/tables/cdi1")),
        cdi2_tables_dir=_resolve_repo_path(repo_root, outputs_config.get("cdi2_tables_dir", "outputs/tables/cdi2")),
        combined_tables_dir=_resolve_repo_path(
            repo_root, outputs_config.get("combined_tables_dir", "outputs/tables/combined")
        ),
        cdi1_comprehension_figures_dir=_resolve_repo_path(
            repo_root,
            outputs_config.get("cdi1_comprehension_figures_dir", "outputs/figures/cdi1/comprehension"),
        ),
        cdi1_production_figures_dir=_resolve_repo_path(
            repo_root,
            outputs_config.get("cdi1_production_figures_dir", "outputs/figures/cdi1/production"),
        ),
        cdi1_items_figures_dir=_resolve_repo_path(
            repo_root, outputs_config.get("cdi1_items_figures_dir", "outputs/figures/cdi1/items")
        ),
        cdi1_categories_figures_dir=_resolve_repo_path(
            repo_root,
            outputs_config.get("cdi1_categories_figures_dir", "outputs/figures/cdi1/categories"),
        ),
        cdi2_production_figures_dir=_resolve_repo_path(
            repo_root,
            outputs_config.get("cdi2_production_figures_dir", "outputs/figures/cdi2/production"),
        ),
        cdi2_items_figures_dir=_resolve_repo_path(
            repo_root, outputs_config.get("cdi2_items_figures_dir", "outputs/figures/cdi2/items")
        ),
        cdi2_categories_figures_dir=_resolve_repo_path(
            repo_root,
            outputs_config.get("cdi2_categories_figures_dir", "outputs/figures/cdi2/categories"),
        ),
        combined_figures_dir=_resolve_repo_path(
            repo_root, outputs_config.get("combined_figures_dir", "outputs/figures/combined")
        ),
        qc_figures_dir=_resolve_repo_path(repo_root, outputs_config.get("qc_figures_dir", "outputs/figures/qc")),
    )

    configured_rules = raw_config.get("form_detection_rules", {})
    form_detection_rules = {
        key: tuple(value)
        for key, value in {**DEFAULT_FORM_DETECTION_RULES, **configured_rules}.items()
    }

    return ProjectConfig(
        paths=paths,
        form_ids=form_ids,
        analysis=analysis,
        eligibility=eligibility,
        exposure=exposure,
        age=age,
        contact=contact,
        education=education,
        completeness=completeness,
        outputs=outputs,
        form_detection_rules=form_detection_rules,
    )
