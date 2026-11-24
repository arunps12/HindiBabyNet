"""Settings and data folders of the CDI processing.

The settings are read from cdi/params.yaml. The data folders are read from
config.yaml in the project root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from hbn_paths import Config


# cdi/params.yaml
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[2] / "params.yaml"

# The five questionnaire forms, in the order in which a parent fills them in.
FORM_ROLES = ("consent", "eligibility", "background", "cdi_8_18", "cdi_19_36")


@dataclass(frozen=True)
class AnalysisSettings:
    age_month_divisor: float
    younger_questionnaire: str
    older_questionnaire: str
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
    cdi1_min_month: int
    cdi1_max_month: int
    cdi2_min_month: int
    cdi2_max_month: int


@dataclass(frozen=True)
class EducationSettings:
    mother_education_column: str
    father_education_column: str


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    params_path: Path
    raw_dir: Path
    word_mapping_path: Path
    output_dir: Path
    form_ids: dict[str, str]
    form_detection_rules: dict[str, tuple[str, ...]]
    analysis: AnalysisSettings
    eligibility: EligibilitySettings
    exposure: ExposureSettings
    age: AgeSettings
    education: EducationSettings


def load_config(
    config_path: str | Path | None = None,
    params_path: str | Path | None = None,
) -> ProjectConfig:
    """Read the settings from params.yaml and the folders from the root config.yaml."""
    params_file = Path(params_path) if params_path is not None else DEFAULT_PARAMS_PATH
    with params_file.open(encoding="utf-8-sig") as handle:
        params = yaml.safe_load(handle) or {}

    folders = Config(config_path)
    raw_dir = folders.input("raw_cdi")
    word_mapping_path = folders.input("word_mapping")

    try:
        forms = params["forms"]
        analysis = params["analysis"]
        eligibility = params["eligibility"]
        exposure = params["exposure"]
        age = params["age"]
        education = params["education"]
        rules = params["form_detection_rules"]
        return ProjectConfig(
            config_path=folders.file,
            params_path=params_file,
            raw_dir=raw_dir,
            word_mapping_path=word_mapping_path,
            output_dir=folders.results("cdi"),
            form_ids={role: str(forms[f"{role}_form_id"]) for role in FORM_ROLES},
            form_detection_rules={role: tuple(str(name) for name in rules[role]) for role in FORM_ROLES},
            analysis=AnalysisSettings(
                age_month_divisor=float(analysis["age_month_divisor"]),
                younger_questionnaire=str(analysis["younger_questionnaire"]),
                older_questionnaire=str(analysis["older_questionnaire"]),
                strict_hindi_threshold=float(analysis["strict_hindi_threshold"]),
                age_discrepancy_flag_months=float(analysis["age_discrepancy_flag_months"]),
            ),
            eligibility=EligibilitySettings(
                consent_column=str(eligibility["consent_column"]),
                mother_tongue_column=str(eligibility["mother_tongue_column"]),
                preterm_column=str(eligibility["preterm_column"]),
                impairment_column=str(eligibility["impairment_column"]),
                age_range_column=str(eligibility["age_range_column"]),
                required_mother_tongue_response=bool(eligibility["required_mother_tongue_response"]),
                allow_preterm=bool(eligibility["allow_preterm"]),
                allow_impairment=bool(eligibility["allow_impairment"]),
            ),
            exposure=ExposureSettings(
                second_language_column=str(exposure["second_language_column"]),
                second_language_percent_column=str(exposure["second_language_percent_column"]),
                third_language_column=str(exposure["third_language_column"]),
                third_language_percent_column=str(exposure["third_language_percent_column"]),
            ),
            age=AgeSettings(
                reported_age_column=str(age["reported_age_column"]),
                birthdate_column=str(age["birthdate_column"]),
                cdi1_min_month=int(age["cdi1_min_month"]),
                cdi1_max_month=int(age["cdi1_max_month"]),
                cdi2_min_month=int(age["cdi2_min_month"]),
                cdi2_max_month=int(age["cdi2_max_month"]),
            ),
            education=EducationSettings(
                mother_education_column=str(education["mother_education_column"]),
                father_education_column=str(education["father_education_column"]),
            ),
        )
    except KeyError as error:
        raise ValueError(f"The setting {error} is missing in {params_file}.") from None

