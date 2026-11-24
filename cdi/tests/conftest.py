"""Helpers for the tests: a small synthetic study in a temporary folder.

No research data is used. Each questionnaire (consent, eligibility, background
and CDI form) is written from a short description, see ``case``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from hindibabynet_cdi.cleaning import (
    CDI8_COMPREHENSION_ONLY,
    CDI8_COMPREHENSION_PRODUCTION,
    CDI19_PRODUCTION,
    NO_VALUES,
    YES_VALUES,
)
from hindibabynet_cdi.columns import (
    ANSWER_TIME_MS,
    CHILD_SEX_COLUMN,
    CREATED_AT,
    FORWARDED_FORM_ID,
    REFERENCE_ID,
    SUBMISSION_ID,
    SUBMISSION_REFERENCE,
)
from hindibabynet_cdi.config import DEFAULT_PARAMS_PATH, ProjectConfig, load_config
from hindibabynet_cdi.eligibility import evaluate_participant_criteria
from hindibabynet_cdi.io import load_forms
from hindibabynet_cdi.linking import build_participant_linkage


# The settings of cdi/params.yaml (column names and form IDs).
PARAMS = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))

# The Hindi answers for yes and no, as the processing recognises them.
YES = sorted(value for value in YES_VALUES if not value.isascii())[0]
NO = sorted(value for value in NO_VALUES if not value.isascii())[0]
UNDERSTANDS = CDI8_COMPREHENSION_ONLY
UNDERSTANDS_AND_SAYS = CDI8_COMPREHENSION_PRODUCTION
SAYS = CDI19_PRODUCTION

# Item ID, Hindi word, English word, Hindi category, English category, questionnaires.
ITEMS = [
    ("item_1", "कुत्ता", "dog", "जानवर", "animals", "8_18 19_36"),
    ("item_2", "बिल्ली", "cat", "जानवर", "animals", "8_18 19_36"),
    ("item_3", "पानी", "water", "खाना और पीना", "food and drink", "8_18 19_36"),
    ("item_4", "दूध", "milk", "खाना और पीना", "food and drink", "8_18 19_36"),
    ("item_5", "टाटा", "bye", "खेल और दिनचर्या", "games and routines", "8_18"),
    ("item_6", "खाना", "eat", "क्रिया शब्द", "action words", "19_36"),
]
# The five items of each questionnaire, in the order of the export.
CDI1_WORDS = [item[1] for item in ITEMS if "8_18" in item[5]]
CDI2_WORDS = [item[1] for item in ITEMS if "19_36" in item[5]]

_ELIGIBILITY = PARAMS["eligibility"]
_EXPOSURE = PARAMS["exposure"]
_AGE = PARAMS["age"]
_EDUCATION = PARAMS["education"]

# Columns of the five exports.
FORM_COLUMNS = {
    "consent": [SUBMISSION_ID, CREATED_AT, _ELIGIBILITY["consent_column"], "fill_date"],
    "eligibility": [
        SUBMISSION_ID,
        CREATED_AT,
        REFERENCE_ID,
        _ELIGIBILITY["mother_tongue_column"],
        _ELIGIBILITY["preterm_column"],
        _ELIGIBILITY["impairment_column"],
        _ELIGIBILITY["age_range_column"],
    ],
    "background": [
        SUBMISSION_ID,
        CREATED_AT,
        SUBMISSION_REFERENCE,
        FORWARDED_FORM_ID,
        _AGE["birthdate_column"],
        _AGE["reported_age_column"],
        CHILD_SEX_COLUMN,
        _EDUCATION["mother_education_column"],
        _EDUCATION["father_education_column"],
        _EXPOSURE["second_language_column"],
        _EXPOSURE["second_language_percent_column"],
        _EXPOSURE["third_language_column"],
        _EXPOSURE["third_language_percent_column"],
    ],
    "cdi_8_18": [SUBMISSION_ID, CREATED_AT, SUBMISSION_REFERENCE, ANSWER_TIME_MS] + CDI1_WORDS,
    "cdi_19_36": [SUBMISSION_ID, CREATED_AT, SUBMISSION_REFERENCE, ANSWER_TIME_MS] + CDI2_WORDS,
}


def case(key: str, form: str = "cdi1", **changes) -> dict:
    """Description of one CDI questionnaire with its consent, eligibility and background form.

    Without changes the questionnaire meets every inclusion rule: a girl born
    on 2020-01-01, CDI-1 filled in at 12 months or CDI-2 at 24 months.
    The submission IDs are consent-<key>, eligibility-<key>, background-<key>
    and cdi-<key>.

    child: key of another case. The questionnaire then belongs to that child
    (same background form) and no new consent, eligibility and background form
    is written.
    cdi_reference, background_reference, eligibility_reference: reference ID
    written in the form, to break the chain.
    """
    first_form = form == "cdi1"
    values = {
        "key": key,
        "form": form,
        "child": None,
        "consent": YES,
        "mother_tongue": YES,
        "preterm": NO,
        "impairment": NO,
        "age_range": YES,
        "birthdate": "2020-01-01",
        "created": "2021-01-05 10:00:00" if first_form else "2022-01-05 10:00:00",
        "reported_age": "12" if first_form else "24",
        "gender": "Female",
        "mother_education": "bachelor",
        "father_education": "master",
        "second_language": "",
        "second_percent": "",
        "third_language": "",
        "third_percent": "",
        "answers": [UNDERSTANDS_AND_SAYS, UNDERSTANDS, "", UNDERSTANDS, ""] if first_form else [SAYS, "", SAYS, "", ""],
        "cdi_reference": None,
        "background_reference": None,
        "eligibility_reference": None,
    }
    unknown = set(changes) - set(values)
    assert not unknown, f"Unknown fields: {unknown}"
    values.update(changes)
    return values


def study_frames(cases: list[dict]) -> dict[str, pd.DataFrame]:
    """The five exports of a study as tables of text."""
    rows: dict[str, list[dict]] = {role: [] for role in FORM_COLUMNS}
    for entry in cases:
        key = entry["key"]
        role = "cdi_8_18" if entry["form"] == "cdi1" else "cdi_19_36"
        if entry["child"] is None:
            rows["consent"].append(
                {
                    SUBMISSION_ID: f"consent-{key}",
                    CREATED_AT: "2021-01-01 09:00:00",
                    _ELIGIBILITY["consent_column"]: entry["consent"],
                    "fill_date": "2021-01-01",
                }
            )
            rows["eligibility"].append(
                {
                    SUBMISSION_ID: f"eligibility-{key}",
                    CREATED_AT: "2021-01-01 09:05:00",
                    REFERENCE_ID: _reference(entry["eligibility_reference"], f"consent-{key}"),
                    _ELIGIBILITY["mother_tongue_column"]: entry["mother_tongue"],
                    _ELIGIBILITY["preterm_column"]: entry["preterm"],
                    _ELIGIBILITY["impairment_column"]: entry["impairment"],
                    _ELIGIBILITY["age_range_column"]: entry["age_range"],
                }
            )
            rows["background"].append(
                {
                    SUBMISSION_ID: f"background-{key}",
                    CREATED_AT: "2021-01-01 09:10:00",
                    SUBMISSION_REFERENCE: _reference(entry["background_reference"], f"eligibility-{key}"),
                    FORWARDED_FORM_ID: str(PARAMS["forms"][f"{role}_form_id"]),
                    _AGE["birthdate_column"]: entry["birthdate"],
                    _AGE["reported_age_column"]: entry["reported_age"],
                    CHILD_SEX_COLUMN: entry["gender"],
                    _EDUCATION["mother_education_column"]: entry["mother_education"],
                    _EDUCATION["father_education_column"]: entry["father_education"],
                    _EXPOSURE["second_language_column"]: entry["second_language"],
                    _EXPOSURE["second_language_percent_column"]: entry["second_percent"],
                    _EXPOSURE["third_language_column"]: entry["third_language"],
                    _EXPOSURE["third_language_percent_column"]: entry["third_percent"],
                }
            )
        words = CDI1_WORDS if entry["form"] == "cdi1" else CDI2_WORDS
        assert len(entry["answers"]) == len(words)
        rows[role].append(
            {
                SUBMISSION_ID: f"cdi-{key}",
                CREATED_AT: entry["created"],
                SUBMISSION_REFERENCE: _reference(entry["cdi_reference"], f"background-{entry['child'] or key}"),
                ANSWER_TIME_MS: "1000",
                **dict(zip(words, entry["answers"])),
            }
        )
    return {role: pd.DataFrame(rows[role], columns=FORM_COLUMNS[role]) for role in FORM_COLUMNS}


def _reference(changed: str | None, default: str) -> str:
    return default if changed is None else changed


def word_mapping() -> pd.DataFrame:
    """A small item list with the columns of word_mapping.csv."""
    rows = []
    for item_id, word, english, category, category_english, questionnaire in ITEMS:
        rows.append(
            {
                "item_id": item_id,
                "word": word,
                "word_clean": word,
                "word_english": english,
                "category": category,
                "category_english": category_english,
                "questionnaire": questionnaire,
                "raw_column_8_18": word if "8_18" in questionnaire else "",
                "raw_column_19_36": word if "19_36" in questionnaire else "",
                "is_duplicate_raw_column": "False",
                "notes": "",
                "in_8_18": str("8_18" in questionnaire),
                "in_19_36": str("19_36" in questionnaire),
            }
        )
    return pd.DataFrame(rows)


def export_name(role: str, suffix: str = ".xlsx") -> str:
    """File name of an export, with the form ID as in Nettskjema."""
    return f"data-{PARAMS['forms'][f'{role}_form_id']}-2026-01-01-1200{suffix}"


def write_export(frame: pd.DataFrame, path: Path) -> None:
    """Save one export as .xlsx or as .txt (separator ';')."""
    if path.suffix == ".xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def exclusion_reasons(config: ProjectConfig) -> dict[str, str]:
    """Primary reason for each CDI submission ID. Empty for an included questionnaire."""
    forms = load_forms(config)
    criteria = evaluate_participant_criteria(build_participant_linkage(config, forms), config)
    return dict(zip(criteria["cdi_submission_id"], criteria["primary_exclusion_reason"]))


@pytest.fixture
def make_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return a function that writes a synthetic study and loads its configuration.

    cases: the questionnaires, see ``case``.
    suffix: ".xlsx" or ".txt" exports.
    paths: folders or files stored in another place (the ``paths`` part of config.yaml).
    params: settings that differ from cdi/params.yaml, for example
        {"analysis": {"strict_hindi_threshold": 80}}.
    The config file is <tmp_path>/project/config.yaml, the data folder <tmp_path>/data.
    """
    monkeypatch.delenv("HBN_CONFIG", raising=False)

    def _make(
        cases: list[dict] | tuple = (),
        suffix: str = ".xlsx",
        paths: dict | None = None,
        params: dict | None = None,
    ) -> ProjectConfig:
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)
        data_dir = tmp_path / "data"
        paths = {key: Path(value) for key, value in (paths or {}).items()}

        raw_dir = paths.get("raw_cdi", data_dir / "raw" / "cdi")
        raw_dir.mkdir(parents=True, exist_ok=True)
        for role, frame in study_frames(list(cases)).items():
            write_export(frame, raw_dir / export_name(role, suffix))
        mapping_file = paths.get("word_mapping", raw_dir / "word_mapping.csv")
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        word_mapping().to_csv(mapping_file, index=False, encoding="utf-8")

        values: dict = {"data_dir": str(data_dir)}
        if paths:
            values["paths"] = {key: str(value) for key, value in paths.items()}
        config_file = project / "config.yaml"
        config_file.write_text(yaml.safe_dump(values), encoding="utf-8")

        params_file = None
        if params:
            settings = yaml.safe_load(DEFAULT_PARAMS_PATH.read_text(encoding="utf-8"))
            for section, changes in params.items():
                settings[section].update(changes)
            params_file = project / "params.yaml"
            params_file.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

        return load_config(config_file, params_file)

    return _make
