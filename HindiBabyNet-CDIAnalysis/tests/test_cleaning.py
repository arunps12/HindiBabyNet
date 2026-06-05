from datetime import datetime

from hindibabynet_cdi.cleaning import (
    SAYS_WORD,
    UNDERSTANDS_AND_SAYS,
    assess_age_quality,
    clean_html_entities,
    code_education,
    code_sex,
    compute_age_days,
    compute_age_months,
    create_participant_id,
    normalize_bool,
    normalize_column_name,
    normalize_response,
    parse_age_months,
    parse_datetime_safe,
    translate_language_value,
    translate_residence_value,
)
from hindibabynet_cdi.config import load_config


def test_load_config_resolves_repo_relative_paths() -> None:
    config = load_config()

    assert config.paths.raw_data.name == "raw"
    assert config.raw_files.cdi_8_18.endswith("539642-2026-06-04-1113.xlsx")
    assert config.forms.cdi_8_18_form_id == 539642
    assert config.analysis.older_questionnaire == "19_36"
    assert config.analysis.younger_age_bins == ("8_10", "11_13", "14_16", "17_18")


def test_normalize_column_name_collapses_unicode_whitespace() -> None:
    assert normalize_column_name("  बच्चे\u00a0का   लिंग  ") == "बच्चे का लिंग"


def test_normalize_response_cleans_spacing_without_changing_label() -> None:
    assert normalize_response("  समझता/समझती   है और   कहता/कहती है ") == UNDERSTANDS_AND_SAYS
    assert normalize_response("  कहता/कहती\u00a0है ") == SAYS_WORD


def test_clean_html_entities_decodes_expected_values() -> None:
    assert clean_html_entities("हिंदी &#43; अन्य") == "हिंदी + अन्य"
    assert clean_html_entities("यदि आपने &#39;अन्य&#39; चुना") == "यदि आपने 'अन्य' चुना"


def test_normalize_bool_handles_hindi_and_english_values() -> None:
    assert normalize_bool(" हाँ ") is True
    assert normalize_bool("No") is False
    assert normalize_bool("") is None


def test_parse_age_months_handles_digits_and_ranges() -> None:
    assert parse_age_months("९") == 9.0
    assert parse_age_months("12") == 12.0
    assert parse_age_months("8 से 18 महीने के बीच") == 13.0


def test_parse_age_months_handles_year_and_month_variants() -> None:
    assert parse_age_months("1 year") == 12.0
    assert parse_age_months("3 saal") == 36.0
    assert parse_age_months("25months") == 25.0
    assert parse_age_months("2 साल 11 महिने") == 35.0


def test_parse_datetime_safe_handles_hindi_digits() -> None:
    parsed = parse_datetime_safe("०४/०६/२०२६ १०:५३")

    assert parsed == datetime(2026, 6, 4, 10, 53)


def test_compute_age_months_returns_expected_value() -> None:
    birthdate = datetime(2025, 6, 4)
    created = datetime(2026, 6, 4)

    assert compute_age_months(birthdate, created) == 11.99


def test_compute_age_days_and_quality_flags() -> None:
    birthdate = datetime(2025, 6, 10)
    created = datetime(2026, 6, 4)

    assert compute_age_days(birthdate, created) == 359
    quality = assess_age_quality(birthdate=birthdate, reference_datetime=created, raw_age_months=9.0)

    assert quality["age_days"] == 359
    assert quality["age_discrepancy_flag"] is True
    assert quality["age_quality_flag"] == "raw_age_mismatch"


def test_code_education_and_sex_cover_expected_categories() -> None:
    assert code_education("Bachelor degree") == 4
    assert code_education("परास्नातक") == 5
    assert code_education("") == 99
    assert code_sex("male") == 1
    assert code_sex("female") == 2
    assert code_sex("unknown") == 99


def test_translate_language_and_residence_values() -> None:
    assert translate_language_value("हिंदी &#43; अन्य") == "Hindi + other"
    assert translate_residence_value(" कस्बा") == "Town"


def test_create_participant_id_is_stable_for_normalized_values() -> None:
    first = create_participant_id("  539547 ", " बच्चा ")
    second = create_participant_id("539547", "बच्चा")

    assert first == second
    assert first.startswith("participant_")