from pathlib import Path

from hindibabynet_cdi.io import (
    find_form_file,
    get_cdi_item_columns,
    get_reference_columns,
    read_form_export,
    validate_form_columns,
    validate_required_columns,
)
from hindibabynet_cdi.metadata import load_word_mapping, match_cdi_columns_to_metadata, summarize_word_mapping


def test_find_form_file_prefers_xlsx_over_txt(tmp_path: Path) -> None:
    txt_path = tmp_path / "data-539642-2026-06-04-1113.txt"
    xlsx_path = tmp_path / "data-539642-2026-06-04-1113.xlsx"
    txt_path.write_text('"$submission_id";"शब्द"\n"1";"हाँ"\n', encoding="utf-8")
    xlsx_path.write_text("placeholder", encoding="utf-8")

    selected = find_form_file(539642, raw_dir=tmp_path)

    assert selected == xlsx_path


def test_read_form_export_preserves_original_columns_and_detects_items(tmp_path: Path) -> None:
    export_path = tmp_path / "data-539642-2026-06-04-1113.txt"
    export_path.write_text(
        '"$submission_id";" SUBMISSION_REFERENCE ";"कू कू ";"$answer_time_ms"\n"1";"2";"केवल समझता/समझती है";"12"\n',
        encoding="utf-8",
    )

    dataframe = read_form_export(export_path)

    assert list(dataframe.columns) == ["$submission_id", " SUBMISSION_REFERENCE ", "कू कू ", "$answer_time_ms"]
    assert get_reference_columns(dataframe) == ["$submission_id", " SUBMISSION_REFERENCE ", "$answer_time_ms"]
    assert get_cdi_item_columns(dataframe) == ["कू कू "]


def test_load_word_mapping_handles_bom_and_questionnaire_splitting(tmp_path: Path) -> None:
    metadata_path = tmp_path / "word_mapping.csv"
    metadata_path.write_text(
        "\ufeffword,word_english,category,category_english,questionnaire\nकू कू,coo,sounds,sounds,8_18;19_36\n",
        encoding="utf-8",
    )

    mapping = load_word_mapping(metadata_path)
    summary = summarize_word_mapping(mapping)

    assert mapping.loc[0, "normalized_word"] == "कू कू"
    assert mapping.loc[0, "questionnaire_list"] == ["8_18", "19_36"]
    assert summary.shared_words == 1


def test_match_cdi_columns_to_metadata_normalizes_spacing() -> None:
    mapping = load_word_mapping(
        Path(__file__).parent / "fixtures_word_mapping.csv"
        if False
        else None
    )

    synthetic_mapping = load_word_mapping.__globals__["pd"].DataFrame(
        {
            "word": ["कू कू", "भों भों"],
            "word_english": ["coo", "woof"],
            "category": ["ध्वनि", "ध्वनि"],
            "category_english": ["sounds", "sounds"],
            "questionnaire": ["8_18;19_36", "8_18"],
            "normalized_word": ["कू कू", "भों भों"],
            "questionnaire_list": [["8_18", "19_36"], ["8_18"]],
        }
    )

    matches = match_cdi_columns_to_metadata(["कू कू ", "भों   भों", "अनजान शब्द"], synthetic_mapping, questionnaire="8_18")

    assert matches["matched"].tolist() == [True, True, False]
    assert matches.loc[0, "word_english"] == "coo"


def test_validate_required_columns_reports_exact_missing_names() -> None:
    synthetic = load_word_mapping.__globals__["pd"].DataFrame(
        {"$submission_id": ["1"], "$created": ["2026-06-04"], "Reference ID": ["abc"]}
    )

    missing = validate_required_columns(synthetic, ["$submission_id", "$created", "क्या आपके बच्चे की मातृभाषा हिंदी है?"])

    assert missing == ["क्या आपके बच्चे की मातृभाषा हिंदी है?"]


def test_validate_form_columns_uses_form_specific_contract() -> None:
    synthetic = load_word_mapping.__globals__["pd"].DataFrame(
        {
            "$submission_id": ["1"],
            "$created": ["2026-06-04"],
            "क्या आपके बच्चे की मातृभाषा हिंदी है?": ["हाँ"],
            "क्या आपका बच्चा प्री-टर्म जन्मा है?": ["नहीं"],
            "क्या आपके बच्चे को बोलने, सुनने या देखने से संबंधित कोई समस्या है?": ["नहीं"],
            "क्या आपके बच्चे की आयु 8 से 36 महीने के बीच है?": ["हाँ"],
            "Reference ID": ["abc"],
            "$answer_time_ms": ["1000"],
        }
    )

    assert validate_form_columns("eligibility", synthetic) == []