"""Tests of the CDI processing with a small synthetic study. No research data is used."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import (
    CDI1_WORDS,
    CDI2_WORDS,
    ITEMS,
    NO,
    SAYS,
    UNDERSTANDS,
    UNDERSTANDS_AND_SAYS,
    YES,
    case,
    exclusion_reasons,
    export_name,
)
from hindibabynet_cdi import io
from hindibabynet_cdi.cleaning import parse_completed_months
from hindibabynet_cdi.cli import OUTPUT_NAMES, main, process_cdi, write_outputs
from hindibabynet_cdi.columns import CDI1_FORM, CDI2_FORM, CREATED_AT, REFERENCE_ID, SUBMISSION_ID
from hindibabynet_cdi.io import load_forms
from hindibabynet_cdi.linking import build_participant_linkage
from hindibabynet_cdi.scoring import ITEM_COLUMNS, SUMMARY_COLUMNS


BLANK = ["", "", "", "", ""]


def by_submission(table: pd.DataFrame) -> pd.DataFrame:
    return table.set_index("submission_id")


# --- Completion rule: a linked questionnaire is complete -----------------------


def test_blank_answers_are_included_and_scored_zero(make_study):
    config = make_study(
        [
            case("few", answers=[UNDERSTANDS_AND_SAYS, "", "", "", ""]),
            case("none", answers=BLANK),
            case("none2", form="cdi2", answers=BLANK),
        ]
    )
    tables = process_cdi(config)

    totals = ["comprehension_total", "production_total", "n_items", "n_answered"]
    cdi1 = by_submission(tables["cdi1_summary"])
    assert sorted(cdi1.index) == ["cdi-few", "cdi-none"]
    assert cdi1.loc["cdi-few", totals].tolist() == [1, 1, 5, 1]
    assert cdi1.loc["cdi-none", totals].tolist() == [0, 0, 5, 0]
    cdi2 = by_submission(tables["cdi2_summary"])
    assert cdi2.loc["cdi-none2", totals[1:]].tolist() == [0, 5, 0]

    items = tables["cdi1_items"]
    few = items.loc[items["submission_id"].eq("cdi-few")]
    assert few["understands"].tolist() == [1, 0, 0, 0, 0]
    assert few["produces"].tolist() == [1, 0, 0, 0, 0]
    none = items.loc[items["submission_id"].eq("cdi-none")]
    assert none["response"].tolist() == BLANK
    assert none["understands"].tolist() == [0, 0, 0, 0, 0]
    assert none["produces"].tolist() == [0, 0, 0, 0, 0]
    assert tables["cdi2_items"]["produces"].tolist() == [0, 0, 0, 0, 0]


# --- Inclusion rules ------------------------------------------------------------

# Key of the case, form, changes to a questionnaire that meets every rule, and
# the primary reason why it is not included ("" = included).
# The child is born on 2020-01-01; "created" is the date of the CDI submission.
RULE_CASES = [
    ("included1", "cdi1", {}, ""),
    ("included2", "cdi2", {}, ""),
    ("unlinked", "cdi1", {"cdi_reference": ""}, "broken_reference_chain"),
    ("nobackground", "cdi2", {"cdi_reference": "background-missing"}, "broken_reference_chain"),
    ("noeligibility", "cdi1", {"background_reference": "eligibility-missing"}, "broken_reference_chain"),
    ("noconsentform", "cdi1", {"eligibility_reference": "consent-missing"}, "broken_reference_chain"),
    ("consentno", "cdi1", {"consent": NO}, "no_consent"),
    ("consentblank", "cdi2", {"consent": ""}, "no_consent"),
    ("nothindi", "cdi1", {"mother_tongue": NO}, "failed_eligibility"),
    ("mothertongueblank", "cdi1", {"mother_tongue": ""}, "failed_eligibility"),
    ("preterm", "cdi1", {"preterm": YES}, "failed_eligibility"),
    ("impairment", "cdi2", {"impairment": YES}, "failed_eligibility"),
    ("agerangeno", "cdi1", {"age_range": NO}, "failed_eligibility"),
    ("pretermblank", "cdi1", {"preterm": "", "impairment": ""}, ""),
    ("exposure75", "cdi1", {"second_language": "English", "second_percent": "25"}, "hindi_exposure_leq_75"),
    ("exposure76", "cdi1", {"second_language": "English", "second_percent": "24"}, ""),
    (
        "exposure70",
        "cdi2",
        {"second_language": "English", "second_percent": "20", "third_language": "Bengali", "third_percent": "10"},
        "hindi_exposure_leq_75",
    ),
    ("exposureunknown", "cdi1", {"second_language": "English"}, "hindi_exposure_unknown"),
    (
        "exposureinvalid",
        "cdi1",
        {"second_language": "English", "second_percent": "60", "third_language": "Bengali", "third_percent": "50"},
        "hindi_exposure_invalid",
    ),
    ("under8months", "cdi1", {"created": "2020-08-31 10:00:00"}, "age_outside_form_range"),
    ("just8months", "cdi1", {"created": "2020-09-01 10:00:00"}, ""),
    ("over36months", "cdi2", {"created": "2023-02-01 10:00:00"}, "age_outside_form_range"),
    ("just36months", "cdi2", {"created": "2023-01-31 10:00:00"}, ""),
    ("nobirthdate", "cdi1", {"birthdate": ""}, "age_outside_form_range"),
    ("under19months", "cdi1", {"created": "2021-08-01 10:00:00"}, ""),
    ("cdi1at19months", "cdi1", {"created": "2021-08-02 10:00:00"}, "age_form_mismatch"),
    ("cdi2at19months", "cdi2", {"created": "2021-08-02 10:00:00"}, ""),
    ("cdi2at12months", "cdi2", {"created": "2021-01-05 10:00:00"}, "age_form_mismatch"),
]


def test_inclusion_rules(make_study):
    config = make_study([case(key, form, **changes) for key, form, changes, _ in RULE_CASES])
    expected = {f"cdi-{key}": reason for key, _, _, reason in RULE_CASES}

    assert exclusion_reasons(config) == expected

    tables = process_cdi(config)
    included = {submission for submission, reason in expected.items() if not reason}
    in_summaries = set(tables["cdi1_summary"]["submission_id"]) | set(tables["cdi2_summary"]["submission_id"])
    in_items = set(tables["cdi1_items"]["submission_id"]) | set(tables["cdi2_items"]["submission_id"])
    assert in_summaries == included
    assert in_items == included


def test_hindi_exposure_limit_comes_from_the_settings(make_study):
    cases = [case("a", second_language="English", second_percent="24")]
    assert exclusion_reasons(make_study(cases)) == {"cdi-a": ""}
    stricter = make_study(cases, params={"analysis": {"strict_hindi_threshold": 80}})
    assert exclusion_reasons(stricter) == {"cdi-a": "hindi_exposure_leq_75"}


# --- Scores and columns ---------------------------------------------------------


def test_cdi1_scores_and_columns(make_study):
    answers = [UNDERSTANDS, UNDERSTANDS_AND_SAYS, "", UNDERSTANDS_AND_SAYS, UNDERSTANDS]
    tables = process_cdi(make_study([case("a", answers=answers)]))

    items = tables["cdi1_items"]
    assert list(items.columns) == [
        "child_id",
        "submission_id",
        "item_id",
        "item_type",
        "category",
        "category_hindi",
        "item",
        "item_english",
        "response",
        "understands",
        "produces",
    ]
    assert items["child_id"].tolist() == ["P000001"] * 5
    assert items["submission_id"].tolist() == ["cdi-a"] * 5
    assert items["item_id"].tolist() == [1, 2, 3, 4, 5]
    assert items["item_type"].tolist() == ["word"] * 5
    assert items["category"].tolist() == [item[4] for item in ITEMS[:5]]
    assert items["category_hindi"].tolist() == [item[3] for item in ITEMS[:5]]
    assert items["item"].tolist() == CDI1_WORDS
    assert items["item_english"].tolist() == ["dog", "cat", "water", "milk", "bye"]
    assert items["response"].tolist() == answers
    assert items["understands"].tolist() == [1, 1, 0, 1, 1]
    assert items["produces"].tolist() == [0, 1, 0, 1, 0]

    summary = tables["cdi1_summary"]
    assert list(summary.columns) == [
        "child_id",
        "submission_id",
        "completion_date",
        "birthdate",
        "age_months",
        "age_months_exact",
        "gender",
        "mother_education",
        "father_education",
        "hindi_exposure_percent",
        "comprehension_total",
        "production_total",
        "n_items",
        "n_answered",
        "age_discrepancy_flag",
        "n_submissions_for_child",
        "exact_duplicate_of",
    ]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["child_id"] == "P000001"
    assert row["submission_id"] == "cdi-a"
    assert row["completion_date"] == "2021-01-05"
    assert row["birthdate"] == "2020-01-01"
    assert row["age_months"] == 12
    assert row["age_months_exact"] == pytest.approx(370 / 30.4375)
    assert row["gender"] == "Female"
    assert row["mother_education"] == "bachelor"
    assert row["father_education"] == "master"
    assert row["hindi_exposure_percent"] == 100.0
    assert row["comprehension_total"] == 4
    assert row["production_total"] == 2
    assert row["n_items"] == 5
    assert row["n_answered"] == 4
    assert not row["age_discrepancy_flag"]
    assert row["n_submissions_for_child"] == 1
    assert row["exact_duplicate_of"] == ""


def test_cdi2_scores_have_no_comprehension(make_study):
    answers = [SAYS, "", SAYS, "", SAYS]
    tables = process_cdi(make_study([case("b", form="cdi2", answers=answers)]))

    items = tables["cdi2_items"]
    assert list(items.columns) == [
        "child_id",
        "submission_id",
        "item_id",
        "item_type",
        "category",
        "category_hindi",
        "item",
        "item_english",
        "response",
        "produces",
    ]
    assert items["item"].tolist() == CDI2_WORDS
    assert items["item_id"].tolist() == [1, 2, 3, 4, 6]
    assert items["response"].tolist() == answers
    assert items["produces"].tolist() == [1, 0, 1, 0, 1]

    summary = tables["cdi2_summary"]
    assert "comprehension_total" not in summary.columns
    assert list(summary.columns) == [
        column for column in SUMMARY_COLUMNS[CDI1_FORM] if column != "comprehension_total"
    ]
    row = summary.iloc[0]
    assert row["completion_date"] == "2022-01-05"
    assert row["age_months"] == 24
    assert row["production_total"] == 3
    assert row["n_items"] == 5
    assert row["n_answered"] == 3

    # No CDI-1 questionnaire: the CDI-1 tables are empty but have their columns.
    assert tables["cdi1_items"].empty
    assert list(tables["cdi1_items"].columns) == ITEM_COLUMNS[CDI1_FORM]
    assert tables["cdi1_summary"].empty
    assert list(tables["cdi1_summary"].columns) == SUMMARY_COLUMNS[CDI1_FORM]


def test_answer_that_is_not_recognised_has_no_score(make_study, capsys):
    tables = process_cdi(make_study([case("a", answers=["something else", UNDERSTANDS, "", "", ""])]))

    items = tables["cdi1_items"]
    assert items["response"].iloc[0] == "something else"
    assert pd.isna(items["understands"].iloc[0])
    assert pd.isna(items["produces"].iloc[0])
    row = tables["cdi1_summary"].iloc[0]
    assert row["comprehension_total"] == 1
    assert row["production_total"] == 0
    assert row["n_items"] == 5
    assert row["n_answered"] == 2
    assert "answers that were not recognised (no score): 1" in capsys.readouterr().out


def test_age_discrepancy_flag(make_study):
    tables = process_cdi(
        make_study([case("same"), case("differs", reported_age="20"), case("unknown", reported_age="")])
    )
    flags = by_submission(tables["cdi1_summary"])["age_discrepancy_flag"]
    assert len(flags) == 3
    assert not flags["cdi-same"]
    assert flags["cdi-differs"]
    assert pd.isna(flags["cdi-unknown"])


def test_missing_information_stays_empty(make_study):
    config = make_study([case("a", father_education=""), case("b", form="cdi2")])
    tables = process_cdi(config)

    assert pd.isna(tables["cdi1_summary"].loc[0, "father_education"])
    assert tables["cdi1_summary"].loc[0, "mother_education"] == "bachelor"
    assert tables["cdi2_summary"].loc[0, "father_education"] == "master"

    write_outputs(tables, config.output_dir)
    saved = pd.read_csv(
        config.output_dir / "cdi1_summary.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    assert saved.loc[0, "father_education"] == ""
    assert saved.loc[0, "exact_duplicate_of"] == ""


# --- Repeated submissions ---------------------------------------------------------


def test_repeated_submissions_are_all_kept(make_study):
    first = [UNDERSTANDS, "", "", "", ""]
    second = [UNDERSTANDS_AND_SAYS, UNDERSTANDS, "", "", ""]
    tables = process_cdi(
        make_study(
            [
                case("first", answers=first),
                case("second", child="first", created="2021-02-05 10:00:00", answers=second),
                case("later", form="cdi2", child="first"),
                case("other", answers=first),
            ]
        )
    )

    cdi1 = by_submission(tables["cdi1_summary"])
    assert sorted(cdi1.index) == ["cdi-first", "cdi-other", "cdi-second"]
    assert cdi1.loc["cdi-first", "child_id"] == cdi1.loc["cdi-second", "child_id"]
    assert cdi1.loc["cdi-first", "child_id"] != cdi1.loc["cdi-other", "child_id"]
    assert cdi1["n_submissions_for_child"].to_dict() == {"cdi-first": 2, "cdi-second": 2, "cdi-other": 1}
    assert cdi1["exact_duplicate_of"].eq("").all()
    assert cdi1.loc["cdi-first", "production_total"] == 0
    assert cdi1.loc["cdi-second", "production_total"] == 1
    assert len(tables["cdi1_items"]) == 15

    # The same child with a CDI-2: counted for CDI-2 only.
    cdi2 = by_submission(tables["cdi2_summary"])
    assert cdi2.loc["cdi-later", "child_id"] == cdi1.loc["cdi-first", "child_id"]
    assert cdi2.loc["cdi-later", "n_submissions_for_child"] == 1


def test_exact_duplicate_names_the_earliest_questionnaire(make_study):
    answers = [UNDERSTANDS, UNDERSTANDS_AND_SAYS, "", "", ""]
    tables = process_cdi(
        make_study(
            [
                # The latest questionnaire is the first row of the export.
                case("late", created="2021-03-05 10:00:00", answers=answers),
                case("early", child="late", created="2021-01-05 10:00:00", answers=answers),
                case("middle", child="late", created="2021-02-05 10:00:00", answers=[UNDERSTANDS, "", "", "", ""]),
                # Another child with the same answers is not a duplicate.
                case("other", answers=answers),
            ]
        )
    )

    cdi1 = by_submission(tables["cdi1_summary"])
    assert cdi1["exact_duplicate_of"].to_dict() == {
        "cdi-early": "",
        "cdi-middle": "",
        "cdi-late": "cdi-early",
        "cdi-other": "",
    }
    assert cdi1["n_submissions_for_child"].to_dict() == {
        "cdi-early": 3,
        "cdi-middle": 3,
        "cdi-late": 3,
        "cdi-other": 1,
    }


def test_item_rows_join_the_summary_exactly(make_study):
    tables = process_cdi(
        make_study(
            [
                case("a"),
                case("b", child="a", created="2021-02-05 10:00:00"),
                case("c"),
                case("d", form="cdi2"),
                case("e", form="cdi2"),
                case("x", consent=NO),
            ]
        )
    )
    for prefix, n_questionnaires in (("cdi1", 3), ("cdi2", 2)):
        items, summary = tables[f"{prefix}_items"], tables[f"{prefix}_summary"]
        summary_keys = list(zip(summary["child_id"], summary["submission_id"]))
        assert len(summary_keys) == len(set(summary_keys)) == n_questionnaires
        assert set(zip(items["child_id"], items["submission_id"])) == set(summary_keys)
        assert not items.duplicated(["submission_id", "item_id"]).any()
        assert items.groupby("submission_id").size().to_dict() == by_submission(summary)["n_items"].to_dict()


# --- Child IDs ----------------------------------------------------------------------


def test_child_ids_follow_the_sorted_background_ids(make_study):
    # Sorted by the background ID in the CDI form, then by the date of the CDI
    # submission. A CDI form that is not linked also takes a number.
    config = make_study(
        [
            case("c", form="cdi2"),
            case("b"),
            case("a2", child="a", created="2021-02-05 10:00:00"),
            case("lost", cdi_reference="background-aa"),
            case("a"),
            case("d", form="cdi2", cdi_reference=""),
        ]
    )

    linkage = build_participant_linkage(config, load_forms(config))
    assert dict(zip(linkage["cdi_submission_id"], linkage["participant_id"])) == {
        "cdi-a": "P000001",
        "cdi-a2": "P000001",
        "cdi-lost": "P000002",
        "cdi-b": "P000003",
        "cdi-c": "P000004",
        "cdi-d": "P000005",
    }

    tables = process_cdi(config)
    assert tables["cdi1_summary"]["submission_id"].tolist() == ["cdi-a", "cdi-a2", "cdi-b"]
    assert tables["cdi1_summary"]["child_id"].tolist() == ["P000001", "P000001", "P000003"]
    assert tables["cdi2_summary"]["submission_id"].tolist() == ["cdi-c"]
    assert tables["cdi2_summary"]["child_id"].tolist() == ["P000004"]


# --- Reading the exports --------------------------------------------------------------


def test_contact_form_is_not_read_and_no_contact_column_is_written(make_study, monkeypatch):
    config = make_study([case("a"), case("b", form="cdi2")])
    contact_file = config.raw_dir / "data-540031-2026-01-01-1200.xlsx"
    pd.DataFrame(
        {
            SUBMISSION_ID: ["contact-a"],
            CREATED_AT: ["2021-01-06 10:00:00"],
            REFERENCE_ID: ["cdi-a"],
            "phone number": ["0000000000"],
            "Email address": ["nobody@example.invalid"],
        }
    ).to_excel(contact_file, index=False)

    files_read = []
    read_raw_file = io.read_raw_file
    monkeypatch.setattr(io, "read_raw_file", lambda path: files_read.append(path) or read_raw_file(path))
    tables = process_cdi(config)

    # Every form is read once; the contact form is not opened.
    assert len(files_read) == 5
    assert contact_file not in files_read
    for table in tables.values():
        assert not table.empty
        for column in table.columns:
            assert not any(word in column.lower() for word in ("mobile", "email", "phone", "contact"))
        assert not table.astype(str).isin(["0000000000", "nobody@example.invalid"]).any().any()


def test_xlsx_export_is_used_before_txt(make_study):
    make_study([case("fromtxt")], suffix=".txt")
    config = make_study([case("fromxlsx")], suffix=".xlsx")

    forms = load_forms(config)
    assert all(form.path.suffix == ".xlsx" for form in forms.values())
    assert [path.name for path in forms["cdi_8_18"].not_used] == [export_name("cdi_8_18", ".txt")]
    assert process_cdi(config)["cdi1_summary"]["submission_id"].tolist() == ["cdi-fromxlsx"]


def test_form_without_form_id_in_the_file_name_is_found_from_its_columns(make_study):
    config = make_study([case("a")])
    (config.raw_dir / export_name("consent")).rename(config.raw_dir / "consent form.xlsx")

    assert load_forms(config)["consent"].path.name == "consent form.xlsx"
    assert len(process_cdi(config)["cdi1_summary"]) == 1


def test_missing_export_stops_with_a_clear_message(make_study):
    config = make_study([case("a")])
    (config.raw_dir / export_name("background")).unlink()

    with pytest.raises(FileNotFoundError, match="background"):
        load_forms(config)


def test_parse_completed_months_handles_digits_and_year_labels():
    assert parse_completed_months("९") == 9
    assert parse_completed_months("1 year") == 12
    assert parse_completed_months("12 months") == 12


# --- The command ----------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".xlsx", ".txt"])
def test_command_writes_the_four_files(make_study, capsys, suffix):
    config = make_study([case("a"), case("b", form="cdi2"), case("c", consent=NO)], suffix=suffix)

    assert main(["--config", str(config.config_path)]) == 0

    assert sorted(path.name for path in config.output_dir.iterdir()) == sorted(
        f"{name}.csv" for name in OUTPUT_NAMES
    )
    for name in OUTPUT_NAMES:
        # UTF-8 with BOM, so that Excel shows the Hindi text.
        assert (config.output_dir / f"{name}.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    saved = {
        name: pd.read_csv(config.output_dir / f"{name}.csv", encoding="utf-8-sig") for name in OUTPUT_NAMES
    }
    assert list(saved["cdi1_items"].columns) == ITEM_COLUMNS[CDI1_FORM]
    assert list(saved["cdi2_items"].columns) == ITEM_COLUMNS[CDI2_FORM]
    assert list(saved["cdi1_summary"].columns) == SUMMARY_COLUMNS[CDI1_FORM]
    assert list(saved["cdi2_summary"].columns) == SUMMARY_COLUMNS[CDI2_FORM]
    assert saved["cdi1_summary"]["submission_id"].tolist() == ["cdi-a"]
    assert saved["cdi1_summary"]["comprehension_total"].tolist() == [3]
    assert saved["cdi1_summary"]["production_total"].tolist() == [1]
    assert saved["cdi2_summary"]["production_total"].tolist() == [2]
    assert saved["cdi1_items"]["item"].tolist() == CDI1_WORDS

    printed = capsys.readouterr().out.splitlines()
    assert f"  CDI-1: {export_name('cdi_8_18', suffix)}" in printed
    cdi1_lines = printed[printed.index("CDI-1 (8 to 18 months)") : printed.index("CDI-2 (19 to 36 months)")]
    assert "  submissions found: 2" in cdi1_lines
    assert "  not included: 1" in cdi1_lines
    assert "    no consent: 1" in cdi1_lines
    assert "  included: 1" in cdi1_lines
    assert "  children with more than one included questionnaire: 0" in cdi1_lines
    assert "  exact duplicates of an earlier questionnaire: 0" in cdi1_lines
    assert "  cdi2_summary.csv" in printed


def test_existing_output_file_stops_the_command_before_processing(make_study, capsys):
    config = make_study([case("a"), case("b", form="cdi2")])
    config.output_dir.mkdir(parents=True)
    old_file = config.output_dir / "cdi2_summary.csv"
    old_file.write_text("old result", encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        main(["--config", str(config.config_path)])

    assert "already exists" in str(stopped.value)
    assert old_file.read_text(encoding="utf-8") == "old result"
    assert [path.name for path in config.output_dir.iterdir()] == ["cdi2_summary.csv"]
    assert "Reading" not in capsys.readouterr().out


def test_write_outputs_never_replaces_a_file(make_study):
    config = make_study([case("a")])
    tables = process_cdi(config)
    config.output_dir.mkdir(parents=True)
    old_file = config.output_dir / "cdi1_summary.csv"
    old_file.write_text("old result", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_outputs(tables, config.output_dir)

    assert old_file.read_text(encoding="utf-8") == "old result"
    assert [path.name for path in config.output_dir.iterdir()] == ["cdi1_summary.csv"]
