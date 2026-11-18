"""Synthetic reuse regressions: no research inputs, outputs, or model fitting."""

from dataclasses import fields
from itertools import product
from unittest.mock import Mock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from hindibabynet_cdi import config as configuration
from hindibabynet_cdi import eligibility, io, item_dictionary, linking, pipeline, scoring
from hindibabynet_cdi.cleaning import (
    CDI8_COMPREHENSION_ONLY,
    CDI8_COMPREHENSION_PRODUCTION,
    CDI19_PRODUCTION,
)
from hindibabynet_cdi.columns import (
    CHILD_SEX_COLUMN,
    CONTACT_EMAIL_COLUMN,
    CONTACT_MOBILE_COLUMN,
    CREATED_AT,
    REFERENCE_ID,
    SUBMISSION_ID,
    SUBMISSION_REFERENCE,
)
from hindibabynet_cdi.reporting import ReportingOutputs


@pytest.fixture
def synthetic_inputs(monkeypatch, tmp_path):
    # Use the loader's existing defaults, but never open the project config.
    monkeypatch.setattr(configuration, "_load_yaml", Mock(return_value={
        "forms": {
            "consent_form_id": "101", "eligibility_form_id": "102",
            "background_form_id": "103", "cdi_8_18_form_id": "104",
            "cdi_19_36_form_id": "105", "contact_form_id": "106",
        },
        "completeness": {"maximum_trailing_blank_run": 0},
    }))
    config = configuration.load_config(tmp_path / "configs" / "synthetic.yaml")
    ids = list(range(5))
    frames = {
        "consent": pd.DataFrame({
            SUBMISSION_ID: [f"consent-{i}" for i in ids],
            CREATED_AT: "2021-01-01",
            config.eligibility.consent_column: "yes",
        }),
        "eligibility": pd.DataFrame({
            SUBMISSION_ID: [f"eligibility-{i}" for i in ids],
            REFERENCE_ID: [f"consent-{i}" for i in ids],
            CREATED_AT: "2021-01-01",
            config.eligibility.mother_tongue_column: "yes",
            config.eligibility.preterm_column: "no",
            config.eligibility.impairment_column: "no",
            config.eligibility.age_range_column: "yes",
        }),
        "background": pd.DataFrame({
            SUBMISSION_ID: [f"background-{i}" for i in ids],
            SUBMISSION_REFERENCE: [f"eligibility-{i}" for i in ids],
            CREATED_AT: "2021-01-01",
            config.age.birthdate_column: "2020-01-01",
            config.age.reported_age_column: [12, 12, 12, 24, 24],
            config.age.child_age_label_column: [12, 12, 12, 24, 24],
            CHILD_SEX_COLUMN: ["Female", "Male", "Female", "Male", "Female"],
            config.education.mother_education_column: "bachelor",
            config.education.father_education_column: "master",
            config.exposure.second_language_column: ["", "English", "", "", ""],
            config.exposure.second_language_percent_column: ["", "25", "", "", ""],
            config.exposure.third_language_column: "",
            config.exposure.third_language_percent_column: "",
        }),
        "cdi_8_18": pd.DataFrame({
            SUBMISSION_ID: [f"cdi-{i}" for i in ids[:3]],
            SUBMISSION_REFERENCE: [f"background-{i}" for i in ids[:3]],
            CREATED_AT: "2021-01-05",
            "नाशपाती": [CDI8_COMPREHENSION_ONLY, CDI8_COMPREHENSION_PRODUCTION, CDI8_COMPREHENSION_ONLY],
            "सेब": [CDI8_COMPREHENSION_PRODUCTION, CDI8_COMPREHENSION_ONLY, ""],
        }),
        "cdi_19_36": pd.DataFrame({
            SUBMISSION_ID: [f"cdi-{i}" for i in ids[3:]],
            SUBMISSION_REFERENCE: [f"background-{i}" for i in ids[3:]],
            CREATED_AT: "2022-01-05",
            "नाशपाती": [CDI19_PRODUCTION, "synthetic unknown"],
            "सेब": [CDI19_PRODUCTION, CDI19_PRODUCTION],
        }),
        "contact": pd.DataFrame({
            SUBMISSION_ID: [f"contact-{i}" for i in ids] + ["contact-extra"],
            REFERENCE_ID: [f"cdi-{i}" for i in ids] + ["cdi-0"],
            CREATED_AT: "2022-01-06",
            CONTACT_MOBILE_COLUMN: "000 000",
            CONTACT_EMAIL_COLUMN: "SYNTHETIC@example.invalid",
        }),
    }
    records = []
    raw_frames = {}
    for role, frame in frames.items():
        # Deliberately discover txt before xlsx, and z before a for role ties.
        stems = ["z", "a"] if role == "cdi_8_18" else ["synthetic"]
        for stem in stems:
            for suffix in (".txt", ".xlsx"):
                path = tmp_path / f"{stem}-{getattr(config.form_ids, role)}{suffix}"
                records.append(io.RawFileRecord(path, "raw"))
                raw_frames[path] = frame.copy(deep=True)
    discovery = Mock(return_value=records)
    reader = Mock(side_effect=lambda path: raw_frames[path].copy(deep=True))
    monkeypatch.setattr(io, "discover_raw_files", discovery)
    monkeypatch.setattr(io, "read_raw_file", reader)
    mapping = pd.DataFrame({
        "item_id": ["2", "1"], "word": ["सेब", "नाशपाती"],
        "word_english": ["apple", "pear"], "category": "फल",
        "category_english": "fruit", "questionnaire": "8_18 19_36",
        "raw_column_8_18": ["सेब", "नाशपाती"],
        "raw_column_19_36": ["सेब", "नाशपाती"], "notes": "",
    })
    mapping_reader = Mock(side_effect=lambda path, dtype: (
        mapping.copy(deep=True)
        if path == config.paths.metadata / "word_mapping.csv" and dtype is str
        else pytest.fail("Unexpected table read outside the synthetic item mapping")
    ))
    monkeypatch.setattr(item_dictionary.pd, "read_csv", mapping_reader)
    return config, discovery, reader, mapping_reader


@pytest.mark.parametrize("module,builder", [
    (linking, linking.build_participant_linkage),
    (item_dictionary, item_dictionary.build_master_item_dictionary),
])
def test_form_reuse_matches_standalone_and_does_not_mutate(monkeypatch, synthetic_inputs, module, builder):
    config, _, _, _ = synthetic_inputs
    forms = io.load_detected_forms(config)
    snapshots = [form.data.copy(deep=True) for form in forms]
    assert [form.role for form in forms] == list(io.ROLE_ORDER)
    assert all(form.path.suffix == ".xlsx" for form in forms)
    assert next(form.path.name for form in forms if form.role == "cdi_8_18") == "a-104.xlsx"
    loader = Mock(wraps=io.load_detected_forms)
    monkeypatch.setattr(module, "load_detected_forms", loader)
    standalone = builder(config)
    reused = builder(config, forms=forms)
    loader.assert_called_once_with(config)
    for field in fields(standalone):
        assert_frame_equal(getattr(standalone, field.name), getattr(reused, field.name), check_exact=True)
    for form, snapshot in zip(forms, snapshots):
        assert_frame_equal(form.data, snapshot, check_exact=True)
    # An explicitly empty snapshot is not a request to rediscover inputs.
    with pytest.raises(KeyError):
        builder(config, forms=[])
    assert loader.call_count == 1


def test_eligibility_reuses_linkage_or_supplied_forms(monkeypatch, synthetic_inputs):
    config, _, _, _ = synthetic_inputs
    forms = io.load_detected_forms(config)
    linkage = linking.build_participant_linkage(config, forms=forms)
    snapshot = linkage.participant_linkage.copy(deep=True)
    builder = Mock(wraps=linking.build_participant_linkage)
    monkeypatch.setattr(eligibility, "build_participant_linkage", builder)
    standalone = eligibility.build_eligibility_outputs(config)
    reused = eligibility.build_eligibility_outputs(config, linkage_outputs=linkage, forms=[])
    assert builder.call_count == 1
    from_forms = eligibility.build_eligibility_outputs(config, forms=forms)
    assert builder.call_count == 2
    assert builder.call_args.kwargs["forms"] is forms
    for field in fields(standalone):
        expected = getattr(standalone, field.name)
        assert_frame_equal(expected, getattr(reused, field.name), check_exact=True)
        assert_frame_equal(expected, getattr(from_forms, field.name), check_exact=True)
    assert_frame_equal(linkage.participant_linkage, snapshot, check_exact=True)
    criteria = reused.participant_criteria
    assert not criteria.loc[criteria["hindi_percentage"].eq(75), "included_final"].any()


@pytest.mark.parametrize("reuse_eligibility,reuse_dictionary,reuse_forms", list(product((False, True), repeat=3)))
def test_scoring_partial_reuse_matches_every_standalone_output(
    monkeypatch, synthetic_inputs, reuse_eligibility, reuse_dictionary, reuse_forms,
):
    config, _, _, mapping_reader = synthetic_inputs
    forms = io.load_detected_forms(config)
    criteria = eligibility.build_eligibility_outputs(config, forms=forms)
    dictionary = item_dictionary.build_master_item_dictionary(config, forms=forms)
    snapshots = [criteria.participant_criteria.copy(deep=True), dictionary.master_dictionary.copy(deep=True)]
    standalone = scoring.build_scoring_outputs(config)
    eligibility_builder = Mock(wraps=eligibility.build_eligibility_outputs)
    dictionary_builder = Mock(wraps=item_dictionary.build_master_item_dictionary)
    linkage_builder = Mock(wraps=linking.build_participant_linkage)
    loaders = []
    for module in (linking, item_dictionary, scoring):
        loader = Mock(wraps=io.load_detected_forms)
        monkeypatch.setattr(module, "load_detected_forms", loader)
        loaders.append(loader)
    monkeypatch.setattr(scoring, "build_eligibility_outputs", eligibility_builder)
    monkeypatch.setattr(scoring, "build_master_item_dictionary", dictionary_builder)
    monkeypatch.setattr(eligibility, "build_participant_linkage", linkage_builder)
    mapping_reader.reset_mock()
    reused = scoring.build_scoring_outputs(
        config,
        eligibility_outputs=criteria if reuse_eligibility else None,
        dictionary_outputs=dictionary if reuse_dictionary else None,
        forms=forms if reuse_forms else None,
    )
    assert eligibility_builder.call_count == int(not reuse_eligibility)
    assert linkage_builder.call_count == int(not reuse_eligibility)
    assert dictionary_builder.call_count == int(not reuse_dictionary)
    assert mapping_reader.call_count == int(not reuse_dictionary)
    assert [loader.call_count for loader in loaders] == [
        int(not reuse_eligibility and not reuse_forms),
        int(not reuse_dictionary and not reuse_forms),
        int(not reuse_forms),
    ]
    for field in fields(standalone):
        assert_frame_equal(getattr(standalone, field.name), getattr(reused, field.name), check_exact=True)
    assert_frame_equal(criteria.participant_criteria, snapshots[0], check_exact=True)
    assert_frame_equal(dictionary.master_dictionary, snapshots[1], check_exact=True)
    assert reused.participant_analysis_cdi2["comprehension_total"].isna().all()
    assert reused.cdi2_items_long["understand"].isna().all()
    assert len(reused.unknown_responses) == 1
    assert "trailing_blank_block" in set(reused.completeness_report["completion_status"])
    assert not {"mobile_number", "email"} & set(reused.participant_analysis_all.columns)


def test_pipeline_computes_once_and_exports_equal_frames(monkeypatch, synthetic_inputs):
    config, discovery, reader, mapping_reader = synthetic_inputs
    # Independent standalone calls reproduce the pre-refactor computation path.
    expected_linkage = linking.build_participant_linkage(config)
    expected_eligibility = eligibility.build_eligibility_outputs(config)
    expected_dictionary = item_dictionary.build_master_item_dictionary(config)
    expected_scoring = scoring.build_scoring_outputs(config)
    expected = {
        "word_mapping_master": expected_dictionary.master_dictionary,
        "participant_linkage": expected_linkage.participant_linkage,
        "participant_crosswalk": expected_linkage.participant_crosswalk,
        "item_mapping_validation": expected_dictionary.validation_report,
        "contact_warnings": expected_linkage.contact_warnings,
        "merge_validation": expected_linkage.merge_validation,
        "participant_flow": expected_eligibility.participant_flow,
        "exclusions": expected_eligibility.exclusions,
        "cdi_all_items_long": pd.concat([
            expected_scoring.cdi1_items_long, expected_scoring.cdi2_items_long,
        ], ignore_index=True, sort=False),
        "sample_characteristics": pipeline.build_sample_characteristics(expected_scoring.participant_analysis_all),
        "age_counts": pipeline.build_age_counts(expected_scoring.participant_analysis_all),
    }
    expected.update({field.name: getattr(expected_scoring, field.name) for field in fields(expected_scoring)})
    qc_frame = pd.DataFrame({"synthetic_qc": [1]})
    report = ReportingOutputs(
        tables={name: pd.DataFrame({"synthetic_table": [i]}) for i, name in enumerate([
            "cdi1_synthetic", "cdi2_synthetic", "combined_synthetic",
        ])},
        model_metadata=pd.DataFrame({"synthetic_model": ["not fitted"]}),
        figure_paths={"synthetic_figure": config.paths.outputs / "synthetic.png"},
    )
    expected.update(report.tables)
    expected.update(model_metadata=report.model_metadata, form_detection=qc_frame, file_variant_comparison=qc_frame)
    reporter = Mock(return_value=report)
    monkeypatch.setattr(pipeline, "build_reporting_outputs", reporter)
    monkeypatch.setattr(io, "build_form_detection_report", Mock(return_value=qc_frame))
    monkeypatch.setattr(io, "compare_raw_file_variants", Mock(return_value=qc_frame))
    directories = Mock()
    writes = Mock()
    monkeypatch.setattr(pipeline, "_ensure_directories", directories)
    monkeypatch.setattr(pipeline, "_write_csv", writes)
    loader = Mock(wraps=io.load_detected_forms)
    for module in (pipeline, linking, item_dictionary, scoring):
        monkeypatch.setattr(module, "load_detected_forms", loader)
    builders = {}
    for name, function, modules in [
        ("linkage", linking.build_participant_linkage, (pipeline, eligibility)),
        ("eligibility", eligibility.build_eligibility_outputs, (pipeline, scoring)),
        ("dictionary", item_dictionary.build_master_item_dictionary, (pipeline, scoring)),
        ("scoring", scoring.build_scoring_outputs, (pipeline,)),
    ]:
        spy = Mock(wraps=function)
        builders[name] = spy
        for module in modules:
            monkeypatch.setattr(module, function.__name__, spy)
    discovery.reset_mock()
    reader.reset_mock()
    mapping_reader.reset_mock()
    paths = pipeline.run_pipeline(config)
    loader.assert_called_once_with(config)
    discovery.assert_called_once_with(config)
    assert reader.call_count == len(discovery.return_value)
    assert mapping_reader.call_count == 1
    assert all(builder.call_count == 1 for builder in builders.values())
    shared_forms = builders["linkage"].call_args.kwargs["forms"]
    assert builders["dictionary"].call_args.kwargs["forms"] is shared_forms
    assert builders["scoring"].call_args.kwargs["forms"] is shared_forms
    reporter.assert_called_once()
    assert reporter.call_args.args[0] is config
    for field in fields(expected_scoring):
        assert_frame_equal(
            getattr(reporter.call_args.args[1], field.name), getattr(expected_scoring, field.name), check_exact=True,
        )
    exported = {call.args[1]: call.args[0] for call in writes.call_args_list}
    assert len(exported) == writes.call_count == len(expected)
    assert set(paths) == set(expected) | set(report.figure_paths)
    for name, frame in expected.items():
        assert_frame_equal(exported[paths[name]], frame, check_exact=True)
    assert paths["cdi1_synthetic"].parent == config.outputs.cdi1_tables_dir
    assert paths["cdi2_synthetic"].parent == config.outputs.cdi2_tables_dir
    assert paths["combined_synthetic"].parent == config.outputs.combined_tables_dir
    assert paths["synthetic_figure"] == report.figure_paths["synthetic_figure"]
    directories.assert_called_once()
    assert not config.paths.processed_data.exists()