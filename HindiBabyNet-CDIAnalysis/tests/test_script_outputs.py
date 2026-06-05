from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(script_name: str):
	script_path = SCRIPTS_DIR / script_name
	spec = importlib.util.spec_from_file_location(f"test_{script_name.replace('.', '_')}", script_path)
	module = importlib.util.module_from_spec(spec)
	assert spec is not None and spec.loader is not None
	spec.loader.exec_module(module)
	return module


def _simple_dataframe(**columns) -> pd.DataFrame:
	return pd.DataFrame(columns)


def _scoring_outputs() -> dict[str, pd.DataFrame]:
	participant_linkage = _simple_dataframe(participant_id=["p1"], questionnaire=["8_18"], included_analysis=[True], exclusion_reason=[""], linkage_quality_flag=["ok"])
	participant_metadata = _simple_dataframe(
		participant_id=["p1"], questionnaire=["8_18"], included_analysis=[True], exclusion_reason=[""],
		age_days=[365], age_months=[12.0], age_bin=["11_13"], sex_raw=["female"], sex_code=[2],
		maternal_education_raw=["Bachelor"], maternal_education_code=[4], father_education_raw=["Master"], father_education_code=[5],
		ses_maternal_education_code=[4], age_quality_flag=["ok"], second_language_percent=[20.0], third_language_percent=[None],
	)
	combined_participant_scores = _simple_dataframe(
		participant_id=["p1"], questionnaire=["8_18"], age_months=[12.0], age_bin=["11_13"], sex_raw=["female"], sex_code=[2],
		maternal_education_code=[4], father_education_code=[5], ses_maternal_education_code=[4],
		production_total=[10], comprehension_total=[15], production_proportion=[0.5], comprehension_proportion=[0.75], comprehension_production_gap=[5], n_words_inventory=[20],
	)
	category_scores = _simple_dataframe(
		participant_id=["p1"], questionnaire=["8_18"], age_months=[12.0], age_bin=["11_13"], sex_raw=["female"], sex_code=[2],
		maternal_education_code=[4], father_education_code=[5], ses_maternal_education_code=[4], category=["ध्वनि"], category_english=["sounds"],
		n_words_category=[5], comprehension_score=[4], production_score=[3], comprehension_proportion=[0.8], production_proportion=[0.6],
	)
	word_level = _simple_dataframe(
		participant_id=["p1"], questionnaire=["8_18"], age_days=[365], age_months=[12.0], age_bin=["11_13"], sex_raw=["female"], sex_code=[2],
		maternal_education_raw=["Bachelor"], maternal_education_code=[4], father_education_raw=["Master"], father_education_code=[5], ses_maternal_education_code=[4],
		item_id=["item_0001"], word=["कू कू"], word_clean=["कू कू"], word_english=["coo"], category=["ध्वनि"], category_english=["sounds"], raw_column_name=["कू कू "],
		response_raw=["केवल समझता/समझती है"], score_code=[1], comprehension=[1], production=[0],
	)
	return {
		"cdi_8_18_scored_wide": _simple_dataframe(participant_id=["p1"], questionnaire=["8_18"], age_days=[365], age_months=[12.0], age_bin=["11_13"], sex_raw=["female"], sex_code=[2], maternal_education_raw=["Bachelor"], maternal_education_code=[4], father_education_raw=["Master"], father_education_code=[5], ses_maternal_education_code=[4], **{"कू कू ":[1]}),
		"cdi_8_18_scored_wide_safe_columns": _simple_dataframe(participant_id=["p1"], questionnaire=["8_18"], age_days=[365], age_months=[12.0], age_bin=["11_13"], sex_raw=["female"], sex_code=[2], maternal_education_raw=["Bachelor"], maternal_education_code=[4], father_education_raw=["Master"], father_education_code=[5], ses_maternal_education_code=[4], item_0001=[1]),
		"cdi_19_36_scored_wide": _simple_dataframe(participant_id=["p2"], questionnaire=["19_36"], age_days=[700], age_months=[23.0], age_bin=["19_24"], sex_raw=["male"], sex_code=[1], maternal_education_raw=["Bachelor"], maternal_education_code=[4], father_education_raw=["Master"], father_education_code=[5], ses_maternal_education_code=[4], **{"शीशा /काँच ":[1]}),
		"cdi_19_36_scored_wide_safe_columns": _simple_dataframe(participant_id=["p2"], questionnaire=["19_36"], age_days=[700], age_months=[23.0], age_bin=["19_24"], sex_raw=["male"], sex_code=[1], maternal_education_raw=["Bachelor"], maternal_education_code=[4], father_education_raw=["Master"], father_education_code=[5], ses_maternal_education_code=[4], item_0002=[1]),
		"cdi_8_18_word_level_long": word_level,
		"cdi_19_36_word_level_long": word_level.assign(questionnaire="19_36"),
		"cdi_combined_word_level_long": pd.concat([word_level, word_level.assign(questionnaire="19_36")], ignore_index=True),
		"cdi_8_18_participant_scores": combined_participant_scores,
		"cdi_19_36_participant_scores": combined_participant_scores.assign(questionnaire="19_36", comprehension_total=pd.NA, comprehension_proportion=pd.NA, comprehension_production_gap=pd.NA),
		"cdi_combined_participant_scores": pd.concat([combined_participant_scores, combined_participant_scores.assign(questionnaire="19_36", comprehension_total=pd.NA, comprehension_proportion=pd.NA, comprehension_production_gap=pd.NA)], ignore_index=True),
		"cdi_category_scores": category_scores,
		"wordbank_age_summary": _simple_dataframe(questionnaire=["8_18"], age_bin=["11_13"], n_children=[1], mean_age_months=[12.0], mean_production=[0.5], mean_comprehension=[0.75]),
		"wordbank_word_by_age": _simple_dataframe(questionnaire=["8_18"], age_bin=["11_13"], word=["कू कू"], word_clean=["कू कू"], word_english=["coo"], category=["ध्वनि"], category_english=["sounds"], n_children=[1], production_rate=[0.0], comprehension_rate=[1.0]),
		"wordbank_category_by_age": _simple_dataframe(questionnaire=["8_18"], age_bin=["11_13"], category=["ध्वनि"], category_english=["sounds"], n_children=[1], mean_production=[0.6], mean_comprehension=[0.8]),
		"wordbank_percentile_curves": _simple_dataframe(questionnaire=["8_18"], age_bin=["11_13"], p10_production=[1], p25_production=[2], p50_production=[3], p75_production=[4], p90_production=[5], p10_comprehension=[2], p25_comprehension=[3], p50_comprehension=[4], p75_comprehension=[5], p90_comprehension=[6]),
		"word_frequency_overall": _simple_dataframe(questionnaire=["8_18"], item_id=["item_0001"], word=["कू कू"], word_clean=["कू कू"], word_english=["coo"], category=["ध्वनि"], category_english=["sounds"], n_children=[1], production_rate=[0.0], comprehension_rate=[1.0]),
		"word_frequency_by_age": _simple_dataframe(questionnaire=["8_18"], age_bin=["11_13"], item_id=["item_0001"], word=["कू कू"], word_clean=["कू कू"], word_english=["coo"], category=["ध्वनि"], category_english=["sounds"], n_children=[1], production_rate=[0.0], comprehension_rate=[1.0]),
		"category_frequency_by_age": _simple_dataframe(questionnaire=["8_18"], age_bin=["11_13"], category=["ध्वनि"], category_english=["sounds"], n_children=[1], mean_production=[0.6], mean_comprehension=[0.8]),
		"shared_word_production_by_age": _simple_dataframe(age_bin=["11_13"], questionnaire=["8_18"], n_children=[1], n_shared_words=[1], mean_shared_word_production=[0.0], production_rate=[0.0]),
		"participant_linkage": participant_linkage,
		"participant_metadata": participant_metadata,
	}


def test_create_participant_metadata_script_writes_output(tmp_path: Path, monkeypatch) -> None:
	module = _load_script_module("create_participant_metadata.py")
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(module, "load_pipeline_forms", lambda: {"background": pd.DataFrame()})
	monkeypatch.setattr(module, "build_participant_linkage", lambda forms: pd.DataFrame({"participant_id": ["p1"]}))
	monkeypatch.setattr(module, "build_participant_metadata", lambda forms, linkage: pd.DataFrame({"participant_id": ["p1"], "questionnaire": ["8_18"]}))

	assert module.main() == 0
	assert (tmp_path / "data/processed/participant_metadata.csv").exists()


def test_score_scripts_write_expected_outputs(tmp_path: Path, monkeypatch) -> None:
	for script_name, expected_files in [
		("score_cdi_8_18.py", ["data/processed/cdi_8_18_scored_wide.csv", "data/processed/cdi_8_18_scored_wide_safe_columns.csv"]),
		("score_cdi_19_36.py", ["data/processed/cdi_19_36_scored_wide.csv", "data/processed/cdi_19_36_scored_wide_safe_columns.csv"]),
	]:
		module = _load_script_module(script_name)
		monkeypatch.chdir(tmp_path)
		monkeypatch.setattr(module, "build_scoring_outputs", _scoring_outputs)
		assert module.main() == 0
		for relative_path in expected_files:
			assert (tmp_path / relative_path).exists()


def test_long_and_aggregate_scripts_write_outputs(tmp_path: Path, monkeypatch) -> None:
	for script_name, expected_files in [
		("create_word_level_long.py", ["data/processed/cdi_8_18_word_level_long.csv", "data/processed/cdi_19_36_word_level_long.csv", "data/processed/cdi_combined_word_level_long.csv"]),
		("compute_participant_scores.py", ["data/processed/cdi_8_18_participant_scores.csv", "data/processed/cdi_19_36_participant_scores.csv", "data/processed/cdi_combined_participant_scores.csv"]),
		("compute_category_scores.py", ["data/processed/cdi_category_scores.csv"]),
		("create_wordbank_tables.py", [
			"outputs/tables/wordbank_age_summary.csv",
			"outputs/tables/wordbank_word_by_age.csv",
			"outputs/tables/wordbank_category_by_age.csv",
			"outputs/tables/wordbank_percentile_curves.csv",
			"outputs/tables/word_frequency_overall.csv",
			"outputs/tables/word_frequency_by_age.csv",
			"outputs/tables/category_frequency_by_age.csv",
			"outputs/tables/shared_word_production_by_age.csv",
		]),
	]:
		module = _load_script_module(script_name)
		monkeypatch.chdir(tmp_path)
		monkeypatch.setattr(module, "build_scoring_outputs", _scoring_outputs)
		assert module.main() == 0
		for relative_path in expected_files:
			assert (tmp_path / relative_path).exists()


def test_run_eda_script_writes_reports_and_figures(tmp_path: Path, monkeypatch) -> None:
	module = _load_script_module("run_eda.py")
	output_root = tmp_path / "outputs"
	config = SimpleNamespace(paths=SimpleNamespace(outputs=output_root))
	monkeypatch.setattr(module, "load_config", lambda: config)
	monkeypatch.setattr(module, "build_scoring_outputs", lambda config=None: _scoring_outputs())

	module.main()
	assert (output_root / "reports/eda_8_18_summary.md").exists()
	assert (output_root / "reports/eda_19_36_summary.md").exists()
	assert (output_root / "reports/eda_combined_summary.md").exists()
	assert (output_root / "figures/age_histogram_8_18.png").exists()
	assert (output_root / "figures/age_histogram_19_36.png").exists()


def test_run_all_executes_expected_script_sequence(monkeypatch) -> None:
	module = _load_script_module("run_all.py")
	called: list[str] = []
	monkeypatch.setattr(module, "_remove_legacy_outputs", lambda: None)
	monkeypatch.setattr(module, "_run_script", lambda script_name: called.append(script_name))

	assert module.main() == 0
	assert called == module.SCRIPT_SEQUENCE