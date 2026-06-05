import pandas as pd

from hindibabynet_cdi.eda_tables import build_eda_markdown_report, build_eda_tables


def test_build_eda_tables_and_markdown_report() -> None:
	participant_linkage = pd.DataFrame(
		[
			{
				"participant_id": "p1",
				"questionnaire": "8_18",
				"included_analysis": True,
				"exclusion_reason": "",
				"linkage_quality_flag": "ok",
			},
			{
				"participant_id": "p2",
				"questionnaire": "8_18",
				"included_analysis": False,
				"exclusion_reason": "age_unusable",
				"linkage_quality_flag": "questionnaire_age_range_mismatch",
			},
		]
	)
	participant_metadata = pd.DataFrame(
		[
			{
				"participant_id": "p1",
				"questionnaire": "8_18",
				"included_analysis": True,
				"age_months": 12.5,
				"age_bin": "11_13",
				"sex_raw": "female",
				"maternal_education_raw": "Bachelor",
				"father_education_raw": "Master",
				"second_language_percent": 20.0,
				"third_language_percent": None,
				"age_quality_flag": "ok",
			},
			{
				"participant_id": "p2",
				"questionnaire": "8_18",
				"included_analysis": False,
				"age_months": None,
				"age_bin": "",
				"sex_raw": "male",
				"maternal_education_raw": "Bachelor",
				"father_education_raw": "Bachelor",
				"second_language_percent": None,
				"third_language_percent": None,
				"age_quality_flag": "age_unusable",
			},
		]
	)
	participant_scores = pd.DataFrame(
		[
			{
				"participant_id": "p1",
				"questionnaire": "8_18",
				"age_months": 12.5,
				"age_bin": "11_13",
				"production_total": 10,
				"comprehension_total": 15,
				"production_proportion": 0.5,
				"comprehension_proportion": 0.75,
			},
		]
	)
	category_scores = pd.DataFrame(
		[
			{
				"participant_id": "p1",
				"questionnaire": "8_18",
				"category": "ध्वनि",
				"category_english": "sounds",
				"production_proportion": 0.5,
				"comprehension_proportion": 0.75,
			}
		]
	)
	word_level_long = pd.DataFrame(
		[
			{
				"participant_id": "p1",
				"questionnaire": "8_18",
				"word": "कू कू",
				"word_english": "coo",
				"category": "ध्वनि",
				"category_english": "sounds",
				"production": 1,
				"comprehension": 1,
			}
		]
	)

	tables = build_eda_tables(
		participant_linkage=participant_linkage,
		participant_metadata=participant_metadata,
		participant_scores=participant_scores,
		category_scores=category_scores,
		word_level_long=word_level_long,
		questionnaire="8_18",
	)
	report = build_eda_markdown_report("Test Report", tables)

	assert {"summary", "exclusion_reasons", "top_produced_words", "metadata_warnings"}.issubset(tables.keys())
	assert int(tables["summary"].loc[tables["summary"]["metric"] == "included_participants", "value"].iloc[0]) == 1
	assert "# Test Report" in report
	assert "## Top Produced Words" in report