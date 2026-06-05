import pandas as pd

from hindibabynet_cdi.scoring import (
	aggregate_scores,
	build_participant_metadata,
	build_scored_wide,
	build_wordbank_tables,
	build_word_level_long,
	score_response,
)


def _forms() -> dict[str, pd.DataFrame]:
	background = pd.DataFrame(
		[
			{
				"$submission_id": "bg-1",
				"$created": "2025-08-31T15:06:43+02:00",
				"birthdate": "2024-01-01",
				"बच्चे का लिंग": "female",
				"बच्चे की उम्र कितने महीनों की है?": "20",
				"बच्चे की आयु": "19-36 महीने",
				"माता की वर्तमान शिक्षा स्तर:": "स्नातक",
				"पिता की वर्तमान शिक्षा स्तर:": "स्नातकोत्तर",
				"माँ कहाँ पली-बढ़ी हैं?": "भारत",
				"पिता कहाँ पले-बढ़े हैं?": "भारत",
				"आप कहाँ रहते हैं?": "भारत",
				"माँ की मातृभाषा क्या है?": "हिंदी",
				"यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया माता की अन्य भाषा बताएं।": "",
				"पिता की मातृभाषा क्या है?": "हिंदी",
				"यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया पिता की अन्य भाषा बताएं।": "",
				"यदि लागू हो, तो आपके बच्चे की दूसरी भाषा क्या है?": "अंग्रेज़ी",
				"आपका बच्चा कितने प्रतिशत समय दूसरी भाषा सुनता है?": "20",
				"यदि लागू हो, तो आपके बच्चे की तीसरी भाषा क्या है?": "",
				"आपका बच्चा कितने प्रतिशत समय तीसरी भाषा सुनता है?": "",
				"बच्चा माँ के संपर्क में कितने प्रतिशत समय रहता है?": "70",
				"बच्चा पिता के संपर्क में कितने प्रतिशत समय रहता है?": "30",
			}
		]
	)
	cdi_8_18 = pd.DataFrame([
		{"$submission_id": "cdi-y-1", "$created": "2025-09-01T11:00:00+02:00", "SUBMISSION_REFERENCE": "bg-1", "कू कू ": "केवल समझता/समझती है", "भों भों  ": "समझता/समझती है और कहता/कहती है"}
	])
	cdi_19_36 = pd.DataFrame([
		{"$submission_id": "cdi-o-1", "$created": "2025-09-01T11:00:00+02:00", "SUBMISSION_REFERENCE": "bg-1", "शीशा /काँच ": "कहता/कहती है"}
	])
	return {"consent": pd.DataFrame(), "eligibility": pd.DataFrame(), "background": background, "cdi_8_18": cdi_8_18, "cdi_19_36": cdi_19_36}


def _tracking(questionnaire: str, cdi_submission_id: str) -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"participant_id": f"participant-{questionnaire}",
				"consent_submission_id": "",
				"eligibility_submission_id": "",
				"background_submission_id": "bg-1",
				"cdi_submission_id": cdi_submission_id,
				"questionnaire": questionnaire,
				"consent_status": "missing_link",
				"eligibility_status": "unclear",
				"included_analysis": True,
				"exclusion_reason": "",
				"linkage_quality_flag": "ok",
			}
		]
	)


def _mapping() -> pd.DataFrame:
	return pd.DataFrame(
		[
			{"item_id": "item_0001", "word": "कू कू", "word_clean": "कू कू", "word_english": "coo", "category": "ध्वनि", "category_english": "sounds", "questionnaire": "8_18", "raw_column_8_18": "कू कू ", "raw_column_19_36": "", "in_8_18": True, "in_19_36": False},
			{"item_id": "item_0002", "word": "भों भों", "word_clean": "भों भों", "word_english": "woof", "category": "ध्वनि", "category_english": "sounds", "questionnaire": "8_18", "raw_column_8_18": "भों भों  ", "raw_column_19_36": "", "in_8_18": True, "in_19_36": False},
			{"item_id": "item_0003", "word": "शीशा / काँच", "word_clean": "शीशा / काँच", "word_english": "glass", "category": "घर", "category_english": "home", "questionnaire": "19_36", "raw_column_8_18": "", "raw_column_19_36": "शीशा /काँच ", "in_8_18": False, "in_19_36": True},
		]
	)


def test_score_response_handles_both_questionnaires() -> None:
	assert score_response("8_18", "केवल समझता/समझती है") == {"comprehension_score": 1, "production_score": 0}
	assert score_response("8_18", "समझता/समझती है और कहता/कहती है") == {"comprehension_score": 1, "production_score": 1}
	assert score_response("19_36", "कहता/कहती है") == {"comprehension_score": None, "production_score": 1}


def test_build_participant_metadata_derives_age_bin_and_codes() -> None:
	participant_metadata = build_participant_metadata(forms=_forms(), linkage=_tracking("19_36", "cdi-o-1"))
	assert len(participant_metadata) == 1
	assert participant_metadata.loc[0, "questionnaire"] == "19_36"
	assert participant_metadata.loc[0, "age_bin"] == "19_24"
	assert participant_metadata.loc[0, "sex_code"] == 2
	assert participant_metadata.loc[0, "maternal_education_code"] == 4


def test_build_scored_wide_creates_raw_and_safe_columns() -> None:
	metadata = build_participant_metadata(forms=_forms(), linkage=_tracking("8_18", "cdi-y-1"))
	raw_wide, safe_wide = build_scored_wide("8_18", forms=_forms(), participant_metadata=metadata, mapping=_mapping())
	assert raw_wide.loc[0, "कू कू "] == 1
	assert raw_wide.loc[0, "भों भों  "] == 2
	assert safe_wide.loc[0, "item_0001"] == 1
	assert safe_wide.loc[0, "item_0002"] == 2


def test_build_word_level_long_and_aggregate_scores() -> None:
	forms = _forms()
	tracking = pd.concat([_tracking("8_18", "cdi-y-1"), _tracking("19_36", "cdi-o-1")], ignore_index=True)
	mapping = _mapping()
	participant_metadata = build_participant_metadata(forms=forms, linkage=tracking)
	word_level = build_word_level_long(forms=forms, tracking=tracking, mapping=mapping, participant_metadata=participant_metadata)
	participant_scores, category_scores, master_dataset = aggregate_scores(word_level, participant_metadata, mapping=mapping)
	older = participant_scores[participant_scores["questionnaire"] == "19_36"].iloc[0]
	younger = participant_scores[participant_scores["questionnaire"] == "8_18"].iloc[0]
	assert len(word_level) == 3
	assert set(["age_days", "sex_raw", "comprehension", "production"]).issubset(word_level.columns)
	assert older["n_words_inventory"] == 1
	assert younger["n_words_inventory"] == 2
	assert float(younger["production_proportion"]) == 0.5
	assert float(older["production_proportion"]) == 1.0
	assert int(category_scores["n_words_category"].max()) >= 1
	assert len(master_dataset) == 2


def test_build_wordbank_tables_returns_required_structures() -> None:
	forms = _forms()
	tracking = pd.concat([_tracking("8_18", "cdi-y-1"), _tracking("19_36", "cdi-o-1")], ignore_index=True)
	mapping = _mapping()
	participant_metadata = build_participant_metadata(forms=forms, linkage=tracking)
	word_level = build_word_level_long(forms=forms, tracking=tracking, mapping=mapping, participant_metadata=participant_metadata)
	participant_scores, _, _ = aggregate_scores(word_level, participant_metadata, mapping=mapping)
	wordbank_tables = build_wordbank_tables(word_level, participant_scores=participant_scores, mapping=mapping)

	assert {"wordbank_age_summary", "wordbank_word_by_age", "wordbank_category_by_age", "wordbank_percentile_curves", "shared_word_production_by_age"}.issubset(wordbank_tables.keys())
	assert {"questionnaire", "age_bin", "word", "production_rate", "comprehension_rate"}.issubset(wordbank_tables["wordbank_word_by_age"].columns)
	assert {"questionnaire", "age_bin", "p50_production", "p50_comprehension"}.issubset(wordbank_tables["wordbank_percentile_curves"].columns)
	assert {"age_bin", "questionnaire", "n_shared_words", "production_rate"}.issubset(wordbank_tables["shared_word_production_by_age"].columns)