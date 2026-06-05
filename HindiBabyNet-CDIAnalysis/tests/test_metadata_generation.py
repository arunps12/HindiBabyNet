import pandas as pd

from hindibabynet_cdi.metadata import generate_metadata_outputs, match_cdi_columns_to_metadata


def _forms() -> dict[str, pd.DataFrame]:
	return {
		"consent": pd.DataFrame(),
		"eligibility": pd.DataFrame(),
		"background": pd.DataFrame(),
		"cdi_8_18": pd.DataFrame(
			[
				{
					"$submission_id": "1",
					"$created": "2026-06-04",
					"SUBMISSION_REFERENCE": "bg-1",
					"$answer_time_ms": "1000",
					"हमारे अध्ययन में भाग लेने के लिए आपका धन्यवाद! अंत में, हम आपको एक पूरी तरह स्वैच्छिक लॉटरी में भाग लेने का अवसर प्रदान कर रहे हैं। <b>क्या आप लॉटरी में भाग लेना चाहते हैं?</b>": "नहीं",
					"टेडी बियर ": "",
					"टेडी बियर .1": "",
					"कू कू ": "",
				}
			]
		),
		"cdi_19_36": pd.DataFrame(
			[
				{
					"$submission_id": "2",
					"$created": "2026-06-04",
					"SUBMISSION_REFERENCE": "bg-1",
					"$answer_time_ms": "1000",
					"हमारे अध्ययन में भाग लेने के लिए आपका धन्यवाद! अंत में, हम आपको एक पूरी तरह स्वैच्छिक लॉटरी में भाग लेने का अवसर प्रदान कर रहे हैं। <b>क्या आप लॉटरी में भाग लेना चाहते हैं?</b>": "नहीं",
					"टेडी बियर ": "",
					"खिड़की .1": "",
				}
			]
		),
	}


def test_generate_metadata_outputs_creates_missing_mapping_rows(tmp_path) -> None:
	reference_mapping = pd.DataFrame(
		{
			"word": ["कू कू", "टेडी बियर"],
			"word_english": ["coo-coo", "teddy bear"],
			"category": ["ध्वनि", "जानवर"],
			"category_english": ["sounds", "animals"],
			"questionnaire": ["8_18;19_36", "8_18;19_36"],
		}
	)
	reference_mapping.to_csv(tmp_path / "word_mapping.csv", index=False)

	from hindibabynet_cdi.config import load_config
	config = load_config()
	synthetic_config = config.__class__(
		repo_root=config.repo_root,
		config_path=config.config_path,
		paths=config.paths.__class__(
			raw_data=config.paths.raw_data,
			interim_data=config.paths.interim_data,
			processed_data=config.paths.processed_data,
			metadata=tmp_path,
			outputs=config.paths.outputs,
		),
		raw_files=config.raw_files,
		forms=config.forms,
		analysis=config.analysis,
	)

	outputs = generate_metadata_outputs(_forms(), config=synthetic_config)

	assert "word_mapping" in outputs
	assert "missing_word_mapping_to_fill" in outputs
	assert len(outputs["word_mapping"]) >= 3
	assert len(outputs["missing_word_mapping_to_fill"]) >= 1
	assert outputs["word_mapping"]["item_id"].str.startswith("item_").all()
	assert outputs["word_mapping"]["is_duplicate_raw_column"].astype(str).str.lower().isin(["true", "false"]).all()
	assert {"word_english", "category_english", "is_duplicate_raw_column"}.issubset(outputs["column_mapping"].columns)
	assert {"questionnaire", "response_raw", "score_code", "comprehension", "production"}.issubset(outputs["response_mapping"].columns)
	assert (outputs["word_mapping_quality_report"]["metric"] == "duplicate_raw_columns").any()


def test_match_cdi_columns_to_metadata_uses_raw_column_assignment() -> None:
	word_mapping = pd.DataFrame(
		{
			"item_id": ["item_0001", "item_0002"],
			"word": ["कू कू", "टेडी बियर"],
			"word_clean": ["कू कू", "टेडी बियर"],
			"word_english": ["coo", "teddy bear"],
			"category": ["ध्वनि", "जानवर"],
			"category_english": ["sounds", "animals"],
			"questionnaire": ["8_18", "19_36"],
			"raw_column_8_18": ["कू कू ", ""],
			"raw_column_19_36": ["", "टेडी बियर .1"],
			"in_8_18": [True, False],
			"in_19_36": [False, True],
			"is_duplicate_raw_column": [False, True],
			"notes": ["", "manual_review_required"],
		}
	)

	matches = match_cdi_columns_to_metadata(["कू कू ", "अनजान शब्द"], word_mapping, questionnaire="8_18")

	assert matches["matched"].tolist() == [True, False]
	assert matches.loc[0, "item_id"] == "item_0001"