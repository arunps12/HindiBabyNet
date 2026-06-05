"""Input helpers for raw Nettskjema CDI exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from hindibabynet_cdi.cleaning import normalize_column_name
from hindibabynet_cdi.config import ProjectConfig, load_config

FORM_FILE_RE = re.compile(r"data-(?P<form_id>\d+)-.*\.(?P<suffix>txt|xlsx)$", re.IGNORECASE)

NON_ITEM_COLUMNS = {
	"$submission_id",
	"$created",
	"$answer_time_ms",
	"$forwarded_to_form",
	"submission_reference",
	"reference id",
	"reference_id",
	"birthdate",
	"बच्चे का लिंग",
	"हमारे अध्ययन में भाग लेने के लिए आपका धन्यवाद! अंत में, हम आपको एक पूरी तरह स्वैच्छिक लॉटरी में भाग लेने का अवसर प्रदान कर रहे हैं। <b>क्या आप लॉटरी में भाग लेना चाहते हैं?</b>",
}

NON_ITEM_COLUMNS_CASEFOLDED = {value.casefold() for value in NON_ITEM_COLUMNS}

FORM_COLUMN_REQUIREMENTS: dict[str, tuple[str, ...]] = {
	"consent": (
		"$submission_id",
		"$created",
		"मुझे ऊपर वर्णित परियोजना के संबंध में जानकारी प्राप्त हो गई है।",
		"मैं सहमति देता/देती हूँ कि मेरी गुमनाम जानकारी ऊपर वर्णित परियोजना में उपयोग की जाएगी।",
		"fill_date",
		"सहमति देने के लिए धन्यवाद। क्या आपके बच्चे का जन्म भारत में हुआ है और क्या वह जन्म से अब तक भारत में ही रह रहा है?",
		"$answer_time_ms",
	),
	"eligibility": (
		"$submission_id",
		"$created",
		"क्या आपके बच्चे की मातृभाषा हिंदी है?",
		"क्या आपका बच्चा प्री-टर्म जन्मा है?",
		"क्या आपके बच्चे को बोलने, सुनने या देखने से संबंधित कोई समस्या है?",
		"क्या आपके बच्चे की आयु 8 से 36 महीने के बीच है?",
		"Reference ID",
		"$answer_time_ms",
	),
	"background": (
		"$submission_id",
		"$created",
		"यदि लागू हो, तो आपके बच्चे की दूसरी भाषा क्या है?",
		"आपका बच्चा कितने प्रतिशत समय दूसरी भाषा सुनता है?",
		"यदि लागू हो, तो आपके बच्चे की तीसरी भाषा क्या है?",
		"आपका बच्चा कितने प्रतिशत समय तीसरी भाषा सुनता है?",
		"माता की वर्तमान शिक्षा स्तर:",
		"पिता की वर्तमान शिक्षा स्तर:",
		"other_education",
		"माँ कहाँ पली-बढ़ी हैं?",
		"पिता कहाँ पले-बढ़े हैं?",
		"यदि आपने &#34;कोई अन्य देश&#34; चुना है, तो कृपया वह(वे) देश बताएं।",
		"आप कहाँ रहते हैं?",
		"माँ की मातृभाषा क्या है?",
		"यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया माता की अन्य भाषा बताएं।",
		"पिता की मातृभाषा क्या है?",
		"यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया पिता की अन्य भाषा बताएं।",
		"बच्चा माँ के संपर्क में कितने प्रतिशत समय रहता है?",
		"बच्चा पिता के संपर्क में कितने प्रतिशत समय रहता है?",
		"बच्चे का लिंग",
		"बच्चे की उम्र कितने महीनों की है?",
		"birthdate",
		"बच्चे की आयु",
		"SUBMISSION_REFERENCE",
		"$answer_time_ms",
		"$forwarded_to_form",
	),
	"cdi_8_18": (
		"$submission_id",
		"$created",
		"हमारे अध्ययन में भाग लेने के लिए आपका धन्यवाद! अंत में, हम आपको एक पूरी तरह स्वैच्छिक लॉटरी में भाग लेने का अवसर प्रदान कर रहे हैं। <b>क्या आप लॉटरी में भाग लेना चाहते हैं?</b>",
		"SUBMISSION_REFERENCE",
		"$answer_time_ms",
	),
	"cdi_19_36": (
		"$submission_id",
		"$created",
		"हमारे अध्ययन में भाग लेने के लिए आपका धन्यवाद! अंत में, हम आपको एक पूरी तरह स्वैच्छिक लॉटरी में भाग लेने का अवसर प्रदान कर रहे हैं। <b>क्या आप लॉटरी में भाग लेना चाहते हैं?</b>",
		"SUBMISSION_REFERENCE",
		"$answer_time_ms",
	),
}


@dataclass(frozen=True)
class RawFormFile:
	form_id: int
	path: Path
	suffix: str


def _get_config(config: ProjectConfig | None = None) -> ProjectConfig:
	return config if config is not None else load_config()


def _iter_form_files(raw_dir: Path) -> Iterable[RawFormFile]:
	for path in raw_dir.iterdir():
		if not path.is_file():
			continue
		match = FORM_FILE_RE.fullmatch(path.name)
		if match is None:
			continue
		yield RawFormFile(
			form_id=int(match.group("form_id")),
			path=path,
			suffix=match.group("suffix").lower(),
		)


def list_form_files(config: ProjectConfig | None = None, raw_dir: str | Path | None = None) -> list[RawFormFile]:
	resolved_raw_dir = Path(raw_dir) if raw_dir is not None else _get_config(config).paths.raw_data
	return sorted(_iter_form_files(resolved_raw_dir), key=lambda item: (item.form_id, item.path.name))


def get_expected_raw_files(config: ProjectConfig | None = None) -> dict[str, Path]:
	project_config = _get_config(config)
	return {
		"consent": project_config.paths.raw_data / project_config.raw_files.consent,
		"eligibility": project_config.paths.raw_data / project_config.raw_files.eligibility,
		"background": project_config.paths.raw_data / project_config.raw_files.background,
		"cdi_8_18": project_config.paths.raw_data / project_config.raw_files.cdi_8_18,
		"cdi_19_36": project_config.paths.raw_data / project_config.raw_files.cdi_19_36,
	}


def validate_raw_file_inventory(config: ProjectConfig | None = None) -> pd.DataFrame:
	project_config = _get_config(config)
	form_id_map = {
		"consent": project_config.forms.consent_form_id,
		"eligibility": project_config.forms.eligibility_form_id,
		"background": project_config.forms.background_form_id,
		"cdi_8_18": project_config.forms.cdi_8_18_form_id,
		"cdi_19_36": project_config.forms.cdi_19_36_form_id,
	}
	detected = {item.form_id: item.path.name for item in list_form_files(config=project_config)}
	rows: list[dict[str, object]] = []
	for form_key, expected_path in get_expected_raw_files(project_config).items():
		form_id = form_id_map[form_key]
		detected_filename = detected.get(form_id, "")
		rows.append(
			{
				"form_key": form_key,
				"form_id": form_id,
				"expected_filename": expected_path.name,
				"expected_path": str(expected_path),
				"exists": expected_path.exists(),
				"detected_filename": detected_filename,
				"filename_matches_expectation": detected_filename == expected_path.name,
			}
		)
	return pd.DataFrame(rows)


def find_form_file(
	form_id: int,
	*,
	config: ProjectConfig | None = None,
	raw_dir: str | Path | None = None,
) -> Path:
	candidates = [item for item in list_form_files(config=config, raw_dir=raw_dir) if item.form_id == int(form_id)]
	if not candidates:
		raise FileNotFoundError(f"No raw export found for form ID {form_id}")
	candidates.sort(key=lambda item: (0 if item.suffix == "xlsx" else 1, item.path.name))
	return candidates[0].path


def read_form_export(path: str | Path) -> pd.DataFrame:
	source_path = Path(path)
	if source_path.suffix.lower() == ".txt":
		return pd.read_csv(
			source_path,
			sep=";",
			quotechar='"',
			dtype=str,
			encoding="utf-8-sig",
			keep_default_na=False,
		)
	if source_path.suffix.lower() == ".xlsx":
		dataframe = pd.read_excel(source_path, dtype=str)
		return dataframe.fillna("")
	raise ValueError(f"Unsupported raw export format: {source_path.suffix}")


def load_form_by_id(
	form_id: int,
	*,
	config: ProjectConfig | None = None,
	raw_dir: str | Path | None = None,
) -> pd.DataFrame:
	return read_form_export(find_form_file(form_id, config=config, raw_dir=raw_dir))


def validate_required_columns(dataframe: pd.DataFrame, required_columns: Iterable[str]) -> list[str]:
	available_normalized = {normalize_column_name(column) for column in dataframe.columns}
	return [column for column in required_columns if normalize_column_name(column) not in available_normalized]


def validate_form_columns(form_key: str, dataframe: pd.DataFrame) -> list[str]:
	if form_key not in FORM_COLUMN_REQUIREMENTS:
		raise KeyError(f"Unknown form key: {form_key}")
	return validate_required_columns(dataframe, FORM_COLUMN_REQUIREMENTS[form_key])


def normalized_column_map(dataframe: pd.DataFrame) -> dict[str, str]:
	return {column: normalize_column_name(column) for column in dataframe.columns}


def find_columns_by_normalized_name(dataframe: pd.DataFrame, *names: str) -> list[str]:
	wanted = {normalize_column_name(name) for name in names}
	return [column for column, normalized in normalized_column_map(dataframe).items() if normalized in wanted]


def get_reference_columns(dataframe: pd.DataFrame) -> list[str]:
	return [
		column
		for column, normalized in normalized_column_map(dataframe).items()
		if normalized.casefold() in NON_ITEM_COLUMNS_CASEFOLDED or normalized.startswith("$")
	]


def get_cdi_item_columns(dataframe: pd.DataFrame) -> list[str]:
	reference_columns = set(get_reference_columns(dataframe))
	return [column for column in dataframe.columns if column not in reference_columns]