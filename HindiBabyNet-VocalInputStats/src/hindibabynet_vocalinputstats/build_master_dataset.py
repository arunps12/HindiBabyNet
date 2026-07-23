"""Build the privacy-safe participant-level master dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hindibabynet_vocalinputstats.config import ProjectConfig, load_config
from hindibabynet_vocalinputstats.durations import resolve_recording_duration, seconds_to_hours
from hindibabynet_vocalinputstats.ids import attach_participant_ids, create_participant_lookup, save_participant_lookup
from hindibabynet_vocalinputstats.io import (
    find_vtc_csv,
    load_vtc_csv,
    read_table,
    write_csv,
    write_dataset_build_report,
    write_validation_report,
)
from hindibabynet_vocalinputstats.vtc_summary import summarize_vtc_dataframe


PUBLIC_COLUMNS = [
    "participant_id",
    "REC_date",
    "birthdate",
    "child_sex",
    "mother_education",
    "father_education",
    "SES",
    "Location",
    "age_days",
    "age_months",
    "age_z",
    "age_z2",
    "recording_duration_sec",
    "recording_duration_hours",
    "adult_female_count",
    "adult_male_count",
    "other_child_count",
    "key_child_count",
    "adult_female_duration_sec",
    "adult_male_duration_sec",
    "other_child_duration_sec",
    "key_child_duration_sec",
    "adult_female_count_hour",
    "adult_male_count_hour",
    "other_child_count_hour",
    "key_child_count_hour",
    "adult_female_duration_hour",
    "adult_male_duration_hour",
    "other_child_duration_hour",
    "key_child_duration_hour",
]

SPEAKERS = ["adult_female", "adult_male", "other_child", "key_child"]
REQUIRED_METADATA_COLUMNS = [
    "REC_date",
    "birthdate",
    "child_sex",
    "mother_education",
    "father_education",
    "Location",
]


def _required_metadata_columns(config: ProjectConfig) -> list[str]:
    return [config.metadata_id_column, *REQUIRED_METADATA_COLUMNS]


def _print_input_path_status(config: ProjectConfig) -> None:
    metadata_exists = config.metadata_path.exists()
    audio_exists = config.audio_root.exists()
    vtc_exists = config.vtc_output_root.exists()

    print(f"Metadata path: {config.metadata_path}")
    print(f"Metadata exists: {metadata_exists}")
    print(f"Audio root: {config.audio_root}")
    print(f"Audio root exists: {audio_exists}")
    print(f"VTC output root: {config.vtc_output_root}")
    print(f"VTC output root exists: {vtc_exists}")

    if not metadata_exists:
        raise FileNotFoundError(
            "Metadata path does not exist. Check configs/config.yaml and verify UNC or network access, "
            "including VPN if needed."
        )
    if not audio_exists:
        print("Warning: audio root does not exist. Continuing, but audio matches will be missing. Check VPN/network access if using a UNC path.")
    if not vtc_exists:
        print("Warning: VTC output root does not exist. Continuing, but VTC matches will be missing. Check VPN/network access if using a UNC path.")


def _load_metadata_table(config: ProjectConfig) -> pd.DataFrame:
    metadata = read_table(config.metadata_path)
    missing = [column for column in _required_metadata_columns(config) if column not in metadata.columns]
    if missing:
        found_columns = ", ".join(str(column) for column in metadata.columns)
        missing_text = ", ".join(missing)
        raise ValueError(f"Metadata table is missing required columns: {missing_text}. Found columns: {found_columns}")
    if config.metadata_id_column != "par_id":
        metadata = metadata.copy()
        metadata["par_id"] = metadata[config.metadata_id_column]
    return metadata


def _parse_dates(metadata: pd.DataFrame) -> pd.DataFrame:
    dataframe = metadata.copy()
    dataframe["REC_date"] = pd.to_datetime(dataframe["REC_date"], errors="coerce")
    dataframe["birthdate"] = pd.to_datetime(dataframe["birthdate"], errors="coerce")
    return dataframe


def _compute_age_fields(dataframe: pd.DataFrame, age_month_denominator: float) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["age_days"] = (enriched["REC_date"] - enriched["birthdate"]).dt.days.astype("float")
    enriched["age_months"] = enriched["age_days"] / age_month_denominator

    valid_age = enriched["age_days"].notna() & (enriched["age_days"] >= 0)
    age_days = enriched.loc[valid_age, "age_days"]
    if age_days.empty:
        enriched["age_z"] = np.nan
    else:
        std = float(age_days.std(ddof=0))
        if std == 0.0:
            enriched["age_z"] = np.where(valid_age, 0.0, np.nan)
        else:
            mean = float(age_days.mean())
            enriched["age_z"] = (enriched["age_days"] - mean) / std
    enriched["age_z2"] = enriched["age_z"] ** 2
    return enriched


def _build_audio_records(metadata: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        audio_match = resolve_recording_duration(
            config.audio_root,
            row.original_par_id,
            config.audio_extensions,
            participant_folder_name=config.audio_layout.participant_folder_name,
            recursive=config.audio_layout.recursive,
            prefer_largest_audio_file=config.audio_layout.prefer_largest_audio_file,
            recording_duration_source=config.recording_duration_source,
            session_selection=config.session_selection,
            session_date_format=config.session_date_format,
        )
        recording_duration_sec = audio_match.duration_sec
        matched_audio = bool(audio_match.matched_audio)
        if not matched_audio:
            warnings.append(
                {
                    "participant_id": row.participant_id,
                    "original_par_id": row.original_par_id,
                    "issue_type": "missing_audio",
                    "message": "No matching audio file found.",
                }
            )
        if audio_match.warning_message is not None:
            warnings.append(
                {
                    "participant_id": row.participant_id,
                    "original_par_id": row.original_par_id,
                    "issue_type": "audio_duration_warning",
                    "message": audio_match.warning_message,
                }
            )
        recording_duration_hours = seconds_to_hours(recording_duration_sec)
        if matched_audio and pd.notna(recording_duration_hours) and recording_duration_hours < config.minimum_recording_hours_warning:
            warnings.append(
                {
                    "participant_id": row.participant_id,
                    "original_par_id": row.original_par_id,
                    "issue_type": "short_recording",
                    "message": (
                        f"Recording duration {recording_duration_hours:.3f} hours is below "
                        f"warning threshold {config.minimum_recording_hours_warning:.3f}."
                    ),
                }
            )
        records.append(
            {
                "participant_id": row.participant_id,
                "original_par_id": row.original_par_id,
                "recording_duration_sec": recording_duration_sec,
                "recording_duration_hours": recording_duration_hours,
                "matched_audio": matched_audio,
            }
        )
        audit_records.append(
            {
                "participant_id": row.participant_id,
                "selected_earliest_folder": audio_match.selected_folder,
                "selected_date": audio_match.selected_date,
                "audio_file_count": audio_match.audio_file_count,
                "total_duration_sec": recording_duration_sec,
                "total_duration_hours": recording_duration_hours,
                "ignored_later_folders": "; ".join(audio_match.ignored_later_folders),
                "warning_or_reason": audio_match.warning_message or audio_match.audit_reason,
            }
        )
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(audit_records), warnings


def _build_age_validation_table(final_master: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["age_days", "age_z", "age_z2"]
    missing_columns = [column for column in required_columns if column not in final_master.columns]
    valid_age = final_master["age_z"].notna()
    valid_age_count = int(valid_age.sum())
    age_z_mean = float(final_master.loc[valid_age, "age_z"].mean()) if valid_age.any() else np.nan
    age_z_sd = float(final_master.loc[valid_age, "age_z"].std(ddof=0)) if valid_age.any() else np.nan
    age_z2_matches = bool(np.allclose(
        final_master.loc[valid_age, "age_z2"],
        final_master.loc[valid_age, "age_z"] ** 2,
        equal_nan=True,
    )) if valid_age.any() else True

    return pd.DataFrame(
        [
            {"check": "required_age_columns_present", "value": len(missing_columns) == 0, "detail": ", ".join(missing_columns)},
            {
                "check": "age_z_mean_approximately_zero",
                "value": bool(valid_age_count < 1 or (np.isfinite(age_z_mean) and abs(age_z_mean) < 1e-9)),
                "detail": age_z_mean if valid_age_count >= 1 else "no valid age rows",
            },
            {
                "check": "age_z_sd_approximately_one",
                "value": bool(valid_age_count < 2 or (np.isfinite(age_z_sd) and abs(age_z_sd - 1.0) < 1e-9)),
                "detail": age_z_sd if valid_age_count >= 2 else "fewer than two valid age rows",
            },
            {"check": "age_z2_equals_square", "value": age_z2_matches, "detail": "age_z2 == age_z^2 for non-missing rows"},
        ]
    )


def _build_vtc_records(metadata: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    summary_frames: list[pd.DataFrame] = []
    matched_records: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    for row in metadata.itertuples(index=False):
        vtc_path = find_vtc_csv(
            config.vtc_output_root,
            row.original_par_id,
            participant_folder_name=config.vtc_layout.participant_folder_name,
            rttm_csv_name=config.vtc_layout.rttm_csv_name,
        )
        matched_vtc = vtc_path is not None
        matched_records.append(
            {
                "participant_id": row.participant_id,
                "original_par_id": row.original_par_id,
                "matched_vtc": matched_vtc,
            }
        )
        if not matched_vtc:
            warnings.append(
                {
                    "participant_id": row.participant_id,
                    "original_par_id": row.original_par_id,
                    "issue_type": "missing_vtc",
                    "message": "No matching VTC rttm.csv found.",
                }
            )
            continue
        try:
            vtc_df = load_vtc_csv(vtc_path)
        except Exception as exc:  # pragma: no cover
            warnings.append(
                {
                    "participant_id": row.participant_id,
                    "original_par_id": row.original_par_id,
                    "issue_type": "invalid_vtc_csv",
                    "message": str(exc),
                }
            )
            continue

        summary, summary_warnings = summarize_vtc_dataframe(
            vtc_df,
            participant_id=row.participant_id,
            original_par_id=row.original_par_id,
        )
        summary_frames.append(summary)
        warnings.extend(summary_warnings)

    if summary_frames:
        summaries = pd.concat(summary_frames, ignore_index=True)
    else:
        summaries = pd.DataFrame(columns=["participant_id", "original_par_id", "speaker", "count", "duration_sec"])
    return summaries, pd.DataFrame.from_records(matched_records), warnings


def _pivot_vtc_summary(summary: pd.DataFrame, matched_vtc: pd.DataFrame) -> pd.DataFrame:
    if matched_vtc.empty:
        columns = ["participant_id", "original_par_id", "matched_vtc"]
        columns.extend(f"{speaker}_count" for speaker in SPEAKERS)
        columns.extend(f"{speaker}_duration_sec" for speaker in SPEAKERS)
        return pd.DataFrame(columns=columns)

    wide = matched_vtc.set_index("participant_id").copy()
    if summary.empty:
        for speaker in SPEAKERS:
            wide[f"{speaker}_count"] = np.where(wide["matched_vtc"], 0.0, np.nan)
            wide[f"{speaker}_duration_sec"] = np.where(wide["matched_vtc"], 0.0, np.nan)
        return wide.reset_index()

    counts = summary.pivot(index="participant_id", columns="speaker", values="count")
    durations = summary.pivot(index="participant_id", columns="speaker", values="duration_sec")
    for speaker in SPEAKERS:
        wide[f"{speaker}_count"] = pd.to_numeric(counts.get(speaker), errors="coerce")
        wide[f"{speaker}_duration_sec"] = pd.to_numeric(durations.get(speaker), errors="coerce")
        wide.loc[wide["matched_vtc"], f"{speaker}_count"] = pd.to_numeric(
            wide.loc[wide["matched_vtc"], f"{speaker}_count"],
            errors="coerce",
        ).fillna(0.0)
        wide.loc[wide["matched_vtc"], f"{speaker}_duration_sec"] = pd.to_numeric(
            wide.loc[wide["matched_vtc"], f"{speaker}_duration_sec"],
            errors="coerce",
        ).fillna(0.0)
    return wide.reset_index()


def _compute_rate_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    for speaker in SPEAKERS:
        count_column = f"{speaker}_count"
        duration_column = f"{speaker}_duration_sec"
        output[f"{speaker}_count_hour"] = np.where(
            output[count_column].notna() & output["recording_duration_hours"].gt(0),
            output[count_column] / output["recording_duration_hours"],
            np.nan,
        )
        output[f"{speaker}_duration_hour"] = np.where(
            output[duration_column].notna() & output["recording_duration_hours"].gt(0),
            output[duration_column] / output["recording_duration_hours"],
            np.nan,
        )
    return output


def _finalize_public_metadata(metadata: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    base = metadata.copy()
    for column in _required_metadata_columns(config):
        if column not in base.columns:
            found_columns = ", ".join(str(item) for item in base.columns)
            raise ValueError(f"Metadata table is missing required column: {column}. Found columns: {found_columns}")
    base["par_id"] = base["par_id"].astype(str).str.strip()
    base = _parse_dates(base)
    base = _compute_age_fields(base, age_month_denominator=config.age_month_denominator)
    base["SES"] = base[config.ses_source]
    return base.rename(columns={"par_id": "original_par_id"})


def _build_dataset_report_lines(config: ProjectConfig, final_master: pd.DataFrame, validation_records: list[dict[str, object]]) -> list[str]:
    issues = pd.DataFrame(validation_records)

    def _count_issues(issue_type: str) -> int:
        if issues.empty:
            return 0
        return int((issues["issue_type"] == issue_type).sum())

    input_long_path = config.derived_data_dir / "input_long.csv"
    input_output_long_path = config.derived_data_dir / "input_output_long.csv"
    input_long_rows = pd.read_csv(input_long_path).shape[0] if input_long_path.exists() else 0
    input_output_long_rows = pd.read_csv(input_output_long_path).shape[0] if input_output_long_path.exists() else 0

    return [
        f"build_datetime_utc: {datetime.now(timezone.utc).isoformat()}",
        f"metadata_participants: {len(final_master)}",
        f"matched_vtc_output: {int(final_master['matched_vtc'].fillna(False).sum())}",
        f"matched_audio_duration: {int(final_master['matched_audio'].fillna(False).sum())}",
        f"missing_vtc: {_count_issues('missing_vtc')}",
        f"missing_audio: {_count_issues('missing_audio')}",
        f"missing_age: {int(final_master['age_days'].isna().sum())}",
        f"negative_or_invalid_age: {int(((final_master['age_days'] < 0) & final_master['age_days'].notna()).sum())}",
        f"final_master_rows: {len(final_master)}",
        f"input_long_rows: {input_long_rows}",
        f"input_output_long_rows: {input_output_long_rows}",
        f"denominator_note: Recording duration source was {config.recording_duration_source} with session_selection={config.session_selection}.",
    ]


def build_master_dataset(config: ProjectConfig) -> pd.DataFrame:
    _print_input_path_status(config)
    metadata = _load_metadata_table(config)
    lookup = create_participant_lookup(metadata, participant_id_digits=config.participant_id_digits)
    save_participant_lookup(lookup, config.private_data_dir / "participant_lookup.csv")

    public_metadata = _finalize_public_metadata(metadata, config)
    public_metadata = attach_participant_ids(public_metadata, lookup)

    validation_records: list[dict[str, object]] = []
    audio_records, duration_audit, audio_warnings = _build_audio_records(public_metadata, config)
    validation_records.extend(audio_warnings)
    vtc_summary, matched_vtc, vtc_warnings = _build_vtc_records(public_metadata, config)
    validation_records.extend(vtc_warnings)

    final_master = public_metadata.merge(audio_records, on=["participant_id", "original_par_id"], how="left", sort=False)
    final_master = final_master.merge(_pivot_vtc_summary(vtc_summary, matched_vtc), on=["participant_id", "original_par_id"], how="left", sort=False)

    for row in final_master.itertuples(index=False):
        if pd.isna(row.age_days):
            validation_records.append(
                {
                    "participant_id": row.participant_id,
                    "original_par_id": row.original_par_id,
                    "issue_type": "missing_age",
                    "message": "Age could not be computed from REC_date and birthdate.",
                }
            )
        elif row.age_days < 0:
            validation_records.append(
                {
                    "participant_id": row.participant_id,
                    "original_par_id": row.original_par_id,
                    "issue_type": "negative_age",
                    "message": "Age in days is negative.",
                }
            )

    final_master = _compute_rate_columns(final_master)
    final_master = final_master.sort_values("participant_id", kind="stable").reset_index(drop=True)

    age_validation = _build_age_validation_table(final_master)
    for row in age_validation.itertuples(index=False):
        if not bool(row.value):
            validation_records.append(
                {
                    "participant_id": "ALL",
                    "original_par_id": "ALL",
                    "issue_type": f"age_validation_failed:{row.check}",
                    "message": str(row.detail),
                }
            )

    public_output = final_master.loc[:, PUBLIC_COLUMNS].copy()
    public_output["REC_date"] = pd.to_datetime(public_output["REC_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    public_output["birthdate"] = pd.to_datetime(public_output["birthdate"], errors="coerce").dt.strftime("%Y-%m-%d")

    write_csv(public_output, config.derived_data_dir / "final_master.csv")
    write_csv(duration_audit, config.results_dir / "recording_duration_audit.csv")
    write_csv(age_validation, config.results_dir / "age_validation.csv")
    write_validation_report(validation_records, config.results_dir / "validation_report.csv")
    report_lines = _build_dataset_report_lines(config, final_master, validation_records)
    write_dataset_build_report(report_lines, config.results_dir / "dataset_build_report.txt")
    return public_output


def run_build_master(config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    return build_master_dataset(config)