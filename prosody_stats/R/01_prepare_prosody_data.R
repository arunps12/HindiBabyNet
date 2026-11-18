source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "table_helpers.R"))
source(file.path("R", "functions", "data_helpers.R"))

load_required_packages()
ensure_analysis_directories()

prosody_data_bundle <- prepare_prosody_long_dataset()

write_csv_output(
  prosody_data_bundle$prosody_long,
  file.path(analysis_paths$data_derived, "prosody_long.csv")
)
write_csv_output(
  prosody_data_bundle$merge_summary,
  file.path(analysis_paths$outputs_tables, "merge_summary.csv")
)
write_csv_output(
  prosody_data_bundle$missingness,
  file.path(analysis_paths$outputs_tables, "missingness_summary.csv")
)
write_csv_output(
  prosody_data_bundle$complete_cases,
  file.path(analysis_paths$outputs_tables, "complete_case_counts.csv")
)
write_csv_output(
  prosody_data_bundle$unmatched,
  file.path(analysis_paths$outputs_tables, "unmatched_identifiers.csv")
)
write_csv_output(
  prosody_data_bundle$variation_checks,
  file.path(analysis_paths$outputs_tables, "variation_checks.csv")
)
write_text_output(
  prosody_data_bundle$merge_notes,
  file.path(analysis_paths$outputs_tables, "merge_summary.txt")
)

message(sprintf(
  "Prepared prosody_long dataset with %d rows across %d participants.",
  nrow(prosody_data_bundle$prosody_long),
  dplyr::n_distinct(prosody_data_bundle$prosody_long$participant_id)
))
