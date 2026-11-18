source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "table_helpers.R"))
source(file.path("R", "functions", "data_helpers.R"))

load_required_packages()
ensure_analysis_directories()

prosody_long <- read_csv_required(file.path(analysis_paths$data_derived, "prosody_long.csv"), "prosody_long")

participant_descriptives <- prosody_long |>
  dplyr::distinct(.data$participant_id, .data$age_days, .data$age_months, .data$age_z, .data$SES, .data$child_sex, .data$city) |>
  dplyr::summarise(
    n_participants = dplyr::n(),
    mean_age_days = mean(.data$age_days, na.rm = TRUE),
    sd_age_days = stats::sd(.data$age_days, na.rm = TRUE),
    mean_age_months = mean(.data$age_months, na.rm = TRUE),
    sd_age_months = stats::sd(.data$age_months, na.rm = TRUE)
  )

speaker_descriptives <- prosody_long |>
  dplyr::group_by(.data$speaker) |>
  dplyr::summarise(
    n_rows = dplyr::n(),
    dplyr::across(dplyr::all_of(analysis_options$primary_acoustic_measures), list(mean = ~mean(.x, na.rm = TRUE), sd = ~stats::sd(.x, na.rm = TRUE))),
    .groups = "drop"
  )

acoustic_correlations <- prosody_long |>
  dplyr::select(dplyr::all_of(analysis_options$primary_acoustic_measures)) |>
  stats::cor(use = "pairwise.complete.obs") |>
  as.data.frame()

pca_summary <- prosody_long |>
  dplyr::summarise(
    dplyr::across(dplyr::matches("^(wav2vec|hubert)_PC[123]$"), list(mean = ~mean(.x, na.rm = TRUE), sd = ~stats::sd(.x, na.rm = TRUE)))
  )

family1_fixed_files <- list.files(analysis_paths$outputs_tables, pattern = "^family1__.*_fixed_effects\\.csv$", full.names = TRUE)
family2_count_files <- list.files(analysis_paths$outputs_tables, pattern = "^family_2_count__.*_fixed_effects\\.csv$", full.names = TRUE)
family2_duration_files <- list.files(analysis_paths$outputs_tables, pattern = "^family_2_duration__.*_fixed_effects\\.csv$", full.names = TRUE)
fit_stat_files <- list.files(analysis_paths$outputs_tables, pattern = "_fit_stats\\.csv$", full.names = TRUE)
vif_files <- list.files(analysis_paths$outputs_tables, pattern = "_vif\\.csv$", full.names = TRUE)
diagnostic_files <- list.files(analysis_paths$outputs_diagnostics, pattern = "_diagnostics_summary\\.csv$", full.names = TRUE)

read_csv_list <- function(paths) {
  bind_rows_safe(lapply(paths, function(path) read.csv(path, stringsAsFactors = FALSE)))
}

write_csv_output(participant_descriptives, file.path(analysis_paths$outputs_tables, "participant_descriptives.csv"))
write_csv_output(speaker_descriptives, file.path(analysis_paths$outputs_tables, "speaker_descriptives.csv"))
write_csv_output(acoustic_correlations, file.path(analysis_paths$outputs_tables, "acoustic_correlations.csv"))
write_csv_output(pca_summary, file.path(analysis_paths$outputs_tables, "pca_summary.csv"))
write_csv_output(read_csv_list(family1_fixed_files), file.path(analysis_paths$outputs_tables, "family1_fixed_effects_summary.csv"))
write_csv_output(read_csv_list(family2_count_files), file.path(analysis_paths$outputs_tables, "family2_count_fixed_effects_summary.csv"))
write_csv_output(read_csv_list(family2_duration_files), file.path(analysis_paths$outputs_tables, "family2_duration_fixed_effects_summary.csv"))
write_csv_output(read_csv_list(fit_stat_files), file.path(analysis_paths$outputs_tables, "model_fit_statistics_summary.csv"))
write_csv_output(read_csv_list(vif_files), file.path(analysis_paths$outputs_tables, "vif_summary.csv"))
write_csv_output(read_csv_list(diagnostic_files), file.path(analysis_paths$outputs_tables, "diagnostics_summary.csv"))
