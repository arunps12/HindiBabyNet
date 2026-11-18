source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "data_prep.R"))
source(file.path("R", "functions", "model_selection.R"))

load_required_packages(attach = FALSE)
ensure_analysis_directories()

datasets_bundle <- load_analysis_datasets(write_summary = FALSE)

input_output_data <- datasets_bundle$datasets$input_output_long

input_output_model_bundles <- list(
	fit_preregistered_model(input_output_data, "key_child_count_hour", "family_2_count", allow_gamma = TRUE),
	fit_preregistered_model(input_output_data, "key_child_duration_min_hour", "family_2_duration", allow_gamma = TRUE)
)

input_output_selection_table <- selection_table_from_bundles(input_output_model_bundles)
utils::write.csv(
	input_output_selection_table,
	file.path(analysis_paths$results_tables, "input_output_model_selection.csv"),
	row.names = FALSE
)

invisible(lapply(input_output_model_bundles, function(bundle) {
	save_model_bundle(bundle, file.path(analysis_paths$results_models, sprintf("%s.rds", bundle$response_name)))
}))

message(sprintf("Loaded %d rows from input_output_long.csv.", nrow(datasets_bundle$datasets$input_output_long)))
message("Model Family 2 fits and selection metadata saved to results/r_models and results/r_tables.")
