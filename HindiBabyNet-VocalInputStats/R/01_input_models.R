source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "data_prep.R"))
source(file.path("R", "functions", "model_selection.R"))

load_required_packages(attach = FALSE)
ensure_analysis_directories()

datasets_bundle <- load_analysis_datasets(write_summary = FALSE)

input_data <- datasets_bundle$datasets$input_long

input_model_bundles <- list(
	fit_preregistered_model(input_data, "input_count_hour", "family_1", allow_gamma = TRUE),
	fit_preregistered_model(input_data, "input_duration_min_hour", "family_1", allow_gamma = TRUE)
)

input_selection_table <- selection_table_from_bundles(input_model_bundles)
utils::write.csv(
	input_selection_table,
	file.path(analysis_paths$results_tables, "input_model_selection.csv"),
	row.names = FALSE
)

invisible(lapply(input_model_bundles, function(bundle) {
	save_model_bundle(bundle, file.path(analysis_paths$results_models, sprintf("%s.rds", bundle$response_name)))
}))

message(sprintf("Loaded %d rows from final_master.csv.", nrow(datasets_bundle$datasets$final_master)))
message("Model Family 1 fits and selection metadata saved to results/r_models and results/r_tables.")
