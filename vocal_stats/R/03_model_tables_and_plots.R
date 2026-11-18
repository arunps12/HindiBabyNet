source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "data_prep.R"))
source(file.path("R", "functions", "model_selection.R"))
source(file.path("R", "functions", "model_diagnostics.R"))
source(file.path("R", "functions", "monte_carlo_ci.R"))
source(file.path("R", "functions", "prediction_plots.R"))
source(file.path("R", "functions", "reporting.R"))

load_required_packages(attach = FALSE)
ensure_analysis_directories()

datasets_bundle <- load_analysis_datasets(write_summary = TRUE)
input_data <- datasets_bundle$datasets$input_long
input_output_data <- datasets_bundle$datasets$input_output_long

model_bundles <- list(
	load_model_bundle(file.path(analysis_paths$results_models, "input_count_hour.rds")),
	load_model_bundle(file.path(analysis_paths$results_models, "input_duration_min_hour.rds")),
	load_model_bundle(file.path(analysis_paths$results_models, "key_child_count_hour.rds")),
	load_model_bundle(file.path(analysis_paths$results_models, "key_child_duration_min_hour.rds"))
)

diagnostics <- lapply(model_bundles, run_model_diagnostics)
names(diagnostics) <- vapply(model_bundles, function(bundle) bundle$response_name, character(1))

predictions <- list(
	input_count_hour = plot_age_speaker_predictions(
		model_bundles[[1]],
		input_data,
		output_stub = "input_count_by_age_speaker",
		y_label = "Predicted input count per hour"
	),
	input_duration_min_hour = plot_age_speaker_predictions(
		model_bundles[[2]],
		input_data,
		output_stub = "input_duration_by_age_speaker",
		y_label = "Predicted input duration (min/hour)"
	),
	key_child_count_hour = plot_input_output_predictions(
		model_bundles[[3]],
		input_output_data,
		predictor_name = "input_count_hour",
		x_label = "Input count per hour",
		y_label = "Predicted key child count per hour",
		output_stub = "key_child_count_by_input"
	),
	key_child_duration_min_hour = plot_input_output_predictions(
		model_bundles[[4]],
		input_output_data,
		predictor_name = "input_duration_min_hour",
		x_label = "Input duration (min/hour)",
		y_label = "Predicted key child duration (min/hour)",
		output_stub = "key_child_duration_by_input"
	)
)

forest_plots <- lapply(model_bundles, function(bundle) {
	plot_fixed_effects_forest(bundle, output_stub = bundle$response_name)
})
names(forest_plots) <- names(diagnostics)

reporting_bundle <- build_reporting_bundle(
	model_bundles = model_bundles,
	diagnostics = diagnostics,
	predictions = predictions,
	dataset_summary = datasets_bundle$dataset_summary
)

bundle_path <- write_reporting_outputs(reporting_bundle)

message(sprintf("Reporting bundle saved to %s", bundle_path))
message("Diagnostics, predictions, and fixed-effect plots saved under results/r_*.")
