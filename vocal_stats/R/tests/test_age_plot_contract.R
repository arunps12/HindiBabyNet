source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "data_prep.R"))
source(file.path("R", "functions", "monte_carlo_ci.R"))
source(file.path("R", "functions", "prediction_plots.R"))

datasets_bundle <- load_analysis_datasets(write_summary = FALSE)

quadratic_bundle <- list(
	response_name = "input_count_hour",
	model_family = "family_1",
	age_specification = "quadratic"
)
none_bundle <- list(
	response_name = "input_count_hour",
	model_family = "family_1",
	age_specification = "none"
)

prediction_grid <- build_age_speaker_grid(quadratic_bundle, datasets_bundle$datasets$input_long, n_points = 5)

expected_age_z <- age_days_to_z(prediction_grid$age_days)
stopifnot(all(abs(expected_age_z - prediction_grid$age_z) < 1e-8))
stopifnot(all(abs((prediction_grid$age_z ^ 2) - prediction_grid$age_z2) < 1e-8))

validate_age_plot_data(prediction_grid, bundle = quadratic_bundle)

prediction_grid_none <- build_age_speaker_grid(none_bundle, datasets_bundle$datasets$input_long, n_points = 5)
stopifnot(!("age_z" %in% names(prediction_grid_none)))
stopifnot(!("age_z2" %in% names(prediction_grid_none)))
validate_age_plot_data(prediction_grid_none, bundle = none_bundle)

message("Age plotting contract test passed.")