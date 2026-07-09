source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "data_prep.R"))
source(file.path("R", "functions", "monte_carlo_ci.R"))
source(file.path("R", "functions", "prediction_plots.R"))

datasets_bundle <- load_analysis_datasets(write_summary = FALSE)
prediction_grid <- build_age_speaker_grid(datasets_bundle$datasets$input_long, n_points = 5)

expected_age_z <- age_days_to_z(prediction_grid$age_days)
stopifnot(all(abs(expected_age_z - prediction_grid$age_z) < 1e-8))

validate_age_plot_data(prediction_grid)
message("Age plotting contract test passed.")