# Tests the age values used in the prediction plots.
# Synthetic data only: no study data is read.
# Run from the project root: Rscript audio-stats/tests/test_age_plot_contract.R
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
root <- normalizePath(file.path(dirname(script), ".."), winslash = "/", mustWork = TRUE)

# 00_setup.R reads config.yaml. A temporary file is enough for this test.
config_file <- tempfile(fileext = ".yaml")
writeLines("data_dir: data", config_file)
old_config <- Sys.getenv("HBN_CONFIG", unset = NA)
Sys.setenv(HBN_CONFIG = config_file)

source(file.path(root, "R", "00_setup.R"))
source(file.path(root, "R", "functions", "data_prep.R"))
source(file.path(root, "R", "functions", "monte_carlo_ci.R"))
source(file.path(root, "R", "functions", "prediction_plots.R"))

if (is.na(old_config)) Sys.unsetenv("HBN_CONFIG") else Sys.setenv(HBN_CONFIG = old_config)
unlink(config_file)

analysis_runtime$age_days_mean <- 400
analysis_runtime$age_days_sd <- 100
synthetic_input <- data.frame(
  age_days = rep(c(300, 400, 500), each = 3),
  speaker = factor(rep(c("adult_female", "adult_male", "other_child"), 3)),
  SES_num = 3,
  child_sex = factor("female"),
  participant_id = factor(rep(c("S001", "S002", "S003"), each = 3)),
  Location = factor("A")
)

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

prediction_grid <- build_age_speaker_grid(quadratic_bundle, synthetic_input, n_points = 5)

expected_age_z <- age_days_to_z(prediction_grid$age_days)
stopifnot(all(abs(expected_age_z - prediction_grid$age_z) < 1e-8))
stopifnot(all(abs((prediction_grid$age_z ^ 2) - prediction_grid$age_z2) < 1e-8))

validate_age_plot_data(prediction_grid, bundle = quadratic_bundle)

prediction_grid_none <- build_age_speaker_grid(none_bundle, synthetic_input, n_points = 5)
stopifnot(!("age_z" %in% names(prediction_grid_none)))
stopifnot(!("age_z2" %in% names(prediction_grid_none)))
validate_age_plot_data(prediction_grid_none, bundle = none_bundle)

message("Age plotting contract test passed.")
