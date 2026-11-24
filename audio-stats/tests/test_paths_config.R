# Tests the folders that R/00_setup.R takes from config.yaml.
# Only temporary files are used: no study data is read.
# Run from the project root: Rscript audio-stats/tests/test_shared_config.R
run_tests <- function() {
  if (!requireNamespace("yaml", quietly = TRUE)) stop("Package 'yaml' is required.")
  script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
  setup <- normalizePath(file.path(dirname(script), "..", "R", "00_setup.R"), winslash = "/", mustWork = TRUE)
  workflow_dir <- dirname(dirname(setup))

  # The environment and the options are restored at the end.
  old_env <- Sys.getenv()
  hbn_names <- grep("^HBN_", names(old_env), value = TRUE)
  old_options <- options("contrasts")
  scratch <- tempfile("hbn-audio-stats-config-")
  on.exit({
    Sys.unsetenv(grep("^HBN_", names(Sys.getenv()), value = TRUE))
    if (length(hbn_names)) do.call(Sys.setenv, as.list(old_env[hbn_names]))
    options(old_options)
    unlink(scratch, recursive = TRUE)
  }, add = TRUE)
  Sys.unsetenv(hbn_names)

  dir.create(file.path(scratch, "project"), recursive = TRUE)
  scratch <- normalizePath(scratch, winslash = "/", mustWork = TRUE)
  project <- file.path(scratch, "project")
  config_file <- file.path(project, "config.yaml")

  load_setup <- function(config_lines) {
    writeLines(config_lines, config_file)
    Sys.setenv(HBN_CONFIG = config_file)
    env <- new.env(parent = globalenv())
    suppressMessages(source(setup, local = env))
    env
  }
  same_path <- function(actual, expected) {
    identical(
      normalizePath(actual, winslash = "/", mustWork = FALSE),
      normalizePath(expected, winslash = "/", mustWork = FALSE)
    )
  }
  expect_folders <- function(env, data_dir, results_dir) {
    paths <- env$analysis_paths
    stopifnot(
      same_path(paths$data_dir, data_dir),
      same_path(paths$preprocessing_results, data_dir),
      same_path(paths$results_report, results_dir),
      same_path(paths$results_models, file.path(results_dir, "models")),
      same_path(paths$results_tables, file.path(results_dir, "tables")),
      same_path(paths$results_diagnostics, file.path(results_dir, "diagnostics")),
      same_path(paths$results_predictions, file.path(results_dir, "predictions")),
      same_path(paths$results_plots, file.path(results_dir, "figures"))
    )
  }
  expect_error <- function(action, pattern) {
    error <- tryCatch({ action(); NULL }, error = identity)
    stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error), fixed = TRUE))
  }

  # 1. Default folders under data_dir. The data folder is outside the project folder.
  study_data <- file.path(scratch, "study data")
  defaults <- load_setup(sprintf('data_dir: "%s"', study_data))
  expect_folders(
    defaults,
    data_dir = file.path(study_data, "derived", "statistical-data"),
    results_dir = file.path(study_data, "results", "audio-stats")
  )
  # The helper files are still taken from the workflow folder.
  stopifnot(
    same_path(defaults$repo_root, workflow_dir),
    same_path(defaults$analysis_paths$r_dir, file.path(workflow_dir, "R")),
    same_path(defaults$analysis_paths$functions_dir, file.path(workflow_dir, "R", "functions"))
  )
  # Reading the configuration creates no folder and no file.
  stopifnot(identical(list.files(scratch, recursive = TRUE), "project/config.yaml"))

  # 2. A relative data_dir starts from the folder of config.yaml.
  relative <- load_setup("data_dir: data")
  expect_folders(
    relative,
    data_dir = file.path(project, "data", "derived", "statistical-data"),
    results_dir = file.path(project, "data", "results", "audio-stats")
  )

  # 3. Folders stored in other places.
  override <- load_setup(c(
    sprintf('data_dir: "%s"', study_data),
    "paths:",
    sprintf('  statistical_data: "%s"', file.path(scratch, "other disk", "statistics")),
    "  results: my results"
  ))
  expect_folders(
    override,
    data_dir = file.path(scratch, "other disk", "statistics"),
    results_dir = file.path(project, "my results", "audio-stats")
  )

  # 4. The analysis options do not depend on the configuration file.
  for (env in list(defaults, relative, override)) {
    stopifnot(
      identical(env$analysis_options$seed, 123),
      identical(env$analysis_options$nsim_default, 1000),
      identical(env$analysis_options$model_robustness_mode, "auto"),
      identical(env$analysis_options$model_config$anova_type, 2L),
      all(unlist(env$analysis_options$model_config$age_specification) == "quadratic"),
      all(unlist(env$analysis_options$model_config$response_transform) == "log1p"),
      identical(env$analysis_options$speaker_order, c("adult_female", "adult_male", "other_child"))
    )
  }

  # 5. A missing or incomplete configuration file is an error.
  expect_error(function() load_setup("wsl_distro: Ubuntu"), "must contain data_dir")
  Sys.setenv(HBN_CONFIG = file.path(scratch, "missing.yaml"))
  expect_error(
    function() suppressMessages(source(setup, local = new.env(parent = globalenv()))),
    "Configuration file not found"
  )

  message("Configuration tests of 00_setup.R passed.")
}
run_tests()