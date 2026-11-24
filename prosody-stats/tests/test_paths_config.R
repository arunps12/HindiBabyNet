# Checks that the analysis reads its data folders from config.yaml and its
# settings from params.yaml. Only temporary folders are used.
# Run from the project root: Rscript prosody-stats/tests/test_paths_config.R
local({
  script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  root <- if (length(script)) {
    dirname(dirname(normalizePath(gsub("~+~", " ", script[[1]], fixed = TRUE))))
  } else getwd()
  root <- normalizePath(root, winslash = "/", mustWork = TRUE)
  setup_file <- file.path(root, "R", "00_setup.R")
  stopifnot(file.exists(setup_file), file.exists(file.path(root, "params.yaml")))

  # All HBN_ environment variables are put back at the end.
  old_environment <- Sys.getenv()
  hbn_names <- grep("^HBN_", names(old_environment), value = TRUE)
  scratch <- tempfile("hbn-prosody-config-")
  on.exit({
    Sys.unsetenv(grep("^HBN_", names(Sys.getenv()), value = TRUE))
    if (length(hbn_names)) do.call(Sys.setenv, as.list(old_environment[hbn_names]))
    unlink(scratch, recursive = TRUE)
  }, add = TRUE)
  Sys.unsetenv(hbn_names)

  # Temporary project. data_dir is written relative to config.yaml and one
  # folder (statistical_data) is stored in another place.
  dir.create(file.path(scratch, "data"), recursive = TRUE)
  dir.create(file.path(scratch, "other", "statistical-data"), recursive = TRUE)
  dir.create(file.path(scratch, "other", "results"))
  scratch <- normalizePath(scratch, winslash = "/", mustWork = TRUE)
  config_file <- file.path(scratch, "config.yaml")
  writeLines(c("data_dir: data", "paths:", "  statistical_data: other/statistical-data"), config_file)
  Sys.setenv(HBN_CONFIG = config_file)

  load_setup <- function(file = setup_file) {
    env <- new.env(parent = globalenv())
    source(file, local = env)
    env
  }
  same_path <- function(found, ...) {
    identical(normalizePath(found, winslash = "/", mustWork = FALSE),
      normalizePath(file.path(...), winslash = "/", mustWork = FALSE))
  }
  fails_with <- function(text, file = setup_file) {
    error <- tryCatch({ load_setup(file); NULL }, error = identity)
    inherits(error, "error") && grepl(text, conditionMessage(error), fixed = TRUE)
  }

  env <- load_setup()
  paths <- env$analysis_paths
  data_dir <- file.path(scratch, "data")
  other_dir <- file.path(scratch, "other", "statistical-data")
  results <- file.path(data_dir, "results", "prosody-stats")
  stopifnot(
    setequal(names(paths), c("repo_root", "config_path", "r_dir", "functions_dir", "data_derived",
      "acoustic_features", "final_master", "input_long", "input_output_long", "participant_lookup",
      "metadata_cleaned", "outputs_models", "outputs_tables", "outputs_diagnostics",
      "outputs_predictions", "outputs_plots", "outputs_reports")),
    same_path(paths$repo_root, root),
    same_path(paths$config_path, config_file),
    same_path(paths$r_dir, root, "R"),
    same_path(paths$functions_dir, root, "R", "functions"),
    # Inputs in the default folders inside data_dir.
    same_path(paths$acoustic_features, data_dir, "derived", "prosody-features",
      "participant_speaker_acoustic_features.csv"),
    same_path(paths$participant_lookup, data_dir, "raw", "metadata", "participant_lookup.csv"),
    same_path(paths$metadata_cleaned, data_dir, "raw", "metadata", "metadata_cleaned.xlsx"),
    # Inputs in the folder that is stored in another place.
    same_path(paths$final_master, other_dir, "final_master.csv"),
    same_path(paths$input_long, other_dir, "input_long.csv"),
    same_path(paths$input_output_long, other_dir, "input_output_long.csv"),
    # All results in results/prosody-stats.
    same_path(paths$data_derived, results),
    same_path(paths$outputs_reports, results),
    same_path(paths$outputs_models, results, "models"),
    same_path(paths$outputs_tables, results, "tables"),
    same_path(paths$outputs_diagnostics, results, "diagnostics"),
    same_path(paths$outputs_predictions, results, "predictions"),
    same_path(paths$outputs_plots, results, "figures")
  )

  # Reading the settings does not create folders. They are created on request.
  stopifnot(!dir.exists(file.path(data_dir, "results")))
  env$ensure_analysis_directories()
  stopifnot(all(dir.exists(file.path(results, c("models", "tables", "diagnostics", "predictions", "figures")))))

  # Settings from params.yaml.
  settings <- env$analysis_options
  stopifnot(
    is.null(env$analysis_config$paths),
    identical(unlist(env$analysis_config$reference_levels),
      c(speaker = "adult_female", child_sex = "female", city = "CNB")),
    identical(unlist(env$analysis_config$ses_levels),
      c("primary_school", "high_school", "Intermediate", "bachelor", "master")),
    identical(settings$primary_acoustic_measures, c("median_F0_Hz", "robust_pitch_range_Hz",
      "wav2vec_PC1", "wav2vec_PC2", "wav2vec_PC3", "hubert_PC1", "hubert_PC2", "hubert_PC3")),
    identical(settings$sensitivity_acoustic_measures,
      c("mean_F0_Hz", "mean_F0_ST", "median_F0_ST", "robust_pitch_range_ST")),
    settings$monte_carlo_draws == 1000,
    settings$nsim_default == 1000,
    settings$seed == 123,
    settings$confidence_level == 0.95,
    settings$prediction_grid_size == 60,
    identical(settings$overwrite_outputs, FALSE),
    identical(settings$model_robustness_mode, "strict"),
    identical(settings$age_specifications,
      list(family_1 = "linear", family_2_count = "linear", family_2_duration = "linear")),
    identical(settings$speaker_order, c("adult_female", "adult_male", "other_child")),
    length(unlist(env$analysis_config$validation$acoustic_numeric_columns)) == 14L
  )

  # The settings of the report (HBN_PROSODY_ variables) still replace the defaults.
  Sys.setenv(HBN_PROSODY_NSIM = "17", HBN_PROSODY_SPEAKER_REF = "adult_male")
  changed <- load_setup()
  stopifnot(
    identical(changed$analysis_options$nsim_default, 17L),
    identical(changed$analysis_options$monte_carlo_draws, 17L),
    identical(changed$analysis_config$reference_levels$speaker, "adult_male"),
    changed$analysis_options$seed == 123
  )
  Sys.unsetenv(c("HBN_PROSODY_NSIM", "HBN_PROSODY_SPEAKER_REF"))

  # A results folder in another place is used for all outputs.
  writeLines(c("data_dir: data", "paths:", "  results: other/results"), config_file)
  moved <- load_setup()$analysis_paths
  stopifnot(
    same_path(moved$outputs_reports, scratch, "other", "results", "prosody-stats"),
    same_path(moved$outputs_plots, scratch, "other", "results", "prosody-stats", "figures"),
    same_path(moved$final_master, data_dir, "derived", "statistical-data", "final_master.csv")
  )

  # A wrong or missing config.yaml stops with a clear message.
  writeLines("paths: {}", config_file)
  stopifnot(fails_with("must contain data_dir"))
  Sys.setenv(HBN_CONFIG = file.path(scratch, "missing.yaml"))
  stopifnot(fails_with("Configuration file not found"))

  # Without HBN_CONFIG, config.yaml is read from the project root (the folder
  # above this analysis). A copy of the code in a temporary project shows this.
  Sys.unsetenv("HBN_CONFIG")
  project <- file.path(scratch, "project")
  dir.create(file.path(project, "prosody-stats", "R"), recursive = TRUE)
  dir.create(file.path(project, "shared"))
  dir.create(file.path(project, "study-data"))
  copied_setup <- file.path(project, "prosody-stats", "R", "00_setup.R")
  stopifnot(
    file.copy(setup_file, copied_setup),
    file.copy(file.path(root, "params.yaml"), file.path(project, "prosody-stats", "params.yaml")),
    file.copy(file.path(dirname(root), "shared", "paths.R"), file.path(project, "shared", "paths.R")),
    fails_with("Configuration file not found", copied_setup)
  )
  writeLines("data_dir: study-data", file.path(project, "config.yaml"))
  in_project <- load_setup(copied_setup)$analysis_paths
  stopifnot(
    same_path(in_project$repo_root, project, "prosody-stats"),
    same_path(in_project$config_path, project, "config.yaml"),
    same_path(in_project$outputs_models, project, "study-data", "results", "prosody-stats", "models")
  )

  message("Paths and settings test passed (temporary folders only).")
})
