# Settings, data locations and packages of the prosody statistical analysis.

required_packages <- c(
  "dplyr",
  "tidyr",
  "readr",
  "stringr",
  "purrr",
  "tibble",
  "ggplot2",
  "yaml",
  "readxl",
  "lme4",
  "lmerTest",
  "glmmTMB",
  "performance",
  "emmeans",
  "broom.mixed",
  "parameters",
  "boot",
  "knitr",
  "rmarkdown",
  "car"
)

analysis_runtime <- new.env(parent = emptyenv())

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

find_repo_root <- function(start = getwd()) {
  current <- normalizePath(start, winslash = "/", mustWork = FALSE)
  repeat {
    candidate <- file.path(current, "R", "00_setup.R")
    if (file.exists(candidate)) {
      return(current)
    }

    parent <- dirname(current)
    if (identical(parent, current)) {
      stop("Could not locate repository root containing R/00_setup.R.", call. = FALSE)
    }
    current <- parent
  }
}

repo_root <- local({
  # The folder of this analysis is found from the location of this file.
  source_files <- lapply(sys.frames(), function(frame) frame$ofile)
  source_files <- Filter(function(path) !is.null(path) && basename(path) == "00_setup.R", source_files)
  if (length(source_files)) find_repo_root(dirname(tail(source_files, 1)[[1]])) else find_repo_root()
})

# Settings of the analysis.
analysis_config <- yaml::read_yaml(file.path(repo_root, "params.yaml"), fileEncoding = "UTF-8")

# Data folders, read from config.yaml in the project root.
source(file.path(dirname(repo_root), "shared", "paths.R"), local = TRUE)
hbn_config <- hbn_read_config(dirname(repo_root))

# All results of this analysis are saved in one results folder.
results_dir <- hbn_results(hbn_config, "prosody-stats")

analysis_paths <- list(
  repo_root = repo_root,
  config_path = hbn_config$file,
  r_dir = file.path(repo_root, "R"),
  functions_dir = file.path(repo_root, "R", "functions"),
  data_derived = results_dir,
  acoustic_features = hbn_path(hbn_config, "prosody_features", analysis_config$files$acoustic_features),
  final_master = hbn_path(hbn_config, "statistical_data", analysis_config$files$final_master),
  input_long = hbn_path(hbn_config, "statistical_data", analysis_config$files$input_long),
  input_output_long = hbn_path(hbn_config, "statistical_data", analysis_config$files$input_output_long),
  participant_lookup = hbn_path(hbn_config, "metadata", analysis_config$files$participant_lookup),
  metadata_cleaned = hbn_path(hbn_config, "metadata", analysis_config$files$metadata_cleaned),
  outputs_models = file.path(results_dir, "models"),
  outputs_tables = file.path(results_dir, "tables"),
  outputs_diagnostics = file.path(results_dir, "diagnostics"),
  outputs_predictions = file.path(results_dir, "predictions"),
  outputs_plots = file.path(results_dir, "figures"),
  outputs_reports = results_dir
)

ensure_analysis_directories <- function(paths = analysis_paths) {
  dir_paths <- c(
    paths$data_derived,
    paths$outputs_models,
    paths$outputs_tables,
    paths$outputs_diagnostics,
    paths$outputs_predictions,
    paths$outputs_plots,
    paths$outputs_reports
  )

  for (directory in unique(dir_paths)) {
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  }

  invisible(paths)
}

analysis_options <- list(
  seed = analysis_config$analysis$random_seed %||% 123,
  confidence_level = analysis_config$analysis$confidence_level %||% 0.95,
  prediction_grid_size = analysis_config$analysis$prediction_grid_size %||% 60,
  monte_carlo_draws = analysis_config$analysis$monte_carlo_draws %||% 1000,
  nsim_default = analysis_config$analysis$monte_carlo_draws %||% 1000,
  overwrite_outputs = isTRUE(analysis_config$analysis$overwrite_outputs),
  model_robustness_mode = "strict",
  speaker_order = unlist(analysis_config$validation$expected_speakers %||% c("adult_female", "adult_male", "other_child")),
  primary_acoustic_measures = unlist(analysis_config$analysis$primary_acoustic_measures %||% character(0)),
  sensitivity_acoustic_measures = unlist(analysis_config$analysis$sensitivity_acoustic_measures %||% character(0)),
  age_specifications = list(
    family_1 = analysis_config$analysis$family_1_age_specification %||% "linear",
    family_2_count = analysis_config$analysis$family_2_count_age_specification %||% analysis_config$analysis$family_2_age_specification %||% "linear",
    family_2_duration = analysis_config$analysis$family_2_duration_age_specification %||% analysis_config$analysis$family_2_age_specification %||% "linear"
  ),
  age_days_mean = NA_real_,
  age_days_sd = NA_real_
)

parse_env_scalar <- function(name, default = "") {
  value <- Sys.getenv(name, unset = "")
  if (!nzchar(value)) {
    return(default)
  }
  trimws(value)
}

parse_env_vector <- function(value, split_pattern = "[;,]") {
  if (!nzchar(value)) {
    return(character(0))
  }

  parts <- trimws(unlist(strsplit(value, split_pattern)))
  unique(parts[nzchar(parts)])
}

normalize_runtime_mode <- function(value) {
  normalized <- trimws(tolower(as.character(value %||% "strict")))
  if (!(normalized %in% c("strict", "auto"))) {
    stop("model_robustness_mode must be one of: strict, auto.", call. = FALSE)
  }
  normalized
}

normalize_age_specification <- function(value) {
  normalized <- trimws(tolower(as.character(value %||% "linear")))
  if (!(normalized %in% c("none", "linear", "quadratic"))) {
    stop("Age specification must be one of: none, linear, quadratic.", call. = FALSE)
  }
  normalized
}

get_family_age_specification <- function(model_family) {
  configured <- switch(
    model_family,
    family_1 = analysis_options$age_specifications$family_1,
    family_2_count = analysis_options$age_specifications$family_2_count,
    family_2_duration = analysis_options$age_specifications$family_2_duration,
    stop(sprintf("Unknown model family: %s", model_family), call. = FALSE)
  )
  normalize_age_specification(configured %||% "linear")
}

active_age_terms <- function(model_family, age_specification = NULL) {
  resolved <- age_specification %||% get_family_age_specification(model_family)
  switch(
    normalize_age_specification(resolved),
    none = character(0),
    linear = c("age_z"),
    quadratic = c("age_z", "age_z2")
  )
}

inverse_transform_values <- function(values, transform_name = "none") {
  if (identical(transform_name, "none")) {
    return(values)
  }
  stop(sprintf("Unsupported transform '%s' in ProsodyStats.", transform_name), call. = FALSE)
}

analysis_runtime$age_days_mean <- analysis_options$age_days_mean
analysis_runtime$age_days_sd <- analysis_options$age_days_sd

apply_runtime_overrides <- function() {
  runtime_nsim <- parse_env_scalar("HBN_PROSODY_NSIM")
  if (nzchar(runtime_nsim)) {
    parsed_nsim <- suppressWarnings(as.integer(runtime_nsim))
    if (is.na(parsed_nsim) || parsed_nsim <= 0) {
      stop("Environment variable HBN_PROSODY_NSIM must be a positive integer.", call. = FALSE)
    }
    analysis_options$nsim_default <<- parsed_nsim
    analysis_options$monte_carlo_draws <<- parsed_nsim
  }

  runtime_confidence <- parse_env_scalar("HBN_PROSODY_CONFIDENCE_LEVEL")
  if (nzchar(runtime_confidence)) {
    parsed_confidence <- suppressWarnings(as.numeric(runtime_confidence))
    if (is.na(parsed_confidence) || parsed_confidence <= 0 || parsed_confidence >= 1) {
      stop("Environment variable HBN_PROSODY_CONFIDENCE_LEVEL must be between 0 and 1.", call. = FALSE)
    }
    analysis_options$confidence_level <<- parsed_confidence
  }

  runtime_mode <- parse_env_scalar("HBN_PROSODY_MODEL_ROBUSTNESS_MODE")
  if (nzchar(runtime_mode)) {
    analysis_options$model_robustness_mode <<- normalize_runtime_mode(runtime_mode)
  }

  runtime_primary <- parse_env_vector(parse_env_scalar("HBN_PROSODY_PRIMARY_MEASURES"))
  if (length(runtime_primary) > 0) {
    analysis_options$primary_acoustic_measures <<- runtime_primary
  }

  runtime_sensitivity <- parse_env_vector(parse_env_scalar("HBN_PROSODY_SENSITIVITY_MEASURES"))
  if (length(runtime_sensitivity) > 0) {
    analysis_options$sensitivity_acoustic_measures <<- runtime_sensitivity
  }

  runtime_family1_age <- parse_env_scalar("HBN_PROSODY_FAMILY1_AGE_SPEC")
  runtime_family2_count_age <- parse_env_scalar("HBN_PROSODY_FAMILY2_COUNT_AGE_SPEC")
  runtime_family2_duration_age <- parse_env_scalar("HBN_PROSODY_FAMILY2_DURATION_AGE_SPEC")

  if (nzchar(runtime_family1_age)) {
    analysis_options$age_specifications$family_1 <<- normalize_age_specification(runtime_family1_age)
  }
  if (nzchar(runtime_family2_count_age)) {
    analysis_options$age_specifications$family_2_count <<- normalize_age_specification(runtime_family2_count_age)
  }
  if (nzchar(runtime_family2_duration_age)) {
    analysis_options$age_specifications$family_2_duration <<- normalize_age_specification(runtime_family2_duration_age)
  }

  runtime_speaker_ref <- parse_env_scalar("HBN_PROSODY_SPEAKER_REF")
  runtime_ses_ref <- parse_env_scalar("HBN_PROSODY_SES_REF")
  runtime_sex_ref <- parse_env_scalar("HBN_PROSODY_CHILD_SEX_REF")
  runtime_city_ref <- parse_env_scalar("HBN_PROSODY_CITY_REF")

  if (nzchar(runtime_speaker_ref)) {
    analysis_config$reference_levels$speaker <<- runtime_speaker_ref
  }
  if (nzchar(runtime_ses_ref)) {
    analysis_config$reference_levels$SES <<- runtime_ses_ref
  }
  if (nzchar(runtime_sex_ref)) {
    analysis_config$reference_levels$child_sex <<- runtime_sex_ref
  }
  if (nzchar(runtime_city_ref)) {
    analysis_config$reference_levels$city <<- runtime_city_ref
  }
}

apply_runtime_overrides()

load_required_packages <- function(required = required_packages, attach = TRUE) {
  missing_packages <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      sprintf(
        "Missing required R packages: %s. Install them before running the workflow.",
        paste(missing_packages, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (isTRUE(attach)) {
    invisible(lapply(required, library, character.only = TRUE))
  }

  invisible(required)
}

source_analysis_file <- function(...) {
  source(file.path(repo_root, ...), local = FALSE)
}