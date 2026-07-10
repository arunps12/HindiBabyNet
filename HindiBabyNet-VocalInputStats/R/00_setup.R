# HindiBabyNet R analysis workflow bootstrap
#
# This file defines package requirements, repo-relative paths, and shared
# runtime helpers used by downstream scripts.

required_packages <- c(
  "tidyverse",
  "lme4",
  "lmerTest",
  "glmmTMB",
  "performance",
  "DHARMa",
  "emmeans",
  "ggeffects",
  "broom.mixed",
  "parameters",
  "boot",
  "ggplot2",
  "yaml",
  "knitr"
)

analysis_runtime <- new.env(parent = emptyenv())

find_repo_root <- function(start = getwd()) {
  current <- normalizePath(start, winslash = "/", mustWork = FALSE)
  repeat {
    candidate <- file.path(current, "data", "derived")
    if (dir.exists(candidate)) {
      return(current)
    }
    parent <- dirname(current)
    if (identical(parent, current)) {
      stop("Could not locate repository root containing data/derived.", call. = FALSE)
    }
    current <- parent
  }
}

repo_root <- find_repo_root()

analysis_paths <- list(
  repo_root = repo_root,
  r_dir = file.path(repo_root, "R"),
  functions_dir = file.path(repo_root, "R", "functions"),
  data_dir = file.path(repo_root, "data", "derived"),
  results_models = file.path(repo_root, "results", "r_models"),
  results_tables = file.path(repo_root, "results", "r_tables"),
  results_diagnostics = file.path(repo_root, "results", "r_diagnostics"),
  results_predictions = file.path(repo_root, "results", "r_predictions"),
  results_plots = file.path(repo_root, "results", "r_plots"),
  results_report = file.path(repo_root, "results", "final_report")
)

ensure_analysis_directories <- function(paths = analysis_paths) {
  dir_paths <- unname(unlist(paths[grepl("results_", names(paths))], use.names = FALSE))
  for (directory in dir_paths) {
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(paths)
}

analysis_options <- list(
  seed = 123,
  nsim_default = 1000,
  nsim_testing = 100,
  age_days_mean = NA_real_,
  age_days_sd = NA_real_,
  model_robustness_mode = "auto",
  family_2_participant_intercept = "auto",
  model_spec_overrides = list(),
  speaker_order = c("adult_female", "adult_male", "other_child")
)

parse_env_vector <- function(value, split_pattern = "[;,]") {
  if (!nzchar(value)) {
    return(character(0))
  }

  parts <- trimws(unlist(strsplit(value, split_pattern)))
  unique(parts[nzchar(parts)])
}

parse_env_toggle <- function(value, default = "with", allowed = c("with", "without", "auto")) {
  if (!nzchar(value)) {
    return(default)
  }

  parsed <- trimws(tolower(value))
  if (!(parsed %in% allowed)) {
    stop(
      sprintf(
        "Expected one of %s but received '%s'.",
        paste(allowed, collapse = ", "),
        value
      ),
      call. = FALSE
    )
  }

  parsed
}

build_model_spec_override <- function(prefix, participant_default) {
  list(
    participant_intercept = parse_env_toggle(
      Sys.getenv(sprintf("%s_PARTICIPANT_INTERCEPT", prefix), unset = ""),
      default = participant_default
    ),
    location_intercept = parse_env_toggle(
      Sys.getenv(sprintf("%s_LOCATION_INTERCEPT", prefix), unset = ""),
      default = "with"
    ),
    drop_fixed_terms = parse_env_vector(Sys.getenv(sprintf("%s_DROP_FIXED_TERMS", prefix), unset = "")),
    extra_fixed_terms = parse_env_vector(Sys.getenv(sprintf("%s_EXTRA_FIXED_TERMS", prefix), unset = "")),
    random_slopes = parse_env_vector(Sys.getenv(sprintf("%s_RANDOM_SLOPES", prefix), unset = ""))
  )
}

runtime_model_robustness <- Sys.getenv("HBN_MODEL_ROBUSTNESS_MODE", unset = "")
if (nzchar(runtime_model_robustness)) {
  analysis_options$model_robustness_mode <- parse_env_toggle(
    runtime_model_robustness,
    default = analysis_options$model_robustness_mode,
    allowed = c("auto", "strict")
  )
}

runtime_nsim <- Sys.getenv("HBN_NSIM", unset = "")
if (nzchar(runtime_nsim)) {
  parsed_nsim <- suppressWarnings(as.integer(runtime_nsim))
  if (is.na(parsed_nsim) || parsed_nsim <= 0) {
    stop("Environment variable HBN_NSIM must be a positive integer.", call. = FALSE)
  }
  analysis_options$nsim_default <- parsed_nsim
}

runtime_family_2_intercept <- Sys.getenv("HBN_FAMILY2_PARTICIPANT_INTERCEPT", unset = "")
if (nzchar(runtime_family_2_intercept)) {
  allowed_modes <- c("auto", "with", "without")
  parsed_mode <- trimws(tolower(runtime_family_2_intercept))
  if (!(parsed_mode %in% allowed_modes)) {
    stop(
      sprintf(
        "Environment variable HBN_FAMILY2_PARTICIPANT_INTERCEPT must be one of: %s.",
        paste(allowed_modes, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  analysis_options$family_2_participant_intercept <- parsed_mode
}

analysis_options$model_spec_overrides <- list(
  family_1 = build_model_spec_override("HBN_FAMILY1", participant_default = "with"),
  family_2 = build_model_spec_override(
    "HBN_FAMILY2",
    participant_default = analysis_options$family_2_participant_intercept
  )
)

analysis_runtime$age_days_mean <- analysis_options$age_days_mean
analysis_runtime$age_days_sd <- analysis_options$age_days_sd

load_required_packages <- function(required = required_packages, attach = TRUE) {
  missing_packages <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      sprintf(
        "Missing required R packages: %s. Install them or restore the renv environment before running the workflow.",
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

read_research_questions <- function(paths = analysis_paths) {
  yaml_path <- file.path(paths$r_dir, "research_questions.yaml")
  if (!file.exists(yaml_path)) {
    stop(sprintf("Research question YAML not found: %s", yaml_path), call. = FALSE)
  }
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package 'yaml' is required to read R/research_questions.yaml.", call. = FALSE)
  }
  yaml::read_yaml(yaml_path)
}

resolve_quarto_command <- function() {
  env_quarto <- Sys.getenv("QUARTO_PATH", unset = "")
  if (nzchar(env_quarto) && file.exists(env_quarto)) {
    return(normalizePath(env_quarto, winslash = "/", mustWork = TRUE))
  }

  discovered <- Sys.which("quarto")
  if (nzchar(discovered)) {
    return(discovered)
  }

  common_candidates <- c(
    "C:/Program Files/Quarto/bin/quarto.exe",
    "C:/Program Files/Quarto/bin/quarto.cmd",
    "C:/Users/arunps/AppData/Local/Programs/Quarto/bin/quarto.exe"
  )

  existing <- common_candidates[file.exists(common_candidates)]
  if (length(existing) > 0) {
    return(normalizePath(existing[[1]], winslash = "/", mustWork = TRUE))
  }

  ""
}

configure_quarto_runtime <- function() {
  quarto_command <- resolve_quarto_command()
  if (nzchar(quarto_command)) {
    Sys.setenv(QUARTO_PATH = quarto_command)
  }
  quarto_command
}

message("HindiBabyNet R analysis scaffold loaded.")
message(sprintf("Repository root: %s", repo_root))
if (Sys.getenv("HBN_NSIM", unset = "") != "") {
  message(sprintf("Runtime nsim override active: %s", analysis_options$nsim_default))
}
if (Sys.getenv("HBN_FAMILY2_PARTICIPANT_INTERCEPT", unset = "") != "") {
  message(sprintf("Runtime Family 2 participant intercept mode active: %s", analysis_options$family_2_participant_intercept))
}
if (Sys.getenv("HBN_MODEL_ROBUSTNESS_MODE", unset = "") != "") {
  message(sprintf("Runtime model robustness mode active: %s", analysis_options$model_robustness_mode))
}
