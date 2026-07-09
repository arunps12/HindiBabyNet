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
  "merTools",
  "boot",
  "ggplot2",
  "quarto"
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
  speaker_order = c("adult_female", "adult_male", "other_child")
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

message("HindiBabyNet R analysis scaffold loaded.")
message(sprintf("Repository root: %s", repo_root))
