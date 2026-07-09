# HindiBabyNet R analysis workflow bootstrap
#
# Phase 1 scaffold only. This file defines package requirements, repo-relative
# paths, and output directories used by downstream scripts.

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
  age_days_sd = NA_real_
)

message("HindiBabyNet R analysis scaffold loaded.")
message(sprintf("Repository root: %s", repo_root))
