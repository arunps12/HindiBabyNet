# Runs the whole report once on small made-up data in a temporary folder.
# No data of the study is read.
# Run from the project root: Rscript prosody-stats/tests/test_synthetic_render.R
# With --check at the end, only the needed packages are listed.
options(error = function() { cat("SYNTHETIC TEST ERROR: ", geterrmessage(), "\n"); traceback(3); quit(status = 1L) })
local({
  packages <- c("dplyr", "tidyr", "readr", "stringr", "purrr", "tibble", "ggplot2", "yaml", "readxl",
    "lme4", "lmerTest", "glmmTMB", "performance", "emmeans", "broom.mixed",
    "parameters", "boot", "knitr", "rmarkdown", "car", "testthat")
  available <- vapply(packages, requireNamespace, logical(1), quietly = TRUE)
  if ("--check" %in% commandArgs(TRUE)) {
    print(available)
    if (available[["rmarkdown"]]) cat("Pandoc available:", rmarkdown::pandoc_available(), "\n")
    return(invisible(NULL))
  }
  if (!all(available)) stop(paste("Synthetic render prerequisites missing:", paste(packages[!available], collapse = ", ")))
  if (!rmarkdown::pandoc_available()) stop("Synthetic render requires Pandoc.")
  script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
  root <- normalizePath(file.path(dirname(gsub("~+~", " ", script, fixed = TRUE)), ".."), winslash = "/")
  scratch <- tempfile("hbn-prosody-synthetic-")
  dir.create(scratch)
  old_environment <- Sys.getenv()
  hbn_names <- grep("^HBN_", names(old_environment), value = TRUE)
  on.exit({
    Sys.unsetenv(grep("^HBN_", names(Sys.getenv()), value = TRUE))
    if (length(hbn_names)) do.call(Sys.setenv, as.list(old_environment[hbn_names]))
    unlink(scratch, recursive = TRUE)
  }, add = TRUE)
  Sys.unsetenv(hbn_names)
  scratch <- normalizePath(scratch, winslash = "/")
  set.seed(1123)
  n <- 48L
  speakers <- c("adult_female", "adult_male", "other_child")
  master <- data.frame(participant_id = sprintf("synthetic_%03d", seq_len(n)),
    age_days = sample(200:800, n), child_sex = rep(c("female", "male"), n / 2),
    SES = rep(c("primary_school", "high_school", "Intermediate", "bachelor", "master", "master"), length.out = n),
    Location = rep(c("CNB", "OTHER", "OTHER", "CNB"), length.out = n))
  master$age_months <- master$age_days / 30.44
  master$age_z <- as.numeric(scale(master$age_days))
  master$age_z2 <- master$age_z ^ 2
  io <- master[rep(seq_len(n), each = 3L), ]
  io$speaker <- rep(speakers, n)
  # The analysis reads the squared age columns from this table.
  io$age_days_sq <- io$age_days ^ 2
  io$age_days_sq_z <- as.numeric(scale(io$age_days_sq))
  io$input_count_hour <- runif(nrow(io), 20, 100)
  io$input_count_min_hour <- io$input_count_hour / 60
  io$input_duration_hour <- runif(nrow(io), 1, 10)
  io$input_duration_min_hour <- io$input_duration_hour / 60
  io$key_child_count_hour <- rep(30 + rnorm(n, sd = 4), each = 3L)
  io$key_child_count_min_hour <- io$key_child_count_hour / 60
  io$key_child_duration_hour <- rep(5 + rnorm(n), each = 3L)
  io$key_child_duration_min_hour <- io$key_child_duration_hour / 60
  acoustic <- io[, c("participant_id", "speaker")]
  acoustic$source_participant_id <- acoustic$participant_id
  acoustic$median_F0_Hz <- 200 + rep(rnorm(n, sd = 15), each = 3L) +
    rep(c(10, -10, 20), n) + 3 * io$age_z + rnorm(nrow(io), sd = 12)
  acoustic$wav2vec_PC1 <- rnorm(nrow(io))
  # The other acoustic columns named in params.yaml get random values.
  settings <- yaml::read_yaml(file.path(root, "params.yaml"), fileEncoding = "UTF-8")
  for (column in setdiff(unlist(settings$validation$acoustic_numeric_columns), names(acoustic))) {
    acoustic[[column]] <- rnorm(nrow(io))
  }
  lookup <- data.frame(participant_id = master$participant_id, original_par_id = master$participant_id)
  input <- io[, c("participant_id", "speaker", "input_count_hour", "input_count_min_hour",
    "input_duration_hour", "input_duration_min_hour")]

  # Temporary project: config.yaml and the input files in the default folders.
  # The optional metadata file is left out.
  data_dir <- file.path(scratch, "data")
  write_input <- function(table, folder, file_name) {
    dir.create(file.path(data_dir, folder), recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(table, file.path(data_dir, folder, file_name), row.names = FALSE)
  }
  write_input(acoustic, "derived/prosody-features", settings$files$acoustic_features)
  write_input(master, "derived/statistical-data", settings$files$final_master)
  write_input(input, "derived/statistical-data", settings$files$input_long)
  write_input(io, "derived/statistical-data", settings$files$input_output_long)
  write_input(lookup, "raw/metadata", settings$files$participant_lookup)
  writeLines("data_dir: data", file.path(scratch, "config.yaml"))
  Sys.setenv(HBN_CONFIG = file.path(scratch, "config.yaml"))
  results_dir <- file.path(data_dir, "results", "prosody-stats")
  work_dir <- file.path(scratch, "work")
  dir.create(work_dir)

  code_files <- function() sort(list.files(root, recursive = TRUE, all.files = TRUE))
  code_files_before <- code_files()
  render <- function(parameters, name) {
    rmarkdown::render(file.path(root, "prosody-stats.Rmd"), params = parameters,
      output_file = name, output_dir = results_dir, intermediates_dir = work_dir,
      envir = new.env(parent = globalenv()), quiet = TRUE, clean = TRUE)
  }
  parameters <- list(nsim = 2L, primary_acoustic_measures = "median_F0_Hz",
    sensitivity_acoustic_measures = "mean_F0_Hz", show_code = FALSE)

  # Without saved results, a report without a new run stops with a clear message.
  no_results <- tryCatch(render(c(parameters, list(run_workflow = FALSE)), "no-results.html"), error = identity)
  stopifnot(inherits(no_results, "error"), grepl("No saved results", conditionMessage(no_results)))

  # Full run: the report and all outputs are saved in the results folder.
  report <- render(parameters, "prosody-stats.html")
  models <- list.files(file.path(results_dir, "models"), pattern = "\\.rds$", full.names = TRUE)
  stopifnot(file.exists(report),
    file.exists(file.path(results_dir, "prosody-stats.html")),
    file.exists(file.path(results_dir, "prosody_long.csv")),
    all(file.exists(file.path(results_dir, "tables",
      c("family1_manifest.csv", "family2_manifest.csv", "diagnostics_summary.csv")))),
    length(models) == 3L,
    all(vapply(lapply(models, readRDS), function(bundle) inherits(bundle$model, "merMod"), logical(1))),
    length(list.files(file.path(results_dir, "predictions"), pattern = "_predictions\\.csv$")) == 3L,
    length(list.files(file.path(results_dir, "diagnostics"), pattern = "_diagnostics_summary\\.csv$")) == 3L,
    length(list.files(file.path(results_dir, "figures"), pattern = "\\.png$")) == 6L)

  # A second report without a new run is built from the saved outputs.
  # The saved models are not changed.
  before <- tools::md5sum(models)
  saved_report <- render(c(parameters, list(run_workflow = FALSE)), "prosody-stats-saved.html")
  stopifnot(file.exists(saved_report), identical(before, tools::md5sum(models)))

  # Nothing was written in the folder of the code.
  stopifnot(identical(code_files_before, code_files()))
  message("Report test passed: 3 models fitted on made-up data, outputs saved in results/prosody-stats, report rebuilt from saved outputs.")
})