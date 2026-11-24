# Checks the structure of the report. With --render it also knits the report with
# synthetic data in a temporary data folder (no study data is read).
# Run from the project root: Rscript --vanilla audio-stats/tests/test_single_report.R --render
local({
  args <- commandArgs(FALSE)
  script <- sub("^--file=", "", grep("^--file=", args, value = TRUE)[[1]])
  root <- normalizePath(file.path(dirname(script), ".."), winslash = "/", mustWork = TRUE)
  report <- file.path(root, "audio-stats.Rmd")
  lines <- readLines(report, warn = FALSE)
  text <- paste(lines, collapse = "\n")
  stopifnot(length(list.files(root, pattern = "[.]Rmd$")) == 1L,
            !any(file.exists(file.path(root, "R", c("01_input_models.R", "02_input_output_models.R",
                                                     "03_model_tables_and_plots.R", "04_final_report.R", "report.qmd")))),
            !file.exists(file.path(root, "R", "functions", "final_report.R")),
            length(gregexpr("fit_preregistered_model\\(", text)[[1]]) == 4L,
            !grepl("cache=TRUE", text, fixed = TRUE),
            match("  html_document:", lines) < match("  pdf_document:", lines))
  starts <- grep("^```\\{r", lines)
  labels <- sub("^```\\{r ([^,}]+).*", "\\1", lines[starts])
  stopifnot(!anyDuplicated(labels))
  for (start in starts) {
    end <- which(seq_along(lines) > start & lines == "```")[[1]]
    parse(text = lines[seq.int(start + 1L, end - 1L)])
  }
  for (file in list.files(file.path(root, "R"), pattern = "[.]R$", recursive = TRUE, full.names = TRUE)) parse(file)
  source(file.path(root, "R", "functions", "question_sections.R"), local = TRUE)
  effects <- data.frame(term = "age_z", estimate = -0.5, std.error = 0.1, statistic = -5,
                        df = 20, p.value = 0.0001, conf.low = -0.7, conf.high = -0.3)
  stopifnot(grepl("negative effect", paste(capture.output(render_effect_bullets(effects)), collapse = " ")))
  message("Single-report structural and numeric narrative tests passed.")
  if (!"--render" %in% commandArgs(TRUE)) return(invisible(NULL))

  old_env <- Sys.getenv()
  hbn <- grep("^HBN_", names(old_env), value = TRUE)
  old_options <- options()
  scratch <- tempfile("hbn-audio-stats-report-")
  on.exit({
    Sys.unsetenv(grep("^HBN_", names(Sys.getenv()), value = TRUE))
    if (length(hbn)) do.call(Sys.setenv, as.list(old_env[hbn]))
    options(old_options)
    unlink(scratch, recursive = TRUE)
  }, add = TRUE)
  Sys.unsetenv(hbn)
  dir.create(scratch)
  scratch <- normalizePath(scratch, winslash = "/", mustWork = TRUE)

  # Temporary config.yaml and data folder. The report is knitted from the source folder.
  writeLines("data_dir: data", file.path(scratch, "config.yaml"))
  Sys.setenv(HBN_CONFIG = file.path(scratch, "config.yaml"))
  derived <- file.path(scratch, "data", "derived", "statistical-data")
  results <- file.path(scratch, "data", "results", "audio-stats")
  dir.create(derived, recursive = TRUE)
  source_files <- function() {
    sort(c(list.files(root, all.files = TRUE, no.. = TRUE),
           list.files(file.path(root, "R"), recursive = TRUE, all.files = TRUE),
           list.files(file.path(root, "tests"), recursive = TRUE, all.files = TRUE)))
  }
  source_files_before <- source_files()

  set.seed(4321)
  n <- 72L
  master <- data.frame(participant_id = sprintf("S%03d", seq_len(n)), REC_date = "2025-01-01",
                       birthdate = "2024-01-01", child_sex = rep(c("female", "male"), n / 2),
                       mother_education = rep(c("high_school", "bachelor", "master"), length.out = n),
                       father_education = "bachelor", SES = rep(c("high_school", "bachelor", "master"), length.out = n),
                       Location = rep(c("A", "B", "C", "D"), length.out = n),
                       age_days = sample(200:850, n), recording_duration_hours = 8)
  master$age_months <- master$age_days / 30.44
  master$age_z <- as.numeric(scale(master$age_days))
  master$age_z2 <- master$age_z ^ 2
  master$key_child_count_hour <- 30 + 2 * master$age_z + rnorm(n, sd = 4)
  master$key_child_duration_hour <- 100 + 5 * master$age_z + rnorm(n, sd = 12)
  long <- master[rep(seq_len(n), each = 3), ]
  long$speaker <- rep(c("adult_female", "adult_male", "other_child"), n)
  participant_effect <- rep(rnorm(n, sd = 3), each = 3)
  long$input_count_hour <- 35 + rep(c(7, -4, 2), n) + 2 * long$age_z + participant_effect + rnorm(n * 3, sd = 5)
  long$input_duration_hour <- 130 + rep(c(15, -12, 4), n) + 5 * long$age_z + 2 * participant_effect + rnorm(n * 3, sd = 15)
  # The participant table also has the rates of each speaker (used by the descriptive statistics).
  for (speaker in c("adult_female", "adult_male", "other_child")) {
    master[[paste0(speaker, "_count_hour")]] <- long$input_count_hour[long$speaker == speaker]
    master[[paste0(speaker, "_duration_hour")]] <- long$input_duration_hour[long$speaker == speaker]
  }
  utils::write.csv(master, file.path(derived, "final_master.csv"), row.names = FALSE)
  utils::write.csv(long, file.path(derived, "input_long.csv"), row.names = FALSE)
  utils::write.csv(long, file.path(derived, "input_output_long.csv"), row.names = FALSE)
  utils::write.csv(data.frame(check = "synthetic_audit_marker", value = TRUE),
                   file.path(derived, "age_validation.csv"), row.names = FALSE)
  render <- function(refit, filename) {
    rmarkdown::render(report, output_format = "html_document", output_file = filename, output_dir = results,
                      params = list(refit_models = refit, nsim = 2L, show_code = FALSE),
                      envir = new.env(parent = globalenv()), quiet = TRUE)
  }
  html <- render(TRUE, "synthetic-refit.html")
  stopifnot(file.exists(file.path(results, "synthetic-refit.html")))
  models <- list.files(file.path(results, "models"), full.names = TRUE)
  stopifnot(length(models) == 4L)
  bundles <- lapply(models, readRDS)
  stopifnot(all(vapply(bundles, function(x) x$selection$model_type[[1]] == "lmm", logical(1))),
            sum(vapply(bundles, function(x) inherits(x$model, "merMod"), logical(1))) == 2L)
  required <- c("input_model_selection.csv", "input_output_model_selection.csv", "all_model_selection.csv",
                "dataset_summary.csv", "model_selection_summary.csv", "fixed_effects_summary.csv",
                "evidence_summary.csv", "research_question_summary.csv", "diagnostics_summary.csv",
                "anova_summary.csv", "collinearity_summary.csv", "prediction_manifest.csv", "reporting_bundle.rds",
                "combined_model_fit_statistics.csv", "combined_fixed_effects.csv", "combined_diagnostics_summary.csv")
  stopifnot(all(file.exists(file.path(results, "tables", required))),
            length(list.files(file.path(results, "predictions"), pattern = "csv$")) == 4L,
            length(list.files(file.path(results, "figures"), pattern = "png$")) == 8L,
            length(list.files(file.path(results, "diagnostics"), pattern = "png$")) == 8L)
  anova <- read.csv(file.path(results, "tables", "anova_summary.csv"))
  stopifnot("age_specification" %in% names(anova), all(anova$age_specification == "quadratic"))

  # Descriptive statistics: 9 tables and 19 figures in the results folder.
  source(file.path(root, "R", "functions", "descriptives.R"), local = TRUE)
  descriptive_table_files <- paste0(c("participant_summary", "missing_values_summary", "recording_duration_summary",
                                      "speaker_count_summary", "speaker_duration_summary", "age_summary",
                                      "sex_distribution", "location_distribution", "education_distribution"), ".csv")
  descriptive_figure_files <- paste0(descriptive_figure_names, ".png")
  descriptive_outputs_exist <- function() {
    identical(sort(list.files(file.path(results, "tables", "descriptive"))), sort(descriptive_table_files)) &&
      identical(sort(list.files(file.path(results, "figures", "descriptive"))), sort(descriptive_figure_files))
  }
  stopifnot(length(descriptive_figure_files) == 19L, descriptive_outputs_exist())
  sex_distribution <- read.csv(file.path(results, "tables", "descriptive", "sex_distribution.csv"))
  stopifnot(identical(sex_distribution$child_sex, c("female", "male")), identical(sex_distribution$count, c(36L, 36L)))

  # Pandoc wraps long HTML lines, so line breaks are replaced by spaces first.
  markup <- gsub("\\s+", " ", paste(readLines(html, warn = FALSE), collapse = " "))
  stopifnot(grepl("synthetic_audit_marker", markup, fixed = TRUE),
            grepl("Descriptive statistics", markup, fixed = TRUE),
            grepl("RQ4. Role of Individual Factors", markup, fixed = TRUE),
            length(gregexpr("<img ", markup, fixed = TRUE)[[1]]) >= 35L)
  before <- tools::md5sum(models)
  unlink(file.path(results, "figures"), recursive = TRUE)
  unlink(file.path(results, "predictions"), recursive = TRUE)
  render(FALSE, "synthetic-reload.html")
  stopifnot(identical(before, tools::md5sum(models)),
            length(list.files(file.path(results, "figures"), pattern = "png$")) == 8L,
            length(list.files(file.path(results, "predictions"), pattern = "csv$")) == 4L,
            descriptive_outputs_exist())

  # Knitting writes only into the results folder, never into the source folder.
  stopifnot(identical(source_files(), source_files_before))
  message("Synthetic HTML refit AND reload renders passed; four models, all output contracts, descriptive outputs, embedded images and routed audits verified.")
})