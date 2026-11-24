source(file.path("..", "R", "00_setup.R"))

testthat::test_that("params.yaml holds the settings and no data locations", {
  params_file <- file.path("..", "params.yaml")
  testthat::expect_true(file.exists(params_file))
  settings <- yaml::read_yaml(params_file, fileEncoding = "UTF-8")
  testthat::expect_false("paths" %in% names(settings))
  # No value is a folder or a path on one computer. Input files are given by name only.
  values <- as.character(unlist(settings))
  testthat::expect_false(any(grepl("[/\\\\~]|^[A-Za-z]:", values)))
  testthat::expect_setequal(names(settings$files), c("acoustic_features", "final_master", "input_long",
    "input_output_long", "participant_lookup", "metadata_cleaned"))
})

testthat::test_that("no file names one computer or the old configuration", {
  # Each word is written in two parts so that this file does not contain it.
  forbidden <- c(paste0("C:/", "Users"), paste0("/ho", "me/"), paste0("hypa", "tia"),
    paste0("HBN_CONFIG", "_PATH"), paste0("project", "_root"))
  files <- list.files("..", pattern = "\\.(R|Rmd|yaml|yml)$", recursive = TRUE,
    full.names = TRUE, ignore.case = TRUE)
  testthat::expect_true(length(files) > 0)
  for (path in files) {
    text <- paste(readLines(path, warn = FALSE), collapse = "\n")
    for (word in forbidden) {
      testthat::expect_false(grepl(word, text, fixed = TRUE, useBytes = TRUE), info = paste(path, word))
    }
  }
})

testthat::test_that("primary and sensitivity acoustic measures are configured", {
  testthat::expect_true("median_F0_Hz" %in% analysis_options$primary_acoustic_measures)
  testthat::expect_true(length(analysis_options$sensitivity_acoustic_measures) > 0)
})
