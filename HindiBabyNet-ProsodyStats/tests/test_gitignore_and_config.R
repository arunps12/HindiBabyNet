source(file.path("..", "R", "00_setup.R"))

testthat::test_that("gitignore covers sensitive derived data and outputs", {
  lines <- readLines(file.path("..", ".gitignore"), warn = FALSE)
  required_patterns <- c(
    "data/raw/*",
    "data/derived/*",
    "outputs/models/*",
    "outputs/diagnostics/*",
    "outputs/predictions/*",
    "outputs/reports/*",
    "*.rds",
    "*.html",
    "*.pdf"
  )
  for (pattern in required_patterns) {
    testthat::expect_true(pattern %in% lines, info = pattern)
  }
})

testthat::test_that("primary and sensitivity acoustic measures are configured", {
  testthat::expect_true("median_F0_Hz" %in% analysis_options$primary_acoustic_measures)
  testthat::expect_true(length(analysis_options$sensitivity_acoustic_measures) > 0)
})
