if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("Package 'testthat' is required to run the ProsodyStats tests.")
}

if (basename(getwd()) != "tests" && dir.exists("tests")) {
  setwd("tests")
}

# Exclude this runner itself: test_dir's default also matches testthat.R.
testthat::test_dir(".", filter = "^(data_helpers|gitignore_and_config|model_helpers)$", reporter = "summary")
