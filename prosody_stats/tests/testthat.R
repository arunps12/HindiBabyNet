if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("Package 'testthat' is required to run the ProsodyStats tests.")
}

if (basename(getwd()) != "tests" && dir.exists("tests")) {
  setwd("tests")
}

# Portable tests use the versioned example, never a private local configuration.
if (!nzchar(Sys.getenv("HBN_CONFIG_PATH"))) {
  Sys.setenv(HBN_CONFIG_PATH = normalizePath(file.path("..", "..", "config.example.yaml")))
}

# Exclude this runner itself: test_dir's default also matches testthat.R.
testthat::test_dir(".", filter = "^(data_helpers|gitignore_and_config|model_helpers)$", reporter = "summary")
