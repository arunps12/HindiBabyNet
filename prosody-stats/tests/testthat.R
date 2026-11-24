# Runs the small tests of the helper functions.
# Run from the project root: Rscript prosody-stats/tests/testthat.R
if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("Package 'testthat' is required to run the ProsodyStats tests.")
}

local({
  # Folder of this file, so that the tests can be started from any folder.
  script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  tests_dir <- if (length(script)) {
    dirname(normalizePath(gsub("~+~", " ", script[[1]], fixed = TRUE)))
  } else if (basename(getwd()) == "tests") getwd() else file.path(getwd(), "tests")

  # The tests use a temporary config.yaml and never the data of the study.
  scratch <- tempfile("hbn-prosody-tests-")
  dir.create(file.path(scratch, "data"), recursive = TRUE)
  writeLines("data_dir: data", file.path(scratch, "config.yaml"))
  old_config <- Sys.getenv("HBN_CONFIG", unset = NA_character_)
  old_wd <- setwd(tests_dir)
  on.exit({
    setwd(old_wd)
    if (is.na(old_config)) Sys.unsetenv("HBN_CONFIG") else Sys.setenv(HBN_CONFIG = old_config)
    unlink(scratch, recursive = TRUE)
  })
  Sys.setenv(HBN_CONFIG = file.path(scratch, "config.yaml"))
  # The other test files in this folder are run on their own with Rscript.
  testthat::test_dir(".", filter = "^(data_helpers|gitignore_and_config|model_helpers)$", reporter = "summary")
})
