# Run from the project root:  Rscript shared/tests/test_paths.R
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
shared_dir <- if (length(script)) dirname(dirname(normalizePath(script))) else file.path(getwd(), "shared")
source(file.path(shared_dir, "paths.R"))

check <- function(condition, label) {
  if (!isTRUE(condition)) stop(sprintf("FAILED: %s", label), call. = FALSE)
  cat(sprintf("ok: %s\n", label))
}
same <- function(a, b) identical(normalizePath(a, winslash = "/", mustWork = FALSE),
                                 normalizePath(b, winslash = "/", mustWork = FALSE))

root <- file.path(tempdir(), "hbn-paths-test")
unlink(root, recursive = TRUE)
dir.create(file.path(root, "project"), recursive = TRUE)
old_env <- Sys.getenv("HBN_CONFIG", unset = NA)
Sys.unsetenv("HBN_CONFIG")

project <- file.path(root, "project")
writeLines(c(
  "data_dir: ../study-data",
  "paths:",
  "  vtc: old place/VTC"
), file.path(project, "config.yaml"))
config <- hbn_read_config(project)

check(same(config$data_dir, file.path(root, "study-data")), "data_dir can be outside the project")
check(same(hbn_path(config, "statistical_data", "final_master.csv"),
           file.path(root, "study-data", "derived", "statistical-data", "final_master.csv")),
      "default folder is under data_dir")
check(same(hbn_path(config, "vtc"), file.path(project, "old place", "VTC")), "one folder can be stored somewhere else")
check(same(hbn_results(config, "audio-stats", "tables"),
           file.path(root, "study-data", "results", "audio-stats", "tables")), "results folder of a workflow")
check(inherits(try(hbn_input(config, "metadata", "x.xlsx"), silent = TRUE), "try-error"), "missing input is an error")
check(inherits(try(hbn_path(config, "nothing"), silent = TRUE), "try-error"), "unknown path name is an error")
check(inherits(try(hbn_read_config(root), silent = TRUE), "try-error"), "missing config.yaml is an error")

if (.Platform$OS.type == "windows") {
  check(identical(hbn_local_path("/mnt/d/study/raw", project), "D:/study/raw"), "Ubuntu mount path is converted on Windows")
  check(grepl("wsl.localhost/Ubuntu/home/me/data$", hbn_local_path("/home/me/data", project)), "Ubuntu home path is converted on Windows")
} else {
  check(identical(hbn_local_path("D:/study/raw", project), "/mnt/d/study/raw"), "Windows drive path is converted on Ubuntu")
}

other <- file.path(root, "other.yaml")
writeLines("data_dir: elsewhere", other)
Sys.setenv(HBN_CONFIG = other)
check(same(hbn_read_config(project)$data_dir, file.path(root, "elsewhere")), "HBN_CONFIG selects another config file")

if (is.na(old_env)) Sys.unsetenv("HBN_CONFIG") else Sys.setenv(HBN_CONFIG = old_env)
unlink(root, recursive = TRUE)
cat("shared/paths.R tests passed\n")
