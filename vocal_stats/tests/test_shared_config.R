# Standalone: Rscript tests/test_shared_config.R (no testthat runner recursion).
local({
  run_tests <- function() {
    if (!requireNamespace("yaml", quietly = TRUE)) stop("Package 'yaml' is required.")
    source_files <- Filter(function(path) !is.null(path) && basename(path) == "test_shared_config.R",
                           lapply(sys.frames(), function(frame) frame$ofile))
    script <- if (length(source_files)) tail(source_files, 1)[[1]] else {
      sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
    }
    setup <- normalizePath(file.path(dirname(script), "..", "R", "00_setup.R"), winslash = "/", mustWork = TRUE)
    old_wd <- getwd()
    old_options <- options("contrasts")
    old_env <- Sys.getenv()
    hbn_names <- grep("^HBN_", names(old_env), value = TRUE)
    had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (had_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
    scratch <- tempfile("hbn-vocal-config-")
    on.exit({
      setwd(old_wd)
      options(old_options)
      Sys.unsetenv(grep("^HBN_", names(Sys.getenv()), value = TRUE))
      if (length(hbn_names)) do.call(Sys.setenv, as.list(old_env[hbn_names]))
      if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv) else {
        if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
      }
      unlink(scratch, recursive = TRUE)
    }, add = TRUE)
    Sys.unsetenv(hbn_names)
    dir.create(file.path(scratch, "code", "R"), recursive = TRUE)
    scratch <- normalizePath(scratch, winslash = "/", mustWork = TRUE)
    code_root <- file.path(scratch, "code")
    stopifnot(file.copy(setup, file.path(code_root, "R", "00_setup.R")))
    dir.create(file.path(scratch, "config"))
    dir.create(file.path(scratch, "external root"))
    dir.create(file.path(scratch, "elsewhere"))
    config_path <- file.path(scratch, "config", "shared.yaml")
    inventory <- function() sort(list.files(scratch, recursive = TRUE, all.files = TRUE, include.dirs = TRUE))
    load_setup <- function() {
      before <- inventory()
      env <- new.env(parent = globalenv())
      suppressMessages(source(file.path(code_root, "R", "00_setup.R"), local = env))
      stopifnot(identical(inventory(), before))
      env
    }
    write_config <- function(lines) writeLines(lines, config_path)
    expect_error <- function(action, pattern) {
      error <- tryCatch({ action(); NULL }, error = identity)
      stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error), fixed = TRUE))
    }
    assert_code <- function(env) {
      stopifnot(identical(env$repo_root, code_root),
                identical(env$analysis_paths$repo_root, code_root),
                identical(env$analysis_paths$r_dir, file.path(code_root, "R")),
                identical(env$analysis_paths$functions_dir, file.path(code_root, "R", "functions")))
      env$source <- function(file, local) file
      stopifnot(identical(env$source_analysis_file("R", "synthetic.R"), file.path(code_root, "R", "synthetic.R")))
    }

    setwd(file.path(scratch, "elsewhere"))
    baseline <- load_setup()
    assert_code(baseline)
    stopifnot(!dir.exists(file.path(code_root, "data", "derived")),
              identical(baseline$find_repo_root(file.path(code_root, "R")), code_root),
              identical(baseline$analysis_paths$data_dir, file.path(code_root, "data", "derived")),
              identical(baseline$analysis_paths$results_models, file.path(code_root, "results", "r_models")),
              identical(baseline$analysis_options$seed, 123),
              identical(baseline$analysis_options$nsim_default, 1000),
              identical(baseline$analysis_options$model_robustness_mode, "auto"))

    write_config(c("vocal:", "  project_root: ../external root", "  derived_data_dir: custom-derived",
                   "  results_dir: custom-results", "  seed: 999", "  anova_type: 3",
                   "prosody_stats:", "  paths:", "    data_derived: wrong-section"))
    Sys.setenv(HBN_CONFIG_PATH = config_path)
    shared <- load_setup()
    assert_code(shared)
    stopifnot(identical(shared$analysis_options, baseline$analysis_options),
              identical(shared$default_model_config, baseline$default_model_config),
              identical(shared$analysis_paths$data_dir, file.path(scratch, "external root", "custom-derived")))
    suffixes <- c(results_models = "r_models", results_tables = "r_tables", results_diagnostics = "r_diagnostics",
                  results_predictions = "r_predictions", results_plots = "r_plots", results_report = "final_report")
    for (name in names(suffixes)) {
      stopifnot(identical(shared$analysis_paths[[name]], file.path(scratch, "external root", "custom-results", suffixes[[name]])))
    }
    # Omitted project_root means code root, not YAML directory or working directory.
    write_config(c("vocal:", "  derived_data_dir: relative-derived", "  results_dir: relative-results"))
    default_root <- load_setup()
    stopifnot(identical(default_root$analysis_paths$data_dir, file.path(code_root, "relative-derived")))
    # An explicit relative HBN_CONFIG_PATH is code-root-relative, matching Python.
    Sys.setenv(HBN_CONFIG_PATH = "../config/shared.yaml")
    relative_config <- load_setup()
    stopifnot(identical(relative_config$analysis_paths, default_root$analysis_paths))
    Sys.setenv(HBN_CONFIG_PATH = config_path)
    yaml::write_yaml(list(vocal = list(project_root = file.path(scratch, "external root"),
                                     derived_data_dir = file.path(scratch, "absolute-derived"),
                                     results_dir = file.path(scratch, "absolute-results"))), config_path)
    absolute <- load_setup()
    stopifnot(identical(absolute$analysis_paths$data_dir, file.path(scratch, "absolute-derived")),
              identical(absolute$analysis_paths$results_models, file.path(scratch, "absolute-results", "r_models")))

    # All pre-existing scientific overrides still take precedence, independently of YAML.
    Sys.setenv(HBN_NSIM = "17", HBN_MODEL_ROBUSTNESS_MODE = "strict", HBN_ANOVA_TYPE = "3",
               HBN_AGE_SPECIFICATION = "linear", HBN_RESPONSE_TRANSFORM_GLOBAL = "none",
               HBN_FAMILY1_PARTICIPANT_INTERCEPT = "without", HBN_FAMILY2_PARTICIPANT_INTERCEPT = "without",
               HBN_FAMILY1_RANDOM_SLOPES = "speaker|participant_id")
    overridden <- load_setup()
    Sys.unsetenv("HBN_CONFIG_PATH")
    native_overridden <- load_setup()
    stopifnot(identical(overridden$analysis_options, native_overridden$analysis_options),
              identical(overridden$analysis_options$seed, 123),
              identical(overridden$analysis_options$nsim_default, 17L),
              identical(overridden$analysis_options$model_config$anova_type, 3L),
              identical(overridden$analysis_options$model_config$age_specification, "linear"),
              identical(overridden$analysis_options$model_config$response_transform, "none"),
              identical(overridden$analysis_options$model_spec_overrides$family_1$participant_intercept, "without"))
    Sys.unsetenv(grep("^HBN_", names(Sys.getenv()), value = TRUE))
    Sys.setenv(HBN_CONFIG_PATH = config_path)
    for (invalid in list(c("prosody_stats: {}"), c("vocal: null"), c("vocal: scalar"),
                         c("vocal: [one, two]"), c("- vocal"))) {
      write_config(invalid)
      expect_error(load_setup, "mapping")
    }
    write_config(c("vocal:", "  project_root: ''"))
    expect_error(load_setup, "non-empty")
    Sys.setenv(HBN_CONFIG_PATH = file.path(scratch, "missing.yaml"))
    expect_error(load_setup, "")
    stopifnot(identical(exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE), had_seed))
    if (had_seed) stopifnot(identical(get(".Random.seed", envir = .GlobalEnv), old_seed))
    message("Vocal shared config tests passed (synthetic fixtures only).")
  }
  run_tests()
})