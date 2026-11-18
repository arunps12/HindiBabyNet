# Standalone: Rscript tests/test_shared_config.R. Never invoke test_dir here.
# The existing testthat.R runner intentionally retains its original three files.
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
    scratch <- tempfile("hbn-prosody-config-")
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
    dir.create(file.path(code_root, "config"))
    dir.create(file.path(scratch, "config"))
    dir.create(file.path(scratch, "external root"))
    dir.create(file.path(scratch, "elsewhere"))
    config_path <- file.path(scratch, "config", "shared.yaml")
    native <- list(paths = list(data_raw = "raw-native", acoustic_features = "features.csv"),
                   analysis = list(random_seed = 321L, confidence_level = 0.91, prediction_grid_size = 27L,
                                   monte_carlo_draws = 41L, overwrite_outputs = FALSE,
                                   primary_acoustic_measures = c("synthetic_primary"),
                                   sensitivity_acoustic_measures = c("synthetic_sensitivity"),
                                   family_1_age_specification = "quadratic", family_2_age_specification = "none"),
                   validation = list(expected_speakers = c("adult_female", "adult_male", "other_child")),
                   reference_levels = list(speaker = "adult_female", SES = "synthetic_SES"))
    yaml::write_yaml(native, file.path(code_root, "config", "config.yaml"))
    inventory <- function() sort(list.files(scratch, recursive = TRUE, all.files = TRUE, include.dirs = TRUE))
    load_setup <- function() {
      before <- inventory()
      env <- new.env(parent = globalenv())
      suppressMessages(source(file.path(code_root, "R", "00_setup.R"), local = env))
      stopifnot(identical(inventory(), before))
      env
    }
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
    stopifnot(identical(baseline$analysis_paths$data_raw, file.path(code_root, "raw-native")),
              identical(baseline$analysis_paths$data_derived, file.path(code_root, "data", "derived")),
              is.na(baseline$analysis_paths$final_master),
              baseline$analysis_options$seed == 321,
              baseline$analysis_options$nsim_default == 41,
              identical(baseline$analysis_options$model_robustness_mode, "strict"),
              identical(baseline$analysis_options$age_specifications$family_2_count, "none"))
    # Legacy flat env-selected YAML retains native-root resolution and all options.
    yaml::write_yaml(native, config_path)
    Sys.setenv(HBN_CONFIG_PATH = config_path)
    flat <- load_setup()
    stopifnot(identical(flat$analysis_config, baseline$analysis_config),
              identical(flat$analysis_options, baseline$analysis_options),
              identical(flat$analysis_paths$data_raw, baseline$analysis_paths$data_raw),
              identical(flat$analysis_paths$config_path, config_path))

    section <- native
    section$project_root <- "../external root"
    section$paths <- list(data_raw = "raw", data_derived = "derived", acoustic_features = "features.csv",
                          final_master = "master.csv", input_long = "input.csv", input_output_long = "io.csv",
                          participant_lookup = "lookup.csv", metadata_cleaned = "metadata.xlsx",
                          outputs_models = "models", outputs_tables = "tables", outputs_diagnostics = "diagnostics",
                          outputs_predictions = "predictions", outputs_plots = "plots", outputs_reports = "reports")
    yaml::write_yaml(list(prosody_stats = section, vocal = list(derived_data_dir = "wrong-section")), config_path)
    # Explicit shared configuration does not depend on a native config existing.
    unlink(file.path(code_root, "config", "config.yaml"))
    shared <- load_setup()
    yaml::write_yaml(native, file.path(code_root, "config", "config.yaml"))
    assert_code(shared)
    stopifnot(identical(shared$analysis_options, baseline$analysis_options),
              identical(shared$analysis_config, shared$read_analysis_config()),
              identical(shared$analysis_paths$config_path, config_path))
    for (name in names(section$paths)) {
      stopifnot(identical(shared$analysis_paths[[name]], file.path(scratch, "external root", section$paths[[name]])))
    }
    section$project_root <- NULL
    yaml::write_yaml(list(prosody_stats = section), config_path)
    default_root <- load_setup()
    stopifnot(identical(default_root$analysis_paths$data_raw, file.path(code_root, "raw")))
    Sys.setenv(HBN_CONFIG_PATH = "../config/shared.yaml")
    relative_config <- load_setup()
    stopifnot(identical(relative_config$analysis_paths$data_raw, default_root$analysis_paths$data_raw))
    Sys.setenv(HBN_CONFIG_PATH = config_path)
    section$project_root <- file.path(scratch, "external root")
    section$paths$data_raw <- file.path(scratch, "absolute-raw")
    yaml::write_yaml(list(prosody_stats = section), config_path)
    absolute <- load_setup()
    stopifnot(identical(absolute$analysis_paths$data_raw, file.path(scratch, "absolute-raw")),
              identical(absolute$analysis_paths$outputs_reports, file.path(scratch, "external root", "reports")))

    Sys.setenv(HBN_PROSODY_NSIM = "17", HBN_PROSODY_CONFIDENCE_LEVEL = "0.87",
               HBN_PROSODY_MODEL_ROBUSTNESS_MODE = "auto", HBN_PROSODY_PRIMARY_MEASURES = "test_a;test_b",
               HBN_PROSODY_SENSITIVITY_MEASURES = "test_c", HBN_PROSODY_FAMILY1_AGE_SPEC = "none",
               HBN_PROSODY_FAMILY2_COUNT_AGE_SPEC = "quadratic", HBN_PROSODY_FAMILY2_DURATION_AGE_SPEC = "linear",
               HBN_PROSODY_SPEAKER_REF = "adult_male", HBN_PROSODY_SES_REF = "test_ses",
               HBN_PROSODY_CHILD_SEX_REF = "test_sex", HBN_PROSODY_CITY_REF = "test_city")
    overridden <- load_setup()
    Sys.unsetenv("HBN_CONFIG_PATH")
    native_overridden <- load_setup()
    stopifnot(identical(overridden$analysis_options, native_overridden$analysis_options),
              identical(overridden$analysis_config$reference_levels, native_overridden$analysis_config$reference_levels),
              overridden$analysis_options$seed == 321,
              identical(overridden$analysis_options$nsim_default, 17L),
              identical(overridden$analysis_options$monte_carlo_draws, 17L),
              identical(overridden$analysis_options$confidence_level, 0.87),
              identical(overridden$analysis_options$model_robustness_mode, "auto"),
              identical(overridden$analysis_options$primary_acoustic_measures, c("test_a", "test_b")),
              identical(overridden$analysis_options$age_specifications$family_2_count, "quadratic"))
    Sys.unsetenv(grep("^HBN_", names(Sys.getenv()), value = TRUE))
    Sys.setenv(HBN_CONFIG_PATH = config_path)
    yaml::write_yaml(list(prosody_stats = list()), config_path)
    defaults <- load_setup()
    stopifnot(identical(defaults$analysis_options$seed, 123),
              identical(defaults$analysis_options$confidence_level, 0.95),
              identical(defaults$analysis_options$nsim_default, 1000),
              identical(defaults$analysis_options$age_specifications$family_1, "linear"))
    for (invalid in list("vocal: {}", "prosody_stats: null", "prosody_stats: scalar",
                         "prosody_stats: [one, two]", "- prosody_stats")) {
      writeLines(invalid, config_path)
      expect_error(load_setup, "mapping")
    }
    writeLines(c("prosody_stats:", "  project_root: ''"), config_path)
    expect_error(load_setup, "non-empty")
    Sys.setenv(HBN_CONFIG_PATH = file.path(scratch, "missing.yaml"))
    expect_error(load_setup, "")
    stopifnot(identical(exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE), had_seed))
    if (had_seed) stopifnot(identical(get(".Random.seed", envir = .GlobalEnv), old_seed))
    message("ProsodyStats shared config tests passed (synthetic fixtures only).")
  }
  run_tests()
})