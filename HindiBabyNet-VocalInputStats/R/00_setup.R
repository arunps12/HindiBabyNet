# HindiBabyNet R analysis workflow bootstrap
#
# This file defines package requirements, repo-relative paths, and shared
# runtime helpers used by downstream scripts.

required_packages <- c(
  "tidyverse",
  "lme4",
  "lmerTest",
  "glmmTMB",
  "performance",
  "DHARMa",
  "emmeans",
  "ggeffects",
  "broom.mixed",
  "parameters",
  "boot",
  "ggplot2",
  "yaml",
  "knitr",
  "car"
)

analysis_runtime <- new.env(parent = emptyenv())

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

find_repo_root <- function(start = getwd()) {
  current <- normalizePath(start, winslash = "/", mustWork = FALSE)
  repeat {
    candidate <- file.path(current, "data", "derived")
    if (dir.exists(candidate)) {
      return(current)
    }
    parent <- dirname(current)
    if (identical(parent, current)) {
      stop("Could not locate repository root containing data/derived.", call. = FALSE)
    }
    current <- parent
  }
}

repo_root <- find_repo_root()

analysis_paths <- list(
  repo_root = repo_root,
  r_dir = file.path(repo_root, "R"),
  functions_dir = file.path(repo_root, "R", "functions"),
  data_dir = file.path(repo_root, "data", "derived"),
  results_models = file.path(repo_root, "results", "r_models"),
  results_tables = file.path(repo_root, "results", "r_tables"),
  results_diagnostics = file.path(repo_root, "results", "r_diagnostics"),
  results_predictions = file.path(repo_root, "results", "r_predictions"),
  results_plots = file.path(repo_root, "results", "r_plots"),
  results_report = file.path(repo_root, "results", "final_report")
)

ensure_analysis_directories <- function(paths = analysis_paths) {
  dir_paths <- unname(unlist(paths[grepl("results_", names(paths))], use.names = FALSE))
  for (directory in dir_paths) {
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(paths)
}

analysis_options <- list(
  seed = 123,
  nsim_default = 1000,
  nsim_testing = 100,
  age_days_mean = NA_real_,
  age_days_sd = NA_real_,
  model_robustness_mode = "auto",
  family_2_participant_intercept = "auto",
  model_spec_overrides = list(),
  speaker_order = c("adult_female", "adult_male", "other_child"),
  model_config = NULL
)

default_model_config <- list(
  age_specification = list(
    family1_count = "quadratic",
    family1_duration = "quadratic",
    family2_count = "quadratic",
    family2_duration = "quadratic"
  ),
  response_transform = list(
    input_count_hour = "log1p",
    input_duration_min_hour = "log1p",
    key_child_count_hour = "log1p",
    key_child_duration_min_hour = "log1p"
  ),
  anova_type = 2
)

normalize_age_specification <- function(value) {
  normalized <- trimws(tolower(as.character(value %||% "quadratic")))
  if (!(normalized %in% c("none", "linear", "quadratic"))) {
    stop("age_specification must be one of: none, linear, quadratic.", call. = FALSE)
  }
  normalized
}

normalize_response_transform_name <- function(value) {
  normalized <- trimws(tolower(as.character(value %||% "none")))
  if (!(normalized %in% c("none", "log", "log1p"))) {
    stop("response_transform must be one of: none, log, log1p.", call. = FALSE)
  }
  normalized
}

normalize_anova_type <- function(value) {
  parsed <- suppressWarnings(as.integer(value %||% 2L))
  if (is.na(parsed) || !(parsed %in% c(2L, 3L))) {
    stop("anova_type must be 2 or 3.", call. = FALSE)
  }
  parsed
}

parse_named_transform_map <- function(value) {
  if (!nzchar(value)) {
    return(list())
  }

  pieces <- trimws(unlist(strsplit(value, "[;,]")))
  pieces <- pieces[nzchar(pieces)]
  transform_map <- list()
  for (piece in pieces) {
    assignment <- trimws(unlist(strsplit(piece, "=", fixed = TRUE)))
    if (length(assignment) != 2 || !all(nzchar(assignment))) {
      stop(sprintf("Invalid response transform mapping: %s", piece), call. = FALSE)
    }
    transform_map[[assignment[[1]]]] <- normalize_response_transform_name(assignment[[2]])
  }
  transform_map
}

parse_named_age_spec_map <- function(value) {
  if (!nzchar(value)) {
    return(list())
  }

  pieces <- trimws(unlist(strsplit(value, "[;,]")))
  pieces <- pieces[nzchar(pieces)]
  age_spec_map <- list()
  for (piece in pieces) {
    assignment <- trimws(unlist(strsplit(piece, "=", fixed = TRUE)))
    if (length(assignment) != 2 || !all(nzchar(assignment))) {
      stop(sprintf("Invalid age specification mapping: %s", piece), call. = FALSE)
    }
    age_spec_map[[assignment[[1]]]] <- normalize_age_specification(assignment[[2]])
  }
  age_spec_map
}

normalize_age_specification_spec <- function(value) {
  if (is.null(value) || length(value) == 0) {
    return("quadratic")
  }

  if (is.list(value)) {
    if (is.null(names(value)) || !length(names(value))) {
      stop("Named age_specification lists must name each model.", call. = FALSE)
    }
    normalized <- lapply(value, normalize_age_specification)
    return(normalized)
  }

  if (length(value) == 1 && is.null(names(value))) {
    return(normalize_age_specification(value))
  }

  named_value <- as.list(value)
  if (is.null(names(named_value)) || !length(names(named_value))) {
    stop("age_specification vectors with multiple values must be named by model.", call. = FALSE)
  }
  lapply(named_value, normalize_age_specification)
}

normalize_response_transform_spec <- function(value) {
  if (is.null(value) || length(value) == 0) {
    return("none")
  }

  if (is.list(value)) {
    if (is.null(names(value)) || !length(names(value))) {
      stop("Named response_transform lists must name each outcome.", call. = FALSE)
    }
    normalized <- lapply(value, normalize_response_transform_name)
    return(normalized)
  }

  if (length(value) == 1 && is.null(names(value))) {
    return(normalize_response_transform_name(value))
  }

  named_value <- as.list(value)
  if (is.null(names(named_value)) || !length(names(named_value))) {
    stop("response_transform vectors with multiple values must be named by outcome.", call. = FALSE)
  }
  lapply(named_value, normalize_response_transform_name)
}

normalize_model_config <- function(config) {
  age_specification <- normalize_age_specification_spec(config$age_specification %||% "quadratic")
  response_transform <- normalize_response_transform_spec(config$response_transform %||% "none")
  list(
    age_specification = age_specification,
    response_transform = response_transform,
    anova_type = normalize_anova_type(config$anova_type %||% 2L)
  )
}

model_age_spec_key <- function(model_family, response_name) {
  if (identical(model_family, "family_1") && identical(response_name, "input_count_hour")) {
    return("family1_count")
  }
  if (identical(model_family, "family_1") && identical(response_name, "input_duration_min_hour")) {
    return("family1_duration")
  }
  if (identical(model_family, "family_2_count") && identical(response_name, "key_child_count_hour")) {
    return("family2_count")
  }
  if (identical(model_family, "family_2_duration") && identical(response_name, "key_child_duration_min_hour")) {
    return("family2_duration")
  }

  stop(
    sprintf("Could not resolve age specification key for model_family='%s', response_name='%s'.", model_family, response_name),
    call. = FALSE
  )
}

get_age_specification_name <- function(response_name = NULL, model_family = NULL, config = analysis_options$model_config) {
  age_specification <- config$age_specification %||% "quadratic"
  if (!is.list(age_specification)) {
    return(normalize_age_specification(age_specification))
  }

  if (is.null(response_name) || is.null(model_family)) {
    stop("response_name and model_family are required when age_specification is configured per model.", call. = FALSE)
  }

  key <- model_age_spec_key(model_family = model_family, response_name = response_name)
  normalize_age_specification(age_specification[[key]] %||% "quadratic")
}

get_response_transform_name <- function(response_name, config = analysis_options$model_config) {
  transform_spec <- config$response_transform %||% "none"
  if (is.list(transform_spec)) {
    selected <- transform_spec[[response_name]] %||% "none"
    return(normalize_response_transform_name(selected))
  }
  normalize_response_transform_name(transform_spec)
}

active_age_specification <- function(response_name = NULL, model_family = NULL, config = analysis_options$model_config) {
  get_age_specification_name(response_name = response_name, model_family = model_family, config = config)
}

active_age_terms <- function(response_name = NULL, model_family = NULL, age_specification = NULL, config = analysis_options$model_config) {
  resolved_age_specification <- age_specification %||% active_age_specification(
    response_name = response_name,
    model_family = model_family,
    config = config
  )

  switch(
    normalize_age_specification(resolved_age_specification),
    none = character(0),
    linear = c("age_z"),
    quadratic = c("age_z", "age_z2")
  )
}

requires_age_predictors <- function(response_name = NULL, model_family = NULL, age_specification = NULL, config = analysis_options$model_config) {
  length(active_age_terms(
    response_name = response_name,
    model_family = model_family,
    age_specification = age_specification,
    config = config
  )) > 0
}

age_interaction_terms <- function(base_term, response_name = NULL, model_family = NULL, age_specification = NULL, config = analysis_options$model_config) {
  terms <- active_age_terms(
    response_name = response_name,
    model_family = model_family,
    age_specification = age_specification,
    config = config
  )
  if (length(terms) == 0) {
    return(character(0))
  }
  sprintf("%s:%s", base_term, terms)
}

format_age_specification_summary <- function(config = analysis_options$model_config) {
  age_specification <- config$age_specification %||% "quadratic"
  if (!is.list(age_specification)) {
    return(normalize_age_specification(age_specification))
  }

  pieces <- vapply(names(age_specification), function(name) {
    sprintf("%s=%s", name, normalize_age_specification(age_specification[[name]]))
  }, character(1))
  paste(pieces, collapse = "; ")
}

validate_response_transform <- function(values, transform_name, response_name) {
  finite_values <- values[is.finite(values)]
  if (length(finite_values) == 0) {
    return(invisible(TRUE))
  }

  if (identical(transform_name, "log") && any(finite_values <= 0)) {
    stop(sprintf("Response '%s' includes non-positive values, which is invalid for log transformation.", response_name), call. = FALSE)
  }
  if (identical(transform_name, "log1p") && any(finite_values < 0)) {
    stop(sprintf("Response '%s' includes negative values, which is invalid for log1p transformation.", response_name), call. = FALSE)
  }
  invisible(TRUE)
}

transform_response_values <- function(values, transform_name, response_name) {
  validate_response_transform(values, transform_name, response_name)
  switch(
    normalize_response_transform_name(transform_name),
    none = values,
    log = log(values),
    log1p = log1p(values)
  )
}

inverse_transform_values <- function(values, transform_name) {
  switch(
    normalize_response_transform_name(transform_name),
    none = values,
    log = exp(values),
    log1p = expm1(values)
  )
}

apply_model_contrasts <- function(config = analysis_options$model_config) {
  if (normalize_anova_type(config$anova_type) == 3L) {
    options(contrasts = c("contr.sum", "contr.poly"))
  } else {
    options(contrasts = c("contr.treatment", "contr.poly"))
  }
  invisible(getOption("contrasts"))
}

current_contrast_label <- function(config = analysis_options$model_config) {
  if (normalize_anova_type(config$anova_type) == 3L) {
    "contr.sum"
  } else {
    "contr.treatment"
  }
}

apply_model_config_overrides <- function(config = default_model_config) {
  updated <- config
  runtime_age_specification <- Sys.getenv("HBN_AGE_SPECIFICATION", unset = "")
  runtime_age_specification_map <- Sys.getenv("HBN_AGE_SPECIFICATION_MAP", unset = "")
  if (nzchar(runtime_age_specification)) {
    updated$age_specification <- runtime_age_specification
  } else if (nzchar(runtime_age_specification_map)) {
    merged_age_specs <- if (is.list(updated$age_specification)) updated$age_specification else list()
    updated$age_specification <- utils::modifyList(merged_age_specs, parse_named_age_spec_map(runtime_age_specification_map))
  }

  runtime_anova_type <- Sys.getenv("HBN_ANOVA_TYPE", unset = "")
  if (nzchar(runtime_anova_type)) {
    updated$anova_type <- runtime_anova_type
  }

  runtime_global_transform <- Sys.getenv("HBN_RESPONSE_TRANSFORM_GLOBAL", unset = "")
  runtime_transform_map <- Sys.getenv("HBN_RESPONSE_TRANSFORM_MAP", unset = "")
  if (nzchar(runtime_global_transform)) {
    updated$response_transform <- runtime_global_transform
  } else if (nzchar(runtime_transform_map)) {
    merged_map <- if (is.list(updated$response_transform)) updated$response_transform else list()
    updated$response_transform <- utils::modifyList(merged_map, parse_named_transform_map(runtime_transform_map))
  }

  normalized <- normalize_model_config(updated)
  analysis_options$model_config <<- normalized
  apply_model_contrasts(normalized)
  normalized
}

parse_env_vector <- function(value, split_pattern = "[;,]") {
  if (!nzchar(value)) {
    return(character(0))
  }

  parts <- trimws(unlist(strsplit(value, split_pattern)))
  unique(parts[nzchar(parts)])
}

parse_env_toggle <- function(value, default = "with", allowed = c("with", "without", "auto")) {
  if (!nzchar(value)) {
    return(default)
  }

  parsed <- trimws(tolower(value))
  if (!(parsed %in% allowed)) {
    stop(
      sprintf(
        "Expected one of %s but received '%s'.",
        paste(allowed, collapse = ", "),
        value
      ),
      call. = FALSE
    )
  }

  parsed
}

build_model_spec_override <- function(prefix, participant_default) {
  list(
    participant_intercept = parse_env_toggle(
      Sys.getenv(sprintf("%s_PARTICIPANT_INTERCEPT", prefix), unset = ""),
      default = participant_default
    ),
    location_intercept = parse_env_toggle(
      Sys.getenv(sprintf("%s_LOCATION_INTERCEPT", prefix), unset = ""),
      default = "with"
    ),
    drop_fixed_terms = parse_env_vector(Sys.getenv(sprintf("%s_DROP_FIXED_TERMS", prefix), unset = "")),
    extra_fixed_terms = parse_env_vector(Sys.getenv(sprintf("%s_EXTRA_FIXED_TERMS", prefix), unset = "")),
    random_slopes = parse_env_vector(Sys.getenv(sprintf("%s_RANDOM_SLOPES", prefix), unset = ""))
  )
}

runtime_model_robustness <- Sys.getenv("HBN_MODEL_ROBUSTNESS_MODE", unset = "")
if (nzchar(runtime_model_robustness)) {
  analysis_options$model_robustness_mode <- parse_env_toggle(
    runtime_model_robustness,
    default = analysis_options$model_robustness_mode,
    allowed = c("auto", "strict")
  )
}

runtime_nsim <- Sys.getenv("HBN_NSIM", unset = "")
if (nzchar(runtime_nsim)) {
  parsed_nsim <- suppressWarnings(as.integer(runtime_nsim))
  if (is.na(parsed_nsim) || parsed_nsim <= 0) {
    stop("Environment variable HBN_NSIM must be a positive integer.", call. = FALSE)
  }
  analysis_options$nsim_default <- parsed_nsim
}

runtime_family_2_intercept <- Sys.getenv("HBN_FAMILY2_PARTICIPANT_INTERCEPT", unset = "")
if (nzchar(runtime_family_2_intercept)) {
  allowed_modes <- c("auto", "with", "without")
  parsed_mode <- trimws(tolower(runtime_family_2_intercept))
  if (!(parsed_mode %in% allowed_modes)) {
    stop(
      sprintf(
        "Environment variable HBN_FAMILY2_PARTICIPANT_INTERCEPT must be one of: %s.",
        paste(allowed_modes, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  analysis_options$family_2_participant_intercept <- parsed_mode
}

analysis_options$model_spec_overrides <- list(
  family_1 = build_model_spec_override("HBN_FAMILY1", participant_default = "with"),
  family_2 = build_model_spec_override(
    "HBN_FAMILY2",
    participant_default = analysis_options$family_2_participant_intercept
  )
)

apply_model_config_overrides(default_model_config)

analysis_runtime$age_days_mean <- analysis_options$age_days_mean
analysis_runtime$age_days_sd <- analysis_options$age_days_sd

load_required_packages <- function(required = required_packages, attach = TRUE) {
  missing_packages <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      sprintf(
        "Missing required R packages: %s. Install them or restore the renv environment before running the workflow.",
        paste(missing_packages, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (isTRUE(attach)) {
    invisible(lapply(required, library, character.only = TRUE))
  }

  invisible(required)
}

source_analysis_file <- function(...) {
  source(file.path(repo_root, ...), local = FALSE)
}

read_research_questions <- function(paths = analysis_paths) {
  yaml_path <- file.path(paths$r_dir, "research_questions.yaml")
  if (!file.exists(yaml_path)) {
    stop(sprintf("Research question YAML not found: %s", yaml_path), call. = FALSE)
  }
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package 'yaml' is required to read R/research_questions.yaml.", call. = FALSE)
  }
  yaml::read_yaml(yaml_path)
}

resolve_quarto_command <- function() {
  env_quarto <- Sys.getenv("QUARTO_PATH", unset = "")
  if (nzchar(env_quarto) && file.exists(env_quarto)) {
    return(normalizePath(env_quarto, winslash = "/", mustWork = TRUE))
  }

  discovered <- Sys.which("quarto")
  if (nzchar(discovered)) {
    return(discovered)
  }

  common_candidates <- c(
    "C:/Program Files/Quarto/bin/quarto.exe",
    "C:/Program Files/Quarto/bin/quarto.cmd",
    "C:/Users/arunps/AppData/Local/Programs/Quarto/bin/quarto.exe"
  )

  existing <- common_candidates[file.exists(common_candidates)]
  if (length(existing) > 0) {
    return(normalizePath(existing[[1]], winslash = "/", mustWork = TRUE))
  }

  ""
}

configure_quarto_runtime <- function() {
  quarto_command <- resolve_quarto_command()
  if (nzchar(quarto_command)) {
    Sys.setenv(QUARTO_PATH = quarto_command)
  }
  quarto_command
}

message("HindiBabyNet R analysis scaffold loaded.")
message(sprintf("Repository root: %s", repo_root))
if (Sys.getenv("HBN_NSIM", unset = "") != "") {
  message(sprintf("Runtime nsim override active: %s", analysis_options$nsim_default))
}
if (Sys.getenv("HBN_FAMILY2_PARTICIPANT_INTERCEPT", unset = "") != "") {
  message(sprintf("Runtime Family 2 participant intercept mode active: %s", analysis_options$family_2_participant_intercept))
}
if (Sys.getenv("HBN_MODEL_ROBUSTNESS_MODE", unset = "") != "") {
  message(sprintf("Runtime model robustness mode active: %s", analysis_options$model_robustness_mode))
}
message(sprintf("Active age specification: %s", format_age_specification_summary(analysis_options$model_config)))
message(sprintf("Active ANOVA type: %s", analysis_options$model_config$anova_type))

fixed_effect_formula <- function(model) {
  if (inherits(model, c("merMod", "glmmTMB"))) {
    return(lme4::nobars(stats::formula(model)))
  }
  stats::formula(model)
}

build_fixed_effect_model_frame <- function(model) {
  stats::model.frame(fixed_effect_formula(model), data = stats::model.frame(model), na.action = stats::na.pass)
}

build_fixed_effect_model_matrix <- function(model) {
  model_frame <- build_fixed_effect_model_frame(model)
  model_terms <- stats::terms(fixed_effect_formula(model), data = model_frame)
  model_matrix <- stats::model.matrix(model_terms, data = model_frame)
  list(matrix = model_matrix, terms = model_terms)
}

term_level_collinearity_from_matrix <- function(model, bundle = NULL) {
  matrix_info <- build_fixed_effect_model_matrix(model)
  design_matrix <- matrix_info$matrix
  intercept_index <- which(colnames(design_matrix) == "(Intercept)")
  if (length(intercept_index) > 0) {
    design_matrix <- design_matrix[, -intercept_index, drop = FALSE]
  }
  if (ncol(design_matrix) == 0) {
    return(data.frame(status = "no_fixed_effect_columns", stringsAsFactors = FALSE))
  }

  assign_index <- attr(matrix_info$matrix, "assign")
  if (length(intercept_index) > 0) {
    assign_index <- assign_index[-intercept_index]
  }
  term_labels <- attr(matrix_info$terms, "term.labels")
  qr_decomp <- qr(design_matrix)
  aliased_columns <- if (qr_decomp$rank < ncol(design_matrix)) qr_decomp$pivot[(qr_decomp$rank + 1):ncol(design_matrix)] else integer(0)

  scale_sds <- apply(design_matrix, 2, stats::sd)
  keep_columns <- is.finite(scale_sds) & scale_sds > 0
  design_matrix <- design_matrix[, keep_columns, drop = FALSE]
  assign_index <- assign_index[keep_columns]
  if (ncol(design_matrix) == 0) {
    return(data.frame(status = "constant_fixed_effect_columns", stringsAsFactors = FALSE))
  }

  correlation_matrix <- stats::cor(scale(design_matrix), use = "pairwise.complete.obs")
  if (any(!is.finite(correlation_matrix))) {
    return(data.frame(status = "invalid_correlation_matrix", stringsAsFactors = FALSE))
  }

  determinant_safe <- function(x) {
    if (length(x) == 1) {
      return(as.numeric(x))
    }
    as.numeric(det(as.matrix(x)))
  }

  rows <- list()
  for (term_index in sort(unique(assign_index))) {
    term_columns <- which(assign_index == term_index)
    term_label <- term_labels[[term_index]]
    if (term_label == "") {
      next
    }

    original_term_columns <- which(assign_index == term_index)
    term_status <- if (any(original_term_columns %in% aliased_columns)) "rank_deficient" else "ok"
    if (identical(term_status, "rank_deficient")) {
      rows[[length(rows) + 1]] <- data.frame(
        term = term_label,
        vif_or_gvif = NA_real_,
        df = length(term_columns),
        adjusted_gvif = NA_real_,
        status = term_status,
        stringsAsFactors = FALSE
      )
      next
    }

    if (length(term_columns) == ncol(correlation_matrix)) {
      gvif_value <- 1.0
    } else {
      other_columns <- setdiff(seq_len(ncol(correlation_matrix)), term_columns)
      numerator <- determinant_safe(correlation_matrix[term_columns, term_columns, drop = FALSE]) *
        determinant_safe(correlation_matrix[other_columns, other_columns, drop = FALSE])
      denominator <- determinant_safe(correlation_matrix)
      gvif_value <- numerator / denominator
    }

    rows[[length(rows) + 1]] <- data.frame(
      term = term_label,
      vif_or_gvif = as.numeric(gvif_value),
      df = length(term_columns),
      adjusted_gvif = as.numeric(gvif_value) ^ (1 / (2 * length(term_columns))),
      status = term_status,
      stringsAsFactors = FALSE
    )
  }

  output <- do.call(rbind, rows)
  if (!is.null(bundle)) {
    output$model <- bundle$model_family
    output$response <- bundle$response_name
  }
  output
}

tidy_car_vif <- function(vif_result, bundle) {
  if (is.matrix(vif_result)) {
    output <- data.frame(
      term = rownames(vif_result),
      vif_or_gvif = as.numeric(vif_result[, "GVIF"]),
      df = as.numeric(vif_result[, "Df"]),
      adjusted_gvif = as.numeric(vif_result[, "GVIF^(1/(2*Df))"]),
      status = "ok",
      stringsAsFactors = FALSE
    )
  } else {
    output <- data.frame(
      term = names(vif_result),
      vif_or_gvif = as.numeric(vif_result),
      df = 1,
      adjusted_gvif = sqrt(as.numeric(vif_result)),
      status = "ok",
      stringsAsFactors = FALSE
    )
  }

  output$model <- bundle$model_family
  output$response <- bundle$response_name
  output
}

vif_table <- function(bundle) {
  try_car <- tryCatch(car::vif(bundle$model), error = function(e) e)
  if (!inherits(try_car, "error")) {
    return(tidy_car_vif(try_car, bundle))
  }

  fallback <- tryCatch(term_level_collinearity_from_matrix(bundle$model, bundle = bundle), error = function(e) {
    data.frame(
      model = bundle$model_family,
      response = bundle$response_name,
      term = NA_character_,
      vif_or_gvif = NA_real_,
      df = NA_real_,
      adjusted_gvif = NA_real_,
      status = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  })
  fallback
}

tidy_anova_table <- function(bundle) {
  anova_type <- analysis_options$model_config$anova_type
  anova_result <- tryCatch(car::Anova(bundle$model, type = anova_type), error = function(e) e)
  if (inherits(anova_result, "error")) {
    return(data.frame(
      response = bundle$response_name,
      response_transform = bundle$response_transform,
      age_specification = analysis_options$model_config$age_specification,
      model_name = bundle$model_family,
      term = NA_character_,
      statistic = NA_real_,
      df = NA_real_,
      p_value = NA_real_,
      anova_type = anova_type,
      status = paste("unsupported:", conditionMessage(anova_result)),
      stringsAsFactors = FALSE
    ))
  }

  anova_frame <- as.data.frame(anova_result, stringsAsFactors = FALSE)
  anova_frame$term <- rownames(anova_frame)
  rownames(anova_frame) <- NULL
  statistic_column <- grep("^F value$|^Chisq$|^LR Chisq$", names(anova_frame), ignore.case = TRUE, value = TRUE)[1]
  df_column <- grep("^Df$", names(anova_frame), ignore.case = TRUE, value = TRUE)[1]
  p_column <- grep("Pr\\(|p.value|p-value", names(anova_frame), ignore.case = TRUE, value = TRUE)[1]

  data.frame(
    response = bundle$response_name,
    response_transform = bundle$response_transform,
    age_specification = analysis_options$model_config$age_specification,
    model_name = bundle$model_family,
    term = anova_frame$term,
    statistic = as.numeric(anova_frame[[statistic_column]]),
    df = as.numeric(anova_frame[[df_column]]),
    p_value = as.numeric(anova_frame[[p_column]]),
    anova_type = anova_type,
    status = "ok",
    stringsAsFactors = FALSE
  )
}