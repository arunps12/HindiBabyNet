# Model selection helpers for the HindiBabyNet R workflow.

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

calculate_skewness <- function(x) {
  values <- x[is.finite(x)]
  if (length(values) < 3) {
    return(NA_real_)
  }

  centered <- values - mean(values)
  variance <- mean(centered ^ 2)
  if (variance <= 0) {
    return(0.0)
  }

  mean(centered ^ 3) / (variance ^ 1.5)
}

inspect_response_variable <- function(data, response_name) {
  values <- data[[response_name]]
  finite_values <- values[is.finite(values)]

  if (length(finite_values) == 0) {
    stop(sprintf("Response '%s' contains no finite values.", response_name), call. = FALSE)
  }

  data.frame(
    response = response_name,
    n = length(finite_values),
    min = min(finite_values),
    max = max(finite_values),
    mean = mean(finite_values),
    sd = stats::sd(finite_values),
    skewness = calculate_skewness(finite_values),
    zeros = sum(finite_values == 0),
    positive_only = all(finite_values > 0),
    stringsAsFactors = FALSE
  )
}

choose_model_type <- function(metrics, allow_gamma = TRUE) {
  if (!isTRUE(metrics$positive_only)) {
    return(list(model_type = "lmm", rationale = "Response includes non-positive values; using Gaussian LMM on the original scale."))
  }

  skewness <- abs(metrics$skewness)
  coefficient_of_variation <- if (isTRUE(metrics$mean == 0)) Inf else metrics$sd / metrics$mean

  if (is.na(skewness) || (skewness < 1.0 && coefficient_of_variation < 0.75)) {
    return(list(model_type = "lmm", rationale = "Response is approximately symmetric with moderate dispersion; using Gaussian LMM on the original scale."))
  }

  if (allow_gamma && skewness >= 2.5) {
    return(list(model_type = "gamma_glmm", rationale = "Response is strongly positively skewed; using Gamma GLMM with a log link."))
  }

  list(model_type = "log_lmm", rationale = "Response is positive and moderately skewed; using log-transform plus Gaussian LMM.")
}

select_model_family <- function(response_name, data, allow_gamma = TRUE) {
  metrics <- inspect_response_variable(data, response_name)
  decision <- choose_model_type(metrics, allow_gamma = allow_gamma)

  cbind(metrics, model_type = decision$model_type, rationale = decision$rationale, stringsAsFactors = FALSE)
}

family_override_key <- function(model_family) {
  if (identical(model_family, "family_1")) {
    return("family_1")
  }
  if (model_family %in% c("family_2_count", "family_2_duration")) {
    return("family_2")
  }

  stop(sprintf("Unknown model family: %s", model_family), call. = FALSE)
}

expand_fixed_term_shorthand <- function(term) {
  term <- trimws(term)
  if (!nzchar(term)) {
    return(character(0))
  }

  if (grepl("\\*", term)) {
    components <- trimws(unlist(strsplit(term, "\\*")))
    if (length(components) == 2) {
      return(c(components, sprintf("%s:%s", components[[1]], components[[2]])))
    }
  }

  term
}

normalize_fixed_terms <- function(terms) {
  unique(unlist(lapply(terms, expand_fixed_term_shorthand), use.names = FALSE))
}

normalize_random_slope_term <- function(term) {
  slope <- trimws(term)
  if (!nzchar(slope)) {
    return(NA_character_)
  }

  if (grepl("^\\(.*\\)$", slope)) {
    return(slope)
  }

  if (grepl("\\|\\|", slope)) {
    pieces <- trimws(unlist(strsplit(slope, "\\|\\|")))
    separator <- "||"
  } else if (grepl("\\|", slope)) {
    pieces <- trimws(unlist(strsplit(slope, "\\|")))
    separator <- "|"
  } else {
    return(NA_character_)
  }

  if (length(pieces) != 2 || !all(nzchar(pieces))) {
    return(NA_character_)
  }

  lhs <- pieces[[1]]
  rhs <- pieces[[2]]
  if (!grepl("^(0|-1|1)\\s*(\\+|$)", lhs)) {
    lhs <- sprintf("1 + %s", lhs)
  }

  sprintf("(%s %s %s)", lhs, separator, rhs)
}

normalize_random_slope_terms <- function(terms) {
  normalized <- vapply(terms, normalize_random_slope_term, character(1))
  unique(normalized[!is.na(normalized) & nzchar(normalized)])
}

default_model_family_spec <- function(model_family, response_name) {
  if (identical(model_family, "family_1")) {
    return(list(
      model_family = model_family,
      response_name = response_name,
      fixed_terms = c("speaker", "age_z", "speaker:age_z", "SES", "child_sex"),
      preserve_fixed_terms = c("speaker", "age_z"),
      simplification_drop_order = c("speaker:age_z", "child_sex", "SES"),
      participant_intercept_mode = "with",
      location_intercept_mode = "with",
      random_slopes = character(0)
    ))
  }

  if (identical(model_family, "family_2_count")) {
    return(list(
      model_family = model_family,
      response_name = response_name,
      fixed_terms = c("input_count_hour", "speaker", "input_count_hour:speaker", "age_z", "SES", "child_sex"),
      preserve_fixed_terms = c("input_count_hour", "speaker"),
      simplification_drop_order = c("input_count_hour:speaker", "child_sex", "SES", "age_z"),
      participant_intercept_mode = analysis_options$family_2_participant_intercept,
      location_intercept_mode = "with",
      random_slopes = character(0)
    ))
  }

  if (identical(model_family, "family_2_duration")) {
    return(list(
      model_family = model_family,
      response_name = response_name,
      fixed_terms = c("input_duration_min_hour", "speaker", "input_duration_min_hour:speaker", "age_z", "SES", "child_sex"),
      preserve_fixed_terms = c("input_duration_min_hour", "speaker"),
      simplification_drop_order = c("input_duration_min_hour:speaker", "child_sex", "SES", "age_z"),
      participant_intercept_mode = analysis_options$family_2_participant_intercept,
      location_intercept_mode = "with",
      random_slopes = character(0)
    ))
  }

  stop(sprintf("Unknown preregistered model family: %s", model_family), call. = FALSE)
}

get_model_family_overrides <- function(model_family) {
  override_key <- family_override_key(model_family)
  overrides <- analysis_options$model_spec_overrides[[override_key]]
  if (is.null(overrides)) {
    return(list())
  }
  overrides
}

build_requested_model_spec <- function(model_family, response_name) {
  spec <- default_model_family_spec(model_family, response_name)
  overrides <- get_model_family_overrides(model_family)

  spec$fixed_terms <- normalize_fixed_terms(c(
    spec$fixed_terms,
    overrides$extra_fixed_terms %||% character(0)
  ))
  spec$fixed_terms <- setdiff(
    spec$fixed_terms,
    normalize_fixed_terms(overrides$drop_fixed_terms %||% character(0))
  )
  spec$participant_intercept_mode <- overrides$participant_intercept %||% spec$participant_intercept_mode
  spec$location_intercept_mode <- overrides$location_intercept %||% spec$location_intercept_mode
  spec$random_slopes <- normalize_random_slope_terms(c(
    spec$random_slopes,
    overrides$random_slopes %||% character(0)
  ))

  spec
}

logical_mode_candidates <- function(mode, default_value = TRUE, allow_fallback = TRUE) {
  normalized_mode <- trimws(tolower(mode %||% if (default_value) "with" else "without"))
  primary <- switch(
    normalized_mode,
    with = TRUE,
    without = FALSE,
    auto = TRUE,
    default_value
  )

  candidates <- primary
  if (identical(normalized_mode, "auto") || isTRUE(allow_fallback)) {
    candidates <- unique(c(candidates, !primary))
  }
  candidates
}

candidate_fixed_term_sets <- function(spec, strict = FALSE) {
  current_terms <- spec$fixed_terms
  candidates <- list(current_terms)
  if (isTRUE(strict)) {
    return(candidates)
  }

  for (term in spec$simplification_drop_order) {
    if (term %in% current_terms && !(term %in% spec$preserve_fixed_terms)) {
      current_terms <- setdiff(current_terms, term)
      if (all(spec$preserve_fixed_terms %in% current_terms)) {
        candidates[[length(candidates) + 1]] <- current_terms
      }
    }
  }

  unique_candidates <- list()
  seen <- character(0)
  for (candidate in candidates) {
    key <- paste(sort(candidate), collapse = ";")
    if (!(key %in% seen)) {
      seen <- c(seen, key)
      unique_candidates[[length(unique_candidates) + 1]] <- candidate
    }
  }

  unique_candidates
}

candidate_random_slope_sets <- function(random_slopes, strict = FALSE) {
  candidates <- list(random_slopes)
  if (isTRUE(strict) || length(random_slopes) == 0) {
    return(candidates)
  }

  if (length(random_slopes) > 1) {
    for (index in seq_along(random_slopes)) {
      candidates[[length(candidates) + 1]] <- random_slopes[-index]
    }
  }
  candidates[[length(candidates) + 1]] <- character(0)

  unique_candidates <- list()
  seen <- character(0)
  for (candidate in candidates) {
    key <- paste(sort(candidate), collapse = ";")
    if (!(key %in% seen)) {
      seen <- c(seen, key)
      unique_candidates[[length(unique_candidates) + 1]] <- candidate
    }
  }

  unique_candidates
}

build_formula_from_spec <- function(spec) {
  fixed_rhs <- if (length(spec$fixed_terms) > 0) {
    paste(c("1", spec$fixed_terms), collapse = " + ")
  } else {
    "1"
  }

  random_terms <- c()
  if (isTRUE(spec$participant_intercept)) {
    random_terms <- c(random_terms, "(1 | participant_id)")
  }
  if (isTRUE(spec$location_intercept)) {
    random_terms <- c(random_terms, "(1 | Location)")
  }
  random_terms <- c(random_terms, spec$random_slopes)

  rhs <- paste(c(fixed_rhs, random_terms), collapse = " + ")
  stats::as.formula(sprintf("%s ~ %s", spec$response_name, rhs))
}

build_candidate_specs <- function(model_family, response_name) {
  strict <- identical(analysis_options$model_robustness_mode, "strict")
  requested_spec <- build_requested_model_spec(model_family, response_name)
  fixed_term_sets <- candidate_fixed_term_sets(requested_spec, strict = strict)
  random_slope_sets <- candidate_random_slope_sets(requested_spec$random_slopes, strict = strict)
  participant_candidates <- logical_mode_candidates(
    requested_spec$participant_intercept_mode,
    default_value = TRUE,
    allow_fallback = !strict
  )
  location_candidates <- logical_mode_candidates(
    requested_spec$location_intercept_mode,
    default_value = TRUE,
    allow_fallback = !strict
  )

  candidates <- list()
  seen <- character(0)
  for (fixed_terms in fixed_term_sets) {
    for (random_slopes in random_slope_sets) {
      for (participant_intercept in participant_candidates) {
        for (location_intercept in location_candidates) {
          spec <- requested_spec
          spec$fixed_terms <- fixed_terms
          spec$random_slopes <- random_slopes
          spec$participant_intercept <- participant_intercept
          spec$location_intercept <- location_intercept
          formula <- build_formula_from_spec(spec)
          key <- paste(deparse(formula), collapse = " ")
          if (!(key %in% seen)) {
            spec$formula <- formula
            spec$requested_formula <- build_formula_from_spec(requested_spec)
            candidates[[length(candidates) + 1]] <- spec
            seen <- c(seen, key)
          }
        }
      }
    }
  }

  candidates
}

build_preregistered_formula <- function(model_family, response_name, include_participant_intercept = TRUE) {
  spec <- build_requested_model_spec(model_family, response_name)
  spec$participant_intercept <- include_participant_intercept
  spec$location_intercept <- identical(spec$location_intercept_mode, "with") || identical(spec$location_intercept_mode, "auto")
  build_formula_from_spec(spec)
}

prepare_model_data <- function(data, formula) {
  stats::na.omit(data[, all.vars(formula), drop = FALSE])
}

formula_has_random_effects <- function(formula) {
  grepl("\\|", paste(deparse(formula), collapse = " "))
}

fit_lmer_with_retries <- function(formula, model_data) {
  control_candidates <- list(
    NULL,
    lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)),
    lme4::lmerControl(optimizer = "Nelder_Mead", optCtrl = list(maxfun = 2e5))
  )

  last_error <- NULL
  for (control in control_candidates) {
    fit_attempt <- tryCatch(
      {
        if (is.null(control)) {
          lmerTest::lmer(formula = formula, data = model_data, REML = FALSE)
        } else {
          lmerTest::lmer(formula = formula, data = model_data, REML = FALSE, control = control)
        }
      },
      error = function(error_condition) {
        last_error <<- error_condition
        NULL
      }
    )

    if (!is.null(fit_attempt)) {
      if (!is.null(control)) {
        fit_attempt@call$control <- control
      }
      return(fit_attempt)
    }
  }

  stop(last_error)
}

fit_glmmTMB_with_retries <- function(formula, model_data) {
  control_candidates <- list(
    NULL,
    glmmTMB::glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4))
  )

  last_error <- NULL
  for (control in control_candidates) {
    fit_attempt <- tryCatch(
      {
        if (is.null(control)) {
          glmmTMB::glmmTMB(formula = formula, data = model_data, family = glmmTMB::Gamma(link = "log"))
        } else {
          glmmTMB::glmmTMB(
            formula = formula,
            data = model_data,
            family = glmmTMB::Gamma(link = "log"),
            control = control
          )
        }
      },
      error = function(error_condition) {
        last_error <<- error_condition
        NULL
      }
    )

    if (!is.null(fit_attempt)) {
      return(fit_attempt)
    }
  }

  stop(last_error)
}

tidy_model_fixed_effects <- function(model, conf.int = TRUE) {
  if (inherits(model, c("merMod", "glmmTMB"))) {
    return(broom.mixed::tidy(model, effects = "fixed", conf.int = conf.int))
  }

  broom::tidy(model, conf.int = conf.int)
}

tidy_model_random_effects <- function(model, conf.int = TRUE) {
  if (inherits(model, c("merMod", "glmmTMB"))) {
    return(broom.mixed::tidy(model, effects = "ran_pars", conf.int = conf.int))
  }

  data.frame(note = "No random effects in fitted model", stringsAsFactors = FALSE)
}

assess_model_fit <- function(model) {
  if (inherits(model, "merMod")) {
    convergence_messages <- tryCatch(model@optinfo$conv$lme4$messages, error = function(e) NULL)
    singular_fit <- tryCatch(lme4::isSingular(model, tol = 1e-4), error = function(e) FALSE)
    problematic <- !is.null(convergence_messages) || isTRUE(singular_fit)
    issues <- c()
    if (!is.null(convergence_messages)) {
      issues <- c(issues, paste(convergence_messages, collapse = " | "))
    }
    if (isTRUE(singular_fit)) {
      issues <- c(issues, "singular fit")
    }

    return(list(problematic = problematic, issue = paste(issues, collapse = " | ")))
  }

  if (inherits(model, "glmmTMB")) {
    pd_hess <- tryCatch(isTRUE(model$sdr$pdHess), error = function(e) FALSE)
    return(list(
      problematic = !pd_hess,
      issue = if (pd_hess) "" else "glmmTMB Hessian is not positive definite"
    ))
  }

  coefficients <- tryCatch(stats::coef(model), error = function(e) NA_real_)
  problematic <- any(!is.finite(coefficients))
  list(problematic = problematic, issue = if (problematic) "non-finite coefficient estimates" else "")
}

fit_selected_model <- function(data, response_name, formula, selection_row) {
  model_type <- selection_row$model_type[[1]]
  model_data <- prepare_model_data(data, formula)
  has_random_effects <- formula_has_random_effects(formula)

  if (nrow(model_data) == 0) {
    stop(sprintf("No complete cases available for response '%s'.", response_name), call. = FALSE)
  }

  if (identical(model_type, "lmm")) {
    model <- if (has_random_effects) {
      fit_lmer_with_retries(formula = formula, model_data = model_data)
    } else {
      stats::lm(formula = formula, data = model_data)
    }
    return(list(
      model = model,
      fitted_formula = formula,
      model_data = model_data,
      transformed_response = FALSE,
      fit_engine = if (has_random_effects) "lmer" else "lm"
    ))
  }

  if (identical(model_type, "log_lmm")) {
    log_response_name <- sprintf("log_%s", response_name)
    model_data[[log_response_name]] <- log(model_data[[response_name]])
    log_formula <- stats::update(formula, stats::as.formula(sprintf("%s ~ .", log_response_name)))
    model <- if (has_random_effects) {
      fit_lmer_with_retries(formula = log_formula, model_data = model_data)
    } else {
      stats::lm(formula = log_formula, data = model_data)
    }
    return(list(
      model = model,
      fitted_formula = log_formula,
      model_data = model_data,
      transformed_response = TRUE,
      fit_engine = if (has_random_effects) "lmer" else "lm"
    ))
  }

  if (identical(model_type, "gamma_glmm")) {
    model <- if (has_random_effects) {
      fit_glmmTMB_with_retries(formula = formula, model_data = model_data)
    } else {
      stats::glm(formula = formula, data = model_data, family = stats::Gamma(link = "log"))
    }
    return(list(
      model = model,
      fitted_formula = formula,
      model_data = model_data,
      transformed_response = FALSE,
      fit_engine = if (has_random_effects) "glmmTMB" else "glm"
    ))
  }

  stop(sprintf("Unsupported model type: %s", model_type), call. = FALSE)
}

fit_preregistered_model <- function(data, response_name, model_family, allow_gamma = TRUE) {
  selection_row <- select_model_family(response_name, data, allow_gamma = allow_gamma)
  candidate_specs <- build_candidate_specs(model_family, response_name)
  last_error <- NULL
  fallback_result <- NULL

  for (candidate_index in seq_along(candidate_specs)) {
    spec <- candidate_specs[[candidate_index]]
    fit_attempt <- tryCatch(
      fit_selected_model(data, response_name, spec$formula, selection_row),
      error = function(error_condition) {
        last_error <<- error_condition
        NULL
      }
    )

    if (!is.null(fit_attempt)) {
      fit_assessment <- assess_model_fit(fit_attempt$model)
      bundle <- list(
        response_name = response_name,
        model_family = model_family,
        selection = selection_row,
        model = fit_attempt$model,
        fitted_formula = fit_attempt$fitted_formula,
        transformed_response = fit_attempt$transformed_response,
        n_complete_cases = nrow(fit_attempt$model_data),
        participant_intercept_mode = if (isTRUE(spec$participant_intercept)) "with" else "without",
        location_intercept_mode = if (isTRUE(spec$location_intercept)) "with" else "without",
        random_slope_terms = paste(spec$random_slopes, collapse = "; "),
        fixed_effect_terms = paste(spec$fixed_terms, collapse = "; "),
        requested_formula = spec$requested_formula,
        fit_engine = fit_attempt$fit_engine,
        fit_status = if (isTRUE(fit_assessment$problematic)) "fallback_problematic" else "accepted",
        fit_issue = fit_assessment$issue
      )

      if (isTRUE(fit_assessment$problematic) && candidate_index < length(candidate_specs)) {
        if (is.null(fallback_result)) {
          fallback_result <- bundle
        }
      } else {
        if (!identical(bundle$fit_status, "accepted")) {
          message(sprintf(
            "Using best available fit for '%s' despite remaining fit issues: %s",
            response_name,
            bundle$fit_issue
          ))
        }
        return(bundle)
      }
    }
  }

  if (!is.null(fallback_result)) {
    message(sprintf(
      "All candidate specifications for '%s' were unstable or simplified; returning the best available fit.",
      response_name
    ))
    return(fallback_result)
  }

  stop(last_error)
}

selection_table_from_bundles <- function(model_bundles) {
  rows <- lapply(model_bundles, function(bundle) {
    data.frame(
      response = bundle$response_name,
      model_family = bundle$model_family,
      model_type = bundle$selection$model_type[[1]],
      rationale = bundle$selection$rationale[[1]],
      n_complete_cases = bundle$n_complete_cases,
      participant_intercept_mode = if (!is.null(bundle$participant_intercept_mode)) bundle$participant_intercept_mode else NA_character_,
      location_intercept_mode = if (!is.null(bundle$location_intercept_mode)) bundle$location_intercept_mode else NA_character_,
      random_slope_terms = if (!is.null(bundle$random_slope_terms) && nzchar(bundle$random_slope_terms)) bundle$random_slope_terms else NA_character_,
      fixed_effect_terms = if (!is.null(bundle$fixed_effect_terms) && nzchar(bundle$fixed_effect_terms)) bundle$fixed_effect_terms else NA_character_,
      fit_engine = if (!is.null(bundle$fit_engine)) bundle$fit_engine else NA_character_,
      fit_status = if (!is.null(bundle$fit_status)) bundle$fit_status else NA_character_,
      fit_issue = if (!is.null(bundle$fit_issue) && nzchar(bundle$fit_issue)) bundle$fit_issue else NA_character_,
      fitted_formula = paste(deparse(bundle$fitted_formula), collapse = " "),
      requested_formula = if (!is.null(bundle$requested_formula)) paste(deparse(bundle$requested_formula), collapse = " ") else NA_character_,
      transformed_response = bundle$transformed_response,
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, rows)
}

save_model_bundle <- function(bundle, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(bundle, output_path)
  invisible(output_path)
}

load_model_bundle <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Model bundle not found: %s", path), call. = FALSE)
  }
  readRDS(path)
}
