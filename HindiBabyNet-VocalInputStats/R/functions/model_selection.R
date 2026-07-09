# Model selection helpers for the HindiBabyNet R workflow.

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

build_preregistered_formula <- function(model_family, response_name) {
  if (identical(model_family, "family_1")) {
    return(stats::as.formula(sprintf("%s ~ speaker * age_z + SES + child_sex + (1|participant_id) + (1|Location)", response_name)))
  }

  if (identical(model_family, "family_2_count")) {
    return(stats::as.formula(sprintf("%s ~ input_count_hour * speaker + age_z + SES + child_sex + (1|participant_id) + (1|Location)", response_name)))
  }

  if (identical(model_family, "family_2_duration")) {
    return(stats::as.formula(sprintf("%s ~ input_duration_min_hour * speaker + age_z + SES + child_sex + (1|participant_id) + (1|Location)", response_name)))
  }

  stop(sprintf("Unknown preregistered model family: %s", model_family), call. = FALSE)
}

prepare_model_data <- function(data, formula) {
  stats::na.omit(data[, all.vars(formula), drop = FALSE])
}

fit_selected_model <- function(data, response_name, formula, selection_row) {
  model_type <- selection_row$model_type[[1]]
  model_data <- prepare_model_data(data, formula)

  if (nrow(model_data) == 0) {
    stop(sprintf("No complete cases available for response '%s'.", response_name), call. = FALSE)
  }

  if (identical(model_type, "lmm")) {
    model <- lme4::lmer(formula = formula, data = model_data, REML = FALSE)
    return(list(model = model, fitted_formula = formula, model_data = model_data, transformed_response = FALSE))
  }

  if (identical(model_type, "log_lmm")) {
    log_response_name <- sprintf("log_%s", response_name)
    model_data[[log_response_name]] <- log(model_data[[response_name]])
    log_formula <- stats::update(formula, stats::as.formula(sprintf("%s ~ .", log_response_name)))
    model <- lme4::lmer(formula = log_formula, data = model_data, REML = FALSE)
    return(list(model = model, fitted_formula = log_formula, model_data = model_data, transformed_response = TRUE))
  }

  if (identical(model_type, "gamma_glmm")) {
    model <- glmmTMB::glmmTMB(formula = formula, data = model_data, family = glmmTMB::Gamma(link = "log"))
    return(list(model = model, fitted_formula = formula, model_data = model_data, transformed_response = FALSE))
  }

  stop(sprintf("Unsupported model type: %s", model_type), call. = FALSE)
}

fit_preregistered_model <- function(data, response_name, model_family, allow_gamma = TRUE) {
  selection_row <- select_model_family(response_name, data, allow_gamma = allow_gamma)
  formula <- build_preregistered_formula(model_family, response_name)
  fit_result <- fit_selected_model(data, response_name, formula, selection_row)

  list(
    response_name = response_name,
    model_family = model_family,
    selection = selection_row,
    model = fit_result$model,
    fitted_formula = fit_result$fitted_formula,
    transformed_response = fit_result$transformed_response,
    n_complete_cases = nrow(fit_result$model_data)
  )
}

selection_table_from_bundles <- function(model_bundles) {
  rows <- lapply(model_bundles, function(bundle) {
    data.frame(
      response = bundle$response_name,
      model_family = bundle$model_family,
      model_type = bundle$selection$model_type[[1]],
      rationale = bundle$selection$rationale[[1]],
      n_complete_cases = bundle$n_complete_cases,
      fitted_formula = paste(deparse(bundle$fitted_formula), collapse = " "),
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
