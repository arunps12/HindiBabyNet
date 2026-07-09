# Monte Carlo confidence interval helpers for the HindiBabyNet R workflow.

mode_or_first_level <- function(values) {
  if (is.factor(values)) {
    counts <- table(values)
    return(factor(names(counts)[which.max(counts)], levels = levels(values)))
  }

  observed <- values[!is.na(values)]
  if (length(observed) == 0) {
    return(NA)
  }

  unique_values <- unique(observed)
  unique_values[[1]]
}

age_days_to_z <- function(age_days, mean_days = analysis_runtime$age_days_mean, sd_days = analysis_runtime$age_days_sd) {
  if (is.na(mean_days) || is.na(sd_days) || sd_days <= 0) {
    stop("age_days_mean and age_days_sd must be available before building prediction grids.", call. = FALSE)
  }
  (age_days - mean_days) / sd_days
}

age_z_to_days <- function(age_z, mean_days = analysis_runtime$age_days_mean, sd_days = analysis_runtime$age_days_sd) {
  if (is.na(mean_days) || is.na(sd_days) || sd_days <= 0) {
    stop("age_days_mean and age_days_sd must be available before converting age_z to age_days.", call. = FALSE)
  }
  (age_z * sd_days) + mean_days
}

build_reference_values <- function(data) {
  reference <- lapply(data, mode_or_first_level)
  as.data.frame(reference, stringsAsFactors = FALSE)
}

build_age_speaker_grid <- function(data, n_points = 60) {
  if (!("speaker" %in% names(data))) {
    stop("Age-by-speaker prediction grids require a speaker column.", call. = FALSE)
  }

  age_days_values <- seq(min(data$age_days, na.rm = TRUE), max(data$age_days, na.rm = TRUE), length.out = n_points)
  speakers <- levels(data$speaker)
  if (is.null(speakers)) {
    speakers <- sort(unique(data$speaker))
  }

  grid <- expand.grid(
    age_days = age_days_values,
    speaker = speakers,
    stringsAsFactors = FALSE
  )

  reference <- build_reference_values(data)[rep(1, nrow(grid)), , drop = FALSE]
  reference$age_days <- grid$age_days
  reference$speaker <- factor(grid$speaker, levels = levels(data$speaker))
  reference$age_z <- age_days_to_z(reference$age_days)
  reference
}

build_input_output_grid <- function(data, predictor_name, n_points = 60) {
  predictor_values <- seq(min(data[[predictor_name]], na.rm = TRUE), max(data[[predictor_name]], na.rm = TRUE), length.out = n_points)
  speakers <- levels(data$speaker)
  if (is.null(speakers)) {
    speakers <- sort(unique(data$speaker))
  }

  grid <- expand.grid(
    predictor_value = predictor_values,
    speaker = speakers,
    stringsAsFactors = FALSE
  )

  reference <- build_reference_values(data)[rep(1, nrow(grid)), , drop = FALSE]
  reference[[predictor_name]] <- grid$predictor_value
  reference$speaker <- factor(grid$speaker, levels = levels(data$speaker))
  reference$age_z <- 0
  reference$age_days <- age_z_to_days(reference$age_z)
  reference
}

compute_monte_carlo_ci <- function(bundle, grid, nsim = analysis_options$nsim_default) {
  model <- bundle$model

  if (inherits(model, "merMod")) {
    response_values <- model.frame(model)[[1]]
    inverse_transform <- if (isTRUE(bundle$transformed_response)) exp else identity

    prediction_function <- function(fitted_model) {
      raw_prediction <- stats::predict(fitted_model, newdata = grid, re.form = NA, allow.new.levels = TRUE)
      inverse_transform(raw_prediction)
    }

    set.seed(analysis_options$seed)
    boot_result <- lme4::bootMer(
      x = model,
      FUN = prediction_function,
      nsim = nsim,
      re.form = NA,
      use.u = FALSE,
      type = "parametric",
      parallel = "no"
    )

    simulated <- as.matrix(boot_result$t)
    if (nrow(simulated) == 0) {
      stop(sprintf("bootMer() returned no simulations for response '%s'.", bundle$response_name), call. = FALSE)
    }

    base_prediction <- inverse_transform(stats::predict(model, newdata = grid, re.form = NA, allow.new.levels = TRUE))

    predictions <- data.frame(
      grid,
      predicted = as.numeric(base_prediction),
      lower = apply(simulated, 2, stats::quantile, probs = 0.025, na.rm = TRUE),
      upper = apply(simulated, 2, stats::quantile, probs = 0.975, na.rm = TRUE),
      interval_method = "bootMer_parametric",
      nsim = nsim,
      stringsAsFactors = FALSE
    )

    return(predictions)
  }

  if (inherits(model, "glmmTMB")) {
    fitted_values <- stats::predict(model, newdata = grid, type = "response", se.fit = TRUE, re.form = NA)
    return(data.frame(
      grid,
      predicted = fitted_values$fit,
      lower = fitted_values$fit - 1.96 * fitted_values$se.fit,
      upper = fitted_values$fit + 1.96 * fitted_values$se.fit,
      interval_method = "glmmTMB_delta_method",
      nsim = nsim,
      stringsAsFactors = FALSE
    ))
  }

  stop(sprintf("Unsupported model class for prediction intervals: %s", class(model)[1]), call. = FALSE)
}
