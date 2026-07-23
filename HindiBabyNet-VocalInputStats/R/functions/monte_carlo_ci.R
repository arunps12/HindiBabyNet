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

augment_age_terms <- function(dataframe) {
  dataframe$age_z <- age_days_to_z(dataframe$age_days)
  dataframe$age_z2 <- dataframe$age_z ^ 2
  dataframe
}

bundle_age_terms <- function(bundle) {
  active_age_terms(
    response_name = bundle$response_name,
    model_family = bundle$model_family,
    age_specification = bundle$age_specification
  )
}

apply_bundle_age_terms <- function(dataframe, bundle, age_days = NULL, age_z = NULL) {
  age_terms <- bundle_age_terms(bundle)

  if (!is.null(age_days)) {
    dataframe$age_days <- age_days
  }

  if (length(age_terms) == 0) {
    if ("age_z" %in% names(dataframe)) {
      dataframe$age_z <- NULL
    }
    if ("age_z2" %in% names(dataframe)) {
      dataframe$age_z2 <- NULL
    }
    return(dataframe)
  }

  if (is.null(dataframe$age_days) && is.null(age_days) && !is.null(age_z)) {
    dataframe$age_days <- age_z_to_days(age_z)
  }

  if (is.null(age_z)) {
    dataframe$age_z <- age_days_to_z(dataframe$age_days)
  } else {
    dataframe$age_z <- age_z
    if (is.null(dataframe$age_days)) {
      dataframe$age_days <- age_z_to_days(age_z)
    }
  }

  if ("age_z2" %in% age_terms) {
    dataframe$age_z2 <- dataframe$age_z ^ 2
  } else if ("age_z2" %in% names(dataframe)) {
    dataframe$age_z2 <- NULL
  }

  dataframe
}

build_reference_values <- function(data) {
  reference <- lapply(data, mode_or_first_level)
  as.data.frame(reference, stringsAsFactors = FALSE)
}

build_age_speaker_grid <- function(bundle, data, n_points = 60) {
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
  apply_bundle_age_terms(reference, bundle = bundle, age_days = grid$age_days)
}

build_input_output_grid <- function(bundle, data, predictor_name, n_points = 60) {
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
  apply_bundle_age_terms(reference, bundle = bundle, age_z = rep(0, nrow(reference)))
}

compute_monte_carlo_ci <- function(bundle, grid, nsim = analysis_options$nsim_default) {
  model <- bundle$model
  response_transform <- bundle$response_transform %||% "none"

  if (inherits(model, "merMod")) {
    prediction_function <- function(fitted_model) {
      raw_prediction <- stats::predict(fitted_model, newdata = grid, re.form = NA, allow.new.levels = TRUE)
      inverse_transform_values(raw_prediction, response_transform)
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

    base_prediction <- inverse_transform_values(
      stats::predict(model, newdata = grid, re.form = NA, allow.new.levels = TRUE),
      response_transform
    )

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

  if (inherits(model, "lm")) {
    fitted_values <- stats::predict(model, newdata = grid, se.fit = TRUE)

    predictions <- data.frame(
      grid,
      predicted = as.numeric(inverse_transform_values(fitted_values$fit, response_transform)),
      lower = as.numeric(inverse_transform_values(fitted_values$fit - 1.96 * fitted_values$se.fit, response_transform)),
      upper = as.numeric(inverse_transform_values(fitted_values$fit + 1.96 * fitted_values$se.fit, response_transform)),
      interval_method = "lm_normal_approx",
      nsim = nsim,
      stringsAsFactors = FALSE
    )

    return(predictions)
  }

  if (inherits(model, "glm")) {
    link_values <- stats::predict(model, newdata = grid, type = "link", se.fit = TRUE)
    inverse_link <- model$family$linkinv

    return(data.frame(
      grid,
      predicted = as.numeric(inverse_link(link_values$fit)),
      lower = as.numeric(inverse_link(link_values$fit - 1.96 * link_values$se.fit)),
      upper = as.numeric(inverse_link(link_values$fit + 1.96 * link_values$se.fit)),
      interval_method = "glm_link_normal_approx",
      nsim = nsim,
      stringsAsFactors = FALSE
    ))
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
