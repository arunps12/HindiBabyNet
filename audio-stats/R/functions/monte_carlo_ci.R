# Monte Carlo confidence interval helpers for the HindiBabyNet R workflow.

# Value of a column at which a prediction is made when the column is not varied:
# the most frequent level of a factor, the mean of a number, else the first value.
reference_value <- function(values) {
  if (is.factor(values)) {
    counts <- table(values)
    return(factor(names(counts)[which.max(counts)], levels = levels(values)))
  }

  observed <- values[!is.na(values)]
  if (length(observed) == 0) {
    return(NA)
  }
  if (is.numeric(observed)) {
    return(mean(observed))
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
  reference <- lapply(data, reference_value)
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

# Grid for a Family 2 model. The data has one row per participant. For each input
# speaker, the input of this speaker goes over its observed range; the inputs of the
# other two speakers, age and SES stay at their mean. The columns `speaker` and the
# common x column (input_count_hour or input_duration_min_hour) say which input is varied.
build_input_output_grid <- function(bundle, data, n_points = 60) {
  input_terms <- family_2_input_terms(bundle$model_family)
  x_name <- family_2_input_axis(bundle$model_family)
  reference <- build_reference_values(data)

  grid <- do.call(rbind, lapply(seq_along(input_terms), function(index) {
    values <- data[[input_terms[[index]]]]
    varied <- seq(min(values, na.rm = TRUE), max(values, na.rm = TRUE), length.out = n_points)
    rows <- reference[rep(1, n_points), , drop = FALSE]
    rows[[input_terms[[index]]]] <- varied
    rows$speaker <- analysis_options$speaker_order[[index]]
    rows[[x_name]] <- varied
    rows
  }))
  rownames(grid) <- NULL
  grid$speaker <- factor(grid$speaker, levels = analysis_options$speaker_order)
  apply_bundle_age_terms(grid, bundle = bundle, age_z = rep(0, nrow(grid)))
}

# The observed values of a Family 2 model in the layout of the grid: one row per
# participant and input speaker.
stack_input_observations <- function(bundle, data) {
  input_terms <- family_2_input_terms(bundle$model_family)
  stacked <- do.call(rbind, lapply(seq_along(input_terms), function(index) {
    rows <- data.frame(
      speaker = analysis_options$speaker_order[[index]],
      input = data[[input_terms[[index]]]],
      response = data[[bundle$response_name]],
      stringsAsFactors = FALSE
    )
    names(rows) <- c("speaker", family_2_input_axis(bundle$model_family), bundle$response_name)
    rows
  }))
  stacked$speaker <- factor(stacked$speaker, levels = analysis_options$speaker_order)
  # A participant without rates has no point in the figure.
  stacked[stats::complete.cases(stacked), , drop = FALSE]
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

  # A glm object is also of class lm: the Gamma model is handled in the next block.
  if (inherits(model, "lm") && !inherits(model, "glm")) {
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
