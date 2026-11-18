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

apply_bundle_age_terms <- function(dataframe, bundle, age_days = NULL, age_z = NULL) {
  age_terms <- active_age_terms(bundle$model_family, age_specification = bundle$age_specification)

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

build_age_speaker_grid <- function(bundle, data, n_points = analysis_options$prediction_grid_size) {
  age_days_values <- seq(min(data$age_days, na.rm = TRUE), max(data$age_days, na.rm = TRUE), length.out = n_points)
  speakers <- levels(data$speaker)
  if (is.null(speakers)) {
    speakers <- sort(unique(data$speaker))
  }

  grid <- expand.grid(age_days = age_days_values, speaker = speakers, stringsAsFactors = FALSE)
  reference <- build_reference_values(data)[rep(1, nrow(grid)), , drop = FALSE]
  reference$age_days <- grid$age_days
  reference$speaker <- factor(grid$speaker, levels = levels(data$speaker))
  apply_bundle_age_terms(reference, bundle = bundle, age_days = grid$age_days)
}

build_predictor_speaker_grid <- function(bundle, data, predictor_name, n_points = analysis_options$prediction_grid_size) {
  predictor_values <- seq(min(data[[predictor_name]], na.rm = TRUE), max(data[[predictor_name]], na.rm = TRUE), length.out = n_points)
  speakers <- levels(data$speaker)
  if (is.null(speakers)) {
    speakers <- sort(unique(data$speaker))
  }

  grid <- expand.grid(predictor_value = predictor_values, speaker = speakers, stringsAsFactors = FALSE)
  reference <- build_reference_values(data)[rep(1, nrow(grid)), , drop = FALSE]
  reference[[predictor_name]] <- grid$predictor_value
  reference$speaker <- factor(grid$speaker, levels = levels(data$speaker))
  apply_bundle_age_terms(reference, bundle = bundle, age_z = rep(0, nrow(reference)))
}

compute_monte_carlo_ci <- function(bundle, grid, nsim = analysis_options$nsim_default) {
  prediction_function <- function(fitted_model) {
    stats::predict(fitted_model, newdata = grid, re.form = NA, allow.new.levels = TRUE)
  }

  set.seed(analysis_options$seed)
  boot_result <- lme4::bootMer(
    x = bundle$model,
    FUN = prediction_function,
    nsim = nsim,
    re.form = NA,
    use.u = FALSE,
    type = "parametric",
    parallel = "no"
  )

  simulated <- as.matrix(boot_result$t)
  base_prediction <- stats::predict(bundle$model, newdata = grid, re.form = NA, allow.new.levels = TRUE)
  alpha <- (1 - analysis_options$confidence_level) / 2

  data.frame(
    grid,
    predicted = as.numeric(base_prediction),
    lower = apply(simulated, 2, stats::quantile, probs = alpha, na.rm = TRUE),
    upper = apply(simulated, 2, stats::quantile, probs = 1 - alpha, na.rm = TRUE),
    interval_method = "bootMer_parametric",
    nsim = nsim,
    stringsAsFactors = FALSE
  )
}
