# Prediction plotting helpers for the HindiBabyNet R workflow.

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

apply_white_plot_theme <- function(plot_object) {
  plot_object +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      legend.background = ggplot2::element_rect(fill = "white", color = NA),
      legend.key = ggplot2::element_rect(fill = "white", color = NA),
      strip.background = ggplot2::element_rect(fill = "white", color = NA)
    )
}

validate_age_plot_data <- function(prediction_data, bundle, tolerance = 1e-8) {
  required_columns <- c(
    "age_days",
    active_age_terms(
      response_name = bundle$response_name,
      model_family = bundle$model_family,
      age_specification = bundle$age_specification
    )
  )
  missing_columns <- setdiff(required_columns, names(prediction_data))
  if (length(missing_columns) > 0) {
    stop(
      sprintf(
        "Prediction data is missing required age columns: %s.",
        paste(missing_columns, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if ("age_z" %in% names(prediction_data)) {
    expected_age_z <- age_days_to_z(prediction_data$age_days)
    if (any(abs(expected_age_z - prediction_data$age_z) > tolerance, na.rm = TRUE)) {
      stop("Prediction data violates the age_days to age_z mapping used for plotting.", call. = FALSE)
    }
  }

  if ("age_z2" %in% required_columns) {
    expected_age_z2 <- prediction_data$age_z ^ 2
    if (any(abs(expected_age_z2 - prediction_data$age_z2) > tolerance, na.rm = TRUE)) {
      stop("Prediction data violates the age_z to age_z2 mapping used for plotting.", call. = FALSE)
    }
  }

  invisible(prediction_data)
}

save_prediction_plot <- function(plot_object, output_path, width = 8, height = 5.5, dpi = 300) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(output_path, plot_object, width = width, height = height, dpi = dpi, limitsize = FALSE, bg = "white")
  invisible(output_path)
}

build_jitter_position <- function(observed_data, x_var, y_var) {
  x_values <- observed_data[[x_var]]
  y_values <- observed_data[[y_var]]
  x_range <- diff(range(x_values, na.rm = TRUE))
  y_range <- diff(range(y_values, na.rm = TRUE))

  ggplot2::position_jitter(
    width = if (is.finite(x_range) && x_range > 0) 0.01 * x_range else 0,
    height = if (is.finite(y_range) && y_range > 0) 0.015 * y_range else 0
  )
}

plot_prediction_with_ci <- function(prediction_data, x_var, y_var, output_stub, color_var = NULL, x_label = NULL, y_label = NULL, title = NULL, facet_var = NULL, observed_data = NULL, observed_x_var = x_var, observed_y_var = y_var, width = 8, height = 5.5) {
  plot_object <- ggplot2::ggplot()

  if (!is.null(observed_data)) {
    point_aesthetics <- if (is.null(color_var)) {
      ggplot2::aes_string(x = observed_x_var, y = observed_y_var)
    } else {
      ggplot2::aes_string(x = observed_x_var, y = observed_y_var, color = color_var, group = color_var)
    }

    plot_object <- plot_object +
      ggplot2::geom_point(
        data = observed_data,
        mapping = point_aesthetics,
        alpha = 0.35,
        size = 1.8,
        position = build_jitter_position(observed_data, observed_x_var, observed_y_var),
        inherit.aes = FALSE
      )
  }

  plot_object <- plot_object +
    ggplot2::geom_ribbon(
      data = prediction_data,
      mapping = ggplot2::aes_string(
        x = x_var,
        ymin = "lower",
        ymax = "upper",
        fill = color_var %||% NULL,
        group = color_var %||% NULL
      ),
      alpha = 0.18,
      color = NA,
      inherit.aes = FALSE,
      show.legend = !is.null(color_var)
    ) +
    ggplot2::geom_line(
      data = prediction_data,
      mapping = ggplot2::aes_string(
        x = x_var,
        y = y_var,
        color = color_var %||% NULL,
        group = color_var %||% NULL
      ),
      linewidth = 1.0,
      inherit.aes = FALSE
    ) +
    ggplot2::labs(
      title = title,
      x = x_label %||% x_var,
      y = y_label %||% y_var,
      color = color_var,
      fill = color_var
    ) +
    ggplot2::theme_minimal(base_size = 12)

  if (!is.null(facet_var)) {
    plot_object <- plot_object + ggplot2::facet_wrap(stats::as.formula(paste("~", facet_var)))
  }

  plot_object <- apply_white_plot_theme(plot_object)

  output_path <- file.path(analysis_paths$results_plots, sprintf("%s.png", output_stub))
  save_prediction_plot(plot_object, output_path, width = width, height = height)
  list(plot = plot_object, path = output_path)
}

create_age_speaker_predictions <- function(bundle, data, nsim = analysis_options$nsim_default, n_points = 60) {
  prediction_grid <- build_age_speaker_grid(bundle, data, n_points = n_points)
  validate_age_plot_data(prediction_grid, bundle = bundle)
  prediction_data <- compute_monte_carlo_ci(bundle, prediction_grid, nsim = nsim)
  validate_age_plot_data(prediction_data, bundle = bundle)
  prediction_data
}

create_input_output_predictions <- function(bundle, data, predictor_name, nsim = analysis_options$nsim_default, n_points = 60) {
  prediction_grid <- build_input_output_grid(bundle, data, predictor_name = predictor_name, n_points = n_points)
  compute_monte_carlo_ci(bundle, prediction_grid, nsim = nsim)
}

plot_age_speaker_predictions <- function(bundle, data, output_stub, y_label, nsim = analysis_options$nsim_default) {
  prediction_data <- create_age_speaker_predictions(bundle, data, nsim = nsim)
  csv_path <- file.path(analysis_paths$results_predictions, sprintf("%s_predictions.csv", output_stub))
  utils::write.csv(prediction_data, csv_path, row.names = FALSE)

  plot_result <- plot_prediction_with_ci(
    prediction_data = prediction_data,
    x_var = "age_days",
    y_var = "predicted",
    output_stub = output_stub,
    color_var = "speaker",
    x_label = "Age (days)",
    y_label = y_label,
    title = sprintf("%s by age and speaker", bundle$response_name),
    observed_data = data,
    observed_x_var = "age_days",
    observed_y_var = bundle$response_name
  )

  list(data = prediction_data, csv_path = csv_path, plot_path = plot_result$path)
}

plot_input_output_predictions <- function(bundle, data, predictor_name, x_label, y_label, output_stub, nsim = analysis_options$nsim_default) {
  prediction_data <- create_input_output_predictions(bundle, data, predictor_name = predictor_name, nsim = nsim)
  csv_path <- file.path(analysis_paths$results_predictions, sprintf("%s_predictions.csv", output_stub))
  utils::write.csv(prediction_data, csv_path, row.names = FALSE)

  plot_result <- plot_prediction_with_ci(
    prediction_data = prediction_data,
    x_var = predictor_name,
    y_var = "predicted",
    output_stub = output_stub,
    color_var = "speaker",
    x_label = x_label,
    y_label = y_label,
    title = sprintf("%s vs %s", bundle$response_name, predictor_name),
    facet_var = "speaker",
    observed_data = data,
    observed_x_var = predictor_name,
    observed_y_var = bundle$response_name,
    width = 10,
    height = 4.8
  )

  list(data = prediction_data, csv_path = csv_path, plot_path = plot_result$path)
}

plot_fixed_effects_forest <- function(bundle, output_stub) {
  effects_table <- tidy_model_fixed_effects(bundle$model, conf.int = TRUE)
  plot_object <- ggplot2::ggplot(effects_table, ggplot2::aes(x = estimate, y = stats::reorder(term, estimate))) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed") +
    ggplot2::geom_errorbarh(ggplot2::aes(xmin = conf.low, xmax = conf.high), height = 0.15) +
    ggplot2::geom_point(size = 2.2) +
    ggplot2::labs(
      title = sprintf("Fixed effects: %s", bundle$response_name),
      x = "Estimate",
      y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12)

  plot_object <- apply_white_plot_theme(plot_object)

  output_path <- file.path(analysis_paths$results_plots, sprintf("%s_forest.png", output_stub))
  save_prediction_plot(plot_object, output_path, width = 8, height = 4.8)
  list(table = effects_table, path = output_path)
}
