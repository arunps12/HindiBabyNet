# Prediction plotting helpers for the HindiBabyNet R workflow.

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

validate_age_plot_data <- function(prediction_data, tolerance = 1e-8) {
  if (!("age_days" %in% names(prediction_data)) || !("age_z" %in% names(prediction_data))) {
    stop("Prediction data must include both age_days and age_z.", call. = FALSE)
  }

  expected_age_z <- age_days_to_z(prediction_data$age_days)
  if (any(abs(expected_age_z - prediction_data$age_z) > tolerance, na.rm = TRUE)) {
    stop("Prediction data violates the age_days to age_z mapping used for plotting.", call. = FALSE)
  }

  invisible(prediction_data)
}

save_prediction_plot <- function(plot_object, output_path, width = 8, height = 5.5, dpi = 300) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(output_path, plot_object, width = width, height = height, dpi = dpi, limitsize = FALSE)
  invisible(output_path)
}

plot_prediction_with_ci <- function(prediction_data, x_var, y_var, output_stub, color_var = NULL, x_label = NULL, y_label = NULL, title = NULL) {
  aesthetics <- ggplot2::aes_string(x = x_var, y = y_var, color = color_var %||% NULL, fill = color_var %||% NULL, group = color_var %||% NULL)
  plot_object <- ggplot2::ggplot(prediction_data, aesthetics) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lower, ymax = upper), alpha = 0.18, color = NA, show.legend = !is.null(color_var)) +
    ggplot2::geom_line(linewidth = 1.0) +
    ggplot2::labs(
      title = title,
      x = x_label %||% x_var,
      y = y_label %||% y_var,
      color = color_var,
      fill = color_var
    ) +
    ggplot2::theme_minimal(base_size = 12)

  output_path <- file.path(analysis_paths$results_plots, sprintf("%s.png", output_stub))
  save_prediction_plot(plot_object, output_path)
  list(plot = plot_object, path = output_path)
}

create_age_speaker_predictions <- function(bundle, data, nsim = analysis_options$nsim_default, n_points = 60) {
  prediction_grid <- build_age_speaker_grid(data, n_points = n_points)
  validate_age_plot_data(prediction_grid)
  prediction_data <- compute_monte_carlo_ci(bundle, prediction_grid, nsim = nsim)
  validate_age_plot_data(prediction_data)
  prediction_data
}

create_input_output_predictions <- function(bundle, data, predictor_name, nsim = analysis_options$nsim_default, n_points = 60) {
  prediction_grid <- build_input_output_grid(data, predictor_name = predictor_name, n_points = n_points)
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
    title = sprintf("%s by age and speaker", bundle$response_name)
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
    title = sprintf("%s vs %s", bundle$response_name, predictor_name)
  )

  list(data = prediction_data, csv_path = csv_path, plot_path = plot_result$path)
}

plot_fixed_effects_forest <- function(bundle, output_stub) {
  effects_table <- broom.mixed::tidy(bundle$model, effects = "fixed", conf.int = TRUE)
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

  output_path <- file.path(analysis_paths$results_plots, sprintf("%s_forest.png", output_stub))
  save_prediction_plot(plot_object, output_path, width = 8, height = 4.8)
  list(table = effects_table, path = output_path)
}
