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

save_plot <- function(plot_object, output_path, width = 8, height = 5.5, dpi = 300) {
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

plot_prediction_with_ci <- function(prediction_data, x_var, y_var, color_var = NULL, x_label = NULL, y_label = NULL, title = NULL, facet_var = NULL, observed_data = NULL, observed_x_var = x_var, observed_y_var = y_var) {
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
    ggplot2::labs(title = title, x = x_label %||% x_var, y = y_label %||% y_var, color = color_var, fill = color_var) +
    ggplot2::theme_minimal(base_size = 12)

  if (!is.null(facet_var)) {
    plot_object <- plot_object + ggplot2::facet_wrap(stats::as.formula(paste("~", facet_var)))
  }

  apply_white_plot_theme(plot_object)
}

plot_fixed_effects_forest <- function(bundle) {
  effects_table <- tidy_model_fixed_effects(bundle$model, conf.int = TRUE)
  ggplot2::ggplot(effects_table, ggplot2::aes(x = estimate, y = stats::reorder(term, estimate))) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed") +
    ggplot2::geom_errorbarh(ggplot2::aes(xmin = conf.low, xmax = conf.high), height = 0.15) +
    ggplot2::geom_point(size = 2.2) +
    ggplot2::labs(title = sprintf("Fixed effects: %s", bundle$response_name), x = "Estimate", y = NULL) +
    ggplot2::theme_minimal(base_size = 12) |>
    apply_white_plot_theme()
}

plot_age_speaker_predictions <- function(bundle, prediction_data, observed_data, y_label) {
  plot_prediction_with_ci(
    prediction_data = prediction_data,
    x_var = "age_days",
    y_var = "predicted",
    color_var = "speaker",
    x_label = "Age (days)",
    y_label = y_label,
    title = sprintf("%s by age and speaker", bundle$response_name),
    observed_data = observed_data,
    observed_x_var = "age_days",
    observed_y_var = bundle$response_name
  )
}

plot_predictor_speaker_predictions <- function(bundle, prediction_data, observed_data, predictor_name, y_label) {
  plot_prediction_with_ci(
    prediction_data = prediction_data,
    x_var = predictor_name,
    y_var = "predicted",
    color_var = "speaker",
    x_label = predictor_name,
    y_label = y_label,
    title = sprintf("%s vs %s", bundle$response_name, predictor_name),
    facet_var = "speaker",
    observed_data = observed_data,
    observed_x_var = predictor_name,
    observed_y_var = bundle$response_name
  )
}
