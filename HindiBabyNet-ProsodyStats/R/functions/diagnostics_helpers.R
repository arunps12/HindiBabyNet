save_diagnostic_plot <- function(plot_object, output_path, width = 7, height = 5, dpi = 300) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(output_path, plot_object, width = width, height = height, dpi = dpi, limitsize = FALSE, bg = "white")
  invisible(output_path)
}

apply_white_diagnostic_theme <- function(plot_object) {
  plot_object +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      legend.background = ggplot2::element_rect(fill = "white", color = NA),
      legend.key = ggplot2::element_rect(fill = "white", color = NA)
    )
}

fixed_effect_formula <- function(model) {
  lme4::nobars(stats::formula(model))
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
    output$model_family <- bundle$model_family
    output$response <- bundle$response_name
    output$acoustic_measure <- bundle$acoustic_measure
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

  output$model_family <- bundle$model_family
  output$response <- bundle$response_name
  output$acoustic_measure <- bundle$acoustic_measure
  output
}

vif_table <- function(bundle) {
  try_car <- tryCatch(car::vif(bundle$model), error = function(e) e)
  if (!inherits(try_car, "error")) {
    return(tidy_car_vif(try_car, bundle))
  }

  term_level_collinearity_from_matrix(bundle$model, bundle = bundle)
}

collect_model_flags <- function(model) {
  convergence_messages <- tryCatch(model@optinfo$conv$lme4$messages, error = function(e) NULL)
  list(
    model_class = class(model)[1],
    singular_fit = lme4::isSingular(model, tol = 1e-4),
    converged = is.null(convergence_messages),
    convergence_messages = if (is.null(convergence_messages)) NA_character_ else paste(convergence_messages, collapse = " | ")
  )
}

build_residual_dataframe <- function(model) {
  data.frame(
    fitted = as.numeric(stats::fitted(model)),
    residual = as.numeric(stats::residuals(model)),
    stringsAsFactors = FALSE
  )
}

build_diagnostic_summary <- function(bundle) {
  flags <- collect_model_flags(bundle$model)
  heteroscedasticity_p <- tryCatch(
    {
      result <- performance::check_heteroscedasticity(bundle$model)
      p_value <- result$p.value
      if (length(p_value) == 0 || all(is.na(p_value))) {
        NA_real_
      } else {
        as.numeric(p_value[[1]])
      }
    },
    error = function(e) NA_real_
  )

  data.frame(
    response = bundle$response_name,
    model_family = bundle$model_family,
    acoustic_measure = bundle$acoustic_measure,
    model_type = bundle$selection$model_type[[1]],
    model_class = flags$model_class,
    n_complete_cases = bundle$n_complete_cases,
    n_participants = bundle$n_participants,
    singular_fit = flags$singular_fit,
    converged = flags$converged,
    convergence_messages = flags$convergence_messages,
    heteroscedasticity_p = heteroscedasticity_p,
    stringsAsFactors = FALSE
  )
}

run_model_diagnostics <- function(bundle, output_stub, output_dir = analysis_paths$outputs_diagnostics) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  residual_data <- build_residual_dataframe(bundle$model)
  residual_plot <- ggplot2::ggplot(residual_data, ggplot2::aes(x = fitted, y = residual)) +
    ggplot2::geom_point(alpha = 0.7, size = 1.8) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
    ggplot2::labs(title = sprintf("Residual vs fitted: %s", output_stub), x = "Fitted values", y = "Residuals") +
    ggplot2::theme_minimal(base_size = 12)
  residual_plot <- apply_white_diagnostic_theme(residual_plot)

  qq_plot <- ggplot2::ggplot(residual_data, ggplot2::aes(sample = residual)) +
    ggplot2::stat_qq(alpha = 0.7, size = 1.8) +
    ggplot2::stat_qq_line() +
    ggplot2::labs(title = sprintf("Residual QQ plot: %s", output_stub), x = "Theoretical quantiles", y = "Sample quantiles") +
    ggplot2::theme_minimal(base_size = 12)
  qq_plot <- apply_white_diagnostic_theme(qq_plot)

  observed_fitted_plot <- ggplot2::ggplot(residual_data, ggplot2::aes(x = fitted + residual, y = fitted)) +
    ggplot2::geom_point(alpha = 0.7, size = 1.8) +
    ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    ggplot2::labs(title = sprintf("Observed vs fitted: %s", output_stub), x = "Observed", y = "Fitted") +
    ggplot2::theme_minimal(base_size = 12)
  observed_fitted_plot <- apply_white_diagnostic_theme(observed_fitted_plot)

  residual_path <- file.path(output_dir, sprintf("%s_residual_vs_fitted.png", output_stub))
  qq_path <- file.path(output_dir, sprintf("%s_qq.png", output_stub))
  observed_fitted_path <- file.path(output_dir, sprintf("%s_observed_vs_fitted.png", output_stub))
  save_diagnostic_plot(residual_plot, residual_path)
  save_diagnostic_plot(qq_plot, qq_path)
  save_diagnostic_plot(observed_fitted_plot, observed_fitted_path)

  summary_table <- build_diagnostic_summary(bundle)
  summary_path <- file.path(output_dir, sprintf("%s_diagnostics_summary.csv", output_stub))
  utils::write.csv(summary_table, summary_path, row.names = FALSE)

  list(
    response = bundle$response_name,
    summary = summary_table,
    paths = list(
      residual_plot = residual_path,
      qq_plot = qq_path,
      observed_fitted_plot = observed_fitted_path,
      summary_csv = summary_path
    )
  )
}
