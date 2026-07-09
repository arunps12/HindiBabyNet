# Model diagnostics helpers for the HindiBabyNet R workflow.

save_diagnostic_plot <- function(plot_object, output_path, width = 7, height = 5, dpi = 300) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  ggplot2::ggsave(output_path, plot_object, width = width, height = height, dpi = dpi, limitsize = FALSE)
  invisible(output_path)
}

collect_model_flags <- function(model) {
  if (inherits(model, "merMod")) {
    convergence_messages <- tryCatch(model@optinfo$conv$lme4$messages, error = function(e) NULL)
    return(list(
      model_class = class(model)[1],
      singular_fit = lme4::isSingular(model, tol = 1e-4),
      converged = is.null(convergence_messages),
      convergence_messages = if (is.null(convergence_messages)) NA_character_ else paste(convergence_messages, collapse = " | ")
    ))
  }

  if (inherits(model, "glmmTMB")) {
    pd_hess <- tryCatch(isTRUE(model$sdr$pdHess), error = function(e) FALSE)
    return(list(
      model_class = class(model)[1],
      singular_fit = NA,
      converged = pd_hess,
      convergence_messages = if (pd_hess) NA_character_ else "glmmTMB Hessian is not positive definite"
    ))
  }

  list(
    model_class = class(model)[1],
    singular_fit = NA,
    converged = NA,
    convergence_messages = NA_character_
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
    performance::check_heteroscedasticity(bundle$model)$p.value[[1]],
    error = function(e) NA_real_
  )

  data.frame(
    response = bundle$response_name,
    model_family = bundle$model_family,
    model_type = bundle$selection$model_type[[1]],
    model_class = flags$model_class,
    n_complete_cases = bundle$n_complete_cases,
    singular_fit = flags$singular_fit,
    converged = flags$converged,
    convergence_messages = flags$convergence_messages,
    heteroscedasticity_p = heteroscedasticity_p,
    stringsAsFactors = FALSE
  )
}

run_model_diagnostics <- function(bundle, output_dir = analysis_paths$results_diagnostics) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  residual_data <- build_residual_dataframe(bundle$model)
  response_stub <- bundle$response_name

  residual_plot <- ggplot2::ggplot(residual_data, ggplot2::aes(x = fitted, y = residual)) +
    ggplot2::geom_point(alpha = 0.7, size = 1.8) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
    ggplot2::labs(
      title = sprintf("Residual vs fitted: %s", response_stub),
      x = "Fitted values",
      y = "Residuals"
    ) +
    ggplot2::theme_minimal(base_size = 12)

  qq_plot <- ggplot2::ggplot(residual_data, ggplot2::aes(sample = residual)) +
    ggplot2::stat_qq(alpha = 0.7, size = 1.8) +
    ggplot2::stat_qq_line() +
    ggplot2::labs(
      title = sprintf("Residual QQ plot: %s", response_stub),
      x = "Theoretical quantiles",
      y = "Sample quantiles"
    ) +
    ggplot2::theme_minimal(base_size = 12)

  residual_path <- file.path(output_dir, sprintf("%s_residual_vs_fitted.png", response_stub))
  qq_path <- file.path(output_dir, sprintf("%s_qq.png", response_stub))
  save_diagnostic_plot(residual_plot, residual_path)
  save_diagnostic_plot(qq_plot, qq_path)

  summary_table <- build_diagnostic_summary(bundle)
  summary_path <- file.path(output_dir, sprintf("%s_diagnostics_summary.csv", response_stub))
  utils::write.csv(summary_table, summary_path, row.names = FALSE)

  dharma_path <- NA_character_
  if (inherits(bundle$model, "glmmTMB")) {
    dharma_path <- file.path(output_dir, sprintf("%s_dharma.png", response_stub))
    tryCatch({
      png(filename = dharma_path, width = 1400, height = 1000, res = 180)
      simulation <- DHARMa::simulateResiduals(bundle$model)
      plot(simulation)
      dev.off()
    }, error = function(e) {
      if (grDevices::dev.cur() > 1) {
        grDevices::dev.off()
      }
      dharma_path <<- NA_character_
    })
  }

  list(
    response = response_stub,
    summary = summary_table,
    paths = list(
      residual_plot = residual_path,
      qq_plot = qq_path,
      summary_csv = summary_path,
      dharma_plot = dharma_path
    )
  )
}
