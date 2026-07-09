# Final report helpers for the HindiBabyNet R workflow.

render_final_report <- function(format = c("pdf", "html"), bundle_path = file.path(analysis_paths$results_tables, "reporting_bundle.rds")) {
  formats <- unique(match.arg(format, choices = c("pdf", "html"), several.ok = TRUE))
  if (!file.exists(bundle_path)) {
    stop(sprintf("Reporting bundle not found: %s. Run R/03_model_tables_and_plots.R first.", bundle_path), call. = FALSE)
  }

  report_input <- file.path(analysis_paths$r_dir, "report.qmd")
  if (!file.exists(report_input)) {
    stop(sprintf("Report template not found: %s", report_input), call. = FALSE)
  }

  dir.create(analysis_paths$results_report, recursive = TRUE, showWarnings = FALSE)

  outputs <- lapply(formats, function(current_format) {
    quarto::quarto_render(
      input = report_input,
      output_format = current_format,
      output_dir = analysis_paths$results_report,
      execute_params = list(reporting_bundle = bundle_path)
    )
  })

  invisible(unlist(outputs))
}
