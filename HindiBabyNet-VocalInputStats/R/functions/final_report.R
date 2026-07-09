# Final report helpers for the HindiBabyNet R workflow.

quarto_render_cli <- function(quarto_command, report_input, current_format, output_dir, bundle_path) {
  output_basename <- sprintf("%s.%s", tools::file_path_sans_ext(basename(report_input)), current_format)
  args <- c(
    "render",
    report_input,
    "--to",
    current_format,
    "--output",
    output_basename,
    "-P",
    sprintf("reporting_bundle:%s", normalizePath(bundle_path, winslash = "/", mustWork = TRUE))
  )

  output <- system2(quarto_command, args = args, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0) {
    stop(paste(output, collapse = "\n"), call. = FALSE)
  }

  rendered_candidates <- c(
    file.path(getwd(), output_basename),
    file.path(dirname(report_input), output_basename)
  )
  rendered_path <- rendered_candidates[file.exists(rendered_candidates)][1]
  final_path <- file.path(output_dir, output_basename)
  if (is.na(rendered_path) || !nzchar(rendered_path)) {
    stop(
      sprintf(
        "Expected rendered report was not created in any known location: %s",
        paste(rendered_candidates, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (file.exists(final_path)) {
    file.remove(final_path)
  }

  if (!file.rename(rendered_path, final_path)) {
    stop(sprintf("Could not move rendered report to %s", final_path), call. = FALSE)
  }

  final_path
}

render_final_report <- function(format = c("pdf", "html"), bundle_path = file.path(analysis_paths$results_tables, "reporting_bundle.rds")) {
  requested_format <- Sys.getenv("HBN_REPORT_FORMAT", unset = "")
  if (nzchar(requested_format)) {
    format <- strsplit(requested_format, ",", fixed = TRUE)[[1]]
  }

  formats <- unique(match.arg(trimws(format), choices = c("pdf", "html"), several.ok = TRUE))
  formats <- c(intersect("html", formats), setdiff(formats, "html"))
  if (!file.exists(bundle_path)) {
    stop(sprintf("Reporting bundle not found: %s. Run R/03_model_tables_and_plots.R first.", bundle_path), call. = FALSE)
  }

  report_input <- file.path(analysis_paths$r_dir, "report.qmd")
  if (!file.exists(report_input)) {
    stop(sprintf("Report template not found: %s", report_input), call. = FALSE)
  }

  dir.create(analysis_paths$results_report, recursive = TRUE, showWarnings = FALSE)
  quarto_command <- configure_quarto_runtime()
  if (!nzchar(quarto_command)) {
    stop(
      "Quarto CLI could not be found. Install Quarto or set QUARTO_PATH before rendering the report.",
      call. = FALSE
    )
  }

  outputs <- c()
  for (current_format in formats) {
    render_result <- tryCatch(
      {
        quarto_render_cli(
          quarto_command = quarto_command,
          report_input = report_input,
          current_format = current_format,
          output_dir = analysis_paths$results_report,
          bundle_path = bundle_path
        )
      },
      error = function(e) {
        if (identical(current_format, "pdf") && grepl("latex|tinytex|pdflatex", conditionMessage(e), ignore.case = TRUE)) {
          message("PDF rendering failed due to missing LaTeX/TinyTeX dependency; HTML output remains supported.")
          return(NA_character_)
        }
        stop(e)
      }
    )
    outputs <- c(outputs, render_result)
  }

  invisible(outputs[!is.na(outputs)])
}
