bind_rows_safe <- function(items) {
  non_empty <- Filter(Negate(is.null), items)
  if (length(non_empty) == 0) {
    return(data.frame())
  }

  all_columns <- unique(unlist(lapply(non_empty, names), use.names = FALSE))
  aligned <- lapply(non_empty, function(item) {
    missing_columns <- setdiff(all_columns, names(item))
    for (column in missing_columns) {
      item[[column]] <- rep(NA, nrow(item))
    }
    item[all_columns]
  })

  do.call(rbind, aligned)
}

write_csv_output <- function(dataframe, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(dataframe, output_path, row.names = FALSE)
  invisible(output_path)
}

write_text_output <- function(lines, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(lines, con = output_path, useBytes = TRUE)
  invisible(output_path)
}
