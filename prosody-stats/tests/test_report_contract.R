# Checks the layout of the report file and that the four analysis parts are
# unchanged. No data is read.
# Run from the project root: Rscript prosody-stats/tests/test_report_contract.R
local({
  script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  root <- if (length(script)) {
    dirname(dirname(normalizePath(gsub("~+~", " ", script[[1]], fixed = TRUE))))
  } else getwd()
  root <- normalizePath(root, winslash = "/", mustWork = TRUE)
  fingerprint <- function(expressions) {
    path <- tempfile()
    on.exit(unlink(path))
    writeLines(deparse(expressions, width.cutoff = 500L), path)
    unname(tools::md5sum(path))
  }
  # One report file and no separate analysis scripts.
  stopifnot(identical(list.files(root, pattern = "\\.Rmd$"), "prosody-stats.Rmd"),
    !length(list.files(file.path(root, "R"), pattern = "^0[1-4].*\\.R$")))
  lines <- readLines(file.path(root, "prosody-stats.Rmd"), warn = FALSE)
  starts <- grep("^```\\{r[ ,}]", lines)
  chunks <- lapply(starts, function(start) {
    end <- start + which(lines[seq.int(start + 1L, length(lines))] == "```")[[1]]
    lines[seq.int(start + 1L, end - 1L)]
  })
  names(chunks) <- sub("^```\\{r ([^,}]+).*", "\\1", lines[starts])
  invisible(lapply(chunks, function(chunk) parse(text = chunk)))
  expected <- c(
    `prepare-prosody-data` = "98df42a54a3041f789a5fbce7b65246e",
    `fit-family1-models` = "225cb8905a38554c81ccc5031bf015eb",
    `fit-family2-models` = "8fcb6ae0558e9d072dae51898b8f27ca",
    `create-tables-plots` = "bdc9986c7ef2abea97807d6fc156dffb"
  )
  actual <- vapply(chunks[names(expected)], function(chunk) fingerprint(parse(text = chunk)), character(1))
  header <- lines[seq_len(which(lines == "---")[[2]])]
  stopifnot(identical(actual, expected), !anyDuplicated(names(chunks)),
    any(header == "  run_workflow: true"), any(header == "  html_document:"),
    !any(grepl("pdf_document:", lines, fixed = TRUE)))
  # The Knit button saves the report in the results folder.
  knit_code <- sub("^knit:\\s*", "", grep("^knit:", header, value = TRUE))
  stopifnot(length(knit_code) == 1L, is.function(eval(parse(text = knit_code))),
    grepl("hbn_results", knit_code, fixed = TRUE), grepl("\"prosody-stats\"", knit_code, fixed = TRUE))
  # The helper files contain only function definitions.
  for (path in list.files(file.path(root, "R", "functions"), pattern = "\\.R$", full.names = TRUE)) {
    expressions <- parse(path)
    stopifnot(all(vapply(expressions, function(x) is.call(x) && identical(x[[1]], as.name("<-")) &&
      is.call(x[[3]]) && identical(x[[3]][[1]], as.name("function")), logical(1))))
  }
  env <- new.env(parent = globalenv())
  source(file.path(root, "R", "functions", "table_helpers.R"), local = env)
  stopifnot(nrow(env$bind_rows_safe(list(data.frame(a = character()), data.frame(b = character())))) == 0L)
  message(sprintf("Report checks passed: %d chunks can be read and the four analysis parts are unchanged.", length(chunks)))
})