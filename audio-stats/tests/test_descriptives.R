# Tests the descriptive tables of R/functions/descriptives.R with a small
# synthetic participant table. All expected values are calculated by hand.
# Run from the project root: Rscript audio-stats/tests/test_descriptives.R
run_tests <- function() {
  script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
  root <- normalizePath(file.path(dirname(script), ".."), winslash = "/", mustWork = TRUE)
  source(file.path(root, "R", "functions", "descriptives.R"), local = TRUE)

  scratch <- tempfile("hbn-audio-stats-descriptives-")
  dir.create(scratch)
  on.exit(unlink(scratch, recursive = TRUE), add = TRUE)

  # Six synthetic participants with missing values and equal counts.
  master <- data.frame(
    participant_id = sprintf("P%03d", 1:6),
    age_days = c(300, 400, NA, 500, 600, 700),
    age_months = c(10, 13, NA, 16, 19, 22),
    age_z = c(-2, -1, NA, 0, 1, 2),
    child_sex = c("M", "F", "F", "M", NA, "M"),
    SES = c("master", "bachelor", "Intermediate", "bachelor", NA, "high_school"),
    Location = c("Delhi", "Jaipur", "Jaipur", "Delhi", NA, "Agra"),
    recording_duration_hours = c(2, 4, 4, 5, 5, 7),
    mother_education = c("master", "bachelor", "Intermediate", "bachelor", NA, "high_school"),
    father_education = c("bachelor", "bachelor", "master", "bachelor", "master", "primary_school"),
    adult_female_count_hour = c(1, 3, 3, 5, NA, NA),
    adult_male_count_hour = c(2, 2, 2, 2, 2, 2),
    other_child_count_hour = NA_real_,
    key_child_count_hour = c(10, 20, 30, 40, 50, 60),
    adult_female_duration_hour = c(0, 10, 20, 30, 40, 50),
    adult_male_duration_hour = c(5, NA, 5, NA, 5, NA),
    other_child_duration_hour = c(1, 2, 3, 4, 5, 6),
    key_child_duration_hour = c(100, 100, 200, 200, NA, NA),
    stringsAsFactors = FALSE
  )
  # Missing values are saved as empty cells, as in final_master.csv.
  master_file <- file.path(scratch, "final_master.csv")
  utils::write.csv(master, master_file, row.names = FALSE, na = "")
  saved_master <- read_descriptive_table(master_file)
  stopifnot(identical(names(saved_master), names(master)), nrow(saved_master) == 6L)

  tables <- build_descriptive_tables(saved_master)

  same_numbers <- function(actual, expected) {
    identical(is.na(actual), is.na(expected)) &&
      isTRUE(all.equal(as.numeric(actual[!is.na(actual)]), as.numeric(expected[!is.na(expected)]), tolerance = 1e-12))
  }
  expect_summary_row <- function(table, row, variable, non_missing, mean, std, min, median, max) {
    stopifnot(
      identical(names(table), c("variable", "non_missing", "mean", "std", "min", "median", "max")),
      identical(table$variable[row], variable),
      table$non_missing[row] == non_missing,
      same_numbers(
        unlist(table[row, c("mean", "std", "min", "median", "max")], use.names = FALSE),
        c(mean, std, min, median, max)
      )
    )
  }

  # Names and order of the nine tables.
  stopifnot(identical(names(tables), c(
    "participant_summary", "missing_values_summary", "recording_duration_summary",
    "speaker_count_summary", "speaker_duration_summary", "age_summary",
    "sex_distribution", "location_distribution", "education_distribution"
  )))

  # participant_summary: eight columns, one row per participant, same order.
  participants <- tables$participant_summary
  stopifnot(
    identical(names(participants), c("participant_id", "age_days", "age_months", "age_z",
                                     "child_sex", "SES", "Location", "recording_duration_hours")),
    identical(participants$participant_id, master$participant_id),
    same_numbers(participants$age_days, master$age_days),
    same_numbers(participants$age_months, master$age_months),
    same_numbers(participants$age_z, master$age_z),
    identical(participants$child_sex, master$child_sex),
    identical(participants$SES, master$SES),
    identical(participants$Location, master$Location),
    same_numbers(participants$recording_duration_hours, master$recording_duration_hours)
  )

  # missing_values_summary: one row per column of the participant table.
  missing_values <- tables$missing_values_summary
  expected_missing <- c(0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2, 0, 6, 0, 0, 3, 0, 2)
  stopifnot(
    identical(names(missing_values), c("column", "missing_count", "missing_percent")),
    identical(missing_values$column, names(master)),
    all(missing_values$missing_count == expected_missing),
    same_numbers(missing_values$missing_percent, expected_missing / 6 * 100)
  )

  # recording_duration_summary. Values 2, 4, 4, 5, 5, 7: mean 4.5,
  # sum of squared differences 13.5, population SD sqrt(13.5 / 6) = 1.5.
  stopifnot(nrow(tables$recording_duration_summary) == 1L)
  expect_summary_row(tables$recording_duration_summary, 1, "recording_duration_hours", 6, 4.5, 1.5, 2, 4.5, 7)
  # The sample SD (divided by n - 1) is a different number.
  stopifnot(abs(stats::sd(master$recording_duration_hours) - 1.5) > 0.1)

  # speaker_count_summary.
  counts <- tables$speaker_count_summary
  stopifnot(nrow(counts) == 4L)
  expect_summary_row(counts, 1, "adult_female_count_hour", 4, 3, sqrt(8 / 4), 1, 3, 5)
  expect_summary_row(counts, 2, "adult_male_count_hour", 6, 2, 0, 2, 2, 2)
  expect_summary_row(counts, 3, "other_child_count_hour", 0, NA, NA, NA, NA, NA)
  expect_summary_row(counts, 4, "key_child_count_hour", 6, 35, sqrt(1750 / 6), 10, 35, 60)

  # speaker_duration_summary.
  durations <- tables$speaker_duration_summary
  stopifnot(nrow(durations) == 4L)
  expect_summary_row(durations, 1, "adult_female_duration_hour", 6, 25, sqrt(1750 / 6), 0, 25, 50)
  expect_summary_row(durations, 2, "adult_male_duration_hour", 3, 5, 0, 5, 5, 5)
  expect_summary_row(durations, 3, "other_child_duration_hour", 6, 3.5, sqrt(17.5 / 6), 1, 3.5, 6)
  expect_summary_row(durations, 4, "key_child_duration_hour", 4, 150, 50, 100, 150, 200)

  # age_summary.
  ages <- tables$age_summary
  stopifnot(nrow(ages) == 3L)
  expect_summary_row(ages, 1, "age_days", 5, 500, sqrt(100000 / 5), 300, 500, 700)
  expect_summary_row(ages, 2, "age_months", 5, 16, sqrt(90 / 5), 10, 16, 22)
  expect_summary_row(ages, 3, "age_z", 5, 0, sqrt(10 / 5), -2, 0, 2)

  # sex_distribution: largest count first, missing values counted as "missing".
  stopifnot(
    identical(names(tables$sex_distribution), c("child_sex", "count")),
    identical(tables$sex_distribution$child_sex, c("M", "F", "missing")),
    identical(tables$sex_distribution$count, c(3L, 2L, 1L))
  )

  # location_distribution: equal counts keep the order of first appearance
  # (Delhi before Jaipur, "missing" before Agra).
  stopifnot(
    identical(names(tables$location_distribution), c("Location", "count")),
    identical(tables$location_distribution$Location, c("Delhi", "Jaipur", "missing", "Agra")),
    identical(tables$location_distribution$count, c(2L, 2L, 1L, 1L))
  )

  # education_distribution: sorted by type, then by value. Capital letters come
  # before small letters ("Intermediate" before "bachelor").
  education <- tables$education_distribution
  stopifnot(
    identical(names(education), c("education_type", "education_value", "count")),
    identical(education$education_type, c(rep("father_education", 3), rep("mother_education", 5))),
    identical(education$education_value, c("bachelor", "master", "primary_school",
                                           "Intermediate", "bachelor", "high_school", "master", "missing")),
    identical(education$count, c(3L, 2L, 1L, 1L, 2L, 1L, 1L, 1L))
  )

  # A participant table without a needed column is an error.
  error <- tryCatch(build_descriptive_tables(saved_master[, names(saved_master) != "Location"]), error = identity)
  stopifnot(inherits(error, "error"), grepl("Location", conditionMessage(error), fixed = TRUE))

  # Names of the 19 figures.
  stopifnot(length(descriptive_figure_names) == 19L, !anyDuplicated(descriptive_figure_names))
  message("Descriptive table tests passed.")

  # The figures need ggplot2. They include panels without data (other_child counts are all missing).
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    message("ggplot2 is not installed: the figure test was skipped.")
    return(invisible(NULL))
  }
  input_long <- do.call(rbind, lapply(descriptive_speakers, function(speaker) {
    data.frame(
      participant_id = saved_master$participant_id,
      speaker = speaker,
      input_count_hour = to_number(saved_master[[paste0(speaker, "_count_hour")]]),
      input_duration_hour = to_number(saved_master[[paste0(speaker, "_duration_hour")]]),
      stringsAsFactors = FALSE
    )
  }))
  figure_dir <- file.path(scratch, "figures")
  figure_paths <- suppressWarnings(save_descriptive_figures(saved_master, input_long, figure_dir))
  stopifnot(
    identical(names(figure_paths), descriptive_figure_names),
    all(file.exists(figure_paths)),
    identical(sort(list.files(figure_dir)), sort(paste0(descriptive_figure_names, ".png")))
  )
  message("Descriptive figure test passed.")
}
run_tests()
