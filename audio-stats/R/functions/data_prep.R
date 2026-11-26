# Data preparation helpers for the HindiBabyNet R workflow.

required_columns <- list(
  final_master = c(
    "participant_id",
    "REC_date",
    "birthdate",
    "child_sex",
    "mother_education",
    "father_education",
    "SES",
    "Location",
    "age_days",
    "age_months",
    "age_z",
    "age_z2",
    "recording_duration_hours",
    "key_child_count_hour",
    "key_child_duration_hour"
  ),
  input_long = c(
    "participant_id",
    "speaker",
    "age_days",
    "age_months",
    "age_z",
    "age_z2",
    "child_sex",
    "SES",
    "mother_education",
    "father_education",
    "Location",
    "recording_duration_hours",
    "input_count_hour",
    "input_duration_hour"
  ),
  # One row per participant: the input of each speaker is in its own column.
  input_output_wide = c(
    "participant_id",
    "age_days",
    "age_months",
    "age_z",
    "age_z2",
    "child_sex",
    "SES",
    "mother_education",
    "father_education",
    "Location",
    "recording_duration_hours",
    "adult_female_count_hour",
    "adult_male_count_hour",
    "other_child_count_hour",
    "adult_female_duration_hour",
    "adult_male_duration_hour",
    "other_child_duration_hour",
    "key_child_count_hour",
    "key_child_duration_hour"
  )
)

read_public_dataset <- function(path, label) {
  if (!file.exists(path)) {
    stop(
      sprintf(
        paste(
          "Required dataset not found: %s. Run the command hbn-audio-stats.",
          "If the other dataset files exist, it adds only the missing input_output_wide.csv."
        ),
        path
      ),
      call. = FALSE
    )
  }

  utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

assert_public_identifiers <- function(dataframe, label) {
  if (!("participant_id" %in% names(dataframe))) {
    stop(sprintf("Dataset '%s' is missing required column: participant_id", label), call. = FALSE)
  }
  if ("original_par_id" %in% names(dataframe)) {
    stop(sprintf("Dataset '%s' contains forbidden private column: original_par_id", label), call. = FALSE)
  }
  invisible(dataframe)
}

assert_required_columns <- function(dataframe, label, columns) {
  missing_columns <- setdiff(columns, names(dataframe))
  if (length(missing_columns) > 0) {
    stop(
      sprintf(
        "Dataset '%s' is missing required columns: %s",
        label,
        paste(missing_columns, collapse = ", ")
      ),
      call. = FALSE
    )
  }
  invisible(dataframe)
}

coerce_common_factors <- function(dataframe) {
  factor_columns <- c("participant_id", "child_sex", "mother_education", "father_education", "Location")
  for (column in intersect(factor_columns, names(dataframe))) {
    dataframe[[column]] <- factor(dataframe[[column]])
  }

  if ("SES" %in% names(dataframe)) {
    ses_levels <- c("primary_school", "high_school", "Intermediate", "bachelor", "master")
    dataframe[["SES"]] <- factor(dataframe[["SES"]], levels = ses_levels, ordered = TRUE)
    dataframe[["SES_num"]] <- as.numeric(dataframe[["SES"]])
  }

  if ("speaker" %in% names(dataframe)) {
    speaker_values <- unique(dataframe[["speaker"]])
    speaker_levels <- unique(c(analysis_options$speaker_order, setdiff(speaker_values, analysis_options$speaker_order)))
    dataframe[["speaker"]] <- factor(dataframe[["speaker"]], levels = speaker_levels)
  }

  dataframe
}

# Every duration per hour (seconds per hour) also gets a column in minutes per hour.
add_duration_minutes_per_hour <- function(dataframe) {
  for (column in grep("_duration_hour$", names(dataframe), value = TRUE)) {
    minutes_column <- sub("_duration_hour$", "_duration_min_hour", column)
    if (!minutes_column %in% names(dataframe)) {
      dataframe[[minutes_column]] <- dataframe[[column]] / 60.0
    }
  }
  dataframe
}

store_age_day_summary <- function(final_master) {
  valid_age_days <- final_master$age_days[!is.na(final_master$age_days)]
  analysis_runtime$age_days_mean <- mean(valid_age_days)
  analysis_runtime$age_days_sd <- stats::sd(valid_age_days)
  analysis_options$age_days_mean <<- analysis_runtime$age_days_mean
  analysis_options$age_days_sd <<- analysis_runtime$age_days_sd
  invisible(final_master)
}

build_dataset_summary <- function(datasets) {
  data.frame(
    dataset = names(datasets),
    n_rows = vapply(datasets, nrow, integer(1)),
    n_columns = vapply(datasets, ncol, integer(1)),
    stringsAsFactors = FALSE
  )
}

load_analysis_datasets <- function(paths = analysis_paths, write_summary = FALSE) {
  dataset_paths <- list(
    final_master = file.path(paths$data_dir, "final_master.csv"),
    input_long = file.path(paths$data_dir, "input_long.csv"),
    input_output_wide = file.path(paths$data_dir, "input_output_wide.csv")
  )

  datasets <- lapply(names(dataset_paths), function(label) {
    dataframe <- read_public_dataset(dataset_paths[[label]], label)
    assert_public_identifiers(dataframe, label)
    assert_required_columns(dataframe, label, required_columns[[label]])
    dataframe <- coerce_common_factors(dataframe)
    add_duration_minutes_per_hour(dataframe)
  })
  names(datasets) <- names(dataset_paths)

  store_age_day_summary(datasets$final_master)

  dataset_summary <- build_dataset_summary(datasets)
  if (isTRUE(write_summary)) {
    dir.create(paths$results_tables, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(dataset_summary, file.path(paths$results_tables, "dataset_summary.csv"), row.names = FALSE)
  }

  list(
    datasets = datasets,
    dataset_summary = dataset_summary,
    age_days_mean = analysis_runtime$age_days_mean,
    age_days_sd = analysis_runtime$age_days_sd
  )
}
