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
    "child_sex",
    "SES",
    "mother_education",
    "father_education",
    "Location",
    "recording_duration_hours",
    "input_count_hour",
    "input_duration_hour"
  ),
  input_output_long = c(
    "participant_id",
    "speaker",
    "age_days",
    "age_months",
    "age_z",
    "child_sex",
    "SES",
    "mother_education",
    "father_education",
    "Location",
    "recording_duration_hours",
    "input_count_hour",
    "input_duration_hour",
    "key_child_count_hour",
    "key_child_duration_hour"
  )
)

read_public_dataset <- function(path, label) {
  if (!file.exists(path)) {
    stop(sprintf("Required dataset not found: %s", path), call. = FALSE)
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
  factor_columns <- c("participant_id", "child_sex", "SES", "mother_education", "father_education", "Location")
  for (column in intersect(factor_columns, names(dataframe))) {
    dataframe[[column]] <- factor(dataframe[[column]])
  }

  if ("speaker" %in% names(dataframe)) {
    speaker_values <- unique(dataframe[["speaker"]])
    speaker_levels <- unique(c(analysis_options$speaker_order, setdiff(speaker_values, analysis_options$speaker_order)))
    dataframe[["speaker"]] <- factor(dataframe[["speaker"]], levels = speaker_levels)
  }

  dataframe
}

add_duration_minutes_per_hour <- function(dataframe) {
  if ("input_duration_hour" %in% names(dataframe) && !"input_duration_min_hour" %in% names(dataframe)) {
    dataframe[["input_duration_min_hour"]] <- dataframe[["input_duration_hour"]] / 60.0
  }
  if ("key_child_duration_hour" %in% names(dataframe) && !"key_child_duration_min_hour" %in% names(dataframe)) {
    dataframe[["key_child_duration_min_hour"]] <- dataframe[["key_child_duration_hour"]] / 60.0
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
    input_output_long = file.path(paths$data_dir, "input_output_long.csv")
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
