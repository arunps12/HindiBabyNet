read_csv_required <- function(path, label) {
  if (!file.exists(path)) {
    stop(sprintf("Required dataset not found: %s", path), call. = FALSE)
  }

  readr::read_csv(path, show_col_types = FALSE)
}

read_excel_optional <- function(path, label) {
  if (!file.exists(path)) {
    return(NULL)
  }

  readxl::read_excel(path)
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

standardize_identifier <- function(values) {
  values <- as.character(values)
  values <- stringr::str_trim(values)
  values[values %in% c("", "NA", "NaN", "NULL")] <- NA_character_
  values
}

standardize_speaker_labels <- function(values, mapping = analysis_config$speaker_mapping) {
  speaker_values <- standardize_identifier(values)
  normalized_mapping <- stats::setNames(unlist(mapping, use.names = FALSE), names(mapping))
  output <- unname(normalized_mapping[speaker_values])
  output[is.na(output)] <- speaker_values[is.na(output)]
  output
}

standardize_dataframe_columns <- function(dataframe, participant_column = analysis_config$variables$participant_id, speaker_column = NULL) {
  if (participant_column %in% names(dataframe)) {
    dataframe[[participant_column]] <- standardize_identifier(dataframe[[participant_column]])
  }

  if (!is.null(speaker_column) && speaker_column %in% names(dataframe)) {
    dataframe[[speaker_column]] <- standardize_speaker_labels(dataframe[[speaker_column]])
  }

  dataframe
}

add_continuous_ses <- function(dataframe, ses_column = analysis_config$variables$ses, ses_num_column = analysis_config$variables$ses_num) {
  if (!(ses_column %in% names(dataframe))) {
    return(dataframe)
  }

  ses_levels <- unlist(analysis_config$ses_levels %||% c("primary_school", "high_school", "Intermediate", "bachelor", "master"), use.names = FALSE)
  dataframe[[ses_column]] <- factor(dataframe[[ses_column]], levels = ses_levels, ordered = TRUE)
  dataframe[[ses_num_column]] <- as.numeric(dataframe[[ses_column]])
  dataframe
}

add_age_squared_terms <- function(dataframe, age_days_column = analysis_config$variables$age_days) {
  if (!(age_days_column %in% names(dataframe))) {
    return(dataframe)
  }

  dataframe[["age_days_sq"]] <- dataframe[[age_days_column]] ^ 2
  valid_rows <- !is.na(dataframe[["age_days_sq"]])
  if (sum(valid_rows) == 0) {
    dataframe[["age_days_sq_z"]] <- NA_real_
    return(dataframe)
  }

  squared_values <- dataframe[["age_days_sq"]][valid_rows]
  sd_value <- stats::sd(squared_values)
  if (is.na(sd_value) || sd_value == 0) {
    dataframe[["age_days_sq_z"]] <- ifelse(valid_rows, 0.0, NA_real_)
    return(dataframe)
  }

  dataframe[["age_days_sq_z"]] <- NA_real_
  dataframe[["age_days_sq_z"]][valid_rows] <- as.numeric(scale(squared_values))
  dataframe
}

assert_unique_keys <- function(dataframe, label, key_columns) {
  duplicate_rows <- dataframe |>
    dplyr::count(dplyr::across(dplyr::all_of(key_columns)), name = "n") |>
    dplyr::filter(.data$n > 1)

  if (nrow(duplicate_rows) > 0) {
    stop(
      sprintf(
        "Dataset '%s' has duplicate rows for key(s): %s",
        label,
        paste(key_columns, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  invisible(dataframe)
}

validate_expected_speakers <- function(dataframe, label, speaker_column = analysis_config$variables$speaker) {
  expected <- unlist(analysis_config$validation$expected_speakers, use.names = FALSE)
  observed <- sort(unique(stats::na.omit(as.character(dataframe[[speaker_column]]))))
  missing_levels <- setdiff(expected, observed)
  invalid_levels <- setdiff(observed, expected)

  if (length(missing_levels) > 0) {
    stop(
      sprintf("Dataset '%s' is missing speaker categories: %s", label, paste(missing_levels, collapse = ", ")),
      call. = FALSE
    )
  }

  if (length(invalid_levels) > 0) {
    stop(
      sprintf("Dataset '%s' has invalid speaker labels: %s", label, paste(invalid_levels, collapse = ", ")),
      call. = FALSE
    )
  }

  invisible(dataframe)
}

validate_numeric_columns <- function(dataframe, label, columns) {
  non_numeric <- columns[!vapply(dataframe[columns], is.numeric, logical(1))]
  if (length(non_numeric) > 0) {
    stop(
      sprintf("Dataset '%s' has non-numeric columns where numeric values were required: %s", label, paste(non_numeric, collapse = ", ")),
      call. = FALSE
    )
  }
  invisible(dataframe)
}

validate_age_range <- function(dataframe, age_column = analysis_config$variables$age_days) {
  min_age <- analysis_config$validation$min_age_days %||% 0
  max_age <- analysis_config$validation$max_age_days %||% Inf
  invalid <- dataframe |>
    dplyr::filter(!is.na(.data[[age_column]]) & (.data[[age_column]] < min_age | .data[[age_column]] > max_age))

  if (nrow(invalid) > 0) {
    stop(sprintf("Found %d rows with impossible age values in '%s'.", nrow(invalid), age_column), call. = FALSE)
  }

  invisible(dataframe)
}

set_reference_levels <- function(dataframe) {
  reference_levels <- analysis_config$reference_levels

  dataframe <- add_continuous_ses(dataframe)
  dataframe <- add_age_squared_terms(dataframe)

  if (analysis_config$variables$speaker %in% names(dataframe)) {
    speaker_levels <- unique(c(
      unlist(analysis_config$validation$expected_speakers, use.names = FALSE),
      setdiff(unique(dataframe[[analysis_config$variables$speaker]]), unlist(analysis_config$validation$expected_speakers, use.names = FALSE))
    ))
    dataframe[[analysis_config$variables$speaker]] <- factor(
      dataframe[[analysis_config$variables$speaker]],
      levels = speaker_levels
    )
    dataframe[[analysis_config$variables$speaker]] <- stats::relevel(
      dataframe[[analysis_config$variables$speaker]],
      ref = reference_levels$speaker
    )
  }

  factor_mappings <- c(
    child_sex = analysis_config$variables$child_sex,
    city = "city"
  )

  for (mapping_name in names(factor_mappings)) {
    column_name <- factor_mappings[[mapping_name]]
    if (column_name %in% names(dataframe)) {
      dataframe[[column_name]] <- factor(dataframe[[column_name]])
      ref_level <- reference_levels[[mapping_name]] %||% NULL
      if (!is.null(ref_level) && ref_level %in% levels(dataframe[[column_name]])) {
        dataframe[[column_name]] <- stats::relevel(dataframe[[column_name]], ref = ref_level)
      }
    }
  }

  dataframe
}

store_age_day_summary <- function(dataframe, age_column = analysis_config$variables$age_days) {
  valid_age_days <- dataframe[[age_column]][!is.na(dataframe[[age_column]])]
  analysis_runtime$age_days_mean <- mean(valid_age_days)
  analysis_runtime$age_days_sd <- stats::sd(valid_age_days)
  analysis_options$age_days_mean <<- analysis_runtime$age_days_mean
  analysis_options$age_days_sd <<- analysis_runtime$age_days_sd
  invisible(dataframe)
}

build_missingness_table <- function(dataframe) {
  data.frame(
    variable = names(dataframe),
    missing_count = vapply(dataframe, function(column) sum(is.na(column)), numeric(1)),
    missing_percent = vapply(dataframe, function(column) mean(is.na(column)) * 100, numeric(1)),
    stringsAsFactors = FALSE
  )
}

count_complete_cases <- function(dataframe, required_columns) {
  complete_rows <- stats::complete.cases(dataframe[, required_columns, drop = FALSE])
  data.frame(
    n_complete_rows = sum(complete_rows),
    n_complete_participants = dplyr::n_distinct(dataframe$participant_id[complete_rows]),
    stringsAsFactors = FALSE
  )
}

build_complete_case_table <- function(dataframe) {
  rows <- list()
  primary_measures <- analysis_options$primary_acoustic_measures

  family1_common <- c("participant_id", "speaker", "age_z", "SES_num", "child_sex", "city")
  for (measure in primary_measures) {
    rows[[length(rows) + 1]] <- cbind(
      data.frame(
        model_family = "family_1",
        outcome = measure,
        predictor = NA_character_,
        stringsAsFactors = FALSE
      ),
      count_complete_cases(dataframe, c(family1_common, measure))
    )
  }

  family2_common_count <- c("participant_id", "speaker", "key_child_count_per_hour", "input_count_per_hour", "age_z", "SES_num", "child_sex", "city")
  family2_common_duration <- c("participant_id", "speaker", "key_child_duration_per_hour", "input_duration_per_hour", "age_z", "SES_num", "child_sex", "city")
  for (measure in primary_measures) {
    rows[[length(rows) + 1]] <- cbind(
      data.frame(
        model_family = "family_2_count",
        outcome = "key_child_count_per_hour",
        predictor = measure,
        stringsAsFactors = FALSE
      ),
      count_complete_cases(dataframe, c(family2_common_count, measure))
    )
    rows[[length(rows) + 1]] <- cbind(
      data.frame(
        model_family = "family_2_duration",
        outcome = "key_child_duration_per_hour",
        predictor = measure,
        stringsAsFactors = FALSE
      ),
      count_complete_cases(dataframe, c(family2_common_duration, measure))
    )
  }

  bind_rows_safe(rows)
}

build_merge_summary_table <- function(datasets, final_dataframe) {
  data.frame(
    dataset = c(names(datasets), "prosody_long"),
    n_rows = c(vapply(datasets, nrow, numeric(1)), nrow(final_dataframe)),
    n_participants = c(vapply(datasets, function(item) dplyr::n_distinct(item$participant_id), numeric(1)), dplyr::n_distinct(final_dataframe$participant_id)),
    stringsAsFactors = FALSE
  )
}

find_unmatched_ids <- function(left_dataframe, right_dataframe, by_columns) {
  left_keys <- dplyr::distinct(left_dataframe, dplyr::across(dplyr::all_of(by_columns)))
  right_keys <- dplyr::distinct(right_dataframe, dplyr::across(dplyr::all_of(by_columns)))

  list(
    left_only = dplyr::anti_join(left_keys, right_keys, by = by_columns),
    right_only = dplyr::anti_join(right_keys, left_keys, by = by_columns)
  )
}

assert_no_unmatched_ids <- function(unmatched, label) {
  left_only <- unmatched$left_only
  right_only <- unmatched$right_only

  if (nrow(left_only) > 0 || nrow(right_only) > 0) {
    stop(
      sprintf(
        "Join validation failed for %s: %d left-only keys and %d right-only keys.",
        label,
        nrow(left_only),
        nrow(right_only)
      ),
      call. = FALSE
    )
  }

  invisible(unmatched)
}

check_repeated_outcomes_constant <- function(dataframe) {
  counts_ok <- dataframe |>
    dplyr::group_by(.data$participant_id) |>
    dplyr::summarise(n_count_values = dplyr::n_distinct(.data$key_child_count_per_hour, na.rm = FALSE), .groups = "drop") |>
    dplyr::filter(.data$n_count_values > 1)

  duration_ok <- dataframe |>
    dplyr::group_by(.data$participant_id) |>
    dplyr::summarise(n_duration_values = dplyr::n_distinct(.data$key_child_duration_per_hour, na.rm = FALSE), .groups = "drop") |>
    dplyr::filter(.data$n_duration_values > 1)

  if (nrow(counts_ok) > 0) {
    stop("key_child_count_per_hour varies across speaker rows within participant.", call. = FALSE)
  }

  if (nrow(duration_ok) > 0) {
    stop("key_child_duration_per_hour varies across speaker rows within participant.", call. = FALSE)
  }

  invisible(dataframe)
}

build_variation_check_table <- function(dataframe) {
  data.frame(
    check = c(
      "key_child_count_per_hour constant within participant",
      "key_child_duration_per_hour constant within participant",
      "input_count_per_hour varies across speaker rows",
      "input_duration_per_hour varies across speaker rows"
    ),
    status = c(
      dataframe |>
        dplyr::group_by(.data$participant_id) |>
        dplyr::summarise(value = dplyr::n_distinct(.data$key_child_count_per_hour, na.rm = FALSE), .groups = "drop") |>
        dplyr::summarise(ok = all(.data$value <= 1)) |>
        dplyr::pull(.data$ok),
      dataframe |>
        dplyr::group_by(.data$participant_id) |>
        dplyr::summarise(value = dplyr::n_distinct(.data$key_child_duration_per_hour, na.rm = FALSE), .groups = "drop") |>
        dplyr::summarise(ok = all(.data$value <= 1)) |>
        dplyr::pull(.data$ok),
      dataframe |>
        dplyr::group_by(.data$participant_id) |>
        dplyr::summarise(value = dplyr::n_distinct(.data$input_count_per_hour, na.rm = TRUE), .groups = "drop") |>
        dplyr::summarise(ok = all(.data$value > 1)) |>
        dplyr::pull(.data$ok),
      dataframe |>
        dplyr::group_by(.data$participant_id) |>
        dplyr::summarise(value = dplyr::n_distinct(.data$input_duration_per_hour, na.rm = TRUE), .groups = "drop") |>
        dplyr::summarise(ok = all(.data$value > 1)) |>
        dplyr::pull(.data$ok)
    ),
    stringsAsFactors = FALSE
  )
}

prepare_prosody_long_dataset <- function() {
  acoustic <- read_csv_required(analysis_paths$acoustic_features, "acoustic_features") |>
    standardize_dataframe_columns(participant_column = "participant_id", speaker_column = "speaker")
  input_output <- read_csv_required(analysis_paths$input_output_long, "input_output_long") |>
    standardize_dataframe_columns(participant_column = "participant_id", speaker_column = "speaker")
  input_long <- read_csv_required(analysis_paths$input_long, "input_long") |>
    standardize_dataframe_columns(participant_column = "participant_id", speaker_column = "speaker")
  final_master <- read_csv_required(analysis_paths$final_master, "final_master") |>
    standardize_dataframe_columns(participant_column = "participant_id")
  participant_lookup <- read_csv_required(analysis_paths$participant_lookup, "participant_lookup") |>
    dplyr::mutate(
      participant_id = standardize_identifier(.data$participant_id),
      original_par_id = standardize_identifier(.data$original_par_id)
    )
  metadata_cleaned <- read_excel_optional(analysis_paths$metadata_cleaned, "metadata_cleaned")

  assert_required_columns(acoustic, "acoustic_features", c("participant_id", "speaker", unlist(analysis_config$validation$acoustic_numeric_columns, use.names = FALSE)))
  assert_required_columns(input_output, "input_output_long", c("participant_id", "speaker", "age_days", "age_months", "age_z", "age_z2", "child_sex", "SES", "Location", "input_count_hour", "input_count_min_hour", "input_duration_hour", "input_duration_min_hour", "key_child_count_hour", "key_child_count_min_hour", "key_child_duration_hour", "key_child_duration_min_hour"))
  assert_required_columns(input_long, "input_long", c("participant_id", "speaker", "input_count_hour", "input_count_min_hour", "input_duration_hour", "input_duration_min_hour"))
  assert_required_columns(final_master, "final_master", c("participant_id", "age_days", "age_months", "age_z", "age_z2", "child_sex", "SES", "Location"))
  assert_required_columns(participant_lookup, "participant_lookup", c("participant_id", "original_par_id"))

  assert_unique_keys(acoustic, "acoustic_features", c("participant_id", "speaker"))
  assert_unique_keys(input_output, "input_output_long", c("participant_id", "speaker"))
  assert_unique_keys(input_long, "input_long", c("participant_id", "speaker"))
  assert_unique_keys(final_master, "final_master", c("participant_id"))
  assert_unique_keys(participant_lookup, "participant_lookup", c("participant_id"))

  validate_expected_speakers(acoustic, "acoustic_features")
  validate_expected_speakers(input_output, "input_output_long")
  validate_expected_speakers(input_long, "input_long")
  validate_numeric_columns(acoustic, "acoustic_features", unlist(analysis_config$validation$acoustic_numeric_columns, use.names = FALSE))
  validate_age_range(input_output, age_column = "age_days")

  unmatched_acoustic <- find_unmatched_ids(acoustic, input_output, by_columns = c("participant_id", "speaker"))
  unmatched_input_long <- find_unmatched_ids(input_long, input_output, by_columns = c("participant_id", "speaker"))
  unmatched_master <- find_unmatched_ids(final_master, input_output, by_columns = c("participant_id"))

  assert_no_unmatched_ids(unmatched_acoustic, "acoustic_features vs input_output_long")
  assert_no_unmatched_ids(unmatched_input_long, "input_long vs input_output_long")
  assert_no_unmatched_ids(unmatched_master, "final_master vs input_output_long")

  joined <- input_output |>
    dplyr::left_join(
      acoustic,
      by = c("participant_id", "speaker"),
      suffix = c("", "_acoustic")
    ) |>
    dplyr::left_join(
      dplyr::select(participant_lookup, "participant_id", lookup_source_participant_id = "original_par_id"),
      by = "participant_id"
    ) |>
    dplyr::mutate(
      source_participant_id = dplyr::coalesce(.data$source_participant_id, .data$lookup_source_participant_id)
    ) |>
    dplyr::rename(
      city = "Location",
      input_count_per_hour = "input_count_hour",
      input_duration_per_hour = "input_duration_hour",
      key_child_count_per_hour = "key_child_count_hour",
      key_child_duration_per_hour = "key_child_duration_hour"
    ) |>
    dplyr::select(
      "participant_id",
      "source_participant_id",
      "speaker",
      "age_days",
      "age_days_sq",
      "age_days_sq_z",
      "age_months",
      "age_z",
      "age_z2",
      "SES",
      "child_sex",
      "city",
      "input_count_per_hour",
      "input_count_min_hour",
      "input_duration_per_hour",
      "input_duration_min_hour",
      "key_child_count_per_hour",
      "key_child_count_min_hour",
      "key_child_duration_per_hour",
      "key_child_duration_min_hour",
      tidyselect::everything()
    ) |>
    dplyr::select(-tidyselect::any_of("lookup_source_participant_id"))

  acoustic_columns <- unlist(analysis_config$validation$acoustic_numeric_columns, use.names = FALSE)
  missing_acoustic <- acoustic_columns[colSums(is.na(joined[, acoustic_columns, drop = FALSE])) > 0]
  if (any(is.na(joined$participant_id)) || any(is.na(joined$speaker))) {
    stop("Joined prosody dataset contains missing participant_id or speaker values.", call. = FALSE)
  }

  assert_unique_keys(joined, "prosody_long", c("participant_id", "speaker"))
  check_repeated_outcomes_constant(joined)
  validate_age_range(joined, age_column = "age_days")

  joined <- set_reference_levels(joined)
  store_age_day_summary(joined)

  datasets <- list(
    acoustic_features = acoustic,
    final_master = final_master,
    input_long = input_long,
    input_output_long = input_output,
    participant_lookup = participant_lookup
  )

  missingness <- build_missingness_table(joined)
  complete_cases <- build_complete_case_table(joined)
  merge_summary <- build_merge_summary_table(datasets, joined)
  unmatched_table <- bind_rows_safe(list(
    dplyr::mutate(unmatched_acoustic$left_only, source = "acoustic_left_only"),
    dplyr::mutate(unmatched_acoustic$right_only, source = "acoustic_right_only"),
    dplyr::mutate(unmatched_input_long$left_only, source = "input_long_left_only"),
    dplyr::mutate(unmatched_input_long$right_only, source = "input_long_right_only"),
    dplyr::mutate(unmatched_master$left_only, source = "final_master_left_only"),
    dplyr::mutate(unmatched_master$right_only, source = "final_master_right_only")
  ))
  variation_checks <- build_variation_check_table(joined)

  merge_notes <- c(
    sprintf("Participants in acoustic_features: %d", dplyr::n_distinct(acoustic$participant_id)),
    sprintf("Participants in final_master: %d", dplyr::n_distinct(final_master$participant_id)),
    sprintf("Participants in input_output_long: %d", dplyr::n_distinct(input_output$participant_id)),
    sprintf("Participant x speaker rows in prosody_long: %d", nrow(joined)),
    sprintf("Missing acoustic variables with at least one NA after join: %s", if (length(missing_acoustic) == 0) "none" else paste(missing_acoustic, collapse = ", ")),
    sprintf("metadata_cleaned.xlsx available: %s", !is.null(metadata_cleaned))
  )

  list(
    prosody_long = joined,
    merge_summary = merge_summary,
    missingness = missingness,
    complete_cases = complete_cases,
    unmatched = unmatched_table,
    variation_checks = variation_checks,
    merge_notes = merge_notes
  )
}
