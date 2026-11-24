# Descriptive tables and figures of the prepared dataset.
#
# The input tables are final_master.csv (one row per participant) and
# input_long.csv (one row per participant and speaker). Read them with
# read_descriptive_table(): the tables must be plain, without factors.

descriptive_speakers <- c("adult_female", "adult_male", "other_child")
descriptive_count_columns <- c(
  "adult_female_count_hour", "adult_male_count_hour",
  "other_child_count_hour", "key_child_count_hour"
)
descriptive_duration_columns <- c(
  "adult_female_duration_hour", "adult_male_duration_hour",
  "other_child_duration_hour", "key_child_duration_hour"
)
descriptive_rate_columns <- c(descriptive_count_columns, descriptive_duration_columns)
descriptive_participant_columns <- c(
  "participant_id", "age_days", "age_months", "age_z",
  "child_sex", "SES", "Location", "recording_duration_hours"
)
descriptive_figure_names <- c(
  "age_distribution_histogram",
  "recording_duration_histogram",
  "child_sex_distribution",
  "location_distribution",
  "mother_education_distribution",
  "father_education_distribution",
  "input_count_hour_by_speaker_boxplot",
  "input_duration_hour_by_speaker_boxplot",
  "age_vs_adult_female_count_hour",
  "age_vs_adult_male_count_hour",
  "age_vs_other_child_count_hour",
  "age_vs_key_child_count_hour",
  "key_child_vs_adult_female_count_hour",
  "key_child_vs_adult_male_count_hour",
  "key_child_vs_other_child_count_hour",
  "hourly_variables_correlation_heatmap",
  "input_composition_per_participant_stacked_bar",
  "mean_input_count_hour_by_speaker",
  "mean_input_duration_hour_by_speaker"
)

read_descriptive_table <- function(path) {
  utils::read.csv(
    path,
    stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN", "N/A", "null"),
    check.names = FALSE,
    fileEncoding = "UTF-8-BOM"
  )
}

check_descriptive_columns <- function(table, columns, label) {
  missing_columns <- setdiff(columns, names(table))
  if (length(missing_columns) > 0) {
    stop(sprintf("%s is missing columns: %s", label, paste(missing_columns, collapse = ", ")), call. = FALSE)
  }
  invisible(table)
}

# Numbers stay as they are. Text that is not a number becomes missing.
to_number <- function(values) {
  if (is.numeric(values)) values else suppressWarnings(as.numeric(as.character(values)))
}

# ---- Tables ----------------------------------------------------------------

# One row for one variable. std is the population standard deviation
# (divided by n, not by n - 1).
numeric_summary_row <- function(values, variable) {
  values <- to_number(values)
  values <- values[!is.na(values)]
  n <- length(values)
  # With no values, every statistic is missing (the calculation is not run).
  value_or_missing <- function(result) if (n > 0) result else NA_real_
  data.frame(
    variable = variable,
    non_missing = n,
    mean = value_or_missing(mean(values)),
    std = value_or_missing(sqrt(mean((values - mean(values))^2))),
    min = value_or_missing(min(values)),
    median = value_or_missing(stats::median(values)),
    max = value_or_missing(max(values)),
    stringsAsFactors = FALSE
  )
}

# Number of rows for each value, largest number first. Missing values are
# counted as "missing". Values with the same number keep the order in which
# they first appear in the data.
value_counts_table <- function(values, name) {
  values <- as.character(values)
  values[is.na(values)] <- "missing"
  found <- unique(values)
  counts <- vapply(found, function(value) sum(values == value), integer(1), USE.NAMES = FALSE)
  largest_first <- order(-counts)
  result <- data.frame(value = found[largest_first], count = counts[largest_first], stringsAsFactors = FALSE)
  names(result)[1] <- name
  result
}

build_descriptive_tables <- function(master) {
  check_descriptive_columns(
    master,
    c(descriptive_participant_columns, descriptive_rate_columns, "mother_education", "father_education"),
    "final_master.csv"
  )
  summary_rows <- function(columns) {
    do.call(rbind, lapply(columns, function(column) numeric_summary_row(master[[column]], column)))
  }

  participant_summary <- master[, descriptive_participant_columns, drop = FALSE]
  rownames(participant_summary) <- NULL

  missing_count <- vapply(master, function(column) sum(is.na(column)), integer(1), USE.NAMES = FALSE)
  missing_values_summary <- data.frame(
    column = names(master),
    missing_count = missing_count,
    missing_percent = missing_count / nrow(master) * 100,
    stringsAsFactors = FALSE
  )

  education <- do.call(rbind, lapply(c("mother_education", "father_education"), function(education_type) {
    counts <- value_counts_table(master[[education_type]], "education_value")
    data.frame(
      education_type = rep(education_type, nrow(counts)),
      education_value = counts$education_value,
      count = counts$count,
      stringsAsFactors = FALSE
    )
  }))
  # Sorted by type, then by value. method = "radix" sorts capital letters
  # before small letters, in the same way on every computer.
  education <- education[order(education$education_type, education$education_value, method = "radix"), , drop = FALSE]
  rownames(education) <- NULL

  list(
    participant_summary = participant_summary,
    missing_values_summary = missing_values_summary,
    recording_duration_summary = summary_rows("recording_duration_hours"),
    speaker_count_summary = summary_rows(descriptive_count_columns),
    speaker_duration_summary = summary_rows(descriptive_duration_columns),
    age_summary = summary_rows(c("age_days", "age_months", "age_z")),
    sex_distribution = value_counts_table(master$child_sex, "child_sex"),
    location_distribution = value_counts_table(master$Location, "Location"),
    education_distribution = education
  )
}

# ---- Figures ---------------------------------------------------------------

descriptive_theme <- function() {
  ggplot2::theme_minimal(base_size = 11)
}

empty_descriptive_figure <- function(title, xlabel = "", ylabel = "") {
  ggplot2::ggplot() +
    ggplot2::annotate("text", x = 0.5, y = 0.5, label = "No data available", size = 4.2) +
    ggplot2::labs(title = title, x = xlabel, y = ylabel) +
    descriptive_theme() +
    ggplot2::theme(
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      panel.grid = ggplot2::element_blank()
    )
}

histogram_figure <- function(values, title, xlabel) {
  values <- to_number(values)
  values <- values[!is.na(values)]
  if (length(values) == 0) {
    return(empty_descriptive_figure(title, xlabel, "Count"))
  }
  bins <- min(20, max(5, length(unique(values))))
  limits <- range(values)
  if (limits[1] == limits[2]) limits <- limits + c(-0.5, 0.5)
  ggplot2::ggplot(data.frame(value = values), ggplot2::aes(x = value)) +
    ggplot2::geom_histogram(
      breaks = seq(limits[1], limits[2], length.out = bins + 1),
      closed = "left",
      fill = "#355c7d",
      colour = "white"
    ) +
    ggplot2::labs(title = title, x = xlabel, y = "Count") +
    descriptive_theme()
}

count_bar_figure <- function(values, title, xlabel) {
  counts <- value_counts_table(values, "value")
  if (nrow(counts) == 0) {
    return(empty_descriptive_figure(title, xlabel, "Count"))
  }
  counts$value <- factor(counts$value, levels = counts$value)
  ggplot2::ggplot(counts, ggplot2::aes(x = value, y = count)) +
    ggplot2::geom_col(fill = "#6c8ead") +
    ggplot2::labs(title = title, x = xlabel, y = "Count") +
    descriptive_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))
}

# Rows of the long table that have a value in one column, for the three
# input speakers in a fixed order.
speaker_values <- function(input_long, column) {
  data <- data.frame(
    speaker = factor(as.character(input_long$speaker), levels = descriptive_speakers),
    value = to_number(input_long[[column]])
  )
  data[!is.na(data$speaker) & !is.na(data$value), , drop = FALSE]
}

speaker_boxplot_figure <- function(input_long, column, title, ylabel) {
  data <- speaker_values(input_long, column)
  if (nrow(data) == 0) {
    return(empty_descriptive_figure(title, "Speaker", ylabel))
  }
  ggplot2::ggplot(data, ggplot2::aes(x = speaker, y = value, fill = speaker)) +
    ggplot2::geom_boxplot() +
    ggplot2::scale_x_discrete(drop = FALSE) +
    ggplot2::scale_fill_brewer(palette = "Blues", drop = FALSE, guide = "none") +
    ggplot2::labs(title = title, x = "Speaker", y = ylabel) +
    descriptive_theme()
}

scatter_figure <- function(master, x, y, title, xlabel, ylabel) {
  data <- data.frame(x = to_number(master[[x]]), y = to_number(master[[y]]))
  data <- data[stats::complete.cases(data), , drop = FALSE]
  if (nrow(data) == 0) {
    return(empty_descriptive_figure(title, xlabel, ylabel))
  }
  ggplot2::ggplot(data, ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_point(size = 2, alpha = 0.8, colour = "#355c7d") +
    ggplot2::geom_smooth(method = "lm", formula = y ~ x, colour = "#c06c84") +
    ggplot2::labs(title = title, x = xlabel, y = ylabel) +
    descriptive_theme()
}

correlation_figure <- function(master) {
  rates <- as.data.frame(lapply(master[descriptive_rate_columns], to_number))
  if (!any(!is.na(rates))) {
    return(empty_descriptive_figure("Correlation heatmap", "Variable", "Variable"))
  }
  # Pearson correlation of each pair, from the rows where both values exist.
  correlation <- suppressWarnings(stats::cor(rates, use = "pairwise.complete.obs"))
  n_columns <- length(descriptive_rate_columns)
  cells <- data.frame(
    row = factor(rep(descriptive_rate_columns, times = n_columns), levels = rev(descriptive_rate_columns)),
    column = factor(rep(descriptive_rate_columns, each = n_columns), levels = descriptive_rate_columns),
    value = as.vector(correlation)
  )
  cells$label <- ifelse(is.na(cells$value), "", sprintf("%.2f", cells$value))
  # White text on the dark cells, black text on the light cells.
  middle <- suppressWarnings(mean(range(cells$value, na.rm = TRUE)))
  cells$dark <- !is.na(cells$value) & cells$value > middle
  ggplot2::ggplot(cells, ggplot2::aes(x = column, y = row, fill = value)) +
    ggplot2::geom_tile(colour = "white") +
    ggplot2::geom_text(ggplot2::aes(label = label, colour = dark), size = 3, show.legend = FALSE) +
    ggplot2::scale_colour_manual(values = c("FALSE" = "black", "TRUE" = "white"), guide = "none") +
    ggplot2::scale_fill_distiller(palette = "Blues", direction = 1, na.value = "white", name = NULL) +
    ggplot2::coord_fixed() +
    ggplot2::labs(title = "Correlation heatmap for count/hour and duration/hour variables", x = NULL, y = NULL) +
    descriptive_theme() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      panel.grid = ggplot2::element_blank()
    )
}

stacked_bar_figure <- function(input_long) {
  data <- data.frame(
    participant_id = as.character(input_long$participant_id),
    speaker = factor(as.character(input_long$speaker), levels = descriptive_speakers),
    value = to_number(input_long$input_count_hour),
    stringsAsFactors = FALSE
  )
  data <- data[!is.na(data$speaker), , drop = FALSE]
  if (all(is.na(data$value))) {
    return(empty_descriptive_figure("Input composition per participant", "Participant", "Input count/hour"))
  }
  data$value[is.na(data$value)] <- 0
  data$participant_id <- factor(data$participant_id, levels = sort(unique(data$participant_id), method = "radix"))
  ggplot2::ggplot(data, ggplot2::aes(x = participant_id, y = value, fill = speaker)) +
    ggplot2::geom_col(position = ggplot2::position_stack(reverse = TRUE)) +
    ggplot2::scale_fill_manual(
      values = c(adult_female = "#355c7d", adult_male = "#6c8ead", other_child = "#c06c84"),
      drop = FALSE,
      name = "Speaker"
    ) +
    ggplot2::labs(title = "Input composition per participant", x = "Participant ID", y = "Input count/hour") +
    descriptive_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}

# Mean of each speaker with a 95 % interval: mean +/- 1.96 * sd / sqrt(n).
# sd is the sample standard deviation (divided by n - 1).
mean_bar_figure <- function(input_long, column, title, ylabel) {
  data <- speaker_values(input_long, column)
  if (nrow(data) == 0) {
    return(empty_descriptive_figure(title, "Speaker", ylabel))
  }
  groups <- split(data$value, data$speaker)
  speaker_means <- data.frame(
    speaker = factor(descriptive_speakers, levels = descriptive_speakers),
    mean = vapply(groups, function(values) if (length(values) > 0) mean(values) else NA_real_, numeric(1)),
    n = vapply(groups, length, integer(1)),
    sd = vapply(groups, function(values) if (length(values) > 1) stats::sd(values) else NA_real_, numeric(1))
  )
  speaker_means$ci95 <- 1.96 * speaker_means$sd / sqrt(speaker_means$n)
  speaker_means$ci95[is.na(speaker_means$ci95)] <- 0
  ggplot2::ggplot(speaker_means, ggplot2::aes(x = speaker, y = mean, fill = speaker)) +
    ggplot2::geom_col(na.rm = TRUE) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = mean - ci95, ymax = mean + ci95),
      width = 0.15,
      colour = "#222222",
      na.rm = TRUE
    ) +
    ggplot2::scale_x_discrete(drop = FALSE) +
    ggplot2::scale_fill_brewer(palette = "Blues", drop = FALSE, guide = "none") +
    ggplot2::labs(title = title, x = "Speaker", y = ylabel) +
    descriptive_theme()
}

save_descriptive_figure <- function(figure, output_dir, name, width = 7, height = 4.5) {
  path <- file.path(output_dir, paste0(name, ".png"))
  ggplot2::ggsave(path, figure, width = width, height = height, dpi = 300, bg = "white")
  path
}

# Create the 19 descriptive figures as PNG files. Returns the file paths.
save_descriptive_figures <- function(master, input_long, output_dir) {
  check_descriptive_columns(
    master,
    c("age_months", "recording_duration_hours", "child_sex", "Location",
      "mother_education", "father_education", descriptive_rate_columns),
    "final_master.csv"
  )
  check_descriptive_columns(
    input_long,
    c("participant_id", "speaker", "input_count_hour", "input_duration_hour"),
    "input_long.csv"
  )
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  age_scatter <- function(y, speaker_label) {
    scatter_figure(
      master, "age_months", y,
      sprintf("Age vs %s count/hour", tolower(speaker_label)),
      "Age (months)", sprintf("%s count/hour", speaker_label)
    )
  }
  key_child_scatter <- function(x, speaker_label) {
    scatter_figure(
      master, x, "key_child_count_hour",
      sprintf("Key child vs %s count/hour", tolower(speaker_label)),
      sprintf("%s count/hour", speaker_label), "Key child count/hour"
    )
  }

  figures <- list(
    age_distribution_histogram = histogram_figure(
      master$age_months, "Age distribution", "Age (months)"
    ),
    recording_duration_histogram = histogram_figure(
      master$recording_duration_hours, "Recording duration distribution", "Recording duration (hours)"
    ),
    child_sex_distribution = count_bar_figure(master$child_sex, "Child sex distribution", "Child sex"),
    location_distribution = count_bar_figure(master$Location, "Location distribution", "Location"),
    mother_education_distribution = count_bar_figure(
      master$mother_education, "Mother education distribution", "Mother education"
    ),
    father_education_distribution = count_bar_figure(
      master$father_education, "Father education distribution", "Father education"
    ),
    input_count_hour_by_speaker_boxplot = speaker_boxplot_figure(
      input_long, "input_count_hour", "Input count/hour by speaker", "Input count/hour"
    ),
    input_duration_hour_by_speaker_boxplot = speaker_boxplot_figure(
      input_long, "input_duration_hour", "Input duration/hour by speaker", "Input duration/hour"
    ),
    age_vs_adult_female_count_hour = age_scatter("adult_female_count_hour", "Adult female"),
    age_vs_adult_male_count_hour = age_scatter("adult_male_count_hour", "Adult male"),
    age_vs_other_child_count_hour = age_scatter("other_child_count_hour", "Other child"),
    age_vs_key_child_count_hour = age_scatter("key_child_count_hour", "Key child"),
    key_child_vs_adult_female_count_hour = key_child_scatter("adult_female_count_hour", "Adult female"),
    key_child_vs_adult_male_count_hour = key_child_scatter("adult_male_count_hour", "Adult male"),
    key_child_vs_other_child_count_hour = key_child_scatter("other_child_count_hour", "Other child"),
    hourly_variables_correlation_heatmap = correlation_figure(master),
    input_composition_per_participant_stacked_bar = stacked_bar_figure(input_long),
    mean_input_count_hour_by_speaker = mean_bar_figure(
      input_long, "input_count_hour", "Mean input count/hour by speaker", "Mean input count/hour"
    ),
    mean_input_duration_hour_by_speaker = mean_bar_figure(
      input_long, "input_duration_hour", "Mean input duration/hour by speaker", "Mean input duration/hour"
    )
  )
  stopifnot(identical(names(figures), descriptive_figure_names))

  vapply(names(figures), function(name) {
    size <- switch(
      name,
      hourly_variables_correlation_heatmap = c(8, 6),
      input_composition_per_participant_stacked_bar = c(10, 5),
      c(7, 4.5)
    )
    save_descriptive_figure(figures[[name]], output_dir, name, width = size[1], height = size[2])
  }, character(1))
}
