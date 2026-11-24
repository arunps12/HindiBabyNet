# Reporting helpers for the HindiBabyNet R workflow.

bind_rows_safe <- function(items) {
  non_empty <- Filter(Negate(is.null), items)
  if (length(non_empty) == 0) {
    return(data.frame())
  }

  all_columns <- unique(unlist(lapply(non_empty, names), use.names = FALSE))
  aligned <- lapply(non_empty, function(item) {
    missing_columns <- setdiff(all_columns, names(item))
    for (column in missing_columns) {
      item[[column]] <- NA
    }
    item[all_columns]
  })

  do.call(rbind, aligned)
}

extract_term_level_statistics <- function(model) {
  if (inherits(model, "glmmTMB")) {
    coefficient_matrix <- summary(model)$coefficients$cond
  } else {
    coefficient_matrix <- coef(summary(model))
  }

  stats_table <- as.data.frame(coefficient_matrix, stringsAsFactors = FALSE)
  stats_table$term <- rownames(stats_table)
  rownames(stats_table) <- NULL

  p_value_column <- grep("Pr\\(|p.value|p-value", names(stats_table), ignore.case = TRUE, value = TRUE)[1]
  df_column <- grep("^df$|df", names(stats_table), ignore.case = TRUE, value = TRUE)[1]

  data.frame(
    term = stats_table$term,
    df = if (!is.na(df_column) && nzchar(df_column)) as.numeric(stats_table[[df_column]]) else NA_real_,
    p.value = if (!is.na(p_value_column) && nzchar(p_value_column)) as.numeric(stats_table[[p_value_column]]) else NA_real_,
    stringsAsFactors = FALSE
  )
}

extract_fixed_effects <- function(bundle) {
  effects <- tidy_model_fixed_effects(bundle$model, conf.int = TRUE)
  term_stats <- extract_term_level_statistics(bundle$model)
  effects <- merge(effects, term_stats, by = "term", all.x = TRUE, sort = FALSE, suffixes = c("", ".term"))

  if (!"df" %in% names(effects) && "df.term" %in% names(effects)) {
    effects$df <- effects$df.term
  } else if ("df.term" %in% names(effects)) {
    effects$df <- ifelse(is.na(effects$df), effects$df.term, effects$df)
  }

  if (!"p.value" %in% names(effects) && "p.value.term" %in% names(effects)) {
    effects$p.value <- effects$p.value.term
  } else if ("p.value.term" %in% names(effects)) {
    effects$p.value <- ifelse(is.na(effects$p.value), effects$p.value.term, effects$p.value)
  }

  effects$df.term <- NULL
  effects$p.value.term <- NULL
  effects$response <- bundle$response_name
  effects$model_family <- bundle$model_family
  effects$model_type <- bundle$selection$model_type[[1]]
  effects$fit_engine <- bundle$fit_engine %||% NA_character_
  effects
}

principal_terms_for_bundle <- function(bundle) {
  age_terms <- switch(
    bundle$age_specification %||% get_age_specification_name(bundle$response_name, bundle$model_family),
    none = character(0),
    linear = c("age_z"),
    quadratic = c("age_z", "age_z2"),
    character(0)
  )

  if (identical(bundle$model_family, "family_1")) {
    interaction_patterns <- unlist(lapply(age_terms, function(term) {
      c(
        sprintf("^speaker.*:%s$", term),
        sprintf("^%s:speaker.*$", term)
      )
    }), use.names = FALSE)
    return(c(sprintf("^%s$", age_terms), "^speaker", interaction_patterns))
  }

  if (identical(bundle$model_family, "family_2_count")) {
    return(c("^input_count_hour$", "^input_count_hour:speaker", "^speaker.*:input_count_hour$"))
  }

  if (identical(bundle$model_family, "family_2_duration")) {
    return(c("^input_duration_min_hour$", "^input_duration_min_hour:speaker", "^speaker.*:input_duration_min_hour$"))
  }

  character(0)
}

filter_focal_effects <- function(fixed_effects, principal_terms) {
  if (length(principal_terms) == 0 || nrow(fixed_effects) == 0) {
    return(fixed_effects[0, , drop = FALSE])
  }

  keep_index <- vapply(fixed_effects$term, function(term) {
    any(vapply(principal_terms, function(pattern) grepl(pattern, term), logical(1)))
  }, logical(1))
  fixed_effects[keep_index, , drop = FALSE]
}

classify_evidence_strength <- function(effects_table, hypothesis_direction = c("positive", "mixed")) {
  hypothesis_direction <- match.arg(hypothesis_direction)
  if (nrow(effects_table) == 0) {
    return("Inconclusive evidence")
  }

  consistent_direction <- if (identical(hypothesis_direction, "positive")) {
    all(effects_table$estimate > 0, na.rm = TRUE)
  } else {
    TRUE
  }

  excludes_zero <- all((effects_table$conf.low > 0) | (effects_table$conf.high < 0), na.rm = TRUE)
  strong_signal <- all(abs(effects_table$estimate / effects_table$std.error) >= 2.5, na.rm = TRUE)

  if (consistent_direction && excludes_zero && strong_signal) {
    return("Strong evidence supporting the hypothesis")
  }
  if (consistent_direction && excludes_zero) {
    return("Moderate evidence supporting the hypothesis")
  }
  if (consistent_direction) {
    return("Weak evidence supporting the hypothesis")
  }
  if (excludes_zero) {
    return("Evidence inconsistent with the hypothesis")
  }

  "Inconclusive evidence"
}

summarise_model_evidence <- function(bundle) {
  fixed_effects <- extract_fixed_effects(bundle)
  principal_terms <- principal_terms_for_bundle(bundle)
  focal_effects <- filter_focal_effects(fixed_effects, principal_terms)
  hypothesis_direction <- if (grepl("family_2", bundle$model_family, fixed = TRUE)) "positive" else "mixed"

  data.frame(
    response = bundle$response_name,
    model_family = bundle$model_family,
    model_type = bundle$selection$model_type[[1]],
    focal_terms = paste(focal_effects$term, collapse = "; "),
    evidence_category = classify_evidence_strength(focal_effects, hypothesis_direction = hypothesis_direction),
    rationale = bundle$selection$rationale[[1]],
    stringsAsFactors = FALSE
  )
}

link_research_questions <- function(evidence_table, questions_yaml) {
  questions <- questions_yaml$research_questions
  rows <- lapply(questions, function(question) {
    if (!is.null(question$response_key)) {
      matching <- evidence_table[evidence_table$response %in% question$response_key, , drop = FALSE]
    } else {
      family_key <- switch(
        question$preregistered_family,
        model_family_1 = "family_1",
        model_family_2 = c("family_2_count", "family_2_duration"),
        question$preregistered_family
      )
      matching <- evidence_table[evidence_table$model_family %in% family_key, , drop = FALSE]
    }

    data.frame(
      research_question_id = question$id,
      title = question$title,
      hypothesis = question$hypothesis,
      evidence_category = if (nrow(matching) == 0) "Inconclusive evidence" else paste(unique(matching$evidence_category), collapse = "; "),
      responses = if (nrow(matching) == 0) NA_character_ else paste(matching$response, collapse = "; "),
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, rows)
}

build_reporting_bundle <- function(model_bundles, diagnostics, predictions, dataset_summary, questions_yaml = read_research_questions()) {
  selection_table <- selection_table_from_bundles(model_bundles)
  fixed_effects <- bind_rows_safe(lapply(model_bundles, extract_fixed_effects))
  evidence_table <- bind_rows_safe(lapply(model_bundles, summarise_model_evidence))
  question_summary <- link_research_questions(evidence_table, questions_yaml)
  diagnostics_table <- bind_rows_safe(lapply(diagnostics, function(item) item$summary))
  anova_table <- bind_rows_safe(lapply(model_bundles, tidy_anova_table))
  collinearity_table <- bind_rows_safe(lapply(model_bundles, vif_table))
  prediction_manifest <- bind_rows_safe(lapply(names(predictions), function(name) {
    item <- predictions[[name]]
    data.frame(
      response = name,
      prediction_csv = item$csv_path,
      plot_path = item$plot_path,
      stringsAsFactors = FALSE
    )
  }))

  list(
    questions = questions_yaml,
    dataset_summary = dataset_summary,
    selection_table = selection_table,
    fixed_effects = fixed_effects,
    evidence_table = evidence_table,
    question_summary = question_summary,
    diagnostics_table = diagnostics_table,
    anova_table = anova_table,
    collinearity_table = collinearity_table,
    prediction_manifest = prediction_manifest
  )
}

write_reporting_outputs <- function(reporting_bundle, output_dir = analysis_paths$results_tables) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(reporting_bundle$dataset_summary, file.path(output_dir, "dataset_summary.csv"), row.names = FALSE)
  utils::write.csv(reporting_bundle$selection_table, file.path(output_dir, "model_selection_summary.csv"), row.names = FALSE)
  utils::write.csv(reporting_bundle$fixed_effects, file.path(output_dir, "fixed_effects_summary.csv"), row.names = FALSE)
  utils::write.csv(reporting_bundle$evidence_table, file.path(output_dir, "evidence_summary.csv"), row.names = FALSE)
  utils::write.csv(reporting_bundle$question_summary, file.path(output_dir, "research_question_summary.csv"), row.names = FALSE)
  utils::write.csv(reporting_bundle$diagnostics_table, file.path(output_dir, "diagnostics_summary.csv"), row.names = FALSE)
  utils::write.csv(reporting_bundle$anova_table, file.path(output_dir, "anova_summary.csv"), row.names = FALSE)
  utils::write.csv(reporting_bundle$collinearity_table, file.path(output_dir, "collinearity_summary.csv"), row.names = FALSE)
  utils::write.csv(reporting_bundle$prediction_manifest, file.path(output_dir, "prediction_manifest.csv"), row.names = FALSE)

  bundle_path <- file.path(output_dir, "reporting_bundle.rds")
  saveRDS(reporting_bundle, bundle_path)
  invisible(bundle_path)
}
