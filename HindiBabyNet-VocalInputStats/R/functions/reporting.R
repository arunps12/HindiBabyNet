# Reporting helpers for the HindiBabyNet R workflow.

extract_fixed_effects <- function(bundle) {
  effects <- broom.mixed::tidy(bundle$model, effects = "fixed", conf.int = TRUE)
  effects$response <- bundle$response_name
  effects$model_family <- bundle$model_family
  effects$model_type <- bundle$selection$model_type[[1]]
  effects
}

principal_terms_for_bundle <- function(bundle) {
  if (identical(bundle$model_family, "family_1")) {
    return(c("age_z", "speakeradult_male", "speakerother_child", "speakeradult_male:age_z", "speakerother_child:age_z"))
  }

  if (identical(bundle$model_family, "family_2_count")) {
    return(c("input_count_hour", "input_count_hour:speakeradult_male", "input_count_hour:speakerother_child"))
  }

  if (identical(bundle$model_family, "family_2_duration")) {
    return(c("input_duration_min_hour", "input_duration_min_hour:speakeradult_male", "input_duration_min_hour:speakerother_child"))
  }

  character(0)
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
  focal_effects <- fixed_effects[fixed_effects$term %in% principal_terms, , drop = FALSE]
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
    family_key <- switch(
      question$preregistered_family,
      model_family_1 = "family_1",
      model_family_2 = c("family_2_count", "family_2_duration"),
      question$preregistered_family
    )

    matching <- evidence_table[evidence_table$model_family %in% family_key, , drop = FALSE]
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
  fixed_effects <- do.call(rbind, lapply(model_bundles, extract_fixed_effects))
  evidence_table <- do.call(rbind, lapply(model_bundles, summarise_model_evidence))
  question_summary <- link_research_questions(evidence_table, questions_yaml)
  diagnostics_table <- do.call(rbind, lapply(diagnostics, function(item) item$summary))
  prediction_manifest <- do.call(rbind, lapply(names(predictions), function(name) {
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
  utils::write.csv(reporting_bundle$prediction_manifest, file.path(output_dir, "prediction_manifest.csv"), row.names = FALSE)

  bundle_path <- file.path(output_dir, "reporting_bundle.rds")
  saveRDS(reporting_bundle, bundle_path)
  invisible(bundle_path)
}
