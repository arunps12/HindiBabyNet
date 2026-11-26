# Research-question presentation migrated from the former Quarto template.
# Uses the in-memory reporting/prediction bundles; never fits or reloads models.
format_number <- function(values, digits = 3) {
  ifelse(is.na(values), "NA", formatC(as.numeric(values), format = "f", digits = digits))
}

format_p_value <- function(values) {
  ifelse(is.na(values), "NA", ifelse(values < 0.001, "< 0.001", formatC(as.numeric(values), format = "f", digits = 3)))
}

label_response <- function(response) {
  switch(as.character(response)[[1]],
    input_count_hour = "Input count per hour",
    input_duration_min_hour = "Input duration (minutes per hour)",
    key_child_count_hour = "Key child count per hour",
    key_child_duration_min_hour = "Key child duration (minutes per hour)", response)
}

label_term <- function(term) {
  switch(term,
    speakeradult_male = "Speaker: adult male vs adult female",
    speakerother_child = "Speaker: other child vs adult female",
    age_z = "Child age (z-scored)", age_z2 = "Child age squared (z-scored)",
    SES_num = "SES (ordered numeric score)",
    child_sexmale = "Child sex: male vs female",
    "speakeradult_male:age_z" = "Adult male by age interaction",
    "speakerother_child:age_z" = "Other child by age interaction",
    adult_female_count_hour = "Adult female input count per hour",
    adult_male_count_hour = "Adult male input count per hour",
    other_child_count_hour = "Other child input count per hour",
    adult_female_duration_min_hour = "Adult female input duration (minutes per hour)",
    adult_male_duration_min_hour = "Adult male input duration (minutes per hour)",
    other_child_duration_min_hour = "Other child input duration (minutes per hour)", term)
}

focal_terms_for_question <- function(question_id, response) {
  if (identical(question_id, "rq1")) return(c("speakeradult_male", "speakerother_child"))
  if (identical(question_id, "rq2")) return(c("age_z", "age_z2", "SES_num", "child_sexmale", "speakeradult_male:age_z", "speakerother_child:age_z"))
  if (identical(question_id, "rq3")) {
    return(family_2_input_terms(if (identical(response, "key_child_count_hour")) "family_2_count" else "family_2_duration"))
  }
  if (identical(question_id, "rq4")) return(c("age_z", "age_z2", "SES_num", "child_sexmale"))
  character(0)
}

prediction_summary_table <- function(prediction_data) {
  # The predictor of the table is the column that changes along the prediction line:
  # the input of a speaker (Family 2) or age (Family 1).
  candidates <- intersect(c("input_count_hour", "input_duration_min_hour", "age_days"), names(prediction_data))
  varies <- vapply(candidates, function(column) length(unique(prediction_data[[column]])) > 1, logical(1))
  predictor <- c(candidates[varies], "age_days")[[1]]
  do.call(rbind, lapply(unique(prediction_data$speaker), function(speaker) {
    rows <- prediction_data[prediction_data$speaker == speaker, , drop = FALSE]
    rows <- rows[order(rows[[predictor]]), , drop = FALSE]
    start <- rows$predicted[[1]]
    end <- rows$predicted[[nrow(rows)]]
    delta <- end - start
    data.frame(speaker = gsub("_", " ", speaker, fixed = TRUE),
               predictor = gsub("_", " ", predictor, fixed = TRUE),
               start_predicted = format_number(start), end_predicted = format_number(end),
               delta = format_number(delta),
               trend = if (abs(delta) < 1e-8) "flat" else if (delta > 0) "increasing" else "decreasing")
  }))
}

effect_display_table <- function(effects) {
  if (!nrow(effects)) return(data.frame(note = "No focal effects available for this question-response pairing."))
  data.frame(term = vapply(effects$term, label_term, character(1)),
             estimate = format_number(effects$estimate), std_error = format_number(effects$std.error),
             statistic = format_number(effects$statistic), df = format_number(effects$df),
             p_value = format_p_value(effects$p.value), conf_low = format_number(effects$conf.low),
             conf_high = format_number(effects$conf.high))
}

render_effect_bullets <- function(effects) {
  if (!nrow(effects)) {
    cat("No focal fixed effects were available for this response under the current question.\n\n")
    return(invisible(NULL))
  }
  for (i in seq_len(nrow(effects))) {
    row <- effects[i, , drop = FALSE]
    direction <- if (is.na(row$estimate)) "estimate unavailable" else if (
      !is.na(row$conf.low) && !is.na(row$conf.high) && row$conf.low > 0
    ) "positive effect with a confidence interval entirely above zero" else if (
      !is.na(row$conf.low) && !is.na(row$conf.high) && row$conf.high < 0
    ) "negative effect with a confidence interval entirely below zero" else if (
      row$estimate > 0
    ) "positive estimate with uncertainty overlapping zero" else if (
      row$estimate < 0
    ) "negative estimate with uncertainty overlapping zero" else "near-zero estimate"
    cat(sprintf("- %s: estimate = %s, 95%% CI [%s, %s], p = %s. This indicates a %s.\n",
                label_term(row$term), format_number(row$estimate), format_number(row$conf.low),
                format_number(row$conf.high), format_p_value(row$p.value), direction))
  }
  cat("\n")
}

render_question_section <- function(question, reporting_bundle, predictions) {
  render_table <- function(data, caption) {
    print(knitr::kable(data, caption = caption))
    cat("\n\n")
  }
  cat(sprintf("## %s\n\n**Research question.** %s\n\n**%s.** %s\n\n",
              question$title, question$hypothesis, question$hypothesis_id, question$expected_pattern))
  summary <- reporting_bundle$question_summary
  row <- summary[summary$research_question_id == question$id, , drop = FALSE]
  if (nrow(row)) cat(sprintf("**Overall evidence category.** %s\n\n", row$evidence_category[[1]]))
  if (question$id %in% c("rq3", "rq4")) {
    cat("### Modeling note for Family 2\n\nThe data has one row per participant. The input of the adult female, the adult male and the other child are three separate predictors in the same model. The model has no random effects. In the prediction summary, the input of one speaker changes and the other inputs, age and SES are at their mean.\n\n")
  }
  # Comparisons that belong to a question: speakers (RQ1) or input coefficients (RQ3).
  comparisons <- switch(question$id,
    rq1 = list(table = reporting_bundle$speaker_pairwise, caption = "Pairwise comparisons of the speakers for"),
    rq3 = list(table = reporting_bundle$input_coefficient_tests, caption = "Tests that compare the input coefficients for"),
    NULL)
  for (response in as.character(question$response_key)) {
    cat(sprintf("### %s\n\n", label_response(response)))
    selection <- reporting_bundle$selection_table
    selected <- selection[selection$response == response,
                          c("response", "model_family", "model_type", "n_complete_cases", "rationale"), drop = FALSE]
    selected$response <- vapply(selected$response, label_response, character(1))
    render_table(selected, paste("Model summary for", label_response(response)))
    render_table(prediction_summary_table(predictions[[response]]$data),
                 paste("Prediction summary for", label_response(response)))
    # Figures already appear in the model sections; reuse rather than regenerate.
    cat("Prediction and forest figures are shown in the corresponding model section above.\n\n")
    effects <- reporting_bundle$fixed_effects
    effects <- effects[effects$response == response & effects$term %in% focal_terms_for_question(question$id, response), , drop = FALSE]
    render_table(effect_display_table(effects), paste("Focal fixed effects for", label_response(response), "under", question$hypothesis_id))
    render_effect_bullets(effects)
    if (!is.null(comparisons) && !is.null(comparisons$table) && nrow(comparisons$table)) {
      rows <- comparisons$table[comparisons$table$response == response, , drop = FALSE]
      rows <- rows[, setdiff(names(rows), c("response", "model_family")), drop = FALSE]
      rownames(rows) <- NULL
      print(knitr::kable(rows, digits = 4, caption = paste(comparisons$caption, label_response(response))))
      cat("\n\n")
    }
  }
}