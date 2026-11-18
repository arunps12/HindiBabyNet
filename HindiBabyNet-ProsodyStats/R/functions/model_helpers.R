extract_term_level_statistics <- function(model) {
  coefficient_matrix <- coef(summary(model))
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

tidy_model_fixed_effects <- function(model, conf.int = TRUE, conf.level = analysis_options$confidence_level) {
  if (inherits(model, c("merMod", "glmmTMB"))) {
    return(broom.mixed::tidy(model, effects = "fixed", conf.int = conf.int, conf.level = conf.level))
  }

  broom::tidy(model, conf.int = conf.int, conf.level = conf.level)
}

tidy_model_random_effects <- function(model, conf.int = TRUE, conf.level = analysis_options$confidence_level) {
  if (inherits(model, c("merMod", "glmmTMB"))) {
    return(broom.mixed::tidy(model, effects = "ran_pars", conf.int = conf.int, conf.level = conf.level))
  }

  data.frame(note = "No random effects in fitted model", stringsAsFactors = FALSE)
}

build_model_stub <- function(model_family, outcome_name, acoustic_measure = NULL) {
  pieces <- c(model_family, outcome_name, acoustic_measure)
  gsub("[^A-Za-z0-9_]+", "_", paste(pieces[!is.na(pieces) & nzchar(pieces)], collapse = "__"))
}

build_family_1_formula <- function(outcome_name) {
  age_terms <- active_age_terms("family_1")
  fixed_terms <- c("speaker", age_terms, "SES_num", "child_sex", "city")
  if ("age_z" %in% age_terms) {
    fixed_terms <- c(fixed_terms, "speaker:age_z")
  }
  if ("age_z2" %in% age_terms) {
    fixed_terms <- c(fixed_terms, "speaker:age_z2")
  }
  stats::as.formula(sprintf("%s ~ %s + (1 | participant_id)", outcome_name, paste(fixed_terms, collapse = " + ")))
}

build_family_2_formula <- function(outcome_name, predictor_name, adjustment_name, model_family) {
  age_terms <- active_age_terms(model_family)
  fixed_terms <- c(sprintf("%s * speaker", predictor_name), adjustment_name, age_terms, "SES_num", "child_sex", "city")
  stats::as.formula(sprintf("%s ~ %s + (1 | participant_id)", outcome_name, paste(fixed_terms, collapse = " + ")))
}

prepare_model_data <- function(data, formula) {
  stats::na.omit(data[, all.vars(formula), drop = FALSE])
}

fit_lmer_with_retries <- function(formula, model_data) {
  control_candidates <- if (identical(analysis_options$model_robustness_mode, "strict")) {
    list(NULL)
  } else {
    list(
      NULL,
      lme4::lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)),
      lme4::lmerControl(optimizer = "Nelder_Mead", optCtrl = list(maxfun = 2e5))
    )
  }

  last_error <- NULL
  for (control in control_candidates) {
    fit_attempt <- tryCatch(
      {
        if (is.null(control)) {
          lmerTest::lmer(formula = formula, data = model_data, REML = FALSE)
        } else {
          lmerTest::lmer(formula = formula, data = model_data, REML = FALSE, control = control)
        }
      },
      error = function(error_condition) {
        last_error <<- error_condition
        NULL
      }
    )

    if (!is.null(fit_attempt)) {
      return(fit_attempt)
    }
  }

  stop(last_error)
}

standardize_predictor <- function(dataframe, predictor_name) {
  standardized_name <- sprintf("%s_z", predictor_name)
  values <- dataframe[[predictor_name]]
  if (all(is.na(values)) || isTRUE(stats::sd(values, na.rm = TRUE) == 0)) {
    stop(sprintf("Predictor '%s' cannot be standardized because it has no variation.", predictor_name), call. = FALSE)
  }
  dataframe[[standardized_name]] <- as.numeric(scale(values))
  list(data = dataframe, standardized_name = standardized_name)
}

fit_prosody_model <- function(data, formula, model_family, response_name, acoustic_measure = NA_character_, standardized_predictor = NA_character_) {
  required_columns <- all.vars(formula)
  complete_rows <- stats::complete.cases(data[, required_columns, drop = FALSE])
  model_data <- prepare_model_data(data, formula)
  analysis_data <- data[complete_rows, , drop = FALSE]
  if (nrow(model_data) == 0) {
    stop(sprintf("No complete cases available for model '%s'.", response_name), call. = FALSE)
  }

  model <- fit_lmer_with_retries(formula = formula, model_data = model_data)
  bundle <- list(
    model = model,
    model_family = model_family,
    response_name = response_name,
    acoustic_measure = acoustic_measure,
    standardized_predictor = standardized_predictor,
    fitted_formula = formula,
    age_specification = get_family_age_specification(model_family),
    response_transform = "none",
    n_complete_cases = nrow(model_data),
    n_participants = dplyr::n_distinct(analysis_data$participant_id),
    model_data = model_data,
    analysis_data = analysis_data,
    selection = data.frame(model_type = "lmm", rationale = "Continuous outcome modeled with Gaussian linear mixed-effects model.", stringsAsFactors = FALSE),
    fit_engine = "lmer"
  )
  bundle
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
  effects$model_family <- bundle$model_family
  effects$response <- bundle$response_name
  effects$acoustic_measure <- bundle$acoustic_measure
  effects$standardized_predictor <- bundle$standardized_predictor
  effects$n_complete_cases <- bundle$n_complete_cases
  effects$n_participants <- bundle$n_participants
  effects
}

model_fit_statistics <- function(bundle) {
  data.frame(
    model_family = bundle$model_family,
    response = bundle$response_name,
    acoustic_measure = bundle$acoustic_measure,
    standardized_predictor = bundle$standardized_predictor,
    n_complete_cases = bundle$n_complete_cases,
    n_participants = bundle$n_participants,
    AIC = AIC(bundle$model),
    BIC = BIC(bundle$model),
    logLik = as.numeric(logLik(bundle$model)),
    formula = paste(deparse(bundle$fitted_formula), collapse = " "),
    stringsAsFactors = FALSE
  )
}

save_model_bundle <- function(bundle, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(bundle, output_path)
  invisible(output_path)
}

load_model_bundle <- function(path) {
  readRDS(path)
}

selection_table_from_bundles <- function(model_bundles) {
  rows <- lapply(model_bundles, function(bundle) {
    data.frame(
      response = bundle$response_name,
      model_family = bundle$model_family,
      acoustic_measure = bundle$acoustic_measure,
      standardized_predictor = bundle$standardized_predictor,
      model_type = bundle$selection$model_type[[1]],
      rationale = bundle$selection$rationale[[1]],
      n_complete_cases = bundle$n_complete_cases,
      n_participants = bundle$n_participants,
      fit_engine = bundle$fit_engine,
      age_specification = bundle$age_specification,
      response_transform = bundle$response_transform,
      fitted_formula = paste(deparse(bundle$fitted_formula), collapse = " "),
      stringsAsFactors = FALSE
    )
  })

  bind_rows_safe(rows)
}

estimate_speaker_emmeans <- function(bundle) {
  result <- tryCatch(
    emmeans::emmeans(bundle$model, specs = ~ speaker),
    error = function(e) NULL
  )

  if (is.null(result)) {
    return(data.frame())
  }

  summary_table <- as.data.frame(summary(result))
  summary_table$model_family <- bundle$model_family
  summary_table$response <- bundle$response_name
  summary_table$acoustic_measure <- bundle$acoustic_measure
  summary_table
}

estimate_speaker_pairs <- function(bundle) {
  result <- tryCatch(
    emmeans::emmeans(bundle$model, specs = ~ speaker),
    error = function(e) NULL
  )

  if (is.null(result)) {
    return(data.frame())
  }

  summary_table <- as.data.frame(summary(emmeans::pairs(result)))
  summary_table$model_family <- bundle$model_family
  summary_table$response <- bundle$response_name
  summary_table$acoustic_measure <- bundle$acoustic_measure
  summary_table
}
