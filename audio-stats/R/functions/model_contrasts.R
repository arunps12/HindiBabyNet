# Comparisons and effect sizes for the HindiBabyNet R workflow.
#
# Family 1: pairwise comparisons of the speakers (estimated marginal means).
# Family 2: tests that compare the coefficients of the three input predictors.
# All models: R2; Family 2 also standardized coefficients.

model_is_gamma <- function(model) {
  inherits(model, c("glmmTMB", "glm")) && !identical(stats::family(model)$family, "gaussian")
}

# Scale on which the estimates of a model are given.
model_estimate_scale <- function(bundle) {
  if (model_is_gamma(bundle$model)) {
    return("log link")
  }
  transform <- bundle$response_transform %||% "none"
  if (identical(transform, "none")) "original scale" else sprintf("%s scale", transform)
}

# Common columns of an emmeans summary (the column names depend on the model class).
tidy_contrast_summary <- function(contrast_object) {
  table <- as.data.frame(summary(contrast_object, infer = c(TRUE, TRUE)))
  pick <- function(pattern) {
    column <- grep(pattern, names(table), value = TRUE)[1]
    if (is.na(column)) rep(NA_real_, nrow(table)) else as.numeric(table[[column]])
  }
  data.frame(
    contrast = as.character(table$contrast),
    estimate = pick("^estimate$"),
    std_error = pick("^SE$"),
    df = pick("^df$"),
    statistic = pick("ratio$"),
    p_value = pick("^p.value$"),
    conf_low = pick("LCL$|^lower.CL$"),
    conf_high = pick("UCL$|^upper.CL$"),
    stringsAsFactors = FALSE
  )
}

# Family 1: pairwise comparisons of the three speakers.
#  - difference between two speakers at the mean age (age_z = 0), averaged over child sex
#    and at the mean SES;
#  - difference between the age slopes of two speakers (only if speaker:age_z is in the model).
# p values: not adjusted (p_value) and Holm-adjusted within each set of three comparisons (p_holm).
# Effect size: Gaussian models, estimate divided by the residual SD; Gamma models, ratio exp(estimate).
speaker_pairwise_table <- function(bundle) {
  model <- bundle$model
  age_terms <- bundle_age_terms(bundle)
  arguments <- list(
    object = model,
    specs = ~ speaker,
    at = as.list(stats::setNames(rep(0, length(age_terms)), age_terms)),
    data = bundle$model_data
  )
  if (inherits(model, "merMod")) {
    arguments$lmer.df <- "satterthwaite"
  }

  # emmeans prints a note because speaker is in an interaction with age. The speakers
  # are compared at the mean age on purpose, so the note is not shown.
  comparisons <- list(
    "speaker difference at mean age" = suppressMessages(do.call(emmeans::emmeans, arguments))
  )
  if ("speaker:age_z" %in% bundle$spec$fixed_terms) {
    comparisons[["difference in age slope (per SD of age)"]] <-
      do.call(emmeans::emtrends, c(arguments, list(var = "age_z")))
  }

  gamma <- model_is_gamma(model)
  rows <- lapply(names(comparisons), function(comparison_type) {
    pairs <- emmeans::contrast(comparisons[[comparison_type]], method = "pairwise", adjust = "none")
    table <- tidy_contrast_summary(pairs)
    table$p_holm <- stats::p.adjust(table$p_value, method = "holm")
    table$effect_size <- if (gamma) exp(table$estimate) else table$estimate / stats::sigma(model)
    cbind(comparison_type = comparison_type, table, stringsAsFactors = FALSE)
  })
  output <- do.call(rbind, rows)
  output$effect_size_type <- if (gamma) "ratio (exp of the estimate)" else "estimate / residual SD"
  output$scale <- model_estimate_scale(bundle)
  cbind(response = bundle$response_name, model_family = bundle$model_family, output, stringsAsFactors = FALSE)
}

# Family 2: do the three input predictors have the same coefficient?
#  - raw: coefficients as estimated (effect of one more unit of input; same unit for all speakers);
#  - standardized: each coefficient multiplied by the SD of its predictor (effect of one SD more input).
# For each scale: the three pairwise differences (t test, Holm-adjusted p in p_holm) and one
# joint F test that all three coefficients are equal (car::linearHypothesis).
input_coefficient_tests <- function(bundle) {
  model <- bundle$model
  input_terms <- family_2_input_terms(bundle$model_family)
  coefficients <- stats::coef(model)
  covariance <- stats::vcov(model)
  if (!all(input_terms %in% names(coefficients)) || anyNA(coefficients[input_terms])) {
    stop("The model does not have a coefficient for each of the three input predictors.", call. = FALSE)
  }
  predictor_sd <- vapply(input_terms, function(term) stats::sd(stats::model.frame(model)[[term]]), numeric(1))
  residual_df <- stats::df.residual(model)
  speakers <- analysis_options$speaker_order
  pairs <- utils::combn(seq_along(input_terms), 2, simplify = FALSE)

  contrast_vector <- function(pair, weights) {
    vector <- stats::setNames(numeric(length(coefficients)), names(coefficients))
    vector[input_terms[pair[[1]]]] <- weights[[pair[[1]]]]
    vector[input_terms[pair[[2]]]] <- -weights[[pair[[2]]]]
    vector
  }

  rows <- lapply(list(raw = rep(1, length(input_terms)), standardized = predictor_sd), function(weights) {
    pairwise <- do.call(rbind, lapply(pairs, function(pair) {
      vector <- contrast_vector(pair, weights)
      estimate <- sum(vector * coefficients)
      std_error <- sqrt(as.numeric(t(vector) %*% covariance %*% vector))
      statistic <- estimate / std_error
      margin <- stats::qt(0.975, residual_df) * std_error
      data.frame(
        test = "pairwise difference",
        contrast = sprintf("%s - %s", speakers[pair[[1]]], speakers[pair[[2]]]),
        estimate = estimate,
        std_error = std_error,
        statistic = statistic,
        statistic_type = "t",
        df = 1,
        df_residual = residual_df,
        p_value = 2 * stats::pt(-abs(statistic), residual_df),
        conf_low = estimate - margin,
        conf_high = estimate + margin,
        stringsAsFactors = FALSE
      )
    }))
    pairwise$p_holm <- stats::p.adjust(pairwise$p_value, method = "holm")

    hypothesis <- rbind(contrast_vector(pairs[[1]], weights), contrast_vector(pairs[[2]], weights))
    joint <- car::linearHypothesis(model, hypothesis.matrix = hypothesis, rhs = c(0, 0), test = "F")
    joint_row <- data.frame(
      test = "joint test: all three coefficients equal",
      contrast = paste(speakers, collapse = " = "),
      estimate = NA_real_,
      std_error = NA_real_,
      statistic = joint[2, "F"],
      statistic_type = "F",
      df = joint[2, "Df"],
      df_residual = joint[2, "Res.Df"],
      p_value = joint[2, "Pr(>F)"],
      conf_low = NA_real_,
      conf_high = NA_real_,
      p_holm = NA_real_,
      stringsAsFactors = FALSE
    )
    rbind(pairwise, joint_row)
  })

  output <- do.call(rbind, lapply(names(rows), function(scale) {
    cbind(coefficient_scale = scale, rows[[scale]], stringsAsFactors = FALSE)
  }))
  rownames(output) <- NULL
  output$scale <- model_estimate_scale(bundle)
  cbind(response = bundle$response_name, model_family = bundle$model_family, output, stringsAsFactors = FALSE)
}

# Effect sizes of a model: R2 (performance::r2) and, for Family 2, the standardized
# coefficient of each input predictor. Gaussian models: b * SD(x) / SD(y). Gamma
# models: b * SD(x) on the log link.
model_effect_sizes <- function(bundle) {
  model <- bundle$model
  # For a singular mixed model a part of R2 cannot be calculated: it is given as NA.
  r2 <- tryCatch(suppressWarnings(performance::r2(model)), error = function(e) NULL)
  r2_values <- unlist(lapply(r2, function(value) if (is.numeric(value)) as.numeric(value)[[1]] else NULL))
  rows <- data.frame(
    measure = gsub("_", " ", names(r2_values), fixed = TRUE),
    term = rep(NA_character_, length(r2_values)),
    value = as.numeric(r2_values),
    stringsAsFactors = FALSE
  )

  if (is_family_2(bundle$model_family)) {
    input_terms <- intersect(family_2_input_terms(bundle$model_family), names(stats::coef(model)))
    frame <- stats::model.frame(model)
    predictor_sd <- vapply(input_terms, function(term) stats::sd(frame[[term]]), numeric(1))
    gamma <- model_is_gamma(model)
    scale_factor <- if (gamma) 1 else stats::sd(stats::model.response(frame))
    rows <- rbind(rows, data.frame(
      measure = if (gamma) "standardized coefficient: b * SD(x), log link" else "standardized coefficient: b * SD(x) / SD(y)",
      term = input_terms,
      value = as.numeric(stats::coef(model)[input_terms] * predictor_sd / scale_factor),
      stringsAsFactors = FALSE
    ))
  }

  rownames(rows) <- NULL
  cbind(response = rep(bundle$response_name, nrow(rows)), model_family = rep(bundle$model_family, nrow(rows)),
        rows, stringsAsFactors = FALSE)
}

# Table for the report: the comparisons of a model, or a note when they cannot be calculated.
comparison_table_or_note <- function(table_function, bundle) {
  tryCatch(
    table_function(bundle),
    error = function(error_condition) {
      data.frame(
        response = bundle$response_name,
        model_family = bundle$model_family,
        note = paste("not available:", conditionMessage(error_condition)),
        stringsAsFactors = FALSE
      )
    }
  )
}
