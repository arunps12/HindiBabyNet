# Tests the structure of the four models, the comparison of the linear and the
# quadratic age model, and the comparisons of speakers and input coefficients.
# Synthetic data only: no study data is read.
# Run from the project root: Rscript --vanilla audio-stats/tests/test_model_structure.R
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
root <- normalizePath(file.path(dirname(script), ".."), winslash = "/", mustWork = TRUE)

# 00_setup.R reads config.yaml and the HBN_ settings. A temporary file is enough here.
old_env <- Sys.getenv()
old_hbn <- grep("^HBN_", names(old_env), value = TRUE)
Sys.unsetenv(old_hbn)
config_file <- tempfile(fileext = ".yaml")
writeLines("data_dir: data", config_file)
Sys.setenv(
  HBN_CONFIG = config_file,
  HBN_FAMILY1_COUNT_LOCATION_INTERCEPT = "with",
  HBN_FAMILY1_DURATION_RANDOM_SLOPES = "speaker|Location",
  HBN_FAMILY1_DROP_FIXED_TERMS = "child_sex"
)
suppressMessages(source(file.path(root, "R", "00_setup.R")))
for (helper in c("model_selection.R", "model_diagnostics.R", "model_contrasts.R", "monte_carlo_ci.R",
                 "prediction_plots.R", "reporting.R", "question_sections.R")) {
  source(file.path(root, "R", "functions", helper))
}
Sys.unsetenv(grep("^HBN_", names(Sys.getenv()), value = TRUE))
if (length(old_hbn)) do.call(Sys.setenv, as.list(old_env[old_hbn]))
unlink(config_file)

formula_text <- function(formula) paste(deparse(formula, width.cutoff = 500L), collapse = " ")
near <- function(x, y, tolerance = 1e-8) isTRUE(all.equal(as.numeric(x), as.numeric(y), tolerance = tolerance))

# ---------------------------------------------------------------------------
# 1. Settings of each model
# ---------------------------------------------------------------------------
# The random effects are read for each Family 1 model, the fixed terms for each family.
overrides <- analysis_options$model_spec_overrides
stopifnot(
  identical(names(overrides), c("family1_count", "family1_duration", "family2_count", "family2_duration")),
  identical(overrides$family1_count$participant_intercept, "with"),
  identical(overrides$family1_count$location_intercept, "with"),
  identical(overrides$family1_count$random_slopes, character(0)),
  identical(overrides$family1_duration$location_intercept, "without"),
  identical(overrides$family1_duration$random_slopes, "speaker|Location"),
  identical(overrides$family1_count$drop_fixed_terms, "child_sex"),
  identical(overrides$family1_duration$drop_fixed_terms, "child_sex"),
  identical(names(overrides$family2_count), c("drop_fixed_terms", "extra_fixed_terms"))
)

analysis_options$model_robustness_mode <- "strict"
count_formula <- formula_text(build_candidate_specs("family_1", "input_count_hour", "linear")[[1]]$formula)
duration_formula <- formula_text(build_candidate_specs("family_1", "input_duration_min_hour", "linear")[[1]]$formula)
stopifnot(
  identical(count_formula,
            "input_count_hour ~ 1 + speaker + age_z + SES_num + speaker:age_z + (1 | participant_id) + (1 | Location)"),
  identical(duration_formula,
            "input_duration_min_hour ~ 1 + speaker + age_z + SES_num + speaker:age_z + (1 | participant_id) + (1 + speaker | Location)")
)

# Defaults: participant intercept, no location intercept, no random slope.
analysis_options$model_spec_overrides <- list()
for (response in c("input_count_hour", "input_duration_min_hour")) {
  candidates <- build_candidate_specs("family_1", response, "quadratic")
  stopifnot(
    length(candidates) == 1L,
    identical(formula_text(candidates[[1]]$formula), sprintf(
      "%s ~ 1 + speaker + age_z + age_z2 + SES_num + child_sex + speaker:age_z + (1 | participant_id)", response
    ))
  )
}

# ---------------------------------------------------------------------------
# 2. Fixed terms
# ---------------------------------------------------------------------------
family_1_terms <- function(age) build_requested_model_spec("family_1", "input_count_hour", age)$fixed_terms
stopifnot(
  identical(family_1_terms("none"), c("speaker", "SES_num", "child_sex")),
  identical(family_1_terms("linear"), c("speaker", "age_z", "SES_num", "child_sex", "speaker:age_z")),
  identical(family_1_terms("quadratic"), c("speaker", "age_z", "age_z2", "SES_num", "child_sex", "speaker:age_z")),
  identical(family_2_input_terms("family_2_count"),
            c("adult_female_count_hour", "adult_male_count_hour", "other_child_count_hour")),
  identical(family_2_input_terms("family_2_duration"),
            c("adult_female_duration_min_hour", "adult_male_duration_min_hour", "other_child_duration_min_hour"))
)

# Family 2 never has random effects: not when they are asked for and not as a fallback.
analysis_options$model_spec_overrides <- list(
  family2_count = list(participant_intercept = "with", location_intercept = "with", random_slopes = "age_z|Location")
)
for (mode in c("strict", "auto")) {
  analysis_options$model_robustness_mode <- mode
  for (family in c("family_2_count", "family_2_duration")) {
    response <- if (family == "family_2_count") "key_child_count_hour" else "key_child_duration_min_hour"
    formulas <- vapply(build_candidate_specs(family, response, "quadratic"), function(spec) formula_text(spec$formula), character(1))
    stopifnot(
      !any(grepl("|", formulas, fixed = TRUE)), !any(grepl("speaker", formulas, fixed = TRUE)),
      identical(formulas[[1]], sprintf("%s ~ 1 + %s + age_z + age_z2 + SES_num + child_sex", response,
                                       paste(family_2_input_terms(family), collapse = " + "))),
      all(vapply(family_2_input_terms(family), function(term) all(grepl(term, formulas, fixed = TRUE)), logical(1)))
    )
  }
}
analysis_options$model_spec_overrides <- list()
analysis_options$model_robustness_mode <- "strict"

# ---------------------------------------------------------------------------
# 3. Synthetic data
# ---------------------------------------------------------------------------
set.seed(2468)
n <- 60L
speakers <- analysis_options$speaker_order
wide <- data.frame(
  participant_id = factor(sprintf("S%03d", seq_len(n))),
  Location = factor(rep(c("A", "B", "C", "D"), length.out = n)),
  child_sex = factor(rep(c("female", "male"), n / 2)),
  SES_num = rep(c(2, 3, 4, 5, 3), length.out = n),
  age_days = sample(200:850, n)
)
wide$age_z <- as.numeric(scale(wide$age_days))
wide$age_z2 <- wide$age_z ^ 2
participant_effect <- rnorm(n, sd = 3)
long <- wide[rep(seq_len(n), each = 3), ]
long$speaker <- factor(rep(speakers, n), levels = speakers)
long$input_count_hour <- 40 + rep(c(8, -5, 2), n) + (2 + rep(c(0, 1, 3), n)) * long$age_z +
  rep(participant_effect, each = 3) + rnorm(3 * n, sd = 4)
long$input_duration_min_hour <- 6 + rep(c(2, -1, 0.5), n) + 0.4 * long$age_z + 0.3 * long$age_z2 +
  rep(participant_effect, each = 3) / 10 + rnorm(3 * n, sd = 0.4)
for (speaker in speakers) {
  wide[[paste0(speaker, "_count_hour")]] <- long$input_count_hour[long$speaker == speaker]
  wide[[paste0(speaker, "_duration_min_hour")]] <- long$input_duration_min_hour[long$speaker == speaker]
}
wide$key_child_count_hour <- 20 + 0.10 * wide$adult_female_count_hour + 0.05 * wide$adult_male_count_hour +
  0.60 * wide$other_child_count_hour + 2 * wide$age_z + rnorm(n, sd = 3)
wide$key_child_duration_min_hour <- 3 + 0.2 * wide$adult_female_duration_min_hour +
  0.5 * wide$other_child_duration_min_hour + 0.3 * wide$age_z + rnorm(n, sd = 0.4)
rownames(long) <- NULL

analysis_options$model_config$age_specification <- list(
  family1_count = "linear", family1_duration = "quadratic", family2_count = "linear", family2_duration = "none"
)

# ---------------------------------------------------------------------------
# 4. Family 1: reported model and comparison of the age models
# ---------------------------------------------------------------------------
# One participant has no age_z2. The linear model alone could use this participant;
# the comparison uses the same rows for both models.
long_with_gap <- long
long_with_gap$age_z2[long_with_gap$participant_id == "S007"] <- NA
count_bundle <- fit_model_with_age_comparison(long_with_gap, "input_count_hour", "family_1")
comparison <- count_bundle$age_comparison
used <- long[long$participant_id != "S007", ]
linear_fit <- lmerTest::lmer(input_count_hour ~ speaker + age_z + SES_num + child_sex + speaker:age_z +
                               (1 | participant_id), data = used, REML = FALSE)
quadratic_fit <- update(linear_fit, . ~ . + age_z2)
reference_test <- anova(linear_fit, quadratic_fit, refit = FALSE)
stopifnot(
  identical(count_bundle$age_specification, "linear"), inherits(count_bundle$model, "lmerModLmerTest"),
  !lme4::isREML(count_bundle$model), count_bundle$n_complete_cases == 3L * (n - 1L),
  identical(comparison$age_specification, c("linear", "quadratic")), identical(comparison$reported, c(TRUE, FALSE)),
  all(comparison$status == "ok"), all(comparison$estimation == "ML"), all(comparison$n_obs == 3L * (n - 1L)),
  all(comparison$fit_engine == "lmer"), all(comparison$random_terms == "(1 | participant_id)"),
  grepl("age_z2", comparison$fitted_formula[[2]], fixed = TRUE), !grepl("age_z2", comparison$fitted_formula[[1]], fixed = TRUE),
  !any(grepl("speaker:age_z2", comparison$fitted_formula, fixed = TRUE)),
  all(grepl("speaker:age_z", comparison$fitted_formula, fixed = TRUE)),
  near(comparison$AIC, c(AIC(linear_fit), AIC(quadratic_fit)), 1e-6),
  near(comparison$BIC, c(BIC(linear_fit), BIC(quadratic_fit)), 1e-6),
  near(comparison$logLik, c(logLik(linear_fit), logLik(quadratic_fit)), 1e-6),
  is.na(comparison$lrt_p[[1]]), comparison$lrt_df[[2]] == 1,
  near(comparison$lrt_chisq[[2]], reference_test$Chisq[[2]], 1e-5),
  near(comparison$lrt_p[[2]], reference_test$`Pr(>Chisq)`[[2]], 1e-5),
  near(comparison$delta_AIC[[2]], AIC(quadratic_fit) - AIC(linear_fit), 1e-6),
  all(comparison$preferred_by_AIC == if (AIC(quadratic_fit) < AIC(linear_fit)) "quadratic" else "linear"),
  all(comparison$preferred_by_LRT == if (reference_test$`Pr(>Chisq)`[[2]] < 0.05) "quadratic" else "linear"),
  all(c("converged", "singular_fit", "heteroscedasticity_p", "residual_test", "residual_test_p") %in% names(comparison)),
  all(comparison$residual_test == "Shapiro-Wilk"), all(is.finite(comparison$residual_test_p))
)

# The data has a real quadratic age effect in the duration: every criterion finds it.
duration_bundle <- fit_model_with_age_comparison(long, "input_duration_min_hour", "family_1")
stopifnot(
  identical(duration_bundle$age_specification, "quadratic"),
  identical(duration_bundle$age_comparison$reported, c(FALSE, TRUE)),
  all(duration_bundle$age_comparison$preferred_by_AIC == "quadratic"),
  all(duration_bundle$age_comparison$preferred_by_BIC == "quadratic"),
  all(duration_bundle$age_comparison$preferred_by_LRT == "quadratic"),
  duration_bundle$age_comparison$delta_AIC[[2]] < 0
)

# ---------------------------------------------------------------------------
# 5. Family 1: pairwise comparisons of the speakers
# ---------------------------------------------------------------------------
pairwise <- speaker_pairwise_table(count_bundle)
effects <- lme4::fixef(count_bundle$model)
at_mean_age <- pairwise[pairwise$comparison_type == "speaker difference at mean age", ]
slopes <- pairwise[pairwise$comparison_type == "difference in age slope (per SD of age)", ]
stopifnot(
  nrow(at_mean_age) == 3L, nrow(slopes) == 3L,
  identical(at_mean_age$contrast, c("adult_female - adult_male", "adult_female - other_child", "adult_male - other_child")),
  # With age_z = 0, the difference to the adult female is the speaker coefficient.
  near(at_mean_age$estimate, c(-effects[["speakeradult_male"]], -effects[["speakerother_child"]],
                               effects[["speakeradult_male"]] - effects[["speakerother_child"]]), 1e-6),
  near(slopes$estimate, c(-effects[["speakeradult_male:age_z"]], -effects[["speakerother_child:age_z"]],
                          effects[["speakeradult_male:age_z"]] - effects[["speakerother_child:age_z"]]), 1e-6),
  near(at_mean_age$p_holm, p.adjust(at_mean_age$p_value, "holm")),
  near(at_mean_age$effect_size, at_mean_age$estimate / sigma(count_bundle$model)),
  all(pairwise$effect_size_type == "estimate / residual SD"), all(pairwise$scale == "original scale"),
  all(at_mean_age$conf_low < at_mean_age$estimate), all(at_mean_age$estimate < at_mean_age$conf_high)
)
# In the quadratic model the comparison is also made at the mean age (age_z = 0 and age_z2 = 0).
quadratic_pairwise <- speaker_pairwise_table(duration_bundle)
quadratic_effects <- lme4::fixef(duration_bundle$model)
stopifnot(near(quadratic_pairwise$estimate[[1]], -quadratic_effects[["speakeradult_male"]], 1e-6))

# ---------------------------------------------------------------------------
# 6. Family 2: three separate predictors, no random effects
# ---------------------------------------------------------------------------
analysis_options$model_robustness_mode <- "auto"
child_count_bundle <- fit_model_with_age_comparison(wide, "key_child_count_hour", "family_2_count")
analysis_options$model_robustness_mode <- "strict"
child_model <- child_count_bundle$model
reference_model <- lm(key_child_count_hour ~ adult_female_count_hour + adult_male_count_hour +
                        other_child_count_hour + age_z + SES_num + child_sex, data = wide)
reference_quadratic <- update(reference_model, . ~ . + age_z2)
child_comparison <- child_count_bundle$age_comparison
stopifnot(
  identical(class(child_model), "lm"), identical(child_count_bundle$fit_engine, "lm"),
  nobs(child_model) == n, near(coef(child_model), coef(reference_model)[names(coef(child_model))]),
  identical(child_count_bundle$participant_intercept_mode, "without"),
  identical(child_count_bundle$location_intercept_mode, "without"),
  all(child_comparison$random_terms == "none"), all(child_comparison$fit_engine == "lm"), all(child_comparison$n_obs == n),
  near(child_comparison$AIC, c(AIC(reference_model), AIC(reference_quadratic))),
  near(child_comparison$BIC, c(BIC(reference_model), BIC(reference_quadratic))),
  near(child_comparison$lrt_chisq[[2]], 2 * (as.numeric(logLik(reference_quadratic)) - as.numeric(logLik(reference_model)))),
  child_comparison$lrt_df[[2]] == 1, is.na(child_comparison$singular_fit[[1]])
)

# With the age specification "none", the linear and the quadratic model are still compared.
child_duration_bundle <- fit_model_with_age_comparison(wide, "key_child_duration_min_hour", "family_2_duration")
stopifnot(
  identical(child_duration_bundle$age_specification, "none"),
  !any(grepl("age_z", names(coef(child_duration_bundle$model)))),
  identical(child_duration_bundle$age_comparison$age_specification, c("linear", "quadratic")),
  !any(child_duration_bundle$age_comparison$reported), all(child_duration_bundle$age_comparison$status == "ok")
)

# ---------------------------------------------------------------------------
# 7. Family 2: tests that compare the three coefficients
# ---------------------------------------------------------------------------
tests <- input_coefficient_tests(child_count_bundle)
raw <- tests[tests$coefficient_scale == "raw", ]
standardized <- tests[tests$coefficient_scale == "standardized", ]
b <- coef(child_model)
terms <- family_2_input_terms("family_2_count")
predictor_sd <- vapply(terms, function(term) sd(wide[[term]]), numeric(1))
single <- car::linearHypothesis(child_model, "adult_female_count_hour = other_child_count_hour")
joint <- car::linearHypothesis(child_model, c("adult_female_count_hour = adult_male_count_hour",
                                              "adult_female_count_hour = other_child_count_hour"))
stopifnot(
  nrow(raw) == 4L, nrow(standardized) == 4L,
  identical(raw$contrast[1:3], c("adult_female - adult_male", "adult_female - other_child", "adult_male - other_child")),
  near(raw$estimate[1:3], c(b[[terms[1]]] - b[[terms[2]]], b[[terms[1]]] - b[[terms[3]]], b[[terms[2]]] - b[[terms[3]]])),
  near(raw$statistic[[2]] ^ 2, single[2, "F"], 1e-6), near(raw$p_value[[2]], single[2, "Pr(>F)"], 1e-6),
  near(raw$p_holm[1:3], p.adjust(raw$p_value[1:3], "holm")),
  raw$statistic_type[[4]] == "F", raw$df[[4]] == 2, raw$df_residual[[4]] == df.residual(child_model),
  near(raw$statistic[[4]], joint[2, "F"], 1e-6), near(raw$p_value[[4]], joint[2, "Pr(>F)"], 1e-6),
  # Standardized: every coefficient is multiplied by the SD of its predictor.
  near(standardized$estimate[1:3],
       c(b[[terms[1]]] * predictor_sd[[1]] - b[[terms[2]]] * predictor_sd[[2]],
         b[[terms[1]]] * predictor_sd[[1]] - b[[terms[3]]] * predictor_sd[[3]],
         b[[terms[2]]] * predictor_sd[[2]] - b[[terms[3]]] * predictor_sd[[3]])),
  # The other child input has the largest coefficient in the synthetic data: the tests find it.
  raw$p_value[[4]] < 0.05, raw$estimate[[2]] < 0, raw$conf_high[[2]] < 0
)

sizes <- model_effect_sizes(child_count_bundle)
standardized_rows <- sizes[!is.na(sizes$term), ]
stopifnot(
  any(sizes$measure == "R2"), any(sizes$measure == "R2 adjusted"),
  near(sizes$value[sizes$measure == "R2"], summary(child_model)$r.squared, 1e-6),
  identical(standardized_rows$term, terms),
  near(standardized_rows$value, b[terms] * predictor_sd / sd(wide$key_child_count_hour))
)
stopifnot(any(grepl("R2", model_effect_sizes(count_bundle)$measure)))

# ---------------------------------------------------------------------------
# 8. Prediction grids
# ---------------------------------------------------------------------------
stopifnot(
  near(reference_value(c(1, 2, 6)), 3), identical(reference_value(c("b", "a")), "b"),
  identical(as.character(reference_value(factor(c("x", "y", "y")))), "y")
)
analysis_runtime$age_days_mean <- mean(wide$age_days)
analysis_runtime$age_days_sd <- sd(wide$age_days)
grid <- build_input_output_grid(child_count_bundle, wide, n_points = 5)
male_rows <- grid[grid$speaker == "adult_male", ]
stopifnot(
  nrow(grid) == 15L, identical(levels(grid$speaker), speakers), all(grid$age_z == 0),
  near(male_rows$adult_male_count_hour, seq(min(wide$adult_male_count_hour), max(wide$adult_male_count_hour), length.out = 5)),
  near(male_rows$input_count_hour, male_rows$adult_male_count_hour),
  near(male_rows$adult_female_count_hour, rep(mean(wide$adult_female_count_hour), 5)),
  near(male_rows$other_child_count_hour, rep(mean(wide$other_child_count_hour), 5)),
  near(male_rows$SES_num, rep(mean(wide$SES_num), 5))
)
predictions <- compute_monte_carlo_ci(child_count_bundle, grid, nsim = 2)
slope <- diff(predictions$predicted[predictions$speaker == "other_child"]) / diff(grid$other_child_count_hour[grid$speaker == "other_child"])
stopifnot(near(slope, rep(b[["other_child_count_hour"]], 4), 1e-6), all(predictions$lower < predictions$upper))
summary_table <- prediction_summary_table(predictions)
stopifnot(nrow(summary_table) == 3L, all(summary_table$predictor == "input count hour"))

observed <- stack_input_observations(child_count_bundle, wide)
stopifnot(
  nrow(observed) == 3L * n, identical(names(observed), c("speaker", "input_count_hour", "key_child_count_hour")),
  near(observed$input_count_hour[observed$speaker == "other_child"], wide$other_child_count_hour),
  near(observed$key_child_count_hour[observed$speaker == "adult_male"], wide$key_child_count_hour)
)

# Family 1: the grid holds SES at its mean, and the age trend is named by age.
age_grid <- build_age_speaker_grid(count_bundle, long, n_points = 4)
stopifnot(near(age_grid$SES_num, rep(mean(long$SES_num), nrow(age_grid))), !("age_z2" %in% names(age_grid)))
age_predictions <- compute_monte_carlo_ci(count_bundle, age_grid, nsim = 2)
stopifnot(all(prediction_summary_table(age_predictions)$predictor == "age days"))

# ---------------------------------------------------------------------------
# 9. Family 2 with a strongly skewed response: Gamma model without random effects
# ---------------------------------------------------------------------------
skewed <- wide
skewed$key_child_count_hour <- exp(3 + 0.02 * wide$other_child_count_hour + 0.3 * wide$age_z + rnorm(n, sd = 0.9))
analysis_options$model_config$response_transform <- "none"
gamma_bundle <- fit_model_with_age_comparison(skewed, "key_child_count_hour", "family_2_count")
gamma_grid <- build_input_output_grid(gamma_bundle, skewed, n_points = 5)
gamma_predictions <- compute_monte_carlo_ci(gamma_bundle, gamma_grid, nsim = 2)
gamma_tests <- input_coefficient_tests(gamma_bundle)
stopifnot(
  identical(gamma_bundle$selection$model_type[[1]], "gamma_glmm"), identical(class(gamma_bundle$model), c("glm", "lm")),
  identical(family(gamma_bundle$model)$family, "Gamma"), all(gamma_bundle$age_comparison$fit_engine == "glm"),
  all(gamma_bundle$age_comparison$residual_test == "DHARMa uniformity (KS)"),
  all(gamma_bundle$age_comparison$status == "ok"), gamma_bundle$age_comparison$lrt_df[[2]] == 1,
  # The predictions are on the scale of the response, not on the log scale.
  near(gamma_predictions$predicted, predict(gamma_bundle$model, newdata = gamma_grid, type = "response")),
  all(gamma_predictions$lower < gamma_predictions$predicted), all(gamma_predictions$predicted < gamma_predictions$upper),
  nrow(gamma_tests) == 8L, all(gamma_tests$scale == "log link"),
  any(grepl("log link", model_effect_sizes(gamma_bundle)$measure, fixed = TRUE))
)

# ---------------------------------------------------------------------------
# 10. Terms of the research-question summaries
# ---------------------------------------------------------------------------
stopifnot(
  identical(focal_terms_for_question("rq3", "key_child_count_hour"), terms),
  identical(focal_terms_for_question("rq3", "key_child_duration_min_hour"), family_2_input_terms("family_2_duration")),
  identical(label_term("other_child_count_hour"), "Other child input count per hour"),
  setequal(strsplit(summarise_model_evidence(child_count_bundle)$focal_terms, "; ", fixed = TRUE)[[1]], terms),
  grepl("speakeradult_male:age_z", summarise_model_evidence(count_bundle)$focal_terms, fixed = TRUE)
)

message("Model structure tests passed: formulas, settings per model, age comparison, speaker comparisons, coefficient tests and grids.")
