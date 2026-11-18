source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "table_helpers.R"))
source(file.path("R", "functions", "data_helpers.R"))
source(file.path("R", "functions", "model_helpers.R"))
source(file.path("R", "functions", "diagnostics_helpers.R"))
source(file.path("R", "functions", "prediction_helpers.R"))
source(file.path("R", "functions", "plotting_helpers.R"))

load_required_packages()
ensure_analysis_directories()

prosody_long <- read_csv_required(file.path(analysis_paths$data_derived, "prosody_long.csv"), "prosody_long")
prosody_long <- set_reference_levels(prosody_long)
store_age_day_summary(prosody_long)

family1_bundles <- list()
family1_fixed_effects <- list()
family1_fit_stats <- list()
family1_diagnostics <- list()
family1_vif_tables <- list()
family1_emmeans <- list()
family1_pairs <- list()

for (outcome_name in analysis_options$primary_acoustic_measures) {
  output_stub <- build_model_stub("family1", outcome_name)
  formula <- build_family_1_formula(outcome_name)
  bundle <- fit_prosody_model(
    data = prosody_long,
    formula = formula,
    model_family = "family_1",
    response_name = outcome_name,
    acoustic_measure = outcome_name
  )

  prediction_grid <- build_age_speaker_grid(bundle, bundle$analysis_data)
  prediction_data <- compute_monte_carlo_ci(bundle, prediction_grid)
  prediction_csv <- file.path(analysis_paths$outputs_predictions, sprintf("%s_predictions.csv", output_stub))
  write_csv_output(prediction_data, prediction_csv)

  prediction_plot <- plot_age_speaker_predictions(bundle, prediction_data, bundle$analysis_data, y_label = outcome_name)
  prediction_plot_path <- file.path(analysis_paths$outputs_plots, sprintf("%s_predictions.png", output_stub))
  save_plot(prediction_plot, prediction_plot_path)

  forest_plot <- plot_fixed_effects_forest(bundle)
  forest_plot_path <- file.path(analysis_paths$outputs_plots, sprintf("%s_forest.png", output_stub))
  save_plot(forest_plot, forest_plot_path, width = 8, height = 4.8)

  diagnostics <- run_model_diagnostics(bundle, output_stub = output_stub)
  vif_summary <- vif_table(bundle)
  emmeans_summary <- estimate_speaker_emmeans(bundle)
  pairs_summary <- estimate_speaker_pairs(bundle)

  save_model_bundle(bundle, file.path(analysis_paths$outputs_models, sprintf("%s.rds", output_stub)))
  write_csv_output(extract_fixed_effects(bundle), file.path(analysis_paths$outputs_tables, sprintf("%s_fixed_effects.csv", output_stub)))
  write_csv_output(model_fit_statistics(bundle), file.path(analysis_paths$outputs_tables, sprintf("%s_fit_stats.csv", output_stub)))
  write_csv_output(vif_summary, file.path(analysis_paths$outputs_tables, sprintf("%s_vif.csv", output_stub)))
  write_csv_output(emmeans_summary, file.path(analysis_paths$outputs_tables, sprintf("%s_emmeans.csv", output_stub)))
  write_csv_output(pairs_summary, file.path(analysis_paths$outputs_tables, sprintf("%s_speaker_pairs.csv", output_stub)))

  family1_bundles[[outcome_name]] <- bundle
  family1_fixed_effects[[outcome_name]] <- extract_fixed_effects(bundle)
  family1_fit_stats[[outcome_name]] <- model_fit_statistics(bundle)
  family1_diagnostics[[outcome_name]] <- diagnostics$summary
  family1_vif_tables[[outcome_name]] <- vif_summary
  family1_emmeans[[outcome_name]] <- emmeans_summary
  family1_pairs[[outcome_name]] <- pairs_summary
}

family1_manifest <- bind_rows_safe(lapply(names(family1_bundles), function(outcome_name) {
  data.frame(
    outcome = outcome_name,
    model_path = file.path(analysis_paths$outputs_models, sprintf("%s.rds", build_model_stub("family1", outcome_name))),
    stringsAsFactors = FALSE
  )
}))
write_csv_output(family1_manifest, file.path(analysis_paths$outputs_tables, "family1_manifest.csv"))
