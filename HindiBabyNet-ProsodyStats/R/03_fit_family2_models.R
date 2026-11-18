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

family2_count_bundles <- list()
family2_duration_bundles <- list()
family2_fixed_effects <- list()
family2_fit_stats <- list()
family2_diagnostics <- list()
family2_vif_tables <- list()

for (acoustic_measure in analysis_options$primary_acoustic_measures) {
  standardized <- standardize_predictor(prosody_long, acoustic_measure)
  prosody_long_with_z <- standardized$data
  predictor_name <- standardized$standardized_name

  for (model_spec in list(
    list(model_family = "family_2_count", response_name = "key_child_count_per_hour", adjustment_name = "input_count_per_hour"),
    list(model_family = "family_2_duration", response_name = "key_child_duration_per_hour", adjustment_name = "input_duration_per_hour")
  )) {
    output_stub <- build_model_stub(model_spec$model_family, model_spec$response_name, acoustic_measure)
    formula <- build_family_2_formula(
      outcome_name = model_spec$response_name,
      predictor_name = predictor_name,
      adjustment_name = model_spec$adjustment_name,
      model_family = model_spec$model_family
    )

    bundle <- fit_prosody_model(
      data = prosody_long_with_z,
      formula = formula,
      model_family = model_spec$model_family,
      response_name = model_spec$response_name,
      acoustic_measure = acoustic_measure,
      standardized_predictor = predictor_name
    )

    prediction_grid <- build_predictor_speaker_grid(bundle, bundle$analysis_data, predictor_name)
    prediction_data <- compute_monte_carlo_ci(bundle, prediction_grid)
    names(prediction_data)[names(prediction_data) == predictor_name] <- predictor_name
    prediction_csv <- file.path(analysis_paths$outputs_predictions, sprintf("%s_predictions.csv", output_stub))
    write_csv_output(prediction_data, prediction_csv)

    prediction_plot <- plot_predictor_speaker_predictions(bundle, prediction_data, bundle$analysis_data, predictor_name, y_label = model_spec$response_name)
    prediction_plot_path <- file.path(analysis_paths$outputs_plots, sprintf("%s_predictions.png", output_stub))
    save_plot(prediction_plot, prediction_plot_path, width = 10, height = 4.8)

    forest_plot <- plot_fixed_effects_forest(bundle)
    forest_plot_path <- file.path(analysis_paths$outputs_plots, sprintf("%s_forest.png", output_stub))
    save_plot(forest_plot, forest_plot_path, width = 8, height = 4.8)

    diagnostics <- run_model_diagnostics(bundle, output_stub = output_stub)
    vif_summary <- vif_table(bundle)

    save_model_bundle(bundle, file.path(analysis_paths$outputs_models, sprintf("%s.rds", output_stub)))
    write_csv_output(extract_fixed_effects(bundle), file.path(analysis_paths$outputs_tables, sprintf("%s_fixed_effects.csv", output_stub)))
    write_csv_output(model_fit_statistics(bundle), file.path(analysis_paths$outputs_tables, sprintf("%s_fit_stats.csv", output_stub)))
    write_csv_output(vif_summary, file.path(analysis_paths$outputs_tables, sprintf("%s_vif.csv", output_stub)))

    if (identical(model_spec$model_family, "family_2_count")) {
      family2_count_bundles[[acoustic_measure]] <- bundle
    } else {
      family2_duration_bundles[[acoustic_measure]] <- bundle
    }
    family2_fixed_effects[[output_stub]] <- extract_fixed_effects(bundle)
    family2_fit_stats[[output_stub]] <- model_fit_statistics(bundle)
    family2_diagnostics[[output_stub]] <- diagnostics$summary
    family2_vif_tables[[output_stub]] <- vif_summary
  }
}

family2_manifest <- bind_rows_safe(c(
  lapply(names(family2_count_bundles), function(acoustic_measure) {
    data.frame(
      model_family = "family_2_count",
      acoustic_measure = acoustic_measure,
      model_path = file.path(analysis_paths$outputs_models, sprintf("%s.rds", build_model_stub("family_2_count", "key_child_count_per_hour", acoustic_measure))),
      stringsAsFactors = FALSE
    )
  }),
  lapply(names(family2_duration_bundles), function(acoustic_measure) {
    data.frame(
      model_family = "family_2_duration",
      acoustic_measure = acoustic_measure,
      model_path = file.path(analysis_paths$outputs_models, sprintf("%s.rds", build_model_stub("family_2_duration", "key_child_duration_per_hour", acoustic_measure))),
      stringsAsFactors = FALSE
    )
  })
))
write_csv_output(family2_manifest, file.path(analysis_paths$outputs_tables, "family2_manifest.csv"))
