source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "data_prep.R"))
source(file.path("R", "functions", "model_selection.R"))

datasets_bundle <- load_analysis_datasets(write_summary = FALSE)

message(sprintf("Loaded %d rows from final_master.csv.", nrow(datasets_bundle$datasets$final_master)))
message("Phase 2 data loading is implemented; model fitting is not implemented yet.")
