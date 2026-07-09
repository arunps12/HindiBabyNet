source(file.path("R", "00_setup.R"))
source(file.path("R", "functions", "final_report.R"))

load_required_packages(attach = FALSE)
ensure_analysis_directories()

render_final_report(format = c("pdf", "html"))

message("Final report render requested for PDF and HTML outputs.")
