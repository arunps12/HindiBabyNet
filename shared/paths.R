# Read config.yaml and give the data folders to the R statistical analyses.
#
# The rules are the same as in shared/hbn_paths.py:
# every folder has a default place under data_dir, a folder stored somewhere
# else is listed under `paths`, and relative paths start from the folder where
# config.yaml is saved.

hbn_default_folders <- c(
  raw_audio = "raw/audio",
  raw_cdi = "raw/cdi",
  metadata = "raw/metadata",
  prepared_audio = "derived/prepared-audio",
  vtc = "derived/vtc-classification",
  xgb = "derived/xgb-classification",
  prosody_features = "derived/prosody-features",
  statistical_data = "derived/statistical-data",
  textgrids = "derived/textgrids",
  annotations = "derived/manual-annotations",
  results = "results",
  cache = "cache"
)

hbn_local_path <- function(value, base, wsl_distro = "Ubuntu") {
  text <- gsub("\\\\", "/", path.expand(trimws(as.character(value))))
  on_windows <- .Platform$OS.type == "windows"
  if (on_windows) {
    if (grepl("^/mnt/[A-Za-z]/", text)) {
      text <- paste0(toupper(substr(text, 6, 6)), ":/", substring(text, 8))
    } else if (grepl("^/", text) && !grepl("^//", text)) {
      text <- paste0("//wsl.localhost/", wsl_distro, text)
    }
  } else if (grepl("^[A-Za-z]:/", text)) {
    text <- paste0("/mnt/", tolower(substr(text, 1, 1)), "/", substring(text, 4))
  } else if (grepl("^//", text)) {
    stop(sprintf("The network path %s cannot be opened from Ubuntu directly. Mount it first.", value), call. = FALSE)
  }
  absolute <- grepl("^(/|[A-Za-z]:/)", text)
  normalizePath(if (absolute) text else file.path(base, text), winslash = "/", mustWork = FALSE)
}

hbn_read_config <- function(repo_dir) {
  config_file <- Sys.getenv("HBN_CONFIG", unset = "")
  if (!nzchar(config_file)) config_file <- file.path(repo_dir, "config.yaml")
  if (!file.exists(config_file)) {
    stop(sprintf(
      "Configuration file not found: %s. Copy config.example.yaml to config.yaml and change the paths.",
      config_file
    ), call. = FALSE)
  }
  if (!requireNamespace("yaml", quietly = TRUE)) stop("Install the R package 'yaml' first.", call. = FALSE)
  config_file <- normalizePath(config_file, winslash = "/", mustWork = TRUE)
  values <- yaml::read_yaml(config_file, fileEncoding = "UTF-8")
  if (!is.list(values) || is.null(values$data_dir)) {
    stop(sprintf("%s must contain data_dir.", config_file), call. = FALSE)
  }
  distro <- if (is.null(values$wsl_distro)) "Ubuntu" else as.character(values$wsl_distro)
  list(
    file = config_file,
    values = values,
    wsl_distro = distro,
    data_dir = hbn_local_path(values$data_dir, dirname(config_file), distro)
  )
}

# Folder (or file) for one key, with optional sub-folders or a file name.
hbn_path <- function(config, key, ...) {
  override <- config$values$paths[[key]]
  if (!is.null(override) && nzchar(as.character(override))) {
    base <- hbn_local_path(override, dirname(config$file), config$wsl_distro)
  } else if (key %in% names(hbn_default_folders)) {
    base <- file.path(config$data_dir, hbn_default_folders[[key]])
  } else {
    stop(sprintf("Unknown path name in config: %s", key), call. = FALSE)
  }
  parts <- c(...)
  if (length(parts)) do.call(file.path, as.list(c(base, parts))) else base
}

# Like hbn_path(), but the file or folder must exist.
hbn_input <- function(config, key, ...) {
  found <- hbn_path(config, key, ...)
  if (!file.exists(found)) {
    stop(sprintf("Not found: %s\nCheck data_dir and paths$%s in %s.", found, key, config$file), call. = FALSE)
  }
  found
}

# Results folder of one workflow, for example hbn_results(config, "audio-stats").
hbn_results <- function(config, workflow, ...) {
  hbn_path(config, "results", workflow, ...)
}
