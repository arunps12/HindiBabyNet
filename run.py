"""Native workflow launcher: explicit setup, local config snapshot, whole-stage resume.

No scientific overrides or participant-level recovery. Unknown, changed, or
incomplete occupied outputs require fresh configured paths, never --force.

configure snapshots native config with every workflow disabled; setup explicitly
syncs enabled Python environments (R dependencies are manual). check requires
already-installed environments and reads inputs without executing workflows.
all runs enabled workflows, excluding reports and interactive annotation.
Use a workflow with --stage for native subcommands, then -- for stage options.
Audio preparation requires --wav or --recordings-parquet; classification also
accepts --participant-id with --wav. TextGrid/annotation use the configured
backend, optionally --input-file/--participant-id. Evaluation reads configured
manual annotations, optionally --participant-id; it has no --input-csv option.
R reports use --stage report, HTML only. No inputs or outputs are redirected.
Manifest-driven audio stages are not auto-resumed: the launcher fingerprints
the parquet itself, but does not decode its referenced paths or membership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml


ORDER = ("audio", "vocal-stats", "prosody", "vocal-models", "prosody-stats", "cdi")
PROJECTS = dict(zip(ORDER, ("HindiBabyNet-Pipeline", "HindiBabyNet-VocalInputStats",
                          "HindiBabyNet-ProsodyModel", "HindiBabyNet-VocalInputStats",
                          "HindiBabyNet-ProsodyStats", "HindiBabyNet-CDIAnalysis")))
SECTIONS = dict(zip(ORDER, ("audio", "vocal", "prosody", "vocal", "prosody_stats", "cdi")))
NATIVE = dict(zip(ORDER, ("configs/config.yaml", "configs/config.yaml", "config/config.yaml",
                         "configs/config.yaml", "config/config.yaml", "configs/config.yaml")))
ROOT = Path(__file__).resolve().parent
OWNER = "HindiBabyNet whole-stage ownership v1"


def read_yaml(path):
    with path.open(encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def path_at(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing/non-string path relative to {base}: {value!r}")
    return (base / Path(value)).resolve()


def configure(destination):
    """Snapshot mappings, not datasets; keep defaults in their native source code."""
    snapshot = {"workflows": {}}
    for name in ORDER:
        project = (ROOT / PROJECTS[name]).resolve()
        section = SECTIONS[name]
        snapshot["workflows"][name] = {"enabled": False, "project": str(project)}
        if section in snapshot:
            continue
        source = project / NATIVE[name]
        payload = read_yaml(source)
        mapping = dict(payload.get(section, payload))
        original_root = path_at(source.parent, mapping["project_root"]) if section in payload and "project_root" in mapping else project
        mapping["project_root"] = str(original_root)
        snapshot[section] = mapping
        if name == "audio":
            sidecar = project / "configs/params.yaml"
            params = payload.get("audio_params", {}) if "audio" in payload else read_yaml(sidecar) if sidecar.exists() else {}
            params = dict(params.get("audio_params", params))
            params["project_root"] = str(path_at(source.parent, params["project_root"]) if "project_root" in params else project)
            snapshot["audio_params"] = params
    with destination.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False, allow_unicode=True)
    print(f"Created {destination}; all workflows disabled. No paths redirected.")


def overlaps(left, right):
    return left == right or left in right.parents or right in left.parents


def r_scripts(name):
    if name == "vocal-models":
        return ["00_setup.R", "01_input_models.R", "02_input_output_models.R",
                "03_model_tables_and_plots.R"]
    return ["00_setup.R", "01_prepare_prosody_data.R", "02_fit_family1_models.R",
            "03_fit_family2_models.R", "04_create_tables_plots.R"]


def stage_arguments(name, stage, extra, project):
    """Parse exact native read-only options; never forward unknown tokens."""
    options = {}
    if name == "audio":
        options = {
            "pipeline": ("limit",),
            "prepare-audio": ("recordings-parquet", "wav", "recording-id", "limit"),
            "classify-xgb": ("recordings-parquet", "wav", "participant-id", "limit"),
            "classify-vtc": ("recordings-parquet", "wav", "participant-id", "limit"),
            "generate-textgrids": ("input-file", "participant-id", "limit"),
            "annotate-segments": ("input-file", "participant-id"),
            "evaluate-models": ("participant-id",),
        }[stage]
    elif name == "prosody":
        options = {"extract": ("participants", "speakers", "device", "no-cache", "log-level"),
                   "validate": (), "inspect": ("participant", "max-segments")}[stage]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    for option in options:
        kwargs = {"action": "store_true"} if option == "no-cache" else {}
        if option in {"limit", "max-segments"}:
            kwargs["type"] = int
        if option in {"participants", "speakers"}:
            kwargs["action"] = "append"
        parser.add_argument("--" + option, **kwargs)
    try:
        parsed, unknown = parser.parse_known_args(extra)
    except argparse.ArgumentError as exc:
        raise ValueError(str(exc)) from exc
    if unknown:
        raise ValueError(f"Unsafe/unsupported stage arguments: {unknown}")
    values = {key: value for key, value in vars(parsed).items() if value is not None}
    for key in ("limit", "max_segments"):
        if key in values and values[key] <= 0:
            raise ValueError(f"{key} must be positive")
    identifiers = [values[key] for key in ("participant_id", "recording_id") if key in values]
    for key in ("wav", "input_file"):
        explicit = "recording_id" if stage == "prepare-audio" else "participant_id"
        if key in values and explicit not in values:
            path = Path(values[key])
            identifiers.append(path.stem if key == "wav" else path.parent.name or path.stem)
    if any(not re.fullmatch(r"[\w .-]+", value) or not value.strip(" .") for value in identifiers):
        raise ValueError("Identifiers must be single safe path components")
    sources = [path_at(project, values[key]) for key in ("recordings_parquet", "wav", "input_file") if key in values]
    if "wav" in values and "recordings_parquet" in values:
        raise ValueError("Provide either --wav or --recordings-parquet, not both")
    if stage == "prepare-audio" and not sources:
        raise ValueError("prepare-audio requires --wav or --recordings-parquet")
    if "recording_id" in values and "wav" not in values:
        raise ValueError("--recording-id requires --wav")
    if stage in {"classify-xgb", "classify-vtc"} and "participant_id" in values and "wav" not in values:
        raise ValueError("Classification --participant-id requires --wav (native batch mode does not filter it)")
    if name == "prosody" and stage == "inspect" and "participant" not in values:
        raise ValueError("prosody inspect requires --participant")
    return values, sources


def workflow(name, settings, base, stage=None, extra=(), shared=None):
    project = path_at(base, settings.get("project", PROJECTS[name]))
    if not project.is_dir():
        raise ValueError(f"{name}: missing project directory: {project}")
    config = path_at(base, settings["config"]) if settings.get("config") else shared
    if name == "vocal-models" and config == shared and shared and "vocal" not in read_yaml(shared):
        config = None  # Legacy R defaults, deliberately not Python's flat YAML.
    if config is None and name != "vocal-models":
        config = project / NATIVE[name]
    payload = read_yaml(config) if config else {}
    section = SECTIONS[name]
    if config is not None and config == shared and section not in payload:
        raise ValueError(f"{name}: shared YAML requires section {section!r} or a legacy config pointer")
    data = payload.get(section, payload)
    if not isinstance(data, dict):
        raise ValueError(f"{section}: expected native mapping")
    configured_root = data.get("project_root")
    if configured_root is not None and name in {"vocal-models", "prosody-stats"}:
        configured_root = os.path.expanduser(configured_root)
    data_root = path_at(config.parent, configured_root) if section in payload and configured_root is not None else project
    if name == "cdi" and section not in payload:
        data_root = config.parent.parent
    params, params_path = {}, None
    if name == "audio":
        if section in payload:
            params = payload.get("audio_params", {})
        else:
            params_path = config.with_name("params.yaml")
            if not params_path.exists():
                params_path = project / "configs/params.yaml"
            params = read_yaml(params_path) if params_path.exists() else {}
            params = params.get("audio_params", params)
        if settings.get("params") and path_at(base, settings["params"]) != params_path:
            raise ValueError("audio: legacy params must be the native sibling/default; prefer configure")
        allowed = ("pipeline", "prepare-audio", "classify-xgb", "classify-vtc",
                   "generate-textgrids", "annotate-segments", "evaluate-models")
        selected = stage or "pipeline"
        if selected not in allowed:
            raise ValueError(f"audio: stage must be one of {allowed}")
        command = ["hindibabynet-" + selected]
    elif name == "vocal-stats":
        selected = stage or "all"
        if selected not in ("all", "build-master", "create-long", "eda", "plots"):
            raise ValueError("vocal-stats: invalid stage")
        command = ["hindibabynet-vocalinputstats", selected, "--config", str(config)]
    elif name == "prosody":
        selected = stage or "extract"
        if selected not in ("extract", "validate", "inspect"):
            raise ValueError("prosody: stage must be extract, validate, or inspect")
        command = ["hindibabynet-prosody", selected, "--config", str(config)]
    elif name == "cdi":
        if stage or extra:
            raise ValueError("cdi: no native CLI stages or pass-through arguments")
        command = ["python", "-c", "import sys; from hindibabynet_cdi.config import load_config; "
                   "from hindibabynet_cdi.pipeline import run_pipeline; "
                   "run_pipeline(load_config(sys.argv[1]))", str(config)]
    else:
        if extra:
            raise ValueError("R workflows do not support pass-through arguments")
        if name == "vocal-models" and config and section not in payload:
            raise ValueError("vocal-models: R requires a shared vocal section, or no config for native defaults")
        scripts = r_scripts(name)
        if stage and stage not in scripts + ["report"]:
            raise ValueError(f"{name}: stage must be one of {scripts}")
        scripts = [stage] if stage and stage != "report" else scripts
        if stage == "report":
            scripts = ["04_final_report.R"] if name == "vocal-models" else []
        for script in scripts:
            if not (project / "R" / script).is_file():
                raise ValueError(f"{name}: missing R script {script}")
        # One R process preserves the scripts' native shared session semantics.
        command = ["-e", "; ".join(f'source("R/{s}")' for s in scripts)]
    options, sources = stage_arguments(name, stage or {"audio": "pipeline", "prosody": "extract"}.get(name), extra, project)
    # Mirror only native filesystem precedence; leave scientific mappings intact.
    effective = dict(data)
    if name in {"vocal-stats", "vocal-models"}:
        for key in ("audio_root", "vtc_output_root", "metadata_path", "metadata_csv", "manual_mapping_csv",
                    "derived_data_dir", "private_data_dir", "figures_dir", "tables_dir", "results_dir"):
            if effective.get(key):
                value = os.path.expanduser(effective[key])
                effective[key] = os.path.expandvars(value) if name == "vocal-stats" else value
    if name == "audio":
        sc = dict(data.get("speaker_classification", {}))
        for backend, key in (("xgb", "model_path"), ("vtc", "repo_path")):
            values = dict(params.get(backend, {}))
            if key in values and "audio" in payload:
                params_root = path_at(config.parent, params.get("project_root", str(data_root)))
                values[key] = str(path_at(params_root, values[key]))
            values.update(sc.get(backend, {}))
            sc[backend] = values
        effective["speaker_classification"] = sc
        # Native data_ingestion.raw_audio_root is CWD-relative even in sections.
        if data.get("data_ingestion", {}).get("raw_audio_root"):
            effective["data_ingestion"] = dict(data["data_ingestion"])
            effective["data_ingestion"]["raw_audio_root"] = str(path_at(project, data["data_ingestion"]["raw_audio_root"]))
    inputs = [p for p in (config, params_path) if p and p.exists()]
    item = {"name": name, "project": project, "config": config, "data": effective,
            "root": data_root, "stage": stage, "extra": list(extra),
            "inputs": inputs + sources, "outputs": [], "optional": [],
            "options": options, "sources": sources, "command": command + list(extra)}
    if name == "audio" and stage in {"generate-textgrids", "annotate-segments"}:
        item["command"] += ["--backend", str(effective.get("speaker_classification", {}).get("backend", "xgb")).lower()]
    stage_outputs(item)
    return item


def stage_outputs(item):
    """One stage-local inventory of native inputs, guarded writes and sentinels."""
    name, stage, root, data = (item[k] for k in ("name", "stage", "root", "data"))
    paths = data.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"{name}: paths must be a mapping")
    p = lambda key, default=None: path_at(root, paths.get(key, default))
    expected = []  # (directory, recursive native filename pattern)
    item["exclude"] = []
    if name == "prosody":
        out = p("output_dir")
        item["inputs"] += [p(key) for key in ("rttm_root", "audio_root", "participant_lookup")]
        if stage in (None, "extract"):
            item["outputs"] = [out, p("cache_dir")]
            expected = [(out, data.get("output", {}).get("final_filename", "participant_speaker_acoustic_features.csv")),
                        (out, "run_metadata.json"), (out / "qc", "processing_summary.csv")]
        else:
            item["outputs"] = [out]
            expected = [(out / "logs", "*.log")]
        for value in data.get("ssl", {}).values():
            if isinstance(value, str) and path_at(item["project"], value).exists():
                item["inputs"].append(path_at(item["project"], value))
    elif name == "vocal-stats":
        d = lambda key: path_at(root, data[key])
        derived, results = d("derived_data_dir"), d("results_dir")
        item["exclude"] = [results / part for part in
                           ("r_models", "r_tables", "r_diagnostics", "r_predictions", "r_plots", "final_report")]
        if stage in (None, "all", "build-master"):
            item["inputs"] += [d(key) for key in ("audio_root", "vtc_output_root")]
            item["inputs"].append(path_at(root, data.get("metadata_path", data.get("metadata_csv"))))
            if data.get("manual_mapping_csv"):
                item["inputs"].append(d("manual_mapping_csv"))
            item["outputs"] = [d(key) for key in ("derived_data_dir", "private_data_dir", "results_dir")]
            expected += [(derived, "final_master.csv"), (d("private_data_dir"), "participant_lookup.csv")]
            expected += [(results, f) for f in ("validation_report.csv", "recording_duration_audit.csv",
                                                "age_validation.csv", "dataset_build_report.txt")]
        else:
            item["inputs"].append(derived / "final_master.csv")
        if stage in (None, "all", "create-long"):
            expected += [(derived, f) for f in ("input_long.csv", "input_output_long.csv")]
        if stage in (None, "all", "eda"):
            item["outputs"].append(d("tables_dir"))
            expected += [(d("tables_dir"), "participant_summary.csv")]
        if stage in (None, "all", "plots"):
            item["outputs"].append(d("figures_dir"))
            expected += [(d("figures_dir"), "*.png"), (d("figures_dir"), "*.pdf")]
        if stage == "plots":
            item["inputs"].append(derived / "input_long.csv")
        elif stage == "create-long":
            validation = results / "validation_report.csv"
            item["inputs"].append(validation)
            item["optional"].append(validation)
            item["outputs"] = [derived / f for f in ("input_long.csv", "input_output_long.csv")]
            item["outputs"].append(results / "dataset_build_report.txt")
    elif name == "audio":
        sc = data.get("speaker_classification", {})
        backend = str(sc.get("backend", "xgb")).lower()
        if backend not in {"xgb", "vtc"}:
            raise ValueError(f"audio: unsupported backend {backend!r}; no substitution")
        item["backend"] = backend
        classification = path_at(root, sc.get("output_root", paths.get("classification_output_root")))
        bookkeeping = [path_at(root, data.get("logs_root", "logs")), path_at(root, data.get("artifacts_root", "artifacts/runs"))]
        if stage in (None, "pipeline") or stage == "prepare-audio" and "wav" not in item["options"]:
            raw = data.get("data_ingestion", {}).get("raw_audio_root") or paths.get("raw_audio_root")
            item["inputs"].append(path_at(root, raw))
        if stage in (None, "pipeline", "classify-xgb", "classify-vtc"):
            if backend == "xgb":
                item["inputs"].append(path_at(root, sc.get("xgb", {}).get("model_path", "models/xgb_egemaps.pkl")))
            else:
                repo = path_at(root, sc.get("vtc", {}).get("repo_path", "external_models/VTC"))
                item["inputs"] += [repo / "scripts/infer.py", repo / "uv.lock",
                                   repo / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")]
            expected = [(classification / backend, "rttm.csv" if backend == "vtc" else "*_segments.parquet")]
            prepared = p("prepared_audio_root")
            if stage in ("classify-xgb", "classify-vtc"):
                if "wav" not in item["options"]:
                    item["inputs"].append(prepared)
                item["outputs"] = [classification] + bookkeeping
            else:
                expected += [(prepared, "*.wav")]
                item["outputs"] = [classification, prepared, p("raw_joined_audio_root")] + bookkeeping
        elif stage == "prepare-audio":
            expected = [(path_at(root, data.get("artifacts_root", "artifacts/runs")), "*_audio_manifest.parquet")]
            item["outputs"] = [p("prepared_audio_root"), p("raw_joined_audio_root")] + bookkeeping
        else:
            key, pattern = {"generate-textgrids": ("textgrid_output_root", "*.TextGrid"),
                            "annotate-segments": ("manual_annotation_root", "speaker_class_annotations.csv"),
                            "evaluate-models": ("evaluation_output_root", "metrics.json")}[stage]
            expected = [(p(key), pattern)]
            item["outputs"] = [p(key)] + bookkeeping
            if stage == "evaluate-models":
                item["inputs"].append(p("manual_annotation_root"))
                expected += [(p(key), f) for f in ("confusion_matrix.csv", "classwise_metrics.csv",
                                                  "evaluated_annotations.csv", "report.md")]
            else:
                if not item["sources"]:
                    item["inputs"].append(classification / backend)
                if stage == "annotate-segments":
                    item["inputs"].append(p("prepared_audio_root"))
        # A forced native classification stage must fingerprint its own backend.
        if stage in ("classify-xgb", "classify-vtc") and stage != "classify-" + backend:
            raise ValueError("Set the configured audio backend to match the selected classification stage")
    elif name == "cdi":
        raw = p("raw_data", "data/raw")
        contact = p("personal_info", "data/personal_info")
        item["inputs"] += [raw, p("metadata", "metadata"), contact]
        item["optional"].append(contact)
        item["inputs"] += [path_at(raw, value) for value in data.get("raw_files", {}).values()]
        item["outputs"] = [p("processed_data", "data/processed")]
        defaults = {f"{group}_tables_dir": f"outputs/tables/{group}" for group in ("qc", "cdi1", "cdi2", "combined")}
        for group, categories in (("cdi1", ("comprehension", "production", "items", "categories")),
                                  ("cdi2", ("production", "items", "categories")), ("combined", ()), ("qc", ())):
            for category in categories or ("",):
                key = f"{group}_{category + '_' if category else ''}figures_dir"
                defaults[key] = f"outputs/figures/{group}" + (f"/{category}" if category else "")
        item["outputs"] += [path_at(root, data.get("outputs", {}).get(key, default)) for key, default in defaults.items()]
        expected = [(p("processed_data", "data/processed"), filename) for filename in
                    ("participant_linkage.csv", "participant_analysis_all.csv", "cdi_all_items_long.csv")]
    elif name == "vocal-models":
        results = path_at(root, data.get("results_dir", "results"))
        tables = results / "r_tables"
        derived = path_at(root, data.get("derived_data_dir", "data/derived"))
        item["inputs"] += [derived / f for f in ("final_master.csv", "input_long.csv", "input_output_long.csv")]
        item["outputs"] = [results / f for f in ("r_models", "r_tables", "r_diagnostics", "r_predictions", "r_plots")]
        if stage == "report":
            tool_path(os.environ.get("QUARTO_PATH") or "quarto")
            template = item["project"] / "R/report.qmd"
            item["inputs"] += [tables, tables / "reporting_bundle.rds", results / "r_plots",
                               results / "r_predictions", results / "r_diagnostics", template]
            # The native helper renders here before moving the result. Guard
            # these transient names too: it can otherwise overwrite old reports.
            item["outputs"] = [results / "final_report"] + [
                directory / ("report." + ext) for directory in (item["project"], item["project"] / "R")
                for ext in ("html", "pdf")]
            item["outputs"] += [directory / "report_files" for directory in (item["project"], item["project"] / "R")]
            expected = [(results / "final_report", "report.html")]
        else:
            expected = [(tables, "reporting_bundle.rds")] if stage in (None, "03_model_tables_and_plots.R") else []
            if stage in (None, "01_input_models.R"):
                expected += [(tables, "input_model_selection.csv")]
            if stage in (None, "02_input_output_models.R"):
                expected += [(tables, "input_output_model_selection.csv")]
            if stage == "00_setup.R":
                raise ValueError("00_setup.R is bootstrap only, not a completed scientific stage")
    else:
        item["inputs"] += [p(key) for key in ("acoustic_features", "final_master", "input_long", "input_output_long",
                             "participant_lookup", "metadata_cleaned")]
        item["outputs"] = [p("data_derived", "data/derived")] + [p("outputs_" + suffix, "outputs/" + suffix)
                    for suffix in ("models", "tables", "diagnostics", "predictions", "plots")]
        tables = p("outputs_tables", "outputs/tables")
        if stage == "report":
            tool_path("pandoc")
            template = item["project"] / "HindiBabyNet_prosody_stat_model_report.Rmd"
            if not template.is_file():
                raise ValueError(f"Missing report template: {template}")
            reports = p("outputs_reports", "outputs/reports")
            item["inputs"] += [tables, p("outputs_models", "outputs/models"),
                               p("outputs_plots", "outputs/plots"), p("outputs_diagnostics", "outputs/diagnostics"), template]
            item["inputs"] += [tables / file for file in ("family1_manifest.csv", "family2_manifest.csv",
                                                          "model_fit_statistics_summary.csv")]
            item["outputs"] = [reports]
            # Explicit HTML, read-only model mode; no scientific YAML changes.
            expression = ('stopifnot(requireNamespace("rmarkdown", quietly=TRUE)); '
                          f'rmarkdown::render({json.dumps(template.as_posix())}, output_format="html_document", '
                          f'output_dir={json.dumps(reports.as_posix())}, intermediates_dir={json.dumps(reports.as_posix())}, '
                          'params=list(run_workflow=FALSE), knit_root_dir=getwd())')
            item["command"] = ["-e", expression]
            expected = [(reports, template.stem + ".html")]
        else:
            if stage == "00_setup.R":
                raise ValueError("00_setup.R is bootstrap only, not a completed scientific stage")
            if stage in (None, "01_prepare_prosody_data.R"):
                expected += [(p("data_derived", "data/derived"), "prosody_long.csv"), (tables, "merge_summary.csv")]
            if stage in (None, "02_fit_family1_models.R"):
                expected += [(tables, "family1_manifest.csv")]
            if stage in (None, "03_fit_family2_models.R"):
                expected += [(tables, "family2_manifest.csv")]
            if stage in (None, "04_create_tables_plots.R"):
                expected += [(tables, "model_fit_statistics_summary.csv")]
            if stage and stage != "01_prepare_prosody_data.R":
                item["outputs"].remove(p("data_derived", "data/derived"))
                item["inputs"].append(p("data_derived", "data/derived") / "prosody_long.csv")
    item["expected"] = expected
    item["outputs"] = list(dict.fromkeys(item["outputs"]))


def fingerprint(paths, exclude=(), ignore_runtime=False):
    """SHA256 every non-audio file; audio uses size+mtime_ns, NOT content proof."""
    result = {}
    for path in sorted(set(paths)):
        files = path.rglob("*") if path.is_dir() else [path]
        for file in files:
            if any(file == skip or skip in file.parents for skip in exclude):
                continue
            if ignore_runtime and any(part in {"__pycache__", ".venv", ".pytest_cache"} for part in file.relative_to(path.parent).parts):
                continue
            if file.is_symlink():
                raise ValueError(f"Symlink identity is unsupported; use a real path: {file}")
            if not file.exists():
                result[str(file)] = None
            elif file.is_file():
                stat = file.stat()
                if file.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aif", ".aiff", ".wma"}:
                    result[str(file)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                else:
                    digest = hashlib.sha256()
                    with file.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    result[str(file)] = {"size": stat.st_size, "sha256": digest.hexdigest()}
    return result


def identity(item):
    project = item["project"]
    code = [project / directory for directory in ("src", "R", "scripts")]
    code += [project / part for part in ("pyproject.toml", "uv.lock", "renv.lock", ".Rprofile", "renv/activate.R")]
    code += list(project.glob("*.Rmd"))
    code += [Path(__file__).resolve()]
    # Hash installed environment metadata without importing scientific packages.
    environment = project / ".venv"
    code += [environment / "pyvenv.cfg", environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")]
    for site in [environment / "Lib/site-packages", *environment.glob("lib/python*/site-packages")]:
        code += [file for info in site.glob("*.dist-info") for file in (info / "RECORD", info / "METADATA")]
    models = []
    if item["name"] == "audio" and item["stage"] in (None, "pipeline", "classify-xgb", "classify-vtc"):
        sc = item["data"].get("speaker_classification", {})
        if item["backend"] == "vtc":
            repo = path_at(item["root"], sc.get("vtc", {}).get("repo_path", "external_models/VTC"))
            # Hash source/weights, not Git internals; unmaterialised LFS weights
            # are only pointer bytes, not a reproducibility guarantee.
            models += [child for child in repo.iterdir() if child.name not in {".git", ".venv", "__pycache__"}]
        else:
            xgb = sc.get("xgb", {})
            model = xgb.get("diarization", {}).get("model", xgb.get("diarization_model", "pyannote/speaker-diarization-3.1"))
            if isinstance(model, str) and path_at(project, model).exists():
                models.append(path_at(project, model))
    env = {k: v for k, v in os.environ.items() if k.startswith("HBN_") and k != "HBN_CONFIG_PATH"}
    env.update({k: os.environ[k] for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                                         "R_LIBS", "R_LIBS_USER", "R_LIBS_SITE", "QUARTO_PATH") if k in os.environ})
    generated = [project / "R" / name for name in ("report.html", "report.pdf", "report_files")]
    files = fingerprint(item["inputs"])
    files.update(fingerprint(code + models, exclude=generated, ignore_runtime=True))
    return {"files": files, "scientific_env": env,
            "command": item["command"], "outputs": [str(p) for p in item["outputs"]]}


def completed_outputs(item):
    if item["name"] == "audio":
        logs = path_at(item["root"], item["data"].get("logs_root", "logs"))
        for log in logs.rglob("*.log"):
            text = log.read_text(encoding="utf-8", errors="replace")
            if re.search(r"\bfail=[1-9]\d*\b|\bFAIL\s*\|", text):
                raise ValueError("Audio reported participant failures; no whole-stage completion")
        if item["stage"] in (None, "pipeline", "classify-xgb", "classify-vtc"):
            root, data = item["root"], item["data"]
            prepared = path_at(root, data["paths"]["prepared_audio_root"])
            sc = data.get("speaker_classification", {})
            classification = path_at(root, sc.get("output_root", data["paths"].get("classification_output_root")))
            options = item["options"]
            if "wav" in options:
                participants = [options.get("participant_id") or Path(options["wav"]).stem]
            elif "recordings_parquet" in options:
                # The native manifest controls membership; no parquet engine is
                # loaded in the launcher. Such stages are never auto-resumed.
                participants = sorted(p.name for p in (classification / item["backend"]).iterdir()
                                      if p.is_dir() and not p.name.startswith("_tmp"))
            else:
                participants = sorted(wav.parent.name for wav in prepared.glob("*/*.wav") if wav.stem == wav.parent.name)
                if options.get("limit") is not None:
                    participants = participants[:options["limit"]]
            if not participants:
                raise ValueError("No native prepared participant audio; no completion")
            for participant in participants:
                directory = classification / item["backend"] / participant
                filenames = ("rttm.csv", "raw_rttm.csv") if item["backend"] == "vtc" else (
                    f"{participant}_segments.parquet", f"{participant}_main_female.wav",
                    f"{participant}_main_male.wav", f"{participant}_summary.json", f"{participant}.TextGrid")
                for filename in filenames:
                    if not (directory / filename).is_file() or not (directory / filename).stat().st_size:
                        raise ValueError("Incomplete native participant classification; no completion")
                if item["backend"] == "vtc" and not all((directory / part).is_dir() for part in ("rttm", "raw_rttm")):
                    raise ValueError("Incomplete native VTC directories; no completion")
    for directory, pattern in item["expected"]:
        matches = [p for p in directory.rglob(pattern) if p.is_file() and p.stat().st_size]
        if not matches:
            raise ValueError(f"{item['name']}: missing/nonempty native output required: {directory / pattern}")
        for path in matches:
            if path.suffix.lower() == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    header = next(csv.reader(handle), [])
                if not header or not any(header):
                    raise ValueError(f"Empty CSV schema: {path}")
    outputs = fingerprint(item["outputs"], item["exclude"])
    if not any(value and value["size"] for value in outputs.values()):
        raise ValueError(f"{item['name']}: no nonempty native outputs")
    return outputs


def run_state(item, config, resume):
    default_stage = {"audio": "pipeline", "prosody": "extract", "vocal-stats": "all"}.get(item["name"])
    key = json.dumps([item["name"], item["stage"] or default_stage, item["extra"], str(item["project"]), str(item["config"])])
    path = config.parent / ".local/runs" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    current = identity(item)
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if (resume and "recordings_parquet" not in item["options"]
                and record.get("ownership") == OWNER and record.get("inputs") == current
                and record.get("outputs") == completed_outputs(item)):
            return path, current, True
    guard_outputs(item)
    return path, current, False


def validate_plan(plan):
    produced = []
    all_outputs = [(item["name"], p) for item in plan for p in item["outputs"]]
    for index, (owner, output) in enumerate(all_outputs):
        for other_owner, other in all_outputs[index + 1:]:
            # Native Python vocal results contain the native R results dirs.
            shared_vocal = {owner, other_owner} == {"vocal-stats", "vocal-models"}
            if owner != other_owner and overlaps(output, other) and not shared_vocal:
                raise ValueError(f"Overlapping workflow output roots: {output} and {other}")
    audio = next((p for p in plan if p["name"] == "audio"), None)
    if audio and str(audio["data"].get("speaker_classification", {}).get("backend", "xgb")).lower() == "xgb":
        for item in plan:
            key = "vtc_output_root" if item["name"] == "vocal-stats" else "rttm_root"
            mapping = item["data"] if key == "vtc_output_root" else item["data"].get("paths", {})
            if item["name"] in {"vocal-stats", "prosody"}:
                consumer = path_at(item["root"], mapping.get(key))
                if any(overlaps(consumer, p) for p in audio["outputs"]):
                    raise ValueError("XGB cannot feed VTC consumers: configure explicit independent existing inputs")
    for item in plan:
        for output in item["outputs"]:
            for other in plan:
                protected = [other["project"] / part for part in
                             ("src", "R", "configs", "config", "scripts", "tests")]
                protected.extend(p for p in other["inputs"] if p == other["config"])
                # Native vocal report uses guarded transient files inside R/;
                # they are known generated files, never source templates.
                if item["name"] == "vocal-models" and item["stage"] == "report":
                    protected = [p for p in protected if p != item["project"] / "R"]
                if any(overlaps(output, p) for p in protected):
                    raise ValueError(f"Output overlaps project code/config: {output}")
        for source in item["inputs"]:
            upstream = any(p == source or p in source.parents for p in produced)
            if not source.exists() and not upstream and source not in item.get("optional", ()):
                raise ValueError(f"{item['name']}: missing input: {source}")
            for output in item["outputs"]:
                if overlaps(source, output):
                    raise ValueError(f"{item['name']}: input/output overlap: {source} and {output}")
        for output in item["outputs"]:
            if output == item["project"] or output in item["project"].parents:
                raise ValueError(f"Refusing project/ancestor output root: {output}")
        produced.extend(item["outputs"])


def guard_outputs(item):
    for output in item["outputs"]:
        excluded = item.get("exclude", ())
        occupied = output.exists() and (not output.is_dir() or any(
            not any(child == skip or skip in child.parents for skip in excluded)
            for child in output.rglob("*")))
        if occupied:
            raise ValueError(f"{item['name']}: refusing existing/non-empty output root: {output}; "
                             "no matching successful ownership record; choose fresh paths. "
                             "Incomplete/per-participant resume is deferred; no overwrite bypass")


def tool_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Tool must be an executable name or full path")
    found = shutil.which(value)
    if found is None:
        raise ValueError(f"Required tool not found: {value}")
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "check", "setup", "all") + ORDER)
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip only complete matching whole stages (default true); never adopt native caches")
    parser.add_argument("--dry-run", action="store_true", help="Read-only plan; no subprocesses")
    parser.add_argument("--stage", help="One native CLI stage; for R use an exact script basename: "
                        + ", ".join(dict.fromkeys(r_scripts("vocal-models") + r_scripts("prosody-stats"))))
    values = list(sys.argv[1:] if argv is None else argv)
    split = values.index("--") if "--" in values else len(values)
    args = parser.parse_args(values[:split])
    extra = values[split + 1:]
    if (args.stage or extra) and args.command not in ORDER:
        parser.error("--stage and pass-through require an individual workflow")
    try:
        config = Path(args.config or os.environ.get("HBN_CONFIG_PATH") or ROOT / "config.local.yaml").resolve()
        if args.command == "configure":
            if args.dry_run:
                print(f"Would exclusively create {config}; native configs unchanged")
                return 0
            configure(config)
            return 0
        settings = read_yaml(config)
        workflows = settings.get("workflows")
        if not isinstance(workflows, dict) or set(workflows) - set(ORDER):
            raise ValueError("workflows must be a mapping of known workflow names")
        for name, value in workflows.items():
            if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
                raise ValueError(f"{name}: enabled must explicitly be true or false")
        names = [name for name in ORDER if workflows.get(name, {}).get("enabled") is True]
        if args.command in ORDER:
            if args.command not in names:
                raise ValueError(f"{args.command}: enable this workflow explicitly in the local config")
            names = [args.command]
        if not names:
            print("No workflows enabled; nothing to do.")
            return 0
        tools = settings.get("tools", {})
        uv = tool_path(tools.get("uv", "uv")) if any(n not in {"vocal-models", "prosody-stats"} for n in names) else None
        rscript = tool_path(tools.get("rscript", "Rscript")) if args.command != "setup" and any(n in {"vocal-models", "prosody-stats"} for n in names) else None
        if "audio" in names and args.command != "setup":
            tool_path("ffmpeg")
        plan = ([{"name": n, "project": path_at(config.parent, workflows[n].get("project", PROJECTS[n]))} for n in names]
            if args.command == "setup" else
            [workflow(n, workflows[n], config.parent, args.stage, extra, config) for n in names])
        commands = []
        for item in plan:
            project = item["project"]
            if not project.is_dir():
                raise ValueError(f"Missing project directory: {project}")
            is_r = item["name"] in {"vocal-models", "prosody-stats"}
            if not is_r:
                for filename in ("pyproject.toml", "uv.lock"):
                    if not (project / filename).is_file():
                        raise ValueError(f"{item['name']}: missing {filename}; frozen setup cannot regenerate it")
                if args.command != "setup" and not (project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")).is_file():
                    raise ValueError(f"{item['name']}: missing local environment; run setup explicitly")
            if args.command == "setup":
                if is_r:
                    print(f"{item['name']}: R packages must be installed separately; no automatic installation")
                    continue
                if any(cwd == project for _, cwd, _ in commands):
                    continue
                command = [uv, "sync", "--frozen"]
            else:
                command = ([rscript] if is_r else [uv, "run", "--frozen", "--no-sync"]) + item["command"]
            commands.append((command, project, item))
        if args.command != "setup":
            validate_plan(plan)
        print(
            "Resume verifies whole-stage local records only. Audio identity is size+mtime_ns, not SHA256. "
            "Remote HuggingFace revisions/weights and installed R packages are not pinned or fully verified. "
            "No participant-level recovery, sandbox, or cross-config concurrency guarantee."
        )
        for command, project, item in commands:
            print(f"[{item['name']}] cwd={project}\n  {subprocess.list2cmdline(command)}")
            if args.command != "setup":
                for output in item["outputs"]:
                    print(f"  output guard: {output}")
        if args.command == "check" or args.dry_run:
            # Report occupied roots, but never execute or mutate anything.
            for item in (plan if args.command != "setup" else []):
                try:
                    _, _, skip = run_state(item, config, args.resume)
                    if skip:
                        print(f"[{item['name']}] verified complete: would skip")
                except ValueError as exc:
                    print(f"BLOCKED for execution: {exc}")
            return 0
        for command, project, item in commands:
            state_path, before, skip = (None, None, False)
            if args.command != "setup":
                state_path, before, skip = run_state(item, config, args.resume)
                if skip:
                    print(f"[{item['name']}] skipped verified complete stage")
                    continue
            env = os.environ.copy()
            # uv must not reuse an activated root/foreign environment.
            env.pop("VIRTUAL_ENV", None)
            env.pop("PYTHONPATH", None)
            for key in ("PYTHONHOME", "UV_PROJECT", "UV_WORKING_DIRECTORY", "UV_ACTIVE", "UV_NO_PROJECT",
                        "UV_PROJECT_ENVIRONMENT", "UV_NO_SYNC", "UV_FROZEN"):
                env.pop(key, None)
            # Use cwd's default .venv, without leaking a pinned env into the
            # audio backend's nested uv process in the external VTC checkout.
            if args.command != "setup":
                env["UV_NO_SYNC"] = "1"
                env["UV_FROZEN"] = "1"
                if item["config"]:
                    env["HBN_CONFIG_PATH"] = str(item["config"])
                else:
                    env.pop("HBN_CONFIG_PATH", None)
                if item["name"] == "vocal-models" and args.stage == "report":
                    if os.environ.get("HBN_REPORT_FORMAT", "html") != "html":
                        raise ValueError("Report stage supports explicit HTML only; PDF toolchain not checked")
                    env["HBN_REPORT_FORMAT"] = "html"
            if state_path is None:
                subprocess.run(command, cwd=project, env=env, check=True)
                continue
            state_path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive per-config-parent execution lock. A crash leaves it in
            # place for manual inspection; never silently adopt partial writes.
            lock = state_path.parent / "active.lock"
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(str(state_path))
            try:
                guard_outputs(item)
                subprocess.run(command, cwd=project, env=env, check=True)
                outputs = completed_outputs(item)
                if identity(item) != before:
                    raise ValueError("Inputs/code/models changed during execution; no completion recorded")
                record = {"ownership": OWNER, "inputs": before, "outputs": outputs}
                temporary = state_path.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(record, handle, sort_keys=True)
                temporary.replace(state_path)
                print(f"[{item['name']}] completed and recorded")
            finally:
                lock.unlink()
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Workflow failed (exit {exc.returncode}); remaining workflows were not started.", file=sys.stderr)
        return exc.returncode if exc.returncode > 0 else 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError) as exc:
        print(f"Launcher error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())