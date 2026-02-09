import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "hindibabynet"

list_of_files = [
    # GitHub
    ".github/workflows/.gitkeep",

    # Source package (src-layout)
    f"src/{project_name}/__init__.py",

    f"src/{project_name}/constant/__init__.py",
    f"src/{project_name}/constant/constants.py",

    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifact_entity.py",

    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/logging/logger.py",

    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/exception/exception.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/io_utils.py",
    f"src/{project_name}/utils/audio_utils.py",

    # Stage 01 (Data Ingestion)
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",

    # Configs and params files
    "configs/config.yaml",
    "configs/params.yaml",

    # Artifacts root folder
    "artifacts/runs/.gitkeep",

    # Data placeholders
    "data/raw/.gitkeep",
    "data/external/.gitkeep",

    # Scripts / tests
    "scripts/run_stage_01.sh",
    "tests/test_smoke.py",

    # Notebooks
    "notebooks/00_research.ipynb",

    #models
    "models/.gitkeep"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if not filepath.exists():
        filepath.touch()
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
