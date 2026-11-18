from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hindibabynet_prosody.config import load_config
from hindibabynet_prosody.reduction.pca import fit_embedding_pca


def test_fit_embedding_pca_projects_three_components(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "config.example.yaml")
    frame = pd.DataFrame(
        {
            "participant_id": ["P001", "P002", "P003", "P004"],
            "speaker": ["adult_female", "adult_female", "adult_male", "other_child"],
            "d1": [0.1, 0.2, 0.3, 0.4],
            "d2": [1.0, 1.1, 1.2, 1.3],
            "d3": [2.0, 2.2, 2.4, 2.6],
        }
    )

    artifacts = fit_embedding_pca(
        embeddings=frame,
        id_columns=["participant_id", "speaker"],
        value_columns=["d1", "d2", "d3"],
        pca_config=config.pca,
        component_prefix="wav2vec",
        artifact_dir=tmp_path,
    )

    assert artifacts.projected.columns.tolist() == [
        "participant_id",
        "speaker",
        "wav2vec_PC1",
        "wav2vec_PC2",
        "wav2vec_PC3",
    ]
    assert len(artifacts.explained_variance) == 3
    assert (tmp_path / "wav2vec_pca.joblib").exists()


def test_fit_embedding_pca_requires_enough_rows() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / "config" / "config.example.yaml")
    frame = pd.DataFrame(
        {
            "participant_id": ["P001", "P002"],
            "speaker": ["adult_female", "adult_male"],
            "d1": [0.1, 0.2],
            "d2": [1.0, 1.1],
            "d3": [2.0, 2.1],
        }
    )

    with pytest.raises(ValueError, match="Need at least"):
        fit_embedding_pca(
            embeddings=frame,
            id_columns=["participant_id", "speaker"],
            value_columns=["d1", "d2", "d3"],
            pca_config=config.pca,
            component_prefix="hubert",
        )
