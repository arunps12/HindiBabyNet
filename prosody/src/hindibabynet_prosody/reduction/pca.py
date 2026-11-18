"""Separate PCA reduction for participant-speaker SSL embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hindibabynet_prosody.config import PCAConfig
from hindibabynet_prosody.io.outputs import write_csv
from hindibabynet_prosody.utils.paths import ensure_directory


@dataclass(frozen=True)
class PCAArtifacts:
    projected: pd.DataFrame
    explained_variance: pd.DataFrame


def fit_embedding_pca(
    *,
    embeddings: pd.DataFrame,
    id_columns: list[str],
    value_columns: list[str],
    pca_config: PCAConfig,
    component_prefix: str,
    artifact_dir: Path | None = None,
) -> PCAArtifacts:
    usable = embeddings.dropna(subset=value_columns).reset_index(drop=True)
    if len(usable) < pca_config.n_components:
        raise ValueError(
            f"Need at least {pca_config.n_components} valid rows for PCA, found {len(usable)}"
        )

    matrix = usable.loc[:, value_columns].to_numpy(dtype=np.float64)
    scaler = StandardScaler() if pca_config.standardize else None
    transformed = scaler.fit_transform(matrix) if scaler is not None else matrix

    pca = PCA(n_components=pca_config.n_components, random_state=pca_config.random_seed)
    scores = pca.fit_transform(transformed)

    projected = usable.loc[:, id_columns].copy()
    for index in range(pca_config.n_components):
        projected[f"{component_prefix}_PC{index + 1}"] = scores[:, index]

    explained_variance = pd.DataFrame(
        {
            "component": [index + 1 for index in range(pca_config.n_components)],
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    if artifact_dir is not None:
        ensure_directory(artifact_dir)
        if scaler is not None:
            joblib.dump(scaler, artifact_dir / f"{component_prefix}_scaler.joblib")
        joblib.dump(pca, artifact_dir / f"{component_prefix}_pca.joblib")
        write_csv(explained_variance, artifact_dir / f"{component_prefix}_explained_variance.csv")

    return PCAArtifacts(projected=projected, explained_variance=explained_variance)
