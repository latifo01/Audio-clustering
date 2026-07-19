"""Audio Clustering — Preprocessing Pipeline Nodes.

Handles scaling, PCA dimensionality reduction, and UMAP embedding.
Saves fitted transformers to the catalog for inference reuse.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns that are metadata, not features
_META_COLS = {"filename", "filepath", "duration_sec", "tag", "environment", "stem"}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def scale_features(raw_features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize features to zero mean and unit variance.

    Args:
        raw_features: Output of feature extraction.

    Returns:
        Tuple of (scaled_features DataFrame, fitted StandardScaler).
    """
    feat_cols = _get_feature_cols(raw_features)
    meta = raw_features[[c for c in _META_COLS if c in raw_features.columns]].copy()

    X = raw_features[feat_cols].values.astype(np.float64)

    # Handle any remaining NaN/Inf (edge-case in feature extraction)
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j]
        logger.warning("Imputed %d non-finite values with column means", nan_mask.sum())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=feat_cols, index=raw_features.index)
    df_scaled = pd.concat([meta.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    logger.info("Scaled feature matrix: %s", df_scaled.shape)
    return df_scaled, scaler


def reduce_pca(
    scaled_features: pd.DataFrame,
    n_components: float = 0.95,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA]:
    """Apply PCA, retaining enough components to explain `n_components` variance.

    Args:
        scaled_features: Output of scale_features.
        n_components: Float = variance threshold (e.g. 0.95) or int = exact components.
        random_state: Random seed.

    Returns:
        Tuple of (PCA-reduced DataFrame, fitted PCA object).
    """
    feat_cols = _get_feature_cols(scaled_features)
    meta = scaled_features[[c for c in _META_COLS if c in scaled_features.columns]].copy()

    X = scaled_features[feat_cols].values

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    logger.info(
        "PCA: %d → %d components | explained variance: %.2f%%",
        X.shape[1],
        pca.n_components_,
        100 * pca.explained_variance_ratio_.sum(),
    )

    pca_cols = [f"pc_{i:02d}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca = pd.concat([meta.reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)

    return df_pca, pca


def embed_umap(
    pca_embeddings: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, object]:
    """Apply UMAP for 2D visualization and optional clustering.

    Args:
        pca_embeddings: Output of reduce_pca.
        n_components: Target embedding dimensionality (2 for visualization).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed.

    Returns:
        Tuple of (UMAP embedding DataFrame, fitted UMAP reducer).
    """
    import umap  # type: ignore

    feat_cols = [c for c in pca_embeddings.columns if c.startswith("pc_")]
    meta = pca_embeddings[[c for c in _META_COLS if c in pca_embeddings.columns]].copy()

    X = pca_embeddings[feat_cols].values

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False,
    )
    X_umap = reducer.fit_transform(X)

    logger.info("UMAP embedding: %s → %s", X.shape, X_umap.shape)

    umap_cols = [f"umap_{i}" for i in range(n_components)]
    df_umap = pd.DataFrame(X_umap, columns=umap_cols)
    df_umap = pd.concat([meta.reset_index(drop=True), df_umap.reset_index(drop=True)], axis=1)

    return df_umap, reducer
