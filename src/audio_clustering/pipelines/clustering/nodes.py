"""Audio Clustering — Clustering Pipeline Nodes.

Fits KMeans, GMM, HDBSCAN, and Agglomerative clustering models.
Performs model selection via Silhouette and BIC criteria.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

_META_COLS = {"filename", "filepath", "duration_sec", "tag", "environment", "stem"}


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feat_cols = [c for c in df.columns if c not in _META_COLS]
    return df[feat_cols].values.astype(np.float64)


def fit_kmeans(
    pca_embeddings: pd.DataFrame,
    k_range: list[int] = list(range(2, 16)),
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans]:
    """Fit KMeans across a range of k, select best by Silhouette score.

    Args:
        pca_embeddings: PCA-reduced feature matrix.
        k_range: List of k values to evaluate.
        random_state: Random seed.

    Returns:
        Tuple of (cluster_labels DataFrame, best KMeans model).
    """
    X = _feature_matrix(pca_embeddings)
    if len(X) < 3:
        raise ValueError("KMeans model selection requires at least 3 samples")
    valid_k = sorted({k for k in k_range if 2 <= k < len(X)})
    if not valid_k:
        raise ValueError("k_range contains no value in [2, n_samples)")
    best_k, best_score, best_model = 2, -1.0, None
    results = []

    for k in valid_k:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        inertia = km.inertia_
        results.append({"k": k, "silhouette": sil, "inertia": inertia})
        logger.info("KMeans k=%d | Silhouette=%.4f | Inertia=%.2f", k, sil, inertia)
        if sil > best_score:
            best_k, best_score, best_model = k, sil, km

    logger.info("Best KMeans: k=%d (Silhouette=%.4f)", best_k, best_score)
    df_results = pd.DataFrame(results)
    from pathlib import Path
    Path("data/07_reporting").mkdir(parents=True, exist_ok=True)
    df_results.to_csv("data/07_reporting/kmeans_selection.csv", index=False)

    labels_df = pca_embeddings[[c for c in _META_COLS if c in pca_embeddings.columns]].copy()
    labels_df["kmeans_label"] = best_model.predict(X)
    return labels_df, best_model


def fit_gmm(
    pca_embeddings: pd.DataFrame,
    n_components_range: list[int] = list(range(2, 9)),
    covariance_type: str = "full",
    random_state: int = 42,
) -> tuple[pd.DataFrame, GaussianMixture]:
    """Fit Gaussian Mixture Models, select best by BIC.

    Args:
        pca_embeddings: PCA-reduced feature matrix.
        n_components_range: Range of Gaussian components to test.
        covariance_type: GMM covariance type.
        random_state: Random seed.

    Returns:
        Tuple of (labels DataFrame with gmm_label and gmm_prob columns, best GMM model).
    """
    X = _feature_matrix(pca_embeddings)
    valid_components = sorted({n for n in n_components_range if 1 < n <= len(X)})
    if not valid_components:
        raise ValueError("n_components_range contains no feasible value")
    best_bic, best_model = np.inf, None

    for n in valid_components:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type=covariance_type,
            random_state=random_state,
            max_iter=200,
            n_init=3,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        logger.info("GMM n=%d | BIC=%.2f | AIC=%.2f", n, bic, aic)
        if bic < best_bic:
            best_bic, best_model = bic, gmm

    logger.info("Best GMM: n_components=%d (BIC=%.2f)", best_model.n_components, best_bic)

    labels_df = pca_embeddings[[c for c in _META_COLS if c in pca_embeddings.columns]].copy()
    labels_df["gmm_label"] = best_model.predict(X)
    probs = best_model.predict_proba(X)
    labels_df["gmm_max_prob"] = probs.max(axis=1)
    return labels_df, best_model


def fit_hdbscan(
    pca_embeddings: pd.DataFrame,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> tuple[pd.DataFrame, object]:
    """Fit HDBSCAN — handles noise and variable-density clusters.

    Label -1 indicates noise points not assigned to any cluster.

    Args:
        pca_embeddings: PCA-reduced feature matrix.
        min_cluster_size: Minimum cluster size.
        min_samples: Determines how conservative clustering is.
        metric: Distance metric.

    Returns:
        Tuple of (labels DataFrame, fitted HDBSCAN object).
    """
    import hdbscan  # type: ignore

    X = _feature_matrix(pca_embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).mean()

    logger.info("HDBSCAN: %d clusters | %.1f%% noise points", n_clusters, noise_pct)

    if n_clusters >= 2:
        valid = labels != -1
        sil = silhouette_score(X[valid], labels[valid]) if valid.sum() > 1 else float("nan")
        logger.info("HDBSCAN Silhouette (excl. noise): %.4f", sil)

    labels_df = pca_embeddings[[c for c in _META_COLS if c in pca_embeddings.columns]].copy()
    labels_df["hdbscan_label"] = labels
    labels_df["hdbscan_prob"] = clusterer.probabilities_
    return labels_df, clusterer


def merge_cluster_labels(
    kmeans_labels: pd.DataFrame,
    gmm_labels: pd.DataFrame,
    hdbscan_labels: pd.DataFrame,
    umap_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all cluster label columns into a single DataFrame.

    Args:
        kmeans_labels: Output of fit_kmeans.
        gmm_labels: Output of fit_gmm.
        hdbscan_labels: Output of fit_hdbscan.
        umap_embeddings: UMAP 2D coordinates for plotting.

    Returns:
        Unified DataFrame with all cluster labels and UMAP coordinates.
    """
    meta_cols = [c for c in kmeans_labels.columns if c in _META_COLS]
    result = kmeans_labels[meta_cols].copy().reset_index(drop=True)

    result["kmeans_label"] = kmeans_labels["kmeans_label"].values
    result["gmm_label"] = gmm_labels["gmm_label"].values
    result["gmm_max_prob"] = gmm_labels["gmm_max_prob"].values
    result["hdbscan_label"] = hdbscan_labels["hdbscan_label"].values
    result["hdbscan_prob"] = hdbscan_labels["hdbscan_prob"].values

    umap_cols = [c for c in umap_embeddings.columns if c.startswith("umap_")]
    for col in umap_cols:
        result[col] = umap_embeddings[col].values

    logger.info("Merged cluster labels: %s", result.shape)
    return result
