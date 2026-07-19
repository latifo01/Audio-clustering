"""Audio Clustering — Evaluation Pipeline Nodes.

Computes: Silhouette, Calinski-Harabasz, Davies-Bouldin,
Adjusted Rand Index (against ground-truth tags),
and cluster stability via bootstrapped Silhouette distributions.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)

_META_COLS = {"filename", "filepath", "duration_sec", "tag", "environment", "stem",
              "kmeans_label", "gmm_label", "gmm_max_prob", "hdbscan_label", "hdbscan_prob"}


def compute_evaluation_metrics(
    cluster_labels: pd.DataFrame,
    pca_embeddings: pd.DataFrame,
) -> pd.DataFrame:
    """Compute internal and external clustering metrics for each algorithm.

    Internal metrics (no ground truth needed):
    - Silhouette score (higher = better, -1 to 1)
    - Calinski-Harabasz index (higher = better)
    - Davies-Bouldin index (lower = better)

    External metrics (require ground-truth tags):
    - Adjusted Rand Index (higher = better, -1 to 1)

    Args:
        cluster_labels: Output of merge_cluster_labels.
        pca_embeddings: PCA feature matrix for distance computation.

    Returns:
        DataFrame with one row per algorithm.
    """
    feat_cols = [c for c in pca_embeddings.columns if c not in _META_COLS and not c.startswith("umap_")]
    X = pca_embeddings[feat_cols].values.astype(np.float64)

    algorithms = {
        "KMeans": cluster_labels["kmeans_label"].values,
        "GMM": cluster_labels["gmm_label"].values,
        "HDBSCAN": cluster_labels["hdbscan_label"].values,
    }

    labeled_mask = cluster_labels["tag"].notna().to_numpy()
    has_ground_truth = bool(labeled_mask.any())
    if has_ground_truth:
        logger.info("Ground-truth ARI computation on %d labeled samples", cluster_labels["tag"].notna().sum())

    records = []
    for algo_name, labels in algorithms.items():
        valid_mask = labels != -1  # Exclude HDBSCAN noise
        X_valid, labels_valid = X[valid_mask], labels[valid_mask]
        n_clusters = len(set(labels_valid))

        if n_clusters < 2 or len(X_valid) < 2:
            logger.warning("%s: too few clusters/samples for metrics.", algo_name)
            continue

        row: dict[str, Any] = {"algorithm": algo_name, "n_clusters": n_clusters}
        row["silhouette"] = round(silhouette_score(X_valid, labels_valid, sample_size=min(5000, len(X_valid))), 4)
        row["calinski_harabasz"] = round(calinski_harabasz_score(X_valid, labels_valid), 2)
        row["davies_bouldin"] = round(davies_bouldin_score(X_valid, labels_valid), 4)
        row["noise_pct"] = round(100 * (~valid_mask).mean(), 2)

        if has_ground_truth:
            ari_mask = valid_mask & labeled_mask
            if ari_mask.sum() >= 2 and len(set(cluster_labels.loc[ari_mask, "tag"])) >= 2:
                row["adjusted_rand_index"] = round(adjusted_rand_score(
                    cluster_labels.loc[ari_mask, "tag"], labels[ari_mask]
                ), 4)
                row["ari_labeled_samples"] = int(ari_mask.sum())
            else:
                row["adjusted_rand_index"] = None
        else:
            row["adjusted_rand_index"] = None

        records.append(row)
        logger.info(
            "%s | n_clusters=%d | Silhouette=%.4f | CH=%.1f | DB=%.4f | ARI=%s",
            algo_name, n_clusters, row["silhouette"], row["calinski_harabasz"],
            row["davies_bouldin"], row.get("adjusted_rand_index", "N/A"),
        )

    return pd.DataFrame(records)


def compute_cluster_stability(
    pca_embeddings: pd.DataFrame,
    best_k: int = 6,
    n_bootstrap: int = 50,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate cluster stability via bootstrapped Silhouette distributions.

    Repeatedly subsample the data and refit KMeans, measuring how consistent
    the Silhouette score is across bootstrap iterations.

    Args:
        pca_embeddings: PCA feature matrix.
        best_k: Number of clusters to evaluate stability for.
        n_bootstrap: Number of bootstrap iterations.
        subsample_ratio: Fraction of data to subsample per iteration.
        random_state: Base random seed.

    Returns:
        DataFrame with stability statistics per k value tested.
    """
    from sklearn.cluster import KMeans

    feat_cols = [c for c in pca_embeddings.columns if c not in _META_COLS and not c.startswith("umap_")]
    X = pca_embeddings[feat_cols].values.astype(np.float64)
    n = len(X)
    if n <= best_k:
        raise ValueError("cluster stability requires n_samples > best_k")
    if n_bootstrap < 1 or not 0 < subsample_ratio <= 1:
        raise ValueError("n_bootstrap must be positive and subsample_ratio in (0, 1]")
    n_sub = min(n, max(best_k + 1, int(n * subsample_ratio)))

    rng = np.random.default_rng(random_state)
    sil_scores = []

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n_sub, replace=False)
        X_sub = X[idx]
        km = KMeans(n_clusters=best_k, random_state=random_state + i, n_init=5)
        labels = km.fit_predict(X_sub)
        if len(set(labels)) >= 2:
            sil = silhouette_score(X_sub, labels, sample_size=min(3000, n_sub))
            sil_scores.append(sil)

    sil_arr = np.array(sil_scores)
    if sil_arr.size == 0:
        raise ValueError("no valid bootstrap sample produced at least two clusters")
    result = pd.DataFrame([{
        "k": best_k,
        "n_bootstrap": n_bootstrap,
        "silhouette_mean": round(float(sil_arr.mean()), 4),
        "silhouette_std": round(float(sil_arr.std()), 4),
        "silhouette_p5": round(float(np.percentile(sil_arr, 5)), 4),
        "silhouette_p95": round(float(np.percentile(sil_arr, 95)), 4),
        "stability_cv": round(float(sil_arr.std() / sil_arr.mean()), 4) if sil_arr.mean() != 0 else None,
    }])
    logger.info("Stability (k=%d): mean=%.4f ± %.4f | 90%% CI=[%.4f, %.4f]",
                best_k, sil_arr.mean(), sil_arr.std(),
                np.percentile(sil_arr, 5), np.percentile(sil_arr, 95))
    return result
