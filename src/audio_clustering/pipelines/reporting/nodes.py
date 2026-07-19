"""Audio Clustering — Reporting Pipeline Nodes.

Generates: cluster signature profiles, UMAP visualization, cluster summary report.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_META_COLS = {"filename", "filepath", "duration_sec", "tag", "environment", "stem",
              "kmeans_label", "gmm_label", "gmm_max_prob", "hdbscan_label", "hdbscan_prob",
              "umap_0", "umap_1"}


def build_cluster_profiles(
    cluster_labels: pd.DataFrame,
    scaled_features: pd.DataFrame,
    label_col: str = "kmeans_label",
) -> pd.DataFrame:
    """Compute mean feature profile per cluster (cluster 'fingerprint').

    Groups by cluster label and computes mean of each feature dimension.
    Only uses the top features by inter-cluster variance for readability.

    Args:
        cluster_labels: Output of merge_cluster_labels.
        scaled_features: Scaled feature matrix (for interpretable values).
        label_col: Which cluster label column to use.

    Returns:
        DataFrame of shape (n_clusters, n_features) with cluster mean profiles.
    """
    feat_cols = [c for c in scaled_features.columns
                 if c not in _META_COLS and not c.startswith("umap_") and not c.startswith("pc_")]

    combined = scaled_features[feat_cols].copy()
    combined[label_col] = cluster_labels[label_col].values

    profiles = combined.groupby(label_col)[feat_cols].mean()

    # Compute inter-cluster variance to rank most discriminative features
    inter_var = profiles.var(axis=0).sort_values(ascending=False)
    top_features = inter_var.head(20).index.tolist()

    logger.info("Cluster profiles computed for %d clusters | Top discriminative features: %s",
                len(profiles), top_features[:5])
    return profiles[top_features]


def plot_umap_clusters(cluster_labels: pd.DataFrame) -> plt.Figure:
    """Generate UMAP scatter plot colored by cluster assignment.

    Args:
        cluster_labels: Must contain umap_0, umap_1, kmeans_label, and optionally 'tag'.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("UMAP 2D Embedding — Audio Cluster Visualization", fontsize=14, fontweight="bold")

    cmap = plt.cm.get_cmap("tab20", cluster_labels["kmeans_label"].nunique())

    # ── Plot 1: colored by KMeans cluster
    ax = axes[0]
    for k, grp in cluster_labels.groupby("kmeans_label"):
        ax.scatter(grp["umap_0"], grp["umap_1"], s=8, alpha=0.6, c=[cmap(k)], label=f"C{k}")
    ax.set_title("KMeans Clusters", fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, loc="best", fontsize=7)

    # ── Plot 2: colored by ground-truth tag (if available)
    ax2 = axes[1]
    if cluster_labels["tag"].notna().any():
        tags = cluster_labels["tag"].fillna("unknown")
        unique_tags = tags.unique()
        tag_cmap = plt.cm.get_cmap("Set3", len(unique_tags))
        tag_to_color = {t: tag_cmap(i) for i, t in enumerate(unique_tags)}
        for tag, grp in cluster_labels.groupby(tags):
            ax2.scatter(grp["umap_0"], grp["umap_1"], s=8, alpha=0.6,
                        c=[tag_to_color[tag]], label=tag)
        ax2.set_title("Ground-Truth Labels", fontweight="bold")
    else:
        hdb_labels = cluster_labels["hdbscan_label"]
        unique_k = sorted(hdb_labels.unique())
        hdb_cmap = plt.cm.get_cmap("tab20", len(unique_k))
        for k in unique_k:
            grp = cluster_labels[hdb_labels == k]
            color = "grey" if k == -1 else hdb_cmap(unique_k.index(k))
            label_str = "Noise" if k == -1 else f"C{k}"
            ax2.scatter(grp["umap_0"], grp["umap_1"], s=8, alpha=0.5, c=[color], label=label_str)
        ax2.set_title("HDBSCAN Clusters", fontweight="bold")

    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.legend(markerscale=3, loc="best", fontsize=7)

    plt.tight_layout()
    return fig
