"""Clustering Pipeline for Audio Clustering."""
from kedro.pipeline import Pipeline, node, pipeline

from audio_clustering.pipelines.clustering.nodes import (
    fit_gmm,
    fit_hdbscan,
    fit_kmeans,
    merge_cluster_labels,
)


def create_clustering_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fit_kmeans,
            inputs={
                "pca_embeddings": "pca_embeddings",
                "k_range": "params:clustering.kmeans.k_range",
                "random_state": "params:clustering.random_state",
            },
            outputs=["kmeans_labels", "kmeans_model"],
            name="fit_kmeans",
            tags=["clustering"],
        ),
        node(
            func=fit_gmm,
            inputs={
                "pca_embeddings": "pca_embeddings",
                "n_components_range": "params:clustering.gmm.n_components_range",
                "covariance_type": "params:clustering.gmm.covariance_type",
                "random_state": "params:clustering.random_state",
            },
            outputs=["gmm_labels", "gmm_model"],
            name="fit_gmm",
            tags=["clustering"],
        ),
        node(
            func=fit_hdbscan,
            inputs={
                "pca_embeddings": "pca_embeddings",
                "min_cluster_size": "params:clustering.hdbscan.min_cluster_size",
                "min_samples": "params:clustering.hdbscan.min_samples",
                "metric": "params:clustering.hdbscan.metric",
            },
            outputs=["hdbscan_labels", "hdbscan_model"],
            name="fit_hdbscan",
            tags=["clustering"],
        ),
        node(
            func=merge_cluster_labels,
            inputs=["kmeans_labels", "gmm_labels", "hdbscan_labels", "umap_embeddings"],
            outputs="cluster_labels",
            name="merge_cluster_labels",
            tags=["clustering"],
        ),
    ])
