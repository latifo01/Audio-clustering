"""Reporting Pipeline for Audio Clustering."""
from kedro.pipeline import Pipeline, node, pipeline

from audio_clustering.pipelines.reporting.nodes import build_cluster_profiles, plot_umap_clusters


def create_reporting_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_cluster_profiles,
            inputs=["cluster_labels", "scaled_features"],
            outputs="cluster_profiles",
            name="build_cluster_profiles",
            tags=["reporting"],
        ),
        node(
            func=plot_umap_clusters,
            inputs="cluster_labels",
            outputs="umap_plot",
            name="plot_umap_clusters",
            tags=["reporting"],
        ),
    ])
