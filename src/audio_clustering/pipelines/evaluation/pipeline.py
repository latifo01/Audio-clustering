"""Evaluation Pipeline for Audio Clustering."""
from kedro.pipeline import Pipeline, node, pipeline

from audio_clustering.pipelines.evaluation.nodes import compute_cluster_stability, compute_evaluation_metrics


def create_evaluation_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_evaluation_metrics,
            inputs=["cluster_labels", "pca_embeddings"],
            outputs="evaluation_metrics",
            name="compute_evaluation_metrics",
            tags=["evaluation"],
        ),
        node(
            func=compute_cluster_stability,
            inputs={
                "pca_embeddings": "pca_embeddings",
                "n_bootstrap": "params:evaluation.stability_n_bootstrap",
                "subsample_ratio": "params:evaluation.stability_subsample_ratio",
                "random_state": "params:evaluation.random_state",
            },
            outputs="stability_report",
            name="compute_cluster_stability",
            tags=["evaluation"],
        ),
    ])
