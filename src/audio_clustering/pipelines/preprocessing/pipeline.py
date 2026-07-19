"""Preprocessing Pipeline for Audio Clustering."""
from kedro.pipeline import Pipeline, node, pipeline

from audio_clustering.pipelines.preprocessing.nodes import embed_umap, reduce_pca, scale_features


def create_preprocessing_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=scale_features,
            inputs="raw_features",
            outputs=["scaled_features", "scaler_model"],
            name="scale_audio_features",
            tags=["preprocessing"],
        ),
        node(
            func=reduce_pca,
            inputs={
                "scaled_features": "scaled_features",
                "n_components": "params:preprocessing.pca_n_components",
                "random_state": "params:preprocessing.random_state",
            },
            outputs=["pca_embeddings", "pca_model"],
            name="reduce_pca",
            tags=["preprocessing"],
        ),
        node(
            func=embed_umap,
            inputs={
                "pca_embeddings": "pca_embeddings",
                "n_components": "params:preprocessing.umap_n_components",
                "n_neighbors": "params:preprocessing.umap_n_neighbors",
                "min_dist": "params:preprocessing.umap_min_dist",
                "random_state": "params:preprocessing.random_state",
            },
            outputs=["umap_embeddings", "umap_model"],
            name="embed_umap",
            tags=["preprocessing"],
        ),
    ])
