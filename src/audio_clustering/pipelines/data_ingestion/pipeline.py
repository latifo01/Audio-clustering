"""Data Ingestion Pipeline for Audio Clustering."""
from kedro.pipeline import Pipeline, node, pipeline

from audio_clustering.pipelines.data_ingestion.nodes import build_audio_manifest, validate_manifest


def create_data_ingestion_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=build_audio_manifest,
            inputs={
                "audio_dir": "params:feature_engineering.audio_dir",
                "min_duration_sec": "params:feature_engineering.min_duration_sec",
            },
            outputs="raw_manifest",
            name="build_audio_manifest",
            tags=["data_ingestion"],
        ),
        node(
            func=validate_manifest,
            inputs="raw_manifest",
            outputs="audio_files_manifest",
            name="validate_manifest",
            tags=["data_ingestion"],
        ),
    ])
