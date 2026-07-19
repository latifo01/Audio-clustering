"""Audio Clustering — Full Kedro Pipeline Registry."""
from kedro.pipeline import Pipeline, pipeline

from audio_clustering.pipelines.data_ingestion.pipeline import create_data_ingestion_pipeline
from audio_clustering.pipelines.feature_engineering.pipeline import create_feature_engineering_pipeline
from audio_clustering.pipelines.preprocessing.pipeline import create_preprocessing_pipeline
from audio_clustering.pipelines.clustering.pipeline import create_clustering_pipeline
from audio_clustering.pipelines.evaluation.pipeline import create_evaluation_pipeline
from audio_clustering.pipelines.reporting.pipeline import create_reporting_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    data_ingestion = create_data_ingestion_pipeline()
    feature_engineering = create_feature_engineering_pipeline()
    preprocessing = create_preprocessing_pipeline()
    clustering = create_clustering_pipeline()
    evaluation = create_evaluation_pipeline()
    reporting = create_reporting_pipeline()

    full_pipeline = (
        data_ingestion
        + feature_engineering
        + preprocessing
        + clustering
        + evaluation
        + reporting
    )

    return {
        "__default__": full_pipeline,
        "data_ingestion": data_ingestion,
        "feature_engineering": feature_engineering,
        "preprocessing": preprocessing,
        "clustering": clustering,
        "evaluation": evaluation,
        "reporting": reporting,
        # Convenience shortcut: run everything except reporting
        "train": data_ingestion + feature_engineering + preprocessing + clustering + evaluation,
    }
