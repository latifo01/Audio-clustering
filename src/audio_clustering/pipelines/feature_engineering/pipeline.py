"""Feature Engineering Pipeline for Audio Clustering."""
from kedro.pipeline import Pipeline, node, pipeline

from audio_clustering.pipelines.feature_engineering.nodes import extract_features


def create_feature_engineering_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=extract_features,
            inputs={
                "manifest": "audio_files_manifest",
                "n_mfcc": "params:feature_engineering.n_mfcc",
                "n_fft": "params:feature_engineering.n_fft",
                "hop_length": "params:feature_engineering.hop_length",
                "sample_rate": "params:feature_engineering.sample_rate",
                "include_delta_mfcc": "params:feature_engineering.include_delta_mfcc",
                "include_delta2_mfcc": "params:feature_engineering.include_delta2_mfcc",
            },
            outputs="raw_features",
            name="extract_audio_features",
            tags=["feature_engineering"],
        ),
    ])
