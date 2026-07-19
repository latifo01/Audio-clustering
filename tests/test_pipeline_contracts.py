import io

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import librosa

from audio_clustering.api.main import _extract_single
from audio_clustering.pipelines.clustering.nodes import fit_kmeans
from audio_clustering.pipelines.evaluation.nodes import compute_cluster_stability, compute_evaluation_metrics
from audio_clustering.pipelines.feature_engineering.nodes import extract_signal_features


def _tone(seconds=1.0, sample_rate=22050):
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    return 0.2 * np.sin(2 * np.pi * 440 * time), sample_rate


def test_training_and_api_share_exact_feature_schema():
    signal, sample_rate = _tone()
    payload = io.BytesIO()
    sf.write(payload, signal, sample_rate, format="WAV", subtype="FLOAT")
    audio_bytes = payload.getvalue()
    decoded, decoded_rate = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    canonical = extract_signal_features(decoded, decoded_rate)
    api_vector = _extract_single(audio_bytes)
    assert len(canonical) == 170
    np.testing.assert_allclose(
        api_vector, np.fromiter(canonical.values(), dtype=float), rtol=1e-4, atol=1e-4
    )


def test_kmeans_filters_impossible_k_values(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame({"pc_0": [0.0, 0.1, 5.0, 5.1], "pc_1": [0.0, 0.2, 5.0, 5.2]})
    labels, model = fit_kmeans(frame, k_range=[1, 2, 4, 20])
    assert model.n_clusters == 2
    assert len(labels) == 4


def test_ari_uses_only_labeled_samples():
    labels = pd.DataFrame({
        "tag": ["a", "a", "b", "b", None],
        "kmeans_label": [0, 0, 1, 1, 0],
        "gmm_label": [0, 0, 1, 1, 0],
        "hdbscan_label": [0, 0, 1, 1, 0],
    })
    embeddings = pd.DataFrame({"pc_0": [0, .1, 5, 5.1, 9], "pc_1": [0, .2, 5, 5.2, 9]})
    report = compute_evaluation_metrics(labels, embeddings)
    assert set(report["adjusted_rand_index"]) == {1.0}
    assert set(report["ari_labeled_samples"]) == {4}


def test_stability_rejects_too_few_samples():
    with pytest.raises(ValueError, match="n_samples"):
        compute_cluster_stability(pd.DataFrame({"pc_0": [0.0, 1.0]}), best_k=2)
