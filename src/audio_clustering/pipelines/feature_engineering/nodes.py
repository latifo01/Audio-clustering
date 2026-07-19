"""Canonical acoustic feature extraction for both training and inference."""
from __future__ import annotations

import logging

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_signal_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
    include_delta_mfcc: bool = True,
    include_delta2_mfcc: bool = True,
) -> dict[str, float]:
    """Return the single, ordered feature contract used everywhere."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size < n_fft:
        raise ValueError(f"audio must be mono and contain at least {n_fft} samples")
    features: dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    for index in range(n_mfcc):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(mfcc[index]))
        features[f"mfcc_{index:02d}_std"] = float(np.std(mfcc[index]))

    if include_delta_mfcc:
        delta = librosa.feature.delta(mfcc)
        for index in range(n_mfcc):
            features[f"delta_mfcc_{index:02d}_mean"] = float(np.mean(delta[index]))
            features[f"delta_mfcc_{index:02d}_std"] = float(np.std(delta[index]))
    if include_delta2_mfcc:
        delta2 = librosa.feature.delta(mfcc, order=2)
        for index in range(n_mfcc):
            features[f"delta2_mfcc_{index:02d}_mean"] = float(np.mean(delta2[index]))
            features[f"delta2_mfcc_{index:02d}_std"] = float(np.std(delta2[index]))

    spectral_groups = {
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length),
    }
    for name, values in spectral_groups.items():
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for band in range(contrast.shape[0]):
        features[f"spectral_contrast_band{band}_mean"] = float(np.mean(contrast[band]))
        features[f"spectral_contrast_band{band}_std"] = float(np.std(contrast[band]))

    texture = {
        "zcr": librosa.feature.zero_crossing_rate(y, hop_length=hop_length),
        "rms": librosa.feature.rms(y=y, hop_length=hop_length),
    }
    for name, values in texture.items():
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for channel in range(12):
        features[f"chroma_{channel:02d}_mean"] = float(np.mean(chroma[channel]))
        features[f"chroma_{channel:02d}_std"] = float(np.std(chroma[channel]))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    features["tempo"] = float(np.atleast_1d(tempo)[0])
    if len(beat_frames) > 2:
        intervals = np.diff(beat_frames).astype(float)
        features["beat_regularity"] = float(1.0 / (1.0 + np.std(intervals) / np.mean(intervals)))
    else:
        features["beat_regularity"] = 0.0
    return features


def extract_features(
    manifest: pd.DataFrame,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
    sample_rate: int = 22050,
    include_delta_mfcc: bool = True,
    include_delta2_mfcc: bool = True,
) -> pd.DataFrame:
    """Extract canonical features for every valid manifest row."""
    feature_rows: list[dict] = []
    failed: list[str] = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting features"):
        try:
            y, sr = librosa.load(row["filepath"], sr=sample_rate, mono=True)
            features: dict = extract_signal_features(
                y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                include_delta_mfcc=include_delta_mfcc,
                include_delta2_mfcc=include_delta2_mfcc,
            )
            features.update({
                "filename": row["filename"],
                "filepath": row["filepath"],
                "duration_sec": row["duration_sec"],
                "tag": row.get("tag", None),
                "environment": row.get("environment", None),
            })
            feature_rows.append(features)
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s", row["filename"], exc)
            failed.append(row["filename"])
    if failed:
        logger.warning("Failed to extract features from %d clips: %s", len(failed), failed[:5])
    result = pd.DataFrame(feature_rows)
    logger.info("Feature matrix shape: %s | Columns: %d", result.shape, len(result.columns))
    return result
