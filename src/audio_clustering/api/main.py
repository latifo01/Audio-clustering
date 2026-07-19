"""Audio Clustering — FastAPI Inference API.

Endpoint: POST /predict
- Accepts: audio file upload (.wav, .mp3)
- Returns: cluster_id, cluster_signature (top features), confidence (GMM prob)
"""
from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from audio_clustering.pipelines.feature_engineering.nodes import extract_signal_features

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Clustering API",
    description="Unsupervised audio scene classifier — returns cluster assignment for uploaded audio clips.",
    version="1.0.0",
)

# ─── Model Loading ────────────────────────────────────────────────────────────
MODEL_DIR = Path("data/06_models")
_scaler = None
_pca = None
_umap = None
_gmm = None
_cluster_profiles = None
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 60.0


def _load_models() -> None:
    global _scaler, _pca, _umap, _gmm, _cluster_profiles
    for attr, filename in [("_scaler", "scaler.pkl"), ("_pca", "pca.pkl"),
                            ("_umap", "umap.pkl"), ("_gmm", "gmm_model.pkl")]:
        path = MODEL_DIR / filename
        if path.exists():
            with open(path, "rb") as f:
                globals()[attr] = pickle.load(f)
        else:
            logger.warning("Model file not found: %s — run `kedro run` first.", path)

    profiles_path = Path("data/07_reporting/cluster_profiles.csv")
    if profiles_path.exists():
        import pandas as pd
        globals()["_cluster_profiles"] = pd.read_csv(profiles_path, index_col=0)


@app.on_event("startup")
def startup_event() -> None:
    _load_models()
    logger.info("Audio Clustering API models loaded.")


# ─── Feature Extraction (single clip) ────────────────────────────────────────

def _extract_single(audio_bytes: bytes, sample_rate: int = 22050, n_mfcc: int = 20) -> np.ndarray:
    import librosa
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"audio duration must be between {MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS} seconds"
        )
    features = extract_signal_features(y, sr, n_mfcc=n_mfcc)
    return np.fromiter(features.values(), dtype=np.float64)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/predict", summary="Predict cluster for an audio file")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Classify an audio clip into a learned cluster.

    Returns:
        JSON with cluster_id, gmm_probability, and top cluster features.
    """
    if _gmm is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run `kedro run` first.")

    allowed = {".wav", ".mp3", ".flac", ".ogg"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds the 20 MiB limit.")
    try:
        raw_features = _extract_single(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {e}")

    # Transform through pipeline
    X = raw_features.reshape(1, -1)
    X_scaled = _scaler.transform(X)
    X_pca = _pca.transform(X_scaled)
    X_umap = _umap.transform(X_pca)

    cluster_id = int(_gmm.predict(X_pca)[0])
    probs = _gmm.predict_proba(X_pca)[0]
    confidence = float(probs[cluster_id])

    response: dict[str, Any] = {
        "filename": file.filename,
        "cluster_id": cluster_id,
        "gmm_confidence": round(confidence, 4),
        "all_cluster_probs": {f"cluster_{i}": round(float(p), 4) for i, p in enumerate(probs)},
        "umap_coordinates": {"umap_0": round(float(X_umap[0, 0]), 4),
                              "umap_1": round(float(X_umap[0, 1]), 4)},
    }

    if _cluster_profiles is not None and cluster_id in _cluster_profiles.index:
        top5 = _cluster_profiles.loc[cluster_id].head(5).to_dict()
        response["cluster_signature_top5"] = {k: round(float(v), 4) for k, v in top5.items()}

    return JSONResponse(content=response)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "models_loaded": str(_gmm is not None)}


@app.get("/clusters/summary")
def cluster_summary() -> JSONResponse:
    """Return cluster profile summaries."""
    if _cluster_profiles is None:
        raise HTTPException(status_code=503, detail="Cluster profiles not available.")
    return JSONResponse(content=_cluster_profiles.to_dict())


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
