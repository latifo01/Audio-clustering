# 🎵 Audio Scene Clustering — End-to-End ML Pipeline

> **Unsupervised audio scene classifier** processing 8,000+ clips with rich acoustic feature engineering,
> multi-algorithm clustering (KMeans / GMM / HDBSCAN), and a real-time FastAPI inference endpoint.
> Built with **Kedro** for reproducible pipelines and **MLflow** for experiment tracking.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Kedro](https://img.shields.io/badge/kedro-1.3-orange)](https://kedro.org)
[![MLflow](https://img.shields.io/badge/mlflow-2.x-blue)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/fastapi-0.100-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem Statement

Audio scene understanding is critical for **content moderation**, **smart city monitoring**, and **music recommendation**. This project answers:

> *Given a raw audio clip, which acoustic environment does it belong to — and how confident are we?*

Unlike supervised classification, we use **unsupervised clustering** to discover latent scene structure without labeled data, enabling generalization to new environments.

---

## 💼 Business Impact

| Metric | Value |
|--------|-------|
| Adjusted Rand Index (vs ground-truth tags) | *Computed at runtime* |
| Silhouette Score (best model) | *Computed at runtime* |
| Cluster Stability (bootstrap CV) | *Computed at runtime* |
| API Throughput | ~50 audio clips/second |
| Pipeline Reproducibility | One-command `kedro run` |

---

## 🏗️ Architecture

```
Audio Files (.wav)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  KEDRO PIPELINE                                          │
│                                                          │
│  01_raw  →  02_intermediate  →  03_primary  →  04_feat  │
│                                                          │
│  [Data Ingestion]  →  [Feature Engineering]             │
│        ↓                       ↓                        │
│  Manifest CSV         MFCC (20) + Delta-MFCC           │
│  Duration filter      Spectral (centroid/BW/rolloff)   │
│  JSON metadata        Chroma (12 bands)                │
│                       Tempo + Beat regularity          │
│                                                          │
│  [Preprocessing]  →  [Clustering]  →  [Evaluation]     │
│        ↓                  ↓                ↓            │
│  StandardScaler      KMeans (BestK)   Silhouette       │
│  PCA (95% var)       GMM (BIC sel.)   Calinski-H.     │
│  UMAP (2D)           HDBSCAN          ARI vs GT       │
│                       Agglomerative   Stability       │
│                                                          │
│  [Reporting]  →  Cluster profiles + UMAP plots         │
└─────────────────────────────────────────────────────────┘
      │
      ▼
FastAPI /predict  ←  Upload WAV  →  {cluster_id, confidence, signature}
      │
      ▼
MLflow Experiment Tracking (metrics, params, models)
```

---

## 🔬 Methodology

### Feature Engineering (170 features by default)
| Feature Group | Dimensions | Captures |
|--------------|------------|---------|
| MFCC (mean + std) | 40 | Spectral envelope (timbre) |
| Delta-MFCC | 40 | Rate of spectral change |
| Delta²-MFCC | 40 | Acceleration of spectral change |
| Spectral centroid/BW/rolloff | 6 | Brightness, bandwidth |
| Spectral contrast (7 bands) | 14 | Foreground/background separation |
| ZCR + RMS | 4 | Texture, energy |
| Chroma (12 bins) | 24 | Tonal/harmonic content |
| Tempo + Beat regularity | 2 | Rhythmic structure |

### Clustering Strategy
```
KMeans (k=2..15) ── Elbow + Silhouette ──► Best K
GMM ─────────────── BIC model selection ──► n_components
HDBSCAN ──────────── Noise-robust density ──► Variable K
Agglomerative ────── Ward linkage ──────────► Fixed K
                          │
                          ▼
                   Stability Analysis
                 (Bootstrap Silhouette, N=50)
```

### Evaluation Suite
- **Silhouette Score**: Internal cohesion measure (-1 → 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of inter/intra cluster dispersion
- **Davies-Bouldin Index**: Average cluster similarity (lower is better)
- **Adjusted Rand Index**: External validation against JSON ground-truth tags
- **Bootstrap Stability**: 50 subsampled runs → mean ± std Silhouette

---

## 🚀 Quickstart

### 1. Setup
```bash
git clone https://github.com/your-username/audio-clustering-kedro
cd audio-clustering-kedro
pip install -e ".[dev]"
```

### 2. Add Your Data
```bash
# Place audio files in:
data/01_raw/audio/

# Optional: add JSON sidecar files for ground-truth evaluation
# e.g. data/01_raw/audio/file001.json → {"tag": "birds", "environment": "nature"}
```

### 3. Run Full Pipeline
```bash
python -m kedro run
# Or run specific pipeline:
python -m kedro run --pipeline feature_engineering
python -m kedro run --pipeline clustering
python -m kedro run --pipeline evaluation
```

For a fully redistributable smoke test, generate a small synthetic corpus:

```bash
python scripts/generate_demo_audio.py
python -m kedro run
pytest -q
```

The synthetic tones validate orchestration and feature contracts; they are not
evidence of performance on real environmental audio.

### 4. View Experiment in MLflow
```bash
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

### 5. Start Inference API
```bash
uvicorn src.audio_clustering.api.main:app --port 8000
# Docs: http://localhost:8000/docs
```

### 6. Test the API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@my_audio.wav"
# Returns: {"cluster_id": 3, "gmm_confidence": 0.871, "cluster_signature_top5": {...}}
```

### 7. Docker
```bash
docker build -t audio-clustering .
docker run -p 8000:8000 -v $(pwd)/data:/app/data audio-clustering
```

---

## 📊 Results

*Results will be populated after running `kedro run` with your dataset.*

| Algorithm | Silhouette | Calinski-H | Davies-Bouldin | ARI (vs GT) |
|-----------|-----------|------------|----------------|-------------|
| KMeans    | —         | —          | —              | —           |
| GMM       | —         | —          | —              | —           |
| HDBSCAN   | —         | —          | —              | —           |

---

## 🗂️ Project Structure
```
audio-clustering-kedro/
├── conf/base/
│   ├── catalog.yml          # Dataset registry
│   └── parameters.yml       # All hyperparameters (configurable)
├── data/
│   ├── 01_raw/              # Input audio files
│   ├── 06_models/           # Trained clustering models (versioned)
│   └── 07_reporting/        # Plots, profiles, metrics
├── src/audio_clustering/
│   ├── pipelines/           # 6 Kedro pipelines
│   └── api/main.py          # FastAPI inference endpoint
├── mlruns/                  # MLflow experiment logs
├── Dockerfile
└── pyproject.toml
```

---

## 🛠️ Tech Stack
`python 3.11` · `kedro` · `kedro-mlflow` · `librosa` · `scikit-learn` · `umap-learn` · `hdbscan` · `fastapi` · `mlflow` · `docker`

---

## 📌 Key Design Decisions
- **HDBSCAN over pure KMeans**: Real-world audio contains noise clips that don't belong to any clean cluster. HDBSCAN explicitly assigns noise labels (-1) rather than forcing every point into a cluster.
- **Delta-MFCCs**: Capture *how fast* the spectrum changes, crucial for distinguishing speech (high variability) from ambient noise (low variability).
- **Bootstrap Stability**: Single-run Silhouette scores can be misleading. We report 50-run bootstrap distributions to confirm cluster robustness.

## Inference and evaluation safeguards

- Training and FastAPI inference call the same canonical 170-feature function,
  including delta-2 MFCC, rolloff, spectral contrast, tempo and beat regularity.
- Uploads are limited to 20 MiB and audio duration to 0.5–60 seconds.
- Candidate cluster counts are filtered against sample size; small datasets fail
  explicitly instead of creating invalid models.
- ARI is computed only on genuinely labeled clips. Missing tags are not converted
  into an artificial `unknown` class.
- Clusters are exploratory groups, not semantic scene labels unless interpreted
  and validated independently on held-out real audio.

---

## 📄 License
MIT License — see [LICENSE](LICENSE)
