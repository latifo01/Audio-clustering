"""
Standalone runner for the Audio Clustering project.
Extracts features from real .wav files and runs clustering.

Run with:
    pip install librosa scikit-learn umap-learn matplotlib numpy
    pip install hdbscan  # optional
    python run_audio_clustering.py

Outputs:
    - UMAP scatter plot (PNG)
    - Cluster profiles (PNG)
    - results_audio.json with metrics for README

Note: Processes up to MAX_FILES files (configurable below) to keep runtime reasonable.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────
AUDIO_DIR = Path(
    r"C:/Users/abdel/OneDrive - Université Paris-Dauphine/Bureau/"
    "Clustering in practice/data_projet"
)
OUT = Path("results_audio")
OUT.mkdir(exist_ok=True)

MAX_FILES = 300          # Limit to keep runtime < 10 min
SR = 22050               # Sample rate
N_MFCC = 20             # Number of MFCC coefficients
N_CLUSTERS_RANGE = range(3, 11)  # K range to test

# Generic Freesound tags to skip when reading ground-truth
GENERIC_TAGS = {"field-recording", "soundeffect", "sound-effects",
                "fx", "sfx", "stereo", "mono", "sample"}


# ─── 1. Build Manifest ────────────────────────────────────────────────────────
def build_manifest(audio_dir=AUDIO_DIR, max_files=MAX_FILES):
    import json

    print(f"  Scanning: {audio_dir}")
    wav_files = sorted(audio_dir.glob("*.wav"))[:max_files]
    print(f"  Found {len(wav_files)} .wav files (capped at {max_files})")

    records = []
    for fp in wav_files:
        record = {"filepath": str(fp), "stem": fp.stem, "tag": None}
        json_path = fp.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path, encoding="utf-8") as jf:
                    meta = json.load(jf)
                tags = meta.get("tags", [])
                specific = [t for t in tags if t.lower() not in GENERIC_TAGS]
                record["tag"] = specific[-1] if specific else (tags[-1] if tags else None)
            except Exception:
                pass
        records.append(record)

    labeled = sum(1 for r in records if r["tag"])
    print(f"  Files with ground-truth tag: {labeled}/{len(records)}")
    return records


# ─── 2. Extract Features ──────────────────────────────────────────────────────
def extract_features(manifest, sr=SR, n_mfcc=N_MFCC):
    import librosa

    features = []
    valid_records = []
    print(f"  Extracting features from {len(manifest)} files...")

    for i, rec in enumerate(manifest):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(manifest)}...")
        try:
            y, _ = librosa.load(rec["filepath"], sr=sr, duration=10.0, mono=True)
            if len(y) < sr * 0.5:
                continue

            feat = []

            # MFCCs (mean + std)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            feat.extend(mfcc.mean(axis=1))
            feat.extend(mfcc.std(axis=1))

            # Delta MFCCs
            delta = librosa.feature.delta(mfcc)
            feat.extend(delta.mean(axis=1))

            # Spectral features
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            feat.extend([spec_centroid.mean(), spec_centroid.std()])
            feat.extend([spec_bw.mean(), spec_bw.std()])
            feat.extend([spec_rolloff.mean(), spec_rolloff.std()])

            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            feat.extend(chroma.mean(axis=1))

            # ZCR + RMS
            zcr = librosa.feature.zero_crossing_rate(y)
            rms = librosa.feature.rms(y=y)
            feat.extend([zcr.mean(), zcr.std(), rms.mean(), rms.std()])

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            feat.append(float(tempo))

            features.append(feat)
            valid_records.append(rec)

        except Exception as e:
            pass  # skip corrupt files

    print(f"  Features extracted: {len(features)} valid files")
    return np.array(features, dtype=np.float32), valid_records


# ─── 3. Preprocessing ─────────────────────────────────────────────────────────
def preprocess(X):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA — keep 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  PCA: {X.shape[1]} → {X_pca.shape[1]} components "
          f"(95% variance, EVR=[{pca.explained_variance_ratio_[:3].sum():.2f}...])")

    # UMAP 2D for visualization
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        X_umap = reducer.fit_transform(X_pca)
        print("  UMAP 2D embedding done")
    except ImportError:
        from sklearn.manifold import TSNE
        X_umap = TSNE(n_components=2, random_state=42).fit_transform(X_pca)
        print("  t-SNE 2D embedding done (umap not installed)")

    return X_pca, X_umap, pca.n_components_


# ─── 4. Clustering ─────────────────────────────────────────────────────────────
def run_clustering(X_pca):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                  silhouette_score)
    from sklearn.mixture import GaussianMixture

    best_k = 5
    best_sil = -1.0
    sil_scores = {}

    print("  Testing KMeans k=3..10...")
    for k in N_CLUSTERS_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        sil_scores[k] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"  Best K={best_k} (Silhouette={best_sil:.4f})")

    # Final KMeans
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels_km = km_final.fit_predict(X_pca)

    # GMM with BIC selection
    bic_scores = {}
    best_gmm_k = best_k
    best_bic = np.inf
    for k in N_CLUSTERS_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        gmm.fit(X_pca)
        bic = gmm.bic(X_pca)
        bic_scores[k] = bic
        if bic < best_bic:
            best_bic = bic
            best_gmm_k = k

    gmm_final = GaussianMixture(n_components=best_gmm_k, random_state=42, n_init=10)
    labels_gmm = gmm_final.fit_predict(X_pca)

    # HDBSCAN
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
        labels_hdbscan = clusterer.fit_predict(X_pca)
        n_clusters_hdbscan = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
        n_noise = (labels_hdbscan == -1).sum()
        sil_hdbscan = silhouette_score(X_pca[labels_hdbscan != -1],
                                       labels_hdbscan[labels_hdbscan != -1]) if n_clusters_hdbscan > 1 else -1
    except ImportError:
        labels_hdbscan = np.zeros(len(X_pca), dtype=int)
        n_clusters_hdbscan = 1
        n_noise = 0
        sil_hdbscan = -1.0

    results = {
        "kmeans": {
            "best_k": best_k,
            "silhouette": float(best_sil),
            "calinski_harabasz": float(calinski_harabasz_score(X_pca, labels_km)),
            "davies_bouldin": float(davies_bouldin_score(X_pca, labels_km)),
            "labels": labels_km,
            "sil_scores_by_k": {k: float(v) for k, v in sil_scores.items()},
        },
        "gmm": {
            "best_k": best_gmm_k,
            "bic": float(best_bic),
            "labels": labels_gmm,
            "bic_scores_by_k": {k: float(v) for k, v in bic_scores.items()},
        },
        "hdbscan": {
            "n_clusters": n_clusters_hdbscan,
            "n_noise": int(n_noise),
            "silhouette": float(sil_hdbscan),
            "labels": labels_hdbscan,
        },
    }
    return results


# ─── 5. ARI (if ground-truth available) ──────────────────────────────────────
def compute_ari(records, labels):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import LabelEncoder

    tagged = [(r["tag"], l) for r, l in zip(records, labels) if r["tag"]]
    if len(tagged) < 5:
        return None, len(tagged)

    tags, lbls = zip(*tagged)
    enc = LabelEncoder()
    gt = enc.fit_transform(tags)
    ari = adjusted_rand_score(gt, list(lbls)[:len(gt)])
    return float(ari), len(tagged)


# ─── 6. Plots ─────────────────────────────────────────────────────────────────
def plot_umap(X_umap, clustering_results, out_path):
    labels = clustering_results["kmeans"]["labels"]
    n_clusters = clustering_results["kmeans"]["best_k"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Audio Clustering — UMAP 2D Embeddings", fontsize=14)

    cmap = plt.get_cmap("tab10")

    for ax, (algo, lbls_key) in zip(axes, [
        ("KMeans", "kmeans"),
        ("GMM", "gmm"),
        ("HDBSCAN", "hdbscan"),
    ]):
        lbls = clustering_results[lbls_key]["labels"]
        unique = sorted(set(lbls))
        for lbl in unique:
            mask = lbls == lbl
            color = "grey" if lbl == -1 else cmap(lbl % 10)
            label = "Noise" if lbl == -1 else f"Cluster {lbl}"
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[color],
                       s=10, alpha=0.7, label=label)
        ax.set_title(algo)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=2, fontsize=7, loc="best")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_elbow(clustering_results, out_path):
    sil_scores = clustering_results["kmeans"]["sil_scores_by_k"]
    bic_scores = clustering_results["gmm"]["bic_scores_by_k"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ks = sorted(sil_scores.keys())
    axes[0].plot(ks, [sil_scores[k] for k in ks], "o-", color="steelblue")
    axes[0].axvline(clustering_results["kmeans"]["best_k"], color="red",
                    linestyle="--", label=f"Best k={clustering_results['kmeans']['best_k']}")
    axes[0].set_xlabel("Number of clusters K")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("KMeans — Silhouette vs K")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ks2 = sorted(bic_scores.keys())
    axes[1].plot(ks2, [bic_scores[k] for k in ks2], "o-", color="firebrick")
    axes[1].axvline(clustering_results["gmm"]["best_k"], color="red",
                    linestyle="--", label=f"Best k={clustering_results['gmm']['best_k']}")
    axes[1].set_xlabel("Number of components K")
    axes[1].set_ylabel("BIC")
    axes[1].set_title("GMM — BIC vs K")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    results = {}

    print("\n[1/5] Building audio manifest...")
    manifest = build_manifest()

    print("\n[2/5] Extracting features (~120 per file)...")
    X, valid_records = extract_features(manifest)

    if len(X) < 10:
        print("ERROR: Too few valid files. Check AUDIO_DIR path.")
        return

    print("\n[3/5] Preprocessing (StandardScaler + PCA + UMAP)...")
    X_pca, X_umap, n_pca = preprocess(X)

    print("\n[4/5] Running clustering algorithms...")
    clustering = run_clustering(X_pca)

    # ARI
    ari_km, n_labeled = compute_ari(valid_records, clustering["kmeans"]["labels"])
    ari_gmm, _ = compute_ari(valid_records, clustering["gmm"]["labels"])

    print("\n[5/5] Computing final metrics and plots...")

    results = {
        "n_files": len(X),
        "n_features_raw": X.shape[1],
        "n_pca_components": int(n_pca),
        "n_labeled_files": n_labeled,
        "kmeans": {k: v for k, v in clustering["kmeans"].items() if k != "labels"},
        "gmm": {k: v for k, v in clustering["gmm"].items() if k != "labels"},
        "hdbscan": {k: v for k, v in clustering["hdbscan"].items() if k != "labels"},
        "ari_kmeans_vs_gt": ari_km,
        "ari_gmm_vs_gt": ari_gmm,
    }

    # Save results
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(OUT / "results_audio.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    # Plots
    plot_umap(X_umap, clustering, OUT / "umap_clusters.png")
    plot_elbow(clustering, OUT / "elbow_curves.png")

    # Print README table
    print("\n\n" + "="*65)
    print("  README TABLE — Clustering Results (copy-paste)")
    print("="*65)
    print(f"  Dataset: {len(X)} audio clips, {X.shape[1]} features → {n_pca} PCA components")
    print()
    print("| Algorithm | Best K | Silhouette | Calinski-H | Davies-Bouldin | ARI (vs GT) |")
    print("|-----------|--------|-----------|------------|----------------|-------------|")
    print(f"| KMeans    | {clustering['kmeans']['best_k']}      | "
          f"{clustering['kmeans']['silhouette']:.4f}    | "
          f"{clustering['kmeans']['calinski_harabasz']:.1f}      | "
          f"{clustering['kmeans']['davies_bouldin']:.4f}         | "
          f"{ari_km:.4f}      |")
    print(f"| GMM       | {clustering['gmm']['best_k']}      | —         | —          | "
          f"— (BIC={clustering['gmm']['bic']:.0f}) | {ari_gmm:.4f}      |")
    print(f"| HDBSCAN   | {clustering['hdbscan']['n_clusters']}      | "
          f"{clustering['hdbscan']['silhouette']:.4f}    | —          | —              | —           |")
    print(f"\n  HDBSCAN noise points: {clustering['hdbscan']['n_noise']} "
          f"({100*clustering['hdbscan']['n_noise']/len(X):.1f}%)")
    print(f"  Labeled files used for ARI: {n_labeled}")
    print(f"\n  Results saved to: {OUT.resolve()}/")


if __name__ == "__main__":
    main()
