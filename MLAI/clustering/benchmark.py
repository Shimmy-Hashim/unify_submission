"""
benchmark.py — Benchmarks 6 clustering models on the processed SCF features.

Models evaluated:
  1. K-Means          (sklearn)
  2. K-Medoids        (sklearn_extra or pure-Python fallback)
  3. Fuzzy C-Means    (skfuzzy)
  4. Gaussian Mixture (sklearn)
  5. DBSCAN           (sklearn)
  6. Agglomerative    (sklearn)

Metrics:
  - Silhouette Score (higher is better)
  - Davies-Bouldin Index (lower is better)
  - Calinski-Harabasz Index (higher is better)
  - BIC / AIC (GMM only, lower is better)
  - Cluster distribution
  - Runtime (seconds)

Reproducibility notes:
  - K-Medoids has two possible implementations (see _get_kmedoids). They produce
    DIFFERENT clusterings on the same data, so the one that actually ran is
    recorded in the results as `kmedoids_impl` rather than left silent.
  - Fuzzy C-Means is seeded (seed=42, as in test_users.py). Without a seed its
    cluster IDs — and so the order of cluster_distribution — permute between
    runs; the metrics themselves are permutation-invariant either way.
  - runtime_seconds is wall-clock and varies between runs by nature.

Reference: Metzler, Zhou & Grace (2021), Quantitative Finance and Economics,
5(3). Cited for the three-tier segmentation precedent only. That paper uses
PAM/Gower on self-reported, mixed-type survey data and explicitly finds K-Means
unsuitable there, so it is NOT a source for the choice of models compared below.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = "MLAI/clustering/results"

def _parse_args():
    """Return (features_path, json_output, csv_output) from CLI args or defaults."""
    features_path = "MLAI/clustering/data/processed/features.csv"
    json_output = os.path.join(RESULTS_DIR, "benchmark_results.json")
    csv_output = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--features" and i + 1 < len(args):
            features_path = args[i + 1]
        elif arg == "--output" and i + 1 < len(args):
            stem = args[i + 1]
            json_output = stem if stem.endswith(".json") else stem + ".json"
            csv_output = json_output.replace(".json", ".csv")
    return features_path, json_output, csv_output

N_CLUSTERS = 3


# ---------------------------------------------------------------------------
# K-Medoids: try sklearn_extra first, fall back to pure Python
# ---------------------------------------------------------------------------

def _get_kmedoids():
    """Return (estimator, implementation_name).

    sklearn_extra is an optional dependency and has been blocked by Windows
    Smart App Control on this project before now. The fallback below is not
    equivalent to it — the two reach different clusterings on the same input —
    so the caller records which one ran instead of letting it pass silently.
    """
    try:
        from sklearn_extra.cluster import KMedoids
        return KMedoids(n_clusters=N_CLUSTERS, random_state=42), "sklearn_extra"
    except ImportError:
        # Pure-Python fallback (PAM-lite: pick medoids by minimising intra-cluster
        # sum of distances to all other cluster members).
        class KMedoidsFallback:
            def __init__(self, n_clusters=3, random_state=42):
                self.n_clusters = n_clusters
                self.random_state = random_state
                self.labels_ = None
                self.cluster_centers_ = None

            def fit_predict(self, X):
                rng = np.random.default_rng(self.random_state)
                n = len(X)
                # Initialise medoid indices randomly
                medoid_idx = rng.choice(n, self.n_clusters, replace=False)

                for _ in range(100):  # max iterations
                    # Assign each point to nearest medoid
                    dists = np.array([
                        np.linalg.norm(X - X[m], axis=1) for m in medoid_idx
                    ])  # shape: (k, n)
                    labels = np.argmin(dists, axis=0)

                    # Update medoids
                    new_medoids = []
                    for k in range(self.n_clusters):
                        members = np.where(labels == k)[0]
                        if len(members) == 0:
                            new_medoids.append(medoid_idx[k])
                            continue
                        intra = np.sum(
                            np.linalg.norm(X[members][:, None] - X[members][None, :], axis=2),
                            axis=1,
                        )
                        new_medoids.append(members[np.argmin(intra)])

                    if np.array_equal(sorted(new_medoids), sorted(medoid_idx)):
                        break
                    medoid_idx = np.array(new_medoids)

                self.labels_ = labels
                self.cluster_centers_ = X[medoid_idx]
                return labels

        return KMedoidsFallback(n_clusters=N_CLUSTERS, random_state=42), "pure_python_fallback"


# ---------------------------------------------------------------------------
# Fuzzy C-Means wrapper
# ---------------------------------------------------------------------------

def run_fuzzy_cmeans(X, n_clusters=N_CLUSTERS):
    # skfuzzy expects shape (features, samples), so transpose. seed matches
    # test_users.py: without it the cluster IDs permute between runs.
    _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T, c=n_clusters, m=2, error=0.005, maxiter=1000, seed=42
    )
    # Hard assignment for metrics: argmax of membership matrix u (shape: c x samples)
    labels = np.argmax(u, axis=0)
    return labels


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(X, labels):
    unique = np.unique(labels[labels >= 0])  # exclude DBSCAN noise (-1)
    if len(unique) < 2:
        return None, None, None

    # For DBSCAN, exclude noise points from metric calculation
    mask = labels >= 0
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return None, None, None

    sil = silhouette_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    ch = calinski_harabasz_score(X[mask], labels[mask])
    return sil, db, ch


def cluster_distribution(labels):
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    return counts.tolist()


def _result(model, labels, metrics, runtime, n_clusters=N_CLUSTERS,
            bic=None, aic=None, **extra):
    """Assemble one result row. `extra` keys are appended after the fixed ones,
    so adding a model-specific field never reorders the CSV columns.

    The `if x else None` guards are the original truthiness semantics, kept
    deliberately: a metric of exactly 0.0 is recorded as None. No model in this
    benchmark produces a 0.0 metric, so this has never actually fired.
    """
    sil, db, ch = metrics
    row = {
        "model": model,
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 4) if sil else None,
        "davies_bouldin_index": round(db, 4) if db else None,
        "calinski_harabasz_index": round(ch, 4) if ch else None,
        "bic": bic,
        "aic": aic,
        "cluster_distribution": cluster_distribution(labels),
        "runtime_seconds": round(runtime, 4),
        "recommended": False,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark():
    features_path, json_output, csv_output = _parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading features from {features_path} ...")
    df = pd.read_csv(features_path)
    X = df.values
    print(f"  Shape: {X.shape}\n")

    results = []

    # ---- 1. K-Means ----
    print("Running KMeans ...")
    t0 = time.time()
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    runtime = time.time() - t0
    results.append(_result("KMeans", labels, compute_metrics(X, labels), runtime))

    # ---- 2. K-Medoids ----
    print("Running KMedoids ...")
    t0 = time.time()
    kmed, kmedoids_impl = _get_kmedoids()
    labels = kmed.fit_predict(X)
    runtime = time.time() - t0
    print(f"  K-Medoids implementation: {kmedoids_impl}")
    results.append(_result("KMedoids", labels, compute_metrics(X, labels), runtime,
                           kmedoids_impl=kmedoids_impl))

    # ---- 3. Fuzzy C-Means ----
    print("Running FuzzyCMeans ...")
    t0 = time.time()
    labels = run_fuzzy_cmeans(X, N_CLUSTERS)
    runtime = time.time() - t0
    results.append(_result("FuzzyCMeans", labels, compute_metrics(X, labels), runtime))

    # ---- 4. Gaussian Mixture Model ----
    print("Running GMM ...")
    t0 = time.time()
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)
    runtime = time.time() - t0
    results.append(_result("GMM", labels, compute_metrics(X, labels), runtime,
                           bic=round(gmm.bic(X), 4), aic=round(gmm.aic(X), 4)))

    # ---- 5. DBSCAN ----
    print("Running DBSCAN ...")
    t0 = time.time()
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    labels = dbscan.fit_predict(X)
    runtime = time.time() - t0
    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    results.append(_result("DBSCAN", labels, compute_metrics(X, labels), runtime,
                           n_clusters=n_found))

    # ---- 6. Agglomerative ----
    print("Running Agglomerative ...")
    t0 = time.time()
    agg = AgglomerativeClustering(n_clusters=N_CLUSTERS)
    labels = agg.fit_predict(X)
    runtime = time.time() - t0
    results.append(_result("Agglomerative", labels, compute_metrics(X, labels), runtime))

    # ---- Determine recommended model ----
    # Rank by silhouette (desc) then Davies-Bouldin (asc); favour silhouette
    scored = [r for r in results if r["silhouette_score"] is not None]
    if scored:
        # Primary: highest silhouette; tiebreak: lowest Davies-Bouldin
        best = max(scored, key=lambda r: (
            r["silhouette_score"],
            -r["davies_bouldin_index"] if r["davies_bouldin_index"] else 0,
        ))
        best["recommended"] = True

    # ---- Save JSON ----
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results -> {json_output}")

    # ---- Save CSV ----
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_output, index=False)
    print(f"Saved benchmark summary -> {csv_output}")

    # ---- Print table ----
    print("\n" + "=" * 105)
    print(f"{'Model':<16} {'Silhouette':>11} {'Davies-Bouldin':>15} {'Calinski-Harabasz':>18} {'BIC':>12} {'AIC':>12} {'Runtime':>9}")
    print("=" * 105)
    for r in results:
        sil_s = f"{r['silhouette_score']:.4f}" if r["silhouette_score"] is not None else "  N/A  "
        db_s = f"{r['davies_bouldin_index']:.4f}" if r["davies_bouldin_index"] is not None else "  N/A  "
        ch_s = f"{r['calinski_harabasz_index']:.1f}" if r["calinski_harabasz_index"] is not None else "  N/A  "
        bic_s = f"{r['bic']:.1f}" if r["bic"] is not None else " null"
        aic_s = f"{r['aic']:.1f}" if r["aic"] is not None else " null"
        rt_s = f"{r['runtime_seconds']:.2f}s"
        print(f"{r['model']:<16} {sil_s:>11} {db_s:>15} {ch_s:>18} {bic_s:>12} {aic_s:>12} {rt_s:>9}")
    print("=" * 105)

    recommended = next((r["model"] for r in results if r["recommended"]), "N/A")
    print(f"\nRecommended: {recommended}\n")


if __name__ == "__main__":
    run_benchmark()
