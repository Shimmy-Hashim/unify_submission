"""
comprehensive_benchmark_11algo.py — Expanded 12-algorithm re-benchmark on the
corrected synthetic dataset (Unify, CCDS25-1170, July 2026).

WHY THIS EXISTS (does NOT replace comprehensive_benchmark.py):
  * The dataset was regenerated with corrections (investment_rate hurdle model,
    net_worth_to_income revision, portfolio_concentration age gradient, Band 5
    emergency-fund fix, 100-minus-age standardization).
  * The Layer 2 feature set was confirmed as the 11 CONTINUOUS features (ablation
    "dilution not distortion": binaries excluded). scoring.py's
    LAYER2_CLUSTERING_FEATURES now holds exactly this set.
  * The algorithm slate is expanded from 5 to 12, mapped to Wani 2024's five-family
    taxonomy (PeerJ CS, DOI 10.7717/peerj-cs.2286) plus one bonus method.

METHODOLOGY IS UNCHANGED from comprehensive_benchmark.py — the composite rank-sum
(silhouette / Davies-Bouldin / Calinski-Harabasz), the 4-metric k-selection score
(sil / inv-DB / CH / size-balance), and both approaches (A = age-stratified,
B = single model with scaled age appended) are IMPORTED from that module, not
reimplemented. Only the algorithm list, the feature set, and the input data change.

This run is CANONICAL as of July 2026. Its outputs use the canonical results/ names
below; the prior 5-algorithm/10-feature run was archived (not deleted) under
results/archive/comprehensive_10feature_5algo/ when this run was promoted:
    results/comprehensive_results.json            (CANONICAL as of July 2026; the prior
                                                   5-algo/10-feature run is archived under
                                                   results/archive/comprehensive_10feature_5algo/)
    results/algorithm_metrics.csv                 (tidy per algo/k/approach)
    results/plots/k_comparison.png
    results/plots/algorithm_heatmap.png           (new output type; no old equivalent)

The fitted production scaler (backend/ml/artifacts/fitted_scaler.pkl) is 10-feature
and is NOT used here; a RobustScaler is fitted IN-MEMORY on the 11 features (same as
the ablation). No production artifact is read or written. --export-centroids is not
provided by this script by design.

Run:
    cd <repo root>
    backend/venv/Scripts/python.exe MLAI/clustering/comprehensive_benchmark_11algo.py
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering, Birch, DBSCAN, HDBSCAN,
)
from sklearn.mixture import GaussianMixture

# Reuse the EXACT scoring methodology from the shared core (no reimplementation).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_core import (  # noqa: E402
    compute_metrics, composite_rank, selection_score, cluster_sizes_and_balance,
)

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# The confirmed 11 continuous Layer 2 features (matches scoring.py
# LAYER2_CLUSTERING_FEATURES; generator column names).
FEATURES_11 = [
    "savings_rate", "emergency_fund_months", "dti", "net_worth_to_income",
    "investment_rate", "portfolio_concentration", "risk_age_deviation",
    "life_coverage_ratio", "ci_coverage_ratio", "healthy_debt_ratio",
    "insurance_spend_ratio",
]

K_RANGE = range(2, 7)
SEED = 42

# k-based algorithms (hard labels at a chosen k). Names map to the taxonomy families.
K_BASED = ["kmeans", "kmedoids", "fuzzy_cmeans", "gmm_full", "gmm_diag",
           "spectral", "agglomerative", "birch", "archetypal"]
# Non-k algorithms: cluster count emerges from the data; reported separately.
NON_K = ["dbscan", "hdbscan", "umap_hdbscan"]

FAMILY = {
    "kmeans": "centroid", "kmedoids": "centroid", "fuzzy_cmeans": "centroid",
    "dbscan": "density", "hdbscan": "density",
    "gmm_full": "distribution", "gmm_diag": "distribution",
    "spectral": "graph", "umap_hdbscan": "graph",
    "agglomerative": "hierarchical", "birch": "hierarchical",
    "archetypal": "bonus (outside taxonomy)",
}

# For identifying the "ideal" cluster in the A-vs-B benchmark-violation check.
# Heuristic only (not production scoring): insurance_spend_ratio is intentionally
# EXCLUDED — it is non-monotonic as a health signal (too low = underinsured, too
# high = over-paying), so it cannot be scored "higher/lower is better".
HIGHER_BETTER = ["savings_rate", "emergency_fund_months", "net_worth_to_income",
                 "investment_rate", "life_coverage_ratio", "ci_coverage_ratio",
                 "healthy_debt_ratio"]
LOWER_BETTER = ["dti", "portfolio_concentration"]
LIA_LIFE_COVER_MIN = 5.0  # same threshold the original run used to flag underinsured "ideal" clusters


# ======================================================================================
# Archetypal Analysis — vendored PCHA (Principal Convex Hull Analysis, Morup & Hansen
# 2012). No maintained package installs on this Python; this is a compact, deterministic
# projected-gradient implementation. X ~= W @ (A @ X): archetypes Z = A @ X are convex
# combinations of data points (A row-stochastic, k x n), and each point is a convex
# combination of archetypes (W row-stochastic, n x k). Hard label = argmax(W).
# ======================================================================================
def _project_rows_to_simplex(V):
    """Euclidean projection of each row of V onto the probability simplex (Duchi 2008)."""
    U = np.sort(V, axis=1)[:, ::-1]
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, V.shape[1] + 1)
    cond = U - css / ind > 0
    rho = cond.sum(axis=1)
    theta = css[np.arange(V.shape[0]), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


def _furthest_sum_init(X, k, seed):
    """Standard AA initialisation: pick k mutually-far data points."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = [int(rng.integers(n))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    while len(idx) < k:
        nxt = int(np.argmax(d))
        if nxt in idx:
            nxt = int(rng.integers(n))
        idx.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
    return idx


def archetypal_labels(X, k, n_iter=120, seed=SEED):
    n = X.shape[0]
    idx = _furthest_sum_init(X, k, seed)
    A = np.zeros((k, n))
    for r, i in enumerate(idx):
        A[r, i] = 1.0
    Z = A @ X                                  # k x d archetypes
    W = np.full((n, k), 1.0 / k)               # n x k weights
    lr_W, lr_A = 0.5, 0.5
    prev = np.inf
    for _ in range(n_iter):
        # Update W (fix Z): grad of ||X - W Z||^2 wrt W = -2 (X - W Z) Z^T
        R = X - W @ Z
        W = _project_rows_to_simplex(W + lr_W * (R @ Z.T) / (np.linalg.norm(Z) ** 2 + 1e-9))
        # Update A (fix W): Z = A X, grad wrt A of ||X - W A X||^2 = -2 W^T (X - W A X) X^T
        R = X - W @ (A @ X)
        A = _project_rows_to_simplex(A + lr_A * (W.T @ R @ X.T) / (np.linalg.norm(W) ** 2 * np.linalg.norm(X) ** 2 + 1e-9))
        Z = A @ X
        err = float(np.linalg.norm(X - W @ Z) ** 2)
        if abs(prev - err) / (prev + 1e-12) < 1e-5:
            break
        prev = err
    return np.argmax(W, axis=1)


# ======================================================================================
# Algorithm dispatch (hard labels)
# ======================================================================================
def run_kbased(name, k, X):
    # sklearn_extra / skfuzzy are imported inside their branches ON PURPOSE, not by
    # oversight: both are optional and have been blocked at import time on this machine by
    # Windows Smart App Control. The callers wrap each algorithm in try/except, so a
    # blocked dependency degrades that one algorithm instead of killing the whole sweep.
    # Do not hoist these to module level.
    if name == "kmeans":
        return KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=SEED).fit_predict(X)
    if name == "kmedoids":
        from sklearn_extra.cluster import KMedoids
        return KMedoids(n_clusters=k, metric="euclidean", init="k-medoids++",
                        random_state=SEED).fit_predict(X)
    if name == "fuzzy_cmeans":
        import skfuzzy as fuzz
        cntr, u, *_ = fuzz.cluster.cmeans(X.T, k, 2, error=0.005, maxiter=1000, init=None, seed=SEED)
        return np.argmax(u, axis=0)
    if name == "gmm_full":
        return GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                               random_state=SEED).fit(X).predict(X)
    if name == "gmm_diag":
        return GaussianMixture(n_components=k, covariance_type="diag", n_init=10,
                               random_state=SEED).fit(X).predict(X)
    if name == "spectral":
        return SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                  n_neighbors=15, assign_labels="kmeans",
                                  random_state=SEED).fit_predict(X)
    if name == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if name == "birch":
        return Birch(n_clusters=k).fit_predict(X)
    if name == "archetypal":
        return archetypal_labels(X, k)
    raise ValueError(name)


def run_nonk(name, X):
    """Return (labels, info) where noise is label -1. Hyperparameters tuned for the
    ~1400-7000 point, 11-12 dim RobustScaler-scaled space."""
    if name == "dbscan":
        # Pick eps from the k-distance knee-ish; small sweep, keep the eps giving the
        # most points clustered with >=2 clusters.
        best = None
        for eps in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            lab = DBSCAN(eps=eps, min_samples=2 * X.shape[1]).fit_predict(X)
            nclust = len(set(lab)) - (1 if -1 in lab else 0)
            clustered = (lab != -1).mean()
            if nclust >= 2:
                score = clustered - abs(nclust - 3) * 0.01  # prefer coverage, mild pull toward interpretable few clusters
                if best is None or score > best[0]:
                    best = (score, eps, lab)
        if best is None:  # fall back to the largest eps
            lab = DBSCAN(eps=3.0, min_samples=2 * X.shape[1]).fit_predict(X)
            return lab, {"eps": 3.0, "min_samples": 2 * X.shape[1]}
        return best[2], {"eps": best[1], "min_samples": 2 * X.shape[1]}
    if name == "hdbscan":
        lab = HDBSCAN(min_cluster_size=max(50, X.shape[0] // 100),
                      min_samples=None).fit_predict(X)
        return lab, {"min_cluster_size": max(50, X.shape[0] // 100)}
    if name == "umap_hdbscan":
        import umap  # optional + SAC-blockable; see the note in run_kbased
        emb = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0,
                        random_state=SEED).fit_transform(X)
        lab = HDBSCAN(min_cluster_size=max(50, X.shape[0] // 100)).fit_predict(emb)
        return lab, {"umap_components": 5, "min_cluster_size": max(50, X.shape[0] // 100)}
    raise ValueError(name)


def nonk_metrics(X, labels):
    """Metrics on non-noise points only; report discovered clusters + noise%."""
    mask = labels != -1
    uniq = set(labels[mask]) if mask.any() else set()
    noise_pct = round(100 * float((~mask).mean()), 3)
    n_clusters = len(uniq)
    if n_clusters >= 2 and mask.sum() > n_clusters:
        m = compute_metrics(X[mask], labels[mask])
    else:
        m = {"silhouette_score": None, "davies_bouldin_index": None,
             "calinski_harabasz_index": None}
    sizes = [int((labels == c).sum()) for c in sorted(uniq)]
    return {"n_clusters_found": n_clusters, "noise_pct": noise_pct,
            "cluster_sizes": sizes, **m}


# ======================================================================================
# Sweep drivers (k-based) — mirror comprehensive_benchmark's run_approach_a/b shape
# ======================================================================================
def sweep_approach_a(df, scaler):
    """Age-stratified: per-band metrics averaged across bands (same basis as the canonical run)."""
    results = {}
    for k in K_RANGE:
        results[k] = {}
        for algo in K_BASED:
            sil, db, ch, sizes = [], [], [], []
            for band in range(1, 6):
                bdf = df[df["age_band"] == band]
                X = scaler.transform(bdf[FEATURES_11])
                try:
                    labels = run_kbased(algo, k, X)
                    m = compute_metrics(X, labels)
                    s, _ = cluster_sizes_and_balance(labels)
                    sil.append(m["silhouette_score"])
                    db.append(m["davies_bouldin_index"])
                    ch.append(m["calinski_harabasz_index"])
                    sizes.extend(s)
                except Exception as e:
                    print(f"    WARN approach A {algo} band {band} k={k}: {e}")
                    sil.append(0.0)
                    db.append(99.0)
                    ch.append(0.0)
                    sizes.append(1)
            results[k][algo] = {
                "silhouette_score": round(float(np.mean(sil)), 4),
                "davies_bouldin_index": round(float(np.mean(db)), 4),
                "calinski_harabasz_index": round(float(np.mean(ch)), 4),
                "size_balance": round(min(sizes) / max(sizes), 4) if sizes else 0.0,
            }
        ranks = composite_rank(results[k])
        for algo in K_BASED:
            results[k][algo]["composite_rank"] = ranks[algo]
        results[k]["_winner"] = min(ranks, key=ranks.get)
    return results


def sweep_approach_b(df, scaler, X_full):
    results = {}
    for k in K_RANGE:
        results[k] = {}
        for algo in K_BASED:
            try:
                labels = run_kbased(algo, k, X_full)
                m = compute_metrics(X_full, labels)
                s, bal = cluster_sizes_and_balance(labels)
                results[k][algo] = {**m, "size_balance": bal}
            except Exception as e:
                print(f"    WARN approach B {algo} k={k}: {e}")
                results[k][algo] = {"silhouette_score": 0.0, "davies_bouldin_index": 99.0,
                                    "calinski_harabasz_index": 0.0, "size_balance": 0.0}
        ranks = composite_rank(results[k])
        for algo in K_BASED:
            results[k][algo]["composite_rank"] = ranks[algo]
        results[k]["_winner"] = min(ranks, key=ranks.get)
    return results


def winner_metric_table(results):
    return {k: {
        "silhouette_score": results[k][results[k]["_winner"]]["silhouette_score"],
        "davies_bouldin_index": results[k][results[k]["_winner"]]["davies_bouldin_index"],
        "calinski_harabasz_index": results[k][results[k]["_winner"]]["calinski_harabasz_index"],
        "size_balance": results[k][results[k]["_winner"]]["size_balance"],
    } for k in K_RANGE}


# ======================================================================================
# Ideal-cluster identification + LIA benchmark-violation check (Step 4 evidence)
# ======================================================================================
def cluster_feature_means(df_subset, labels):
    tmp = df_subset[FEATURES_11].copy()
    tmp["__c"] = labels
    keep = tmp["__c"] != -1
    tmp = tmp[keep]
    return tmp.groupby("__c")[FEATURES_11].mean()


def _scaled_against(column, value, higher_better):
    """Min-max `value` against `column`'s spread. A flat column scores a neutral 0.5."""
    rng = column.max() - column.min()
    if rng <= 0:
        return 0.5
    scaled = (value - column.min()) / rng
    return scaled if higher_better else 1 - scaled


def ideal_cluster_id(prof):
    """Highest summed normalised health score (heuristic, insurance_spend excluded)."""
    abs_dev = prof["risk_age_deviation"].abs()
    scores = {}
    for cid in prof.index:
        s = 0.0
        for col in HIGHER_BETTER:
            s += _scaled_against(prof[col], prof.loc[cid, col], True)
        for col in LOWER_BETTER:
            s += _scaled_against(prof[col], prof.loc[cid, col], False)
        # risk_age_deviation is scored on distance from 0 — nearer 0 is healthier.
        s += _scaled_against(abs_dev, abs(prof.loc[cid, "risk_age_deviation"]), False)
        scores[cid] = s
    return max(scores, key=scores.get)


def benchmark_violation_check(df, scaler, X_full, approach, algo, k):
    """Return the winning approach's 'ideal' cluster life_coverage_ratio(s) and whether
    any violate the LIA 9-10x recommendation (proxy threshold < 5x, as the original used)."""
    findings = []
    if approach == "A":
        for band in range(1, 6):
            bdf = df[df["age_band"] == band]
            X = scaler.transform(bdf[FEATURES_11])
            labels = run_kbased(algo, k, X)
            prof = cluster_feature_means(bdf, labels)
            cid = ideal_cluster_id(prof)
            lc = float(prof.loc[cid, "life_coverage_ratio"])
            findings.append({"band": band, "ideal_cluster": int(cid),
                             "life_coverage_ratio": round(lc, 2),
                             "violates_lia": bool(lc < LIA_LIFE_COVER_MIN)})
    else:
        labels = run_kbased(algo, k, X_full)
        prof = cluster_feature_means(df, labels)
        cid = ideal_cluster_id(prof)
        lc = float(prof.loc[cid, "life_coverage_ratio"])
        findings.append({"band": "all", "ideal_cluster": int(cid),
                         "life_coverage_ratio": round(lc, 2),
                         "violates_lia": bool(lc < LIA_LIFE_COVER_MIN)})
    return findings


# ======================================================================================
# Plots
# ======================================================================================
def _norm(vals, higher=True):
    v = np.array(vals, float)
    mn, mx = v.min(), v.max()
    if mx == mn:
        return np.full_like(v, 0.5)
    out = (v - mn) / (mx - mn)
    return out if higher else 1 - out


def plot_k_comparison(res_a, res_b, sel_a, sel_b, rk_a, rk_b):
    ks = list(K_RANGE)
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.2)
    for col, (res, sel, rk, title) in enumerate([
        (res_a, sel_a, rk_a, "Approach A (Age-stratified)"),
        (res_b, sel_b, rk_b, "Approach B (Single model)"),
    ]):
        wm = winner_metric_table(res)
        sil = [wm[k]["silhouette_score"] for k in ks]
        db = [wm[k]["davies_bouldin_index"] for k in ks]
        ch = [wm[k]["calinski_harabasz_index"] for k in ks]
        ax = fig.add_subplot(gs[0, col])
        ax.plot(ks, _norm(sil, True), "o-", label="silhouette (norm)", color="#4C72B0")
        ax.plot(ks, _norm(db, False), "s-", label="DB inverted (norm)", color="#55A868")
        ax.plot(ks, _norm(ch, True), "^-", label="CH (norm)", color="#C44E52")
        ax.axvline(rk, color="black", ls="--", alpha=0.7)
        ax.set_title(f"Per-k winner — {title}")
        ax.set_xlabel("k")
        ax.set_xticks(ks)
        ax.set_ylabel("normalised metric")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=9)
        ax_t = fig.add_subplot(gs[1, col])
        ax_t.axis("off")
        cell = [[str(k), res[k]["_winner"], f"{sel[k]:.3f}"] for k in ks]
        t = ax_t.table(cellText=cell, colLabels=["k", "winning algorithm", "selection score"],
                       loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.4)
    fig.suptitle("12-algorithm k=2..6 selection comparison (11 continuous features)", fontsize=15)
    p = os.path.join(PLOTS_DIR, "k_comparison.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def plot_algo_heatmap(res_a, res_b):
    """Silhouette heatmap algos x k, one panel per approach — the readable way to show
    all algorithms at once (an 11-line overlay would be unreadable)."""
    ks = list(K_RANGE)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (res, title) in zip(axes, [(res_a, "Approach A"), (res_b, "Approach B")]):
        M = np.array([[res[k][a]["silhouette_score"] for k in ks] for a in K_BASED])
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_yticks(range(len(K_BASED)))
        ax.set_yticklabels(K_BASED)
        ax.set_xlabel("k")
        ax.set_title(f"Silhouette — {title}")
        for i in range(len(K_BASED)):
            for j in range(len(ks)):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Silhouette by algorithm and k (k-based methods)", fontsize=14)
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "algorithm_heatmap.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  saved {p}")


# ======================================================================================
# Main
# ======================================================================================
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    assert all(f in df.columns for f in FEATURES_11), "dataset missing an 11-feature column"
    print("=" * 80)
    print(f"12-ALGORITHM RE-BENCHMARK — {len(df)} users, {len(FEATURES_11)} continuous features")
    print(f"k-based ({len(K_BASED)}): {K_BASED}")
    print(f"non-k ({len(NON_K)}): {NON_K}")
    print("=" * 80)

    # In-memory RobustScaler on the 11 features (production 10-feature scaler NOT used).
    scaler = RobustScaler().fit(df[FEATURES_11])
    age_scaler = StandardScaler().fit(df[["age_midpoint"]])
    X_full = np.hstack([scaler.transform(df[FEATURES_11]), age_scaler.transform(df[["age_midpoint"]])])
    X_noage = scaler.transform(df[FEATURES_11])

    # ---- k-based sweep, both approaches ----
    print("\n[Approach A] age-stratified sweep...")
    res_a = sweep_approach_a(df, scaler)
    sel_a = selection_score(winner_metric_table(res_a))
    rk_a = max(sel_a, key=sel_a.get)
    algo_a = res_a[rk_a]["_winner"]

    print("[Approach B] single-model sweep...")
    res_b = sweep_approach_b(df, scaler, X_full)
    sel_b = selection_score(winner_metric_table(res_b))
    rk_b = max(sel_b, key=sel_b.get)
    algo_b = res_b[rk_b]["_winner"]

    # ---- print full metric tables ----
    for label, res in [("A (age-stratified)", res_a), ("B (single model)", res_b)]:
        print(f"\n=== Approach {label}: silhouette / DB / CH / size-balance ===")
        for k in K_RANGE:
            print(f"  k={k} (winner: {res[k]['_winner']}):")
            for a in K_BASED:
                r = res[k][a]
                print(f"    {a:<14} sil={r['silhouette_score']:>7.4f}  db={r['davies_bouldin_index']:>7.4f}"
                      f"  ch={r['calinski_harabasz_index']:>9.1f}  bal={r['size_balance']:>6.4f}"
                      f"  rank={r['composite_rank']}")
    sel_a_rounded = {k: round(sel_a[k], 4) for k in K_RANGE}
    sel_b_rounded = {k: round(sel_b[k], 4) for k in K_RANGE}
    print(f"\nApproach A: selection scores {sel_a_rounded} -> rec_k={rk_a}, algo={algo_a}")
    print(f"Approach B: selection scores {sel_b_rounded} -> rec_k={rk_b}, algo={algo_b}")

    # ---- non-k methods, reported separately (approach-B space, whole dataset) ----
    print("\n=== Non-k methods (natural cluster count; single-model feature space) ===")
    nonk_results = {}
    for name in NON_K:
        try:
            labels, info = run_nonk(name, X_full if name != "umap_hdbscan" else X_noage)
            m = nonk_metrics(X_full if name != "umap_hdbscan" else X_noage, labels)
            nonk_results[name] = {"params": info, **m, "ran": True}
            print(f"  {name:<14} clusters={m['n_clusters_found']}  noise={m['noise_pct']}%  "
                  f"sil={m['silhouette_score']}  db={m['davies_bouldin_index']}  ch={m['calinski_harabasz_index']}")
        except Exception as e:
            nonk_results[name] = {"ran": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  {name:<14} FAILED: {type(e).__name__}: {e}")

    # ---- Step 6: HDBSCAN noise check on the pure 11-feature matrix (as the original did).
    #      Swept over min_cluster_size since the discovered count/noise are sensitive to it;
    #      reporting the sensitivity honestly rather than tuning to reproduce the old number.
    print("\n=== Step 6: HDBSCAN noise check (pure 11-feature space, no age) ===")
    hdb_sweep = []
    for mcs in [30, 50, 70, 100, 150, 200]:
        lab = HDBSCAN(min_cluster_size=mcs).fit_predict(X_noage)
        nc = len(set(lab)) - (1 if -1 in lab else 0)
        npct = round(100 * float((lab == -1).mean()), 3)
        hdb_sweep.append({"min_cluster_size": mcs, "clusters_found": nc, "noise_pct": npct})
        print(f"  min_cluster_size={mcs:<4} -> {nc} clusters, {npct}% noise")
    print("  (original corroboration on the OLD data was 3 clusters / 0.13% noise)")

    # ---- winning approach + benchmark-violation evidence ----
    sil_a = res_a[rk_a][algo_a]["silhouette_score"]
    sil_b = res_b[rk_b][algo_b]["silhouette_score"]
    win_approach = "A" if sil_a >= sil_b else "B"
    win_k = rk_a if win_approach == "A" else rk_b
    win_algo = algo_a if win_approach == "A" else algo_b
    print(f"\n=== Winning approach: {win_approach} ({win_algo}, k={win_k}) — "
          f"winner silhouette A={sil_a} vs B={sil_b} ===")
    violation = benchmark_violation_check(df, scaler, X_full, win_approach, win_algo, win_k)
    print("  LIA life-coverage check on the winning approach's ideal cluster(s):")
    for f in violation:
        flag = "  <-- VIOLATES LIA (<5x)" if f["violates_lia"] else ""
        print(f"    band {f['band']}: ideal cluster {f['ideal_cluster']}, "
              f"life_coverage={f['life_coverage_ratio']}x{flag}")

    # ---- persist ----
    def res_to_json(res, sel, rk, algo):
        return {
            "results_by_k": {str(k): {"winner": res[k]["_winner"],
                                      "algorithms": {a: res[k][a] for a in K_BASED}}
                             for k in K_RANGE},
            "selection_scores_by_k": {str(k): sel[k] for k in K_RANGE},
            "recommended_k": rk, "recommended_algorithm": algo,
        }

    out = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "supersedes_note": "CANONICAL 12-algorithm / 11-continuous-feature run (July 2026). "
                           "Supersedes the prior 5-algo/10-feature run, archived under "
                           "results/archive/comprehensive_10feature_5algo/.",
        "dataset": {"n_users": int(len(df)), "n_features": len(FEATURES_11),
                    "features": FEATURES_11, "scaler": "RobustScaler (in-memory, 11-feature)"},
        "algorithm_families": FAMILY,
        "k_based_algorithms": K_BASED,
        "non_k_algorithms": NON_K,
        "seed": SEED,
        "approach_a": res_to_json(res_a, sel_a, rk_a, algo_a),
        "approach_b": res_to_json(res_b, sel_b, rk_b, algo_b),
        "non_k_results": nonk_results,
        "hdbscan_noise_check": {"space": "pure 11-feature (no age)",
                                "min_cluster_size_sweep": hdb_sweep,
                                "original_corroboration": "3 clusters / 0.13% noise (OLD data)"},
        "winning_approach": win_approach, "winning_algorithm": win_algo, "winning_k": win_k,
        "winner_silhouette_A": sil_a, "winner_silhouette_B": sil_b,
        "benchmark_violation_check": violation,
    }
    jp = os.path.join(RESULTS_DIR, "comprehensive_results.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved {jp}")

    # tidy CSV
    rows = []
    for approach, res in [("A", res_a), ("B", res_b)]:
        for k in K_RANGE:
            for a in K_BASED:
                r = res[k][a]
                rows.append({"approach": approach, "k": k, "algorithm": a, "family": FAMILY[a],
                             "silhouette": r["silhouette_score"], "davies_bouldin": r["davies_bouldin_index"],
                             "calinski_harabasz": r["calinski_harabasz_index"],
                             "size_balance": r["size_balance"], "composite_rank": r["composite_rank"],
                             "is_k_winner": a == res[k]["_winner"]})
    csvp = os.path.join(RESULTS_DIR, "algorithm_metrics.csv")
    pd.DataFrame(rows).to_csv(csvp, index=False)
    print(f"  saved {csvp}")

    plot_k_comparison(res_a, res_b, sel_a, sel_b, rk_a, rk_b)
    plot_algo_heatmap(res_a, res_b)

    print("\n" + "=" * 80)
    print(f"DONE. Winner: Approach {win_approach}, {win_algo}, k={win_k} "
          f"(selection score chose k; composite rank chose algo).")
    print("=" * 80)


if __name__ == "__main__":
    main()
