"""
k3_rerun.py — K=3 targeted rerun of the Unify cross-domain clustering benchmark.

The k=2 benchmark separated injected mismatch profiles (notably protection_first) and age
gradients rather than a financial-health spectrum. HDBSCAN independently found 3 clusters
(0.13% noise). This script reruns ONLY the algorithm comparison at k=3 and fixes the ideal-
centroid selection to a normalised composite financial-health score, enabling an
At Risk / Building / Stable tier structure.

Reuses the already-fitted scaler (transform only). Does NOT rerun Hopkins, EDA, scaling,
k search, or the dendrogram, and does NOT overwrite any existing results file.

Run standalone:
    cd <repo root>
    python MLAI/clustering/k3_rerun.py

Outputs:
    results/benchmark_results_k3.json
    results/archetype_profiles_k3.csv
    results/cluster_assignments_k3.csv
    results/plots/algorithm_comparison_k3.png
"""

import os
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn_extra.cluster import KMedoids
import skfuzzy as fuzz

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Paths (resolved relative to repo root so the script runs from any cwd)
# --------------------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
SCALER_PATH = os.path.join(REPO_ROOT, "backend/ml/artifacts/fitted_scaler.pkl")
RESULTS_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
K2_JSON = os.path.join(RESULTS_DIR, "benchmark_results.json")

K = 3
RANDOM_STATE = 42

CLUSTERING_FEATURES = [
    "savings_rate", "emergency_fund_months", "dti", "net_worth_to_income",
    "investment_rate", "portfolio_concentration", "risk_age_deviation",
    "life_coverage_ratio", "ci_coverage_ratio", "hospitalisation_coverage",
]

AGE_BANDS = [1, 2, 3, 4, 5]
BAND_LABELS = {1: "20-29", 2: "30-39", 3: "40-49", 4: "50-59", 5: "60-70"}

# Higher-is-better / lower-is-better feature groups for the composite health score.
HIGHER_BETTER = [
    "savings_rate", "emergency_fund_months", "net_worth_to_income", "investment_rate",
    "life_coverage_ratio", "ci_coverage_ratio", "hospitalisation_coverage",
]
LOWER_BETTER = ["dti", "portfolio_concentration"]
TIER_ORDER = ["At Risk", "Building", "Stable"]  # lowest -> highest composite


# ======================================================================================
# Primary algorithms (k=3)
# ======================================================================================
def run_kmeans(X, k):
    m = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RANDOM_STATE)
    return m.fit_predict(X), {}


def run_kmedoids(X, k):
    m = KMedoids(n_clusters=k, metric="euclidean", init="k-medoids++",
                 random_state=RANDOM_STATE)
    return m.fit_predict(X), {}


def run_gmm(X, k):
    m = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                        random_state=RANDOM_STATE)
    m.fit(X)
    labels = m.predict(X)  # argmax of posterior probabilities
    return labels, {"bic": round(float(m.bic(X)), 4), "aic": round(float(m.aic(X)), 4)}


def run_fuzzy(X, k):
    # cmeans returns (centers, u, u0, d, jm, p, fpc); only u and fpc are used here.
    _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        X.T, k, 2, error=0.005, maxiter=1000, init=None, seed=RANDOM_STATE
    )
    return np.argmax(u, axis=0), {"fuzzy_partition_coefficient": round(float(fpc), 4)}


PRIMARY_ALGOS = {
    "kmeans": run_kmeans,
    "kmedoids": run_kmedoids,
    "gmm": run_gmm,
    "fuzzy_cmeans": run_fuzzy,
}


def base_metrics(X, labels):
    sizes = np.bincount(labels, minlength=K).tolist()
    return {
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz_index": float(calinski_harabasz_score(X, labels)),
        "cluster_sizes": sizes,
        "size_balance": float(min(sizes) / max(sizes)),
    }


# ======================================================================================
# Composite rank (rank-sum across silhouette, DB, CH)
# ======================================================================================
def composite_rank(results):
    names = list(results.keys())
    sil = {n: results[n]["metrics"]["silhouette_score"] for n in names}
    db = {n: results[n]["metrics"]["davies_bouldin_index"] for n in names}
    ch = {n: results[n]["metrics"]["calinski_harabasz_index"] for n in names}

    def rank_map(values, higher_better):
        order = sorted(values, key=lambda n: values[n], reverse=higher_better)
        return {n: i + 1 for i, n in enumerate(order)}

    r_sil, r_db, r_ch = rank_map(sil, True), rank_map(db, False), rank_map(ch, True)
    totals = {n: r_sil[n] + r_db[n] + r_ch[n] for n in names}
    for n in names:
        results[n]["metrics"]["composite_rank"] = totals[n]
    winner = min(totals, key=lambda n: totals[n])
    return winner, {"silhouette": r_sil, "davies_bouldin": r_db,
                    "calinski_harabasz": r_ch, "total": totals}


def print_rank_table(name, ranks):
    print(f"\n  Composite ranking — {name}:")
    print(f"    {'algo':<14}{'sil_rk':>8}{'db_rk':>8}{'ch_rk':>8}{'TOTAL':>8}")
    for n in sorted(ranks["total"], key=lambda n: ranks["total"][n]):
        print(f"    {n:<14}{ranks['silhouette'][n]:>8}{ranks['davies_bouldin'][n]:>8}"
              f"{ranks['calinski_harabasz'][n]:>8}{ranks['total'][n]:>8}")


# ======================================================================================
# Ideal centroid — FIXED composite-score logic
# ======================================================================================
def _normalise(series, higher_better):
    """Min-max a per-cluster series onto 0-1. A flat series scores a neutral 0.5."""
    rng = series.max() - series.min()
    if rng <= 0:
        return 0.5
    scaled = (series - series.min()) / rng
    return scaled if higher_better else 1 - scaled


def composite_scores(archetype_df):
    """Return a per-cluster composite financial-health score (higher = healthier)."""
    scores = pd.DataFrame(index=archetype_df.index)

    for col in HIGHER_BETTER:
        scores[col] = _normalise(archetype_df[col], True)
    for col in LOWER_BETTER:
        scores[col] = _normalise(archetype_df[col], False)

    # risk_age_deviation is scored on absolute deviation from 0 — nearer 0 is healthier.
    scores["risk_age_deviation"] = _normalise(archetype_df["risk_age_deviation"].abs(), False)

    return scores.sum(axis=1)


def identify_ideal_cluster(archetype_df):
    """Cluster id with the highest composite financial-health score."""
    return composite_scores(archetype_df).idxmax()


def assign_tiers(comp_series):
    """Map the 3 clusters in comp_series to At Risk / Building / Stable by composite rank."""
    order = comp_series.sort_values().index.tolist()  # ascending -> lowest first
    return {cid: TIER_ORDER[i] for i, cid in enumerate(order)}


# ======================================================================================
# Archetype profile (means on ORIGINAL unscaled feature values)
# ======================================================================================
def archetype_df_from_labels(df, labels):
    tmp = df[CLUSTERING_FEATURES].copy()
    tmp["__cluster"] = labels
    rows = {}
    for c in sorted(pd.unique(labels)):
        sub = tmp[tmp["__cluster"] == c]
        rows[int(c)] = {f: round(float(sub[f].mean()), 4) for f in CLUSTERING_FEATURES}
    return pd.DataFrame.from_dict(rows, orient="index")[CLUSTERING_FEATURES]


# ======================================================================================
# Approach B feature matrix (10 robust-scaled + standardised age_midpoint)
# ======================================================================================
def build_approach_b_matrix(df, scaler):
    X10 = scaler.transform(df[CLUSTERING_FEATURES].values)
    age = df["age_midpoint"].values.astype(float).reshape(-1, 1)
    age_std = StandardScaler().fit_transform(age)  # new scaler, age only — Approach B only
    return np.hstack([X10, age_std])


# ======================================================================================
# Approach A — fit one algorithm per band, average within-band metrics
# ======================================================================================
def approach_a_fit(df, scaler, algo_name):
    fn = PRIMARY_ALGOS[algo_name]
    labels = pd.Series(index=df.index, dtype="int64")
    per_band_metrics, per_band_extra = {}, {}
    for b in AGE_BANDS:
        idx = df.index[df["age_band"] == b]
        Xb = scaler.transform(df.loc[idx, CLUSTERING_FEATURES].values)
        lb, extra = fn(Xb, K)
        labels.loc[idx] = [b * 10 + int(c) for c in lb]  # {band}{cluster} notation
        per_band_metrics[b] = base_metrics(Xb, lb)
        per_band_extra[b] = extra
    return labels, per_band_metrics, per_band_extra


def average_band_metrics(per_band_metrics, per_band_extra):
    bands = list(per_band_metrics.keys())
    out = {
        "silhouette_score": round(float(np.mean([per_band_metrics[b]["silhouette_score"] for b in bands])), 4),
        "davies_bouldin_index": round(float(np.mean([per_band_metrics[b]["davies_bouldin_index"] for b in bands])), 4),
        "calinski_harabasz_index": round(float(np.mean([per_band_metrics[b]["calinski_harabasz_index"] for b in bands])), 4),
        "cluster_sizes": [s for b in bands for s in per_band_metrics[b]["cluster_sizes"]],
        "size_balance": round(float(np.mean([per_band_metrics[b]["size_balance"] for b in bands])), 4),
        "metric_basis": "mean of per-band within-band scores (age-stratified model)",
    }
    # Average algo-specific extras across bands where present.
    keys = set().union(*[set(per_band_extra[b].keys()) for b in bands]) if bands else set()
    for key in keys:
        vals = [per_band_extra[b][key] for b in bands if key in per_band_extra[b]]
        if vals:
            out[key] = round(float(np.mean(vals)), 4)
    return out


def round_metrics(m):
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


# ======================================================================================
# Comparison plot
# ======================================================================================
def plot_comparison(results_a, results_b):
    algos = list(PRIMARY_ALGOS.keys())

    def collect(results):
        sil = [results[a]["metrics"]["silhouette_score"] for a in algos]
        db = [results[a]["metrics"]["davies_bouldin_index"] for a in algos]
        ch = [results[a]["metrics"]["calinski_harabasz_index"] for a in algos]
        db_inv = [1.0 / (1.0 + v) for v in db]
        ch_arr = np.array(ch)
        ch_norm = (ch_arr / ch_arr.max()).tolist() if ch_arr.max() > 0 else ch_arr.tolist()
        return sil, db_inv, ch_norm

    sil_a, dbinv_a, chn_a = collect(results_a)
    sil_b, dbinv_b, chn_b = collect(results_b)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metric_sets = [
        ("Silhouette (higher better)", sil_a, sil_b),
        ("1/(1+Davies-Bouldin) (higher better)", dbinv_a, dbinv_b),
        ("Calinski-Harabasz (normalised)", chn_a, chn_b),
    ]
    x = np.arange(len(algos))
    w = 0.38
    for ax, (title, va, vb) in zip(axes, metric_sets):
        ax.bar(x - w / 2, va, w, label="Approach A", color="#4C72B0")
        ax.bar(x + w / 2, vb, w, label="Approach B", color="#DD8452", hatch="//")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=20)
        ax.legend()
    fig.suptitle("Primary Algorithm Comparison (k=3) — Approach A vs B", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(PLOTS_DIR, "algorithm_comparison_k3.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ======================================================================================
# Main
# ======================================================================================
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("=" * 80)
    print("K=3 TARGETED RERUN — cross-domain clustering benchmark")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"  Loaded {len(df)} users; reusing fitted scaler (transform only).")

    # ---------------------------------------------------------------- Approach A
    print("\n" + "=" * 80)
    print("APPROACH A — age-stratified, k=3 per band (metrics = mean of per-band scores)")
    print("=" * 80)
    results_a, a_labels = {}, {}
    for name in PRIMARY_ALGOS:
        labels, pbm, pbe = approach_a_fit(df, scaler, name)
        metrics = average_band_metrics(pbm, pbe)
        results_a[name] = {"metrics": metrics}
        a_labels[name] = labels
        print(f"    {name:<14} sil={metrics['silhouette_score']:.4f}  "
              f"DB={metrics['davies_bouldin_index']:.4f}  "
              f"CH={metrics['calinski_harabasz_index']:.1f}  balance={metrics['size_balance']:.2f}")
    winner_a, ranks_a = composite_rank(results_a)
    print_rank_table("Approach A", ranks_a)
    print(f"  Approach A winner: {winner_a}")

    # ---------------------------------------------------------------- Approach B
    print("\n" + "=" * 80)
    print("APPROACH B — single model, k=3, age_midpoint as 11th feature")
    print("=" * 80)
    X_b = build_approach_b_matrix(df, scaler)
    results_b, b_labels = {}, {}
    for name, fn in PRIMARY_ALGOS.items():
        labels, extra = fn(X_b, K)
        metrics = round_metrics(base_metrics(X_b, labels))
        metrics.update(extra)
        results_b[name] = {"metrics": metrics}
        b_labels[name] = labels
        print(f"    {name:<14} sil={metrics['silhouette_score']:.4f}  "
              f"DB={metrics['davies_bouldin_index']:.4f}  "
              f"CH={metrics['calinski_harabasz_index']:.1f}  balance={metrics['size_balance']:.2f}")
    winner_b, ranks_b = composite_rank(results_b)
    print_rank_table("Approach B", ranks_b)
    print(f"  Approach B winner: {winner_b}")

    # ---------------------------------------------------------------- Ideal centroid + tiers
    print("\n" + "=" * 80)
    print("IDEAL CENTROID SELECTION (composite financial-health score)")
    print("=" * 80)

    # Approach A: ideal per band, then overall ideal = highest composite across all bands.
    a_winner_labels = a_labels[winner_a]
    arche_a = archetype_df_from_labels(df, a_winner_labels.values)  # index = band-cluster id
    a_tiers, a_comp_all = {}, {}
    for b in AGE_BANDS:
        band_ids = [b * 10 + c for c in range(K)]
        band_arche = arche_a.loc[band_ids]
        comp = composite_scores(band_arche)
        a_comp_all.update(comp.to_dict())
        a_tiers.update(assign_tiers(comp))
    ideal_a = int(max(a_comp_all, key=lambda c: a_comp_all[c]))
    print(f"\n  Approach A ({winner_a}) — per-band tiers, overall ideal across bands:")
    _print_archetypes(arche_a, a_tiers, ideal_a, grouped_by_band=True)
    print(f"  -> Overall ideal: cluster {ideal_a} (tier {a_tiers[ideal_a]}, "
          f"highest composite {a_comp_all[ideal_a]:.3f} across all bands)")

    # Approach B: ideal across the 3 full-dataset clusters.
    b_winner_labels = b_labels[winner_b]
    arche_b = archetype_df_from_labels(df, b_winner_labels)
    comp_b = composite_scores(arche_b)
    b_tiers = assign_tiers(comp_b)
    ideal_b = int(comp_b.idxmax())
    print(f"\n  Approach B ({winner_b}) — 3 clusters:")
    _print_archetypes(arche_b, b_tiers, ideal_b, grouped_by_band=False)
    print(f"  -> Ideal: cluster {ideal_b} (tier {b_tiers[ideal_b]})")

    print(f"\n  Ideal centroid identified: Cluster {ideal_a} for Approach A / "
          f"Cluster {ideal_b} for Approach B")

    # ---------------------------------------------------------------- archetype CSV
    arche_rows = []
    for cid in arche_a.index:
        row = {"approach": "A", "algorithm": winner_a, "cluster_id": int(cid)}
        row.update(arche_a.loc[cid].to_dict())
        arche_rows.append(row)
    for cid in arche_b.index:
        row = {"approach": "B", "algorithm": winner_b, "cluster_id": int(cid)}
        row.update(arche_b.loc[cid].to_dict())
        arche_rows.append(row)
    arche_out = pd.DataFrame(arche_rows)[
        ["approach", "algorithm", "cluster_id"] + CLUSTERING_FEATURES]
    arche_path = os.path.join(RESULTS_DIR, "archetype_profiles_k3.csv")
    arche_out.to_csv(arche_path, index=False)
    print(f"\n  Saved {arche_path}")

    # ---------------------------------------------------------------- assignments CSV
    assign = pd.DataFrame({
        "user_id": df["user_id"],
        "age_band": df["age_band"],
        "profile_type": df["profile_type"],
        "approach_a_cluster_k3": a_winner_labels.astype(int).values,
        "approach_b_cluster_k3": b_winner_labels.astype(int),
    })
    assign_path = os.path.join(RESULTS_DIR, "cluster_assignments_k3.csv")
    assign.to_csv(assign_path, index=False)
    print(f"  Saved {assign_path}")

    # ---------------------------------------------------------------- k2 vs k3 comparison
    with open(K2_JSON) as f:
        k2 = json.load(f)
    k2_a = k2["algorithm_comparison"]["approach_a"]["results"]
    k2_b = k2["algorithm_comparison"]["approach_b"]["results"]
    label = {"kmeans": "K-Means", "kmedoids": "K-Medoids",
             "gmm": "GMM", "fuzzy_cmeans": "Fuzzy C-Means"}
    print("\n" + "=" * 80)
    print("k=2 vs k=3 silhouette comparison")
    print("=" * 80)
    print(f"{'':16}{'Approach A':>26}{'Approach B':>26}")
    print(f"{'Algorithm':16}{'Silhouette (k2 -> k3)':>26}{'Silhouette (k2 -> k3)':>26}")
    for algo in PRIMARY_ALGOS:
        a2 = k2_a[algo]["silhouette_score"]
        a3 = results_a[algo]["metrics"]["silhouette_score"]
        b2 = k2_b[algo]["silhouette_score"]
        b3 = results_b[algo]["metrics"]["silhouette_score"]
        print(f"{label[algo]:16}{f'{a2:.4f}  ->  {a3:.4f}':>26}{f'{b2:.4f}  ->  {b3:.4f}':>26}")

    # ---------------------------------------------------------------- plot
    print("\n" + "=" * 80)
    print("OUTPUT")
    print("=" * 80)
    plot_comparison(results_a, results_b)

    # ---------------------------------------------------------------- recommendation
    sil_a = results_a[winner_a]["metrics"]["silhouette_score"]
    sil_b = results_b[winner_b]["metrics"]["silhouette_score"]
    if sil_a >= sil_b:
        rec_approach, rec_algo = "A", winner_a
        reasoning = (
            f"Approach A ({winner_a}) attains the higher silhouette at k=3 "
            f"({sil_a} vs Approach B {winner_b} {sil_b}). Age-stratified k=3 models give a "
            f"clean At Risk / Building / Stable tier per band, which maps directly onto the "
            f"age-band-relative Financial Health Score.")
    else:
        rec_approach, rec_algo = "B", winner_b
        reasoning = (
            f"Approach B ({winner_b}) attains the higher silhouette at k=3 "
            f"({sil_b} vs Approach A {winner_a} {sil_a}). A single age-aware k=3 model is "
            f"simpler to operate and yields one global At Risk / Building / Stable tiering.")

    # ---------------------------------------------------------------- JSON
    def arche_labels_dict(tiers):
        return {f"cluster_{cid}": tiers[cid] for cid in tiers}

    results_json = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "k": K,
        "rationale_for_k3": (
            "k=2 archetype analysis showed mismatch profile separation and age gradients "
            "rather than financial health archetypes. HDBSCAN independently found 3 clusters "
            "(0.13% noise). k=3 selected to enable At Risk / Building / Stable tier structure."),
        "algorithm_comparison": {
            "approach_a": {
                "k_used": K,
                "results": {n: results_a[n]["metrics"] for n in results_a},
                "winner": winner_a,
                "ideal_cluster_id": ideal_a,
                "ideal_cluster_label": a_tiers[ideal_a],
                "archetype_labels": arche_labels_dict(a_tiers),
            },
            "approach_b": {
                "k_used": K,
                "results": {n: results_b[n]["metrics"] for n in results_b},
                "winner": winner_b,
                "ideal_cluster_id": ideal_b,
                "ideal_cluster_label": b_tiers[ideal_b],
                "archetype_labels": arche_labels_dict(b_tiers),
            },
        },
        "k2_vs_k3_delta": {
            "approach_a_winner_silhouette_k2": k2["algorithm_comparison"]["approach_a"]["results"][
                k2["algorithm_comparison"]["approach_a"]["winner"]]["silhouette_score"],
            "approach_a_winner_silhouette_k3": sil_a,
            "approach_b_winner_silhouette_k2": k2["algorithm_comparison"]["approach_b"]["results"][
                k2["algorithm_comparison"]["approach_b"]["winner"]]["silhouette_score"],
            "approach_b_winner_silhouette_k3": sil_b,
        },
        "recommended_approach": rec_approach,
        "recommended_algorithm": rec_algo,
        "recommendation_reasoning": reasoning,
    }
    json_path = os.path.join(RESULTS_DIR, "benchmark_results_k3.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved {json_path}")

    print("\n" + "=" * 80)
    print(f"DONE. Recommended at k=3: Approach {rec_approach} with {rec_algo}")
    print(reasoning)
    print("=" * 80)


def _print_archetypes(arche, tiers, ideal_id, grouped_by_band):
    cols = CLUSTERING_FEATURES
    header = "    " + f"{'cluster':<9}{'tier':<10}" + "".join(f"{c[:11]:>13}" for c in cols)
    print(header)
    if grouped_by_band:
        for b in AGE_BANDS:
            for c in range(K):
                cid = b * 10 + c
                _print_row(arche, cid, tiers, ideal_id, cols)
            print()
    else:
        for cid in arche.index:
            _print_row(arche, cid, tiers, ideal_id, cols)


def _print_row(arche, cid, tiers, ideal_id, cols):
    marker = "  <- IDEAL" if cid == ideal_id else ""
    vals = "".join(f"{arche.loc[cid, c]:>13.3f}" for c in cols)
    print(f"    {cid:<9}{tiers.get(cid, ''):<10}{vals}{marker}")


if __name__ == "__main__":
    main()
