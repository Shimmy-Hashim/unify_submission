"""
validate_clusters.py — Stage 2: Validates K-Means clusters (age 21–35) against the absolute
benchmarks derived from the full SCF 2022 dataset. Assigns tier labels (High Performer /
Stable / At Risk) based on how each cluster centroid scores against population percentile
thresholds.
"""

import json
import os
import pickle

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES_PATH = "MLAI/clustering/data/processed/features.csv"
BENCHMARKS_PATH = "MLAI/clustering/results/benchmarks.json"
SCALER_PATH = "MLAI/clustering/data/processed/scaler.pkl"
OUTPUT_PATH = "MLAI/clustering/results/cluster_validation.json"
ANALYSIS_PATH = "MLAI/clustering/results/cluster_count_analysis.json"

N_CLUSTERS = 3
K_RANGE = range(2, 6)

# Ranked best-to-worst; a cluster's tier is its rank in this list.
TIERS = ["High Performer", "Stable", "At Risk"]

FEATURE_COLS = [
    "savings_rate",
    "emergency_fund_ratio",
    "debt_to_income",
    "expense_to_income",
    "healthy_debt_ratio",
]

SG_BENCHMARKS = {
    "savings_rate": 0.20,
    "emergency_fund_ratio": 3.0,
    "debt_to_income": 0.36,
}


def score_centroid(centroid, thresholds):
    score = 0.0
    # Higher savings_rate = better
    if centroid["savings_rate"] >= thresholds["savings_rate"]["p66"]:
        score += 1
    elif centroid["savings_rate"] >= thresholds["savings_rate"]["p33"]:
        score += 0.5
    # Higher emergency_fund_ratio = better
    if centroid["emergency_fund_ratio"] >= thresholds["emergency_fund_ratio"]["p66"]:
        score += 1
    elif centroid["emergency_fund_ratio"] >= thresholds["emergency_fund_ratio"]["p33"]:
        score += 0.5
    # Lower debt_to_income = better
    if centroid["debt_to_income"] <= thresholds["debt_to_income"]["p33"]:
        score += 1
    elif centroid["debt_to_income"] <= thresholds["debt_to_income"]["p66"]:
        score += 0.5
    # Lower expense_to_income = better
    if centroid["expense_to_income"] <= thresholds["expense_to_income"]["p33"]:
        score += 1
    elif centroid["expense_to_income"] <= thresholds["expense_to_income"]["p66"]:
        score += 0.5
    # Higher healthy_debt_ratio = better
    if centroid["healthy_debt_ratio"] >= thresholds["healthy_debt_ratio"]["p66"]:
        score += 1
    elif centroid["healthy_debt_ratio"] >= thresholds["healthy_debt_ratio"]["p33"]:
        score += 0.5
    return score


def validate_clusters():
    # Load inputs
    print("Loading features ...")
    features_df = pd.read_csv(FEATURES_PATH)

    print("Loading benchmarks ...")
    with open(BENCHMARKS_PATH) as f:
        benchmarks = json.load(f)
    thresholds = benchmarks["thresholds"]

    print("Loading scaler ...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Re-fit K-Means (must match benchmark.py: n_clusters=3, random_state=42)
    print("Fitting K-Means (n=3, random_state=42) ...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
    labels = km.fit_predict(features_df[FEATURE_COLS].values)
    centroids_normalised = km.cluster_centers_

    # Inverse transform centroids to original (ratio) scale
    centroids_original = scaler.inverse_transform(centroids_normalised)

    # Build centroid dicts and compute cluster counts
    centroid_dicts = []
    cluster_counts = []
    scores = []

    for i in range(N_CLUSTERS):
        count = int((labels == i).sum())
        cluster_counts.append(count)
        c = {feat: float(centroids_original[i, j]) for j, feat in enumerate(FEATURE_COLS)}
        centroid_dicts.append(c)
        scores.append(score_centroid(c, thresholds))

    # Rank all clusters: primary key = score, tie-break = savings_rate (both descending)
    ranked = sorted(
        range(N_CLUSTERS),
        key=lambda i: (scores[i], centroid_dicts[i]["savings_rate"]),
        reverse=True,
    )
    tier_map = {cluster_id: TIERS[rank] for rank, cluster_id in enumerate(ranked)}

    # Check SG benchmark warnings per cluster
    cluster_records = []
    for i in range(N_CLUSTERS):
        c = centroid_dicts[i]
        tier = tier_map[i]
        warnings = []
        if tier == "High Performer":
            if c["savings_rate"] < SG_BENCHMARKS["savings_rate"]:
                warnings.append(
                    f"'High Performer' cluster savings_rate ({c['savings_rate']:.4f}) is below "
                    f"SG benchmark ({SG_BENCHMARKS['savings_rate']})"
                )
            if c["emergency_fund_ratio"] < SG_BENCHMARKS["emergency_fund_ratio"]:
                warnings.append(
                    f"'High Performer' cluster emergency_fund_ratio ({c['emergency_fund_ratio']:.4f}) is below "
                    f"SG benchmark ({SG_BENCHMARKS['emergency_fund_ratio']} months)"
                )
        cluster_records.append(
            {
                "cluster_id": i,
                "tier": tier,
                "score": scores[i],
                "centroid": {k: round(v, 4) for k, v in c.items()},
                "count": cluster_counts[i],
                "warnings": warnings,
            }
        )

    # Sort display by tier order
    tier_order = {"High Performer": 0, "Stable": 1, "At Risk": 2}
    cluster_records_sorted = sorted(cluster_records, key=lambda r: tier_order[r["tier"]])

    # Print validation table
    sep = "=" * 85
    print(f"\n{sep}")
    print(f"Cluster Validation — K-Means on SCF 2022 (Age 21–35, n={len(features_df)})")
    print(f"Benchmarks derived from full SCF 2022 dataset ({benchmarks['source']})")
    print(sep)
    header = f"{'Cluster':<9} {'Tier':<12} {'savings_rate':>12} {'emerg_fund':>11} {'debt_to_inc':>12} {'score':>7} {'Count':>7}"
    print(header)
    print("-" * 85)
    for r in cluster_records_sorted:
        c = r["centroid"]
        print(
            f"{r['cluster_id']:<9} {r['tier']:<12} {c['savings_rate']:>12.4f} "
            f"{c['emergency_fund_ratio']:>11.4f} {c['debt_to_income']:>12.4f} "
            f"{r['score']:>7.1f} {r['count']:>7}"
        )
    print("-" * 85)
    print(
        f"{'SG Benchmark':<22} "
        f"{SG_BENCHMARKS['savings_rate']:>12.2f} "
        f"{SG_BENCHMARKS['emergency_fund_ratio']:>11.1f} "
        f"{SG_BENCHMARKS['debt_to_income']:>12.2f}"
    )
    print(sep)

    # Print warnings
    all_warnings = [w for r in cluster_records for w in r["warnings"]]
    if all_warnings:
        print("WARNINGS:")
        for w in all_warnings:
            print(f"  WARNING: {w}")
    else:
        print("No SG benchmark warnings for the 'High Performer' cluster.")
    print(sep)

    # Save output
    output = {
        "model": "KMeans",
        "n_clusters": N_CLUSTERS,
        "dataset": f"SCF 2022 age 21-35 (n={len(features_df)})",
        "benchmarks_source": benchmarks["source"],
        "clusters": cluster_records,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved cluster validation to: {OUTPUT_PATH}")

    # --- Cluster count analysis (k = 2..5) ---
    cluster_count_analysis(features_df[FEATURE_COLS].values, len(features_df))


def cluster_count_analysis(X, n_samples):
    print("\n" + "=" * 55)
    print("Cluster Count Analysis (K-Means, k=2..5)")
    print("=" * 55)
    print(f"{'k':<5} {'silhouette':>12}  {'cluster sizes'}")
    print("-" * 55)

    results = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        sil = float(round(silhouette_score(X, labels), 4))
        sizes = sorted([int((labels == i).sum()) for i in range(k)], reverse=True)
        print(f"{k:<5} {sil:>12.4f}  {sizes}")
        results.append({"k": k, "silhouette_score": sil, "cluster_sizes": sizes})
    print("=" * 55)

    output = {
        "dataset": f"SCF 2022 age 21-35 (n={n_samples})",
        "model": "KMeans",
        "random_state": 42,
        "results": results,
    }
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved cluster count analysis to: {ANALYSIS_PATH}")


if __name__ == "__main__":
    validate_clusters()
