"""
ablation_binary_features.py — Binary-feature exclusion ablation (July 2026).

Unify (CCDS25-1170). Empirically tests the literature finding (Hennig & Liao 2013;
Foss et al. 2018) that dummy-coded binary variables can distort distance-based
clustering — against OUR OWN synthetic data, not just by citation. Compares K-Means++
(the locked algorithm) across k=2..6 under both approaches (A = age-stratified,
B = single model with age as a feature) on:

    Config A — 11 continuous-only features (the proposed Layer 2 design):
               the 9 original continuous clustering features (risk_age_deviation in
               place of raw equity_pct, as production does) + healthy_debt_ratio +
               insurance_spend_ratio (both new, July 2026)
    Config B — all 14 candidates: Config A + 3 binaries (has_investments,
               hospitalisation_coverage, income_protection_coverage), unweighted,
               RobustScaler-scaled like everything else
    Config C — not a third clustering run: an explicit RobustScaler degeneracy check
               on Config B's 3 binary columns (IQR per column; sklearn substitutes
               scale=1.0 when IQR=0, so a skewed binary is NOT divided by a tiny IQR —
               whether distortion still occurs is measured, not assumed)

has_investments is read directly from the generated column (July 2026: the generator
now draws a hurdle/mixture — INVESTOR_PROB = 0.88 per OCBC FWI 2024, non-investors get
investment_rate exactly 0.0 — so the flag is the mixture draw, not a threshold). Its
prevalence is reported against the OCBC 88% anchor.

Scalers are fitted IN-MEMORY per configuration. backend/ml/artifacts/ (including
fitted_scaler.pkl) is never read or written by this script.

Run:
    cd <repo root>
    python MLAI/clustering/ablation_binary_features.py

Output:
    results/ablation_binary_features.json  (+ console report)
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_core import (  # noqa: E402  (same metric methodology, not reimplemented)
    compute_metrics, cluster_sizes_and_balance, selection_score,
)

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results")
OUT_PATH = os.path.join(RESULTS_DIR, "ablation_binary_features.json")

CONTINUOUS_11 = [
    "savings_rate", "emergency_fund_months", "dti", "net_worth_to_income",
    "investment_rate", "portfolio_concentration", "risk_age_deviation",
    "life_coverage_ratio", "ci_coverage_ratio", "healthy_debt_ratio",
    "insurance_spend_ratio",
]
BINARY_3 = ["has_investments", "hospitalisation_coverage", "income_protection_coverage"]
MIXED_14 = CONTINUOUS_11 + BINARY_3

K_RANGE = range(2, 7)
PROD_K = 3  # the locked production k
KMEANS_KW = {"init": "k-means++", "n_init": 20, "random_state": 42}


def kmeans_labels(X, k):
    return KMeans(n_clusters=k, **KMEANS_KW).fit(X).labels_


# --------------------------------------------------------------------------------------
# Benchmark runs (mirrors comprehensive_benchmark's approach structure, K-Means only)
# --------------------------------------------------------------------------------------
def run_approach_a(df, scaler, features):
    """Age-stratified: metrics are the mean of per-band within-band scores."""
    out = {}
    for k in K_RANGE:
        band_sil, band_db, band_ch, band_sizes = [], [], [], []
        for band in range(1, 6):
            band_df = df[df["age_band"] == band]
            X = scaler.transform(band_df[features])
            labels = kmeans_labels(X, k)
            m = compute_metrics(X, labels)
            sizes, _ = cluster_sizes_and_balance(labels)
            band_sil.append(m["silhouette_score"])
            band_db.append(m["davies_bouldin_index"])
            band_ch.append(m["calinski_harabasz_index"])
            band_sizes.extend(sizes)
        out[k] = {
            "silhouette_score": round(float(np.mean(band_sil)), 4),
            "davies_bouldin_index": round(float(np.mean(band_db)), 4),
            "calinski_harabasz_index": round(float(np.mean(band_ch)), 4),
            "size_balance": round(min(band_sizes) / max(band_sizes), 4),
            "metric_basis": "mean of per-band within-band scores",
        }
    return out


def run_approach_b(df, scaler, features):
    """Single model with age: scaled features + StandardScaler-scaled age_midpoint."""
    age_scaled = StandardScaler().fit_transform(df[["age_midpoint"]])
    X_full = np.hstack([scaler.transform(df[features]), age_scaled])
    out = {}
    labels_by_k = {}
    for k in K_RANGE:
        labels = kmeans_labels(X_full, k)
        m = compute_metrics(X_full, labels)
        sizes, balance = cluster_sizes_and_balance(labels)
        out[k] = {**m, "cluster_sizes": sizes, "size_balance": balance}
        labels_by_k[k] = labels
    return out, X_full, labels_by_k


# --------------------------------------------------------------------------------------
# Config C — RobustScaler degeneracy check on binary columns
# --------------------------------------------------------------------------------------
def scaler_degeneracy_report(df, scaler, features):
    """Per-feature quartiles, IQR, the scale RobustScaler actually used, and each scaled
    column's share of total scaled variance (its leverage on squared Euclidean distance)."""
    X_scaled = scaler.transform(df[features])
    variances = X_scaled.var(axis=0)
    total_var = float(variances.sum())
    rows = {}
    for i, f in enumerate(features):
        q25, q75 = np.percentile(df[f], [25, 75])
        iqr = float(q75 - q25)
        rows[f] = {
            "q25": round(float(q25), 4),
            "q75": round(float(q75), 4),
            "iqr": round(iqr, 4),
            "iqr_is_zero": bool(iqr == 0.0),
            "robust_center": round(float(scaler.center_[i]), 4),
            "robust_scale_used": round(float(scaler.scale_[i]), 4),
            "scaled_variance": round(float(variances[i]), 4),
            "pct_of_total_scaled_variance": round(100 * float(variances[i]) / total_var, 2),
        }
    return rows


# --------------------------------------------------------------------------------------
# Composition analysis at the production k
# --------------------------------------------------------------------------------------
def binary_purity(df, labels, binaries):
    """Per-cluster mean of each binary; near-pure flag = cluster mean <0.05 or >0.95
    while the population prevalence is not itself extreme (0.2..0.8)."""
    out = {}
    for b in binaries:
        prev = float(df[b].mean())
        cluster_means = {int(c): round(float(df.loc[labels == c, b].mean()), 3)
                         for c in np.unique(labels)}
        near_pure = [c for c, m in cluster_means.items()
                     if (m < 0.05 or m > 0.95) and 0.2 <= prev <= 0.8]
        out[b] = {"population_prevalence": round(prev, 3),
                  "cluster_means": cluster_means,
                  "near_pure_clusters": near_pure}
    return out


def profile_crosstab(df, labels):
    """profile_type x cluster percentage table (row-normalised)."""
    ct = pd.crosstab(df["profile_type"], pd.Series(labels, index=df.index, name="cluster"),
                     normalize="index")
    return {pt: {str(c): round(100 * v, 1) for c, v in row.items()}
            for pt, row in ct.iterrows()}


def cluster_feature_means(df, labels, features):
    tmp = df[features].copy()
    tmp["__c"] = labels
    prof = tmp.groupby("__c")[features].mean().round(4)
    return {str(int(c)): prof.loc[c].to_dict() for c in prof.index}


def nmi_labels_vs_binaries(df, labels, binaries):
    return {b: round(float(normalized_mutual_info_score(labels, df[b])), 4)
            for b in binaries}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA_PATH)
    if "has_investments" not in df.columns:
        raise SystemExit("Dataset predates the hurdle-model generator — regenerate "
                         "synthetic_users.csv (has_investments column missing)")
    prev_hi = float(df["has_investments"].mean())

    print("=" * 78)
    print("BINARY-FEATURE ABLATION — K-Means++, k=2..6, both approaches")
    print("=" * 78)
    print(f"Dataset: {len(df)} users | has_investments from the generator's "
          f"hurdle/mixture draw -> prevalence {prev_hi:.3f} "
          f"(OCBC FWI 2024 anchor: 0.88)")
    for b in ["hospitalisation_coverage", "income_protection_coverage"]:
        print(f"  {b}: prevalence {df[b].mean():.3f}")

    results = {}
    scalers = {}
    labels_b_at_prod_k = {}

    for cfg_name, features in [("A_continuous_11", CONTINUOUS_11), ("B_mixed_14", MIXED_14)]:
        print(f"\n--- Config {cfg_name} ({len(features)} features) ---")
        scaler = RobustScaler().fit(df[features])
        scalers[cfg_name] = scaler

        res_a = run_approach_a(df, scaler, features)
        res_b, X_full, labels_by_k = run_approach_b(df, scaler, features)
        labels_b_at_prod_k[cfg_name] = labels_by_k[PROD_K]

        sel_a = selection_score(res_a)
        sel_b = selection_score(res_b)

        for label, res, sel in [("A (age-stratified)", res_a, sel_a),
                                ("B (single model)", res_b, sel_b)]:
            print(f"  Approach {label}:")
            print(f"    {'k':<4}{'silhouette':>12}{'davies-bouldin':>16}"
                  f"{'calinski-h':>12}{'balance':>10}{'sel.score':>11}")
            for k in K_RANGE:
                r = res[k]
                print(f"    {k:<4}{r['silhouette_score']:>12.4f}"
                      f"{r['davies_bouldin_index']:>16.4f}"
                      f"{r['calinski_harabasz_index']:>12.1f}"
                      f"{r['size_balance']:>10.4f}{sel[k]:>11.4f}")

        # Composition at the production k (approach B labels for the cross-tab; approach A
        # per-band binary purity as well, since production is age-stratified).
        comp = {
            "approach_B_k3": {
                "cluster_sizes": res_b[PROD_K]["cluster_sizes"],
                "cluster_feature_means": cluster_feature_means(df, labels_by_k[PROD_K],
                                                               features),
                "profile_type_crosstab_pct": profile_crosstab(df, labels_by_k[PROD_K]),
                "binary_purity": binary_purity(df, labels_by_k[PROD_K], BINARY_3),
                "nmi_labels_vs_binaries": nmi_labels_vs_binaries(df, labels_by_k[PROD_K],
                                                                 BINARY_3),
            },
            "approach_A_k3_binary_purity_by_band": {},
        }
        for band in range(1, 6):
            band_df = df[df["age_band"] == band]
            X = scaler.transform(band_df[features])
            labels = kmeans_labels(X, PROD_K)
            comp["approach_A_k3_binary_purity_by_band"][str(band)] = binary_purity(
                band_df, labels, BINARY_3)

        results[cfg_name] = {
            "features": features,
            "approach_A": {str(k): res_a[k] for k in K_RANGE},
            "approach_A_selection_scores": {str(k): sel_a[k] for k in K_RANGE},
            "approach_B": {str(k): res_b[k] for k in K_RANGE},
            "approach_B_selection_scores": {str(k): sel_b[k] for k in K_RANGE},
            "composition_at_k3": comp,
        }

    # How much does adding the binaries actually change the partition?
    ari = float(adjusted_rand_score(labels_b_at_prod_k["A_continuous_11"],
                                    labels_b_at_prod_k["B_mixed_14"]))
    print(f"\nPartition agreement (approach B, k={PROD_K}) between configs A and B: "
          f"ARI = {ari:.4f}")

    # Config C — scaler degeneracy check on the mixed-14 scaler.
    print("\n--- Config C: RobustScaler degeneracy check (on Config B's scaler) ---")
    deg = scaler_degeneracy_report(df, scalers["B_mixed_14"], MIXED_14)
    print(f"  {'feature':<28}{'q25':>8}{'q75':>8}{'IQR':>8}{'scale used':>12}"
          f"{'scaled var':>12}{'% tot var':>10}")
    for f in MIXED_14:
        d = deg[f]
        zero_note = "  <-- IQR=0, sklearn fallback scale=1.0" if d["iqr_is_zero"] else ""
        print(f"  {f:<28}{d['q25']:>8.3f}{d['q75']:>8.3f}{d['iqr']:>8.3f}"
              f"{d['robust_scale_used']:>12.3f}{d['scaled_variance']:>12.4f}"
              f"{d['pct_of_total_scaled_variance']:>9.2f}%{zero_note}")
    binary_var_share = sum(deg[b]["pct_of_total_scaled_variance"] for b in BINARY_3)
    print(f"  Binary features' combined share of total scaled variance: "
          f"{binary_var_share:.2f}% (proportionate share would be 3/14 = 21.4%)")

    out = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_users": int(len(df)),
            "source": "synthetic_users.csv regenerated July 2026 with healthy_debt_ratio, "
                      "insurance_spend_ratio, income_protection_coverage added; consolidated "
                      "citation-fix regeneration (investment_rate hurdle/mixture, "
                      "net_worth_to_income revision, 100-minus-age standardization)",
            "has_investments_derivation": "generated column (hurdle/mixture, "
                                          "INVESTOR_PROB=0.88; no threshold)",
            "binary_prevalences": {b: round(float(df[b].mean()), 4) for b in BINARY_3},
        },
        "methodology": {
            "algorithm": "KMeans(init='k-means++', n_init=20, random_state=42)",
            "k_range": list(K_RANGE),
            "production_k": PROD_K,
            "scaling": "RobustScaler fitted in-memory per configuration (artifacts untouched)",
            "metrics": "same compute_metrics/selection_score as comprehensive_benchmark.py",
        },
        "configs": results,
        "partition_ari_configA_vs_configB_at_k3": round(ari, 4),
        "scaler_degeneracy_check": {
            "per_feature": deg,
            "binary_combined_pct_of_scaled_variance": round(binary_var_share, 2),
            "proportionate_share_pct": round(100 * 3 / 14, 2),
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
