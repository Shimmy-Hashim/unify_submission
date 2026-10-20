"""
build_production_artifacts.py — refit the three LOCKED domain-split winners on the current
canonical dataset and export them as the production artifact set (Unify, CCDS25-1170,
July 2026).

WHAT THIS PRODUCES
  backend/ml/artifacts/domain_split/ — the NEW production artifact set for the domain-split
  (three-model) Layer 2 architecture. It is deliberately a separate directory from the old
  single-combined-model artifacts that sit directly in backend/ml/artifacts/
  (fitted_scaler.pkl, synthetic_ideal_centroids.json, archetype_profiles_k3.csv), which
  belong to a superseded architecture and are left untouched by this script.

THE THREE LOCKED WINNERS (brief Section 5, "FINAL ALGORITHM/TIER DECISIONS — LOCKED";
standalone record at results/domain_split/FINAL_TIER_DECISIONS.md):
  Banking     A|kmeans|2         K-Means++,     k=2, Approach A (age-stratified)  -> 2 tiers
  Investment  A|fuzzy_cmeans|2   Fuzzy C-Means, k=2, Approach A (age-stratified)  -> 2 tiers
  Insurance   B|kmedoids|3       K-Medoids,     k=3, Approach B (single model)    -> 3 native
                                 clusters, health-ranks 1 and 2 merged post-hoc  -> 2 tiers

METHODOLOGY IS IMPORTED, NOT REIMPLEMENTED. The scaler construction, algorithm
hyperparameters, health-ordering rule and metric definitions all come from the scripts that
produced the validated results (domain_split_clustering.py -> comprehensive_benchmark_11algo.py
/ benchmark_core.py / final_selection.py, and risk_age_diagnostic.install_bench for the
Investment domain's canonical V2 benchmark set). The point of this script is to REPRODUCE the
validated fit and serialize it — not to re-derive it a second, subtly different way.

REPRODUCTION GATE. Every winner's recorded metrics (macro/micro silhouette, Davies-Bouldin,
Calinski-Harabasz, smallest-cluster share, pooled tier shares, ordering rho, normal-population
floor) are re-checked against the values in results/domain_split/domain_split_grid.csv and
FINAL_TIER_DECISIONS.md. If any check fails the script writes NOTHING and exits non-zero —
per the task's "stop and report the discrepancy" instruction.

SCOPE. Artifacts only. This script does not wire anything into scoring.py, does not build the
Phase 3 inference endpoint, and defines no Layer 3 combination rules.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/build_production_artifacts.py
    backend/venv/Scripts/python.exe MLAI/clustering/build_production_artifacts.py --check-only
"""

import os
import sys
import json
import pickle
import argparse
import hashlib
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import (  # noqa: E402
    DATA_PATH, SPECS, MIN_CLUSTER_SHARE, cluster_health_order, meets_mask, health_signal,
)
from comprehensive_benchmark_11algo import SEED  # noqa: E402
from benchmark_core import compute_metrics  # noqa: E402
from final_selection import macro_silhouette, cluster_shares  # noqa: E402
from risk_age_diagnostic import install_bench, NORMAL_FLOOR  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "backend/ml/artifacts/domain_split")
GRID_CSV = os.path.join(REPO_ROOT,
                        "MLAI/clustering/results/domain_split/domain_split_grid.csv")

ARTIFACT_SCHEMA_VERSION = "1.0.0"

AGE_RANGES = {1: "20-29", 2: "30-39", 3: "40-49", 4: "50-59", 5: "60-70"}

# ======================================================================================
# The locked winners, and the values each must reproduce.
# ======================================================================================
# `expect_*` values are transcribed from the two committed evidence files, which are the
# record of what was actually validated:
#   grid metrics  -> results/domain_split/domain_split_grid.csv (the four clustering metrics
#                    + smallest-cluster share, as re-read from the CSV at runtime)
#   tier shares / ordering rho / normal floor
#                 -> results/domain_split/FINAL_TIER_DECISIONS.md sections 1, 3-5
# `tier_labels` are the FINAL user-facing wording, locked in the brief (Section 5, July 2026)
# for all three domains. They are listed healthiest-first and are attached by HEALTH RANK, not
# by cluster id — see health_order()/build_domain(); a cluster id means different things in
# different age bands, so the label must never be pinned to an id.
WINNERS = {
    "banking": {
        "spec": "banking_core4",
        "combination": "A|kmeans|2",
        "approach": "A",
        "algorithm": "kmeans",
        "algorithm_display": "K-Means++",
        "k": 2,
        "shipped_tiers": 2,
        "merged": False,
        "tier_labels": ["Stable", "Building"],
        "expect_pooled_shares_pct": [47.9, 52.1],
        "expect_min_rho": 1.000,
        "expect_normal_floor_pct": 40.2,          # worst age band (Approach A)
        "normal_floor_level": "worst_band",
    },
    "investment": {
        "spec": "investment_core3",
        "combination": "A|fuzzy_cmeans|2",
        "approach": "A",
        "algorithm": "fuzzy_cmeans",
        "algorithm_display": "Fuzzy C-Means",
        "k": 2,
        "shipped_tiers": 2,
        "merged": False,
        "tier_labels": ["On Track", "Getting Started"],
        "expect_pooled_shares_pct": [38.1, 61.9],
        "expect_min_rho": 1.000,
        "expect_normal_floor_pct": 29.8,          # worst age band (Approach A)
        "normal_floor_level": "worst_band",
    },
    "insurance": {
        "spec": "insurance_core3",
        "combination": "B|kmedoids|3",
        "approach": "B",
        "algorithm": "kmedoids",
        "algorithm_display": "K-Medoids",
        "k": 3,
        "shipped_tiers": 2,
        "merged": True,
        "merge_health_ranks": [1, 2],             # collapse 2nd + 3rd healthiest
        "tier_labels": ["Covered", "Not Covered"],
        "expect_native_shares_pct": [33.7, 37.7, 28.6],
        "expect_pooled_shares_pct": [33.7, 66.3],  # after the merge
        "expect_min_rho": 1.000,                   # merged system
        "expect_normal_floor_pct": 31.9,           # pooled (Approach B)
        "normal_floor_level": "pooled",
    },
}

# Tolerances. These are drift detectors, not fudge factors: silhouette/DB are reported to 4dp
# in the CSV, CH to 4dp but on a ~1e3 scale, shares to 1dp in the markdown, so each tolerance
# sits just under the reported precision of the value it guards.
#
# ⚠️ The refit is deterministic to ~1e-15, NOT to the byte. Banking's KMeans (n_init=20)
# produces centroid coordinates that differ in the last 1-2 ULP between consecutive runs of
# this script — same seed, same data, same library versions — because threaded BLAS
# reductions do not fix summation order. Two runs back to back disagree with each other, and
# disagree differently each time. Investment (Fuzzy C-Means) and Insurance (K-Medoids, whose
# centres are real observed users) reproduce byte-identically.
#
# So do NOT treat a changed banking_model.json hash as evidence the model drifted: the
# manifest's `files` block verifies integrity AT REST, not reproducibility. What establishes
# that the refit reproduced the validated model is the reproduction gate below — every
# metric agrees to 4dp, and the drift is ~13 orders of magnitude smaller than the closest
# user's distance to a cluster boundary, so no user's tier can move.
TOL_SILHOUETTE = 5e-4
TOL_DB = 5e-4
TOL_CH_REL = 1e-4
TOL_SHARE_PP = 0.06
TOL_RHO = 1e-3
TOL_FLOOR_PP = 0.06


# ======================================================================================
# Fitting — one function per domain, each returning the fitted objects plus real centroids
# ======================================================================================
def build_scalers(df, feats, approach):
    """Exactly domain_split_clustering.main()'s construction: a RobustScaler fitted on the
    WHOLE population for the domain's features, plus (Approach B only) a StandardScaler on
    age_midpoint, which enters the model as one extra input dimension."""
    scaler = RobustScaler().fit(df[feats])
    age_scaler = StandardScaler().fit(df[["age_midpoint"]]) if approach == "B" else None
    return scaler, age_scaler


def design_matrix(df, feats, scaler, age_scaler):
    X = scaler.transform(df[feats])
    if age_scaler is not None:
        X = np.hstack([X, age_scaler.transform(df[["age_midpoint"]])])
    return X


def fit_kmeans(X, k):
    """Matches run_kbased('kmeans', ...) exactly; .fit() rather than .fit_predict() so the
    real cluster_centers_ come out with the labels."""
    m = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=SEED).fit(X)
    return m.labels_, m.cluster_centers_


def fit_kmedoids(X, k):
    """Matches run_kbased('kmedoids', ...). cluster_centers_ are the MEDOIDS — real observed
    users, not synthetic means, which is exactly the robustness property K-Medoids is here
    for."""
    from sklearn_extra.cluster import KMedoids
    m = KMedoids(n_clusters=k, metric="euclidean", init="k-medoids++",
                 random_state=SEED).fit(X)
    return m.labels_, m.cluster_centers_


def fit_fuzzy_cmeans(X, k):
    """Matches run_kbased('fuzzy_cmeans', ...). skfuzzy returns the fuzzy centres directly;
    hard labels are argmax of the membership matrix. For m=2 the membership is a strictly
    decreasing function of distance to each centre, so argmax(u) == argmin(distance) —
    asserted below rather than assumed, because production inference will assign by nearest
    centroid and must reproduce these labels exactly."""
    import skfuzzy as fuzz
    cntr, u, *_ = fuzz.cluster.cmeans(X.T, k, 2, error=0.005, maxiter=1000, init=None,
                                      seed=SEED)
    return np.argmax(u, axis=0), np.asarray(cntr)


def nearest_centroid(X, centroids):
    d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(d, axis=1)


# ======================================================================================
# Metrics, reproduced the way the validation run aggregated them
# ======================================================================================
def metrics_for(X, labels, k):
    m = compute_metrics(X, labels)
    mac, per_cluster = macro_silhouette(X, labels)
    shares, n_found = cluster_shares(labels, k)
    ids, counts = np.unique(labels, return_counts=True)
    return {
        "silhouette_micro": m["silhouette_score"],
        "silhouette_macro": round(float(mac), 4),
        "davies_bouldin": m["davies_bouldin_index"],
        "calinski_harabasz": m["calinski_harabasz_index"],
        "n_clusters_found": n_found,
        # NOTE the two are different views: `cluster_shares_pct_sorted` is the validation
        # run's own field (final_selection.cluster_shares returns shares SORTED ascending,
        # not keyed by cluster id) and is kept so the numbers line up with the committed
        # grid; `share_pct_by_cluster_id` is the one a consumer actually wants.
        "cluster_shares_pct_sorted": [round(100 * s, 2) for s in shares],
        "share_pct_by_cluster_id": {str(int(c)): round(100 * n / len(labels), 2)
                                    for c, n in zip(ids, counts)},
        "min_cluster_pct": round(100 * min(shares), 2),
        "per_cluster_silhouette": {str(c): round(v, 4) for c, v in per_cluster.items()},
    }


def aggregate_band_metrics(per_band):
    """Approach A reports the MEAN across the five per-band fits, and the WORST single band's
    smallest cluster — identical to run_spec_grid()."""
    keys = ("silhouette_micro", "silhouette_macro", "davies_bouldin", "calinski_harabasz")
    agg = {k: round(float(np.mean([per_band[b][k] for b in per_band])), 4) for k in keys}
    agg["min_cluster_pct"] = round(min(per_band[b]["min_cluster_pct"] for b in per_band), 2)
    agg["worst_band"] = int(min(per_band, key=lambda b: per_band[b]["min_cluster_pct"]))
    return agg


# ======================================================================================
# Tier ordering + verification
# ======================================================================================
def health_order(sub, labels, domain):
    """Rank this partition's clusters healthiest-first using the SAME composite health ranking
    used throughout validation (cluster_health_order: for each out-of-model benchmark variable,
    rank the clusters on the variable's health signal, sum the ranks, lowest total is
    healthiest). Labels are attached by the caller, from the rank — never from a cluster id."""
    order, _per_var = cluster_health_order(sub, labels, domain)
    return [int(c) for c in order]


def ordering_rho(sub, labels, domain, order):
    """Spearman rho between each benchmark variable's per-cluster health-signal mean and the
    health ranking — the `minRho` column of FINAL_TIER_DECISIONS.md. rho = 1.0 means the tier
    ladder never contradicts that variable."""
    hr = {c: i for i, c in enumerate(order)}
    out = {}
    for var in ds.BENCH_VARS[domain]:
        sig = health_signal(sub, domain, var)
        vals = [float(sig[labels == c].mean()) for c in order]
        ranks = [-hr[c] for c in order]
        out[var] = (None if len(set(vals)) < 2
                    else round(float(stats.spearmanr(vals, ranks).statistic), 3))
    vals = [v for v in out.values() if v is not None]
    return out, (min(vals) if vals else None)


def benchmark_profile(sub, labels, domain, order):
    """Per-tier mean and % clearing each out-of-model benchmark — the evidence a tier label
    means something, carried into the artifact so it is auditable without re-running."""
    prof = {}
    for rank, c in enumerate(order):
        cl = sub[labels == c]
        prof[rank] = {
            "n": int(len(cl)),
            "share_pct": round(100 * len(cl) / len(sub), 2),
            "benchmarks": {
                var: {"mean": round(float(cl[var].mean()), 4),
                      "pct_clearing": round(100 * float(meets_mask(cl, domain, var).mean()), 1)}
                for var in ds.BENCH_VARS[domain]
            },
        }
    return prof


# ======================================================================================
# Build
# ======================================================================================
def build_domain(df, domain, cfg, report):
    feats = SPECS[cfg["spec"]][1]
    approach, k = cfg["approach"], cfg["k"]
    scaler, age_scaler = build_scalers(df, feats, approach)
    fitter = {"kmeans": fit_kmeans, "kmedoids": fit_kmedoids,
              "fuzzy_cmeans": fit_fuzzy_cmeans}[cfg["algorithm"]]

    art = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "domain": domain,
        "combination": cfg["combination"],
        "algorithm": cfg["algorithm"],
        "algorithm_display": cfg["algorithm_display"],
        "approach": approach,
        "approach_description": ("A — age-stratified: one independent model per age band"
                                 if approach == "A" else
                                 "B — single model fitted on the pooled population, with "
                                 "standardized age_midpoint as an extra input dimension"),
        "k_native": k,
        "shipped_tiers": cfg["shipped_tiers"],
        "clustering_features": feats,
        "random_seed": SEED,
        "scaler": {
            "type": "RobustScaler",
            "fitted_on": "all 7,000 synthetic users (not per band) — matches validation",
            "features": feats,
            "center_": [float(x) for x in scaler.center_],
            "scale_": [float(x) for x in scaler.scale_],
        },
        "assignment_rule": ("nearest centroid (Euclidean) in the scaled space, within the "
                            "user's own age band" if approach == "A" else
                            "nearest medoid (Euclidean) in the scaled space, including the "
                            "standardized age dimension"),
    }
    if age_scaler is not None:
        art["age_scaler"] = {
            "type": "StandardScaler",
            "feature": "age_midpoint",
            "note": "appended as the final input dimension, AFTER the RobustScaler columns",
            "mean_": [float(x) for x in age_scaler.mean_],
            "scale_": [float(x) for x in age_scaler.scale_],
        }

    # ---------------------------------------------------------------- fit + real centroids
    if approach == "A":
        per_band_metrics, bands_out, all_labels = {}, {}, pd.Series(index=df.index, dtype=int)
        for b in range(1, 6):
            sub = df[df["age_band"] == b]
            X = design_matrix(sub, feats, scaler, None)
            labels, cent = fitter(X, k)
            # Production will assign by nearest centroid; the fitted labels must agree.
            assert (nearest_centroid(X, cent) == labels).all(), (
                f"{domain} band {b}: nearest-centroid assignment does not reproduce the "
                f"fitted labels — the artifact would not be faithful to the validated model")
            per_band_metrics[b] = metrics_for(X, labels, k)
            order = health_order(sub, labels, domain)
            rhos, min_rho = ordering_rho(sub, labels, domain, order)
            bands_out[str(b)] = {
                "age_range": AGE_RANGES[b],
                "n_users": int(len(sub)),
                "centroids_scaled": {str(c): [float(x) for x in cent[c]] for c in range(k)},
                "centroids_unscaled": {
                    str(c): dict(zip(feats,
                                     [float(x) for x in scaler.inverse_transform(
                                         cent[c].reshape(1, -1))[0]]))
                    for c in range(k)},
                "tier_map": {str(c): {"health_rank": order.index(c),
                                      "tier_label": cfg["tier_labels"][order.index(c)]}
                             for c in range(k)},
                "health_order_best_to_worst": order,
                "spearman_rho_vs_health_order": rhos,
                "min_spearman_rho": min_rho,
                "metrics": per_band_metrics[b],
                "tier_benchmark_profile": benchmark_profile(sub, labels, domain, order),
            }
            all_labels.loc[sub.index] = [order.index(int(c)) for c in labels]
        art["bands"] = bands_out
        art["metrics"] = aggregate_band_metrics(per_band_metrics)
        art["metrics"]["aggregation"] = ("mean across the 5 per-band fits; min_cluster_pct is "
                                         "the worst single band's smallest cluster")
        min_rho_overall = min(bands_out[b]["min_spearman_rho"] for b in bands_out)
    else:
        X = design_matrix(df, feats, scaler, age_scaler)
        labels, cent = fitter(X, k)
        assert (nearest_centroid(X, cent) == labels).all(), (
            f"{domain}: nearest-medoid assignment does not reproduce the fitted labels")
        native_metrics = metrics_for(X, labels, k)
        order = health_order(df, labels, domain)
        native_dims = feats + ["age_midpoint__standardized"]
        art["model"] = {
            "n_users": int(len(df)),
            "input_dimensions": native_dims,
            "centroids_scaled": {str(c): [float(x) for x in cent[c]] for c in range(k)},
            "centroids_unscaled": {
                str(c): dict(zip(
                    feats + ["age_midpoint"],
                    [float(x) for x in scaler.inverse_transform(
                        cent[c][:len(feats)].reshape(1, -1))[0]]
                    + [float(age_scaler.inverse_transform(
                        cent[c][len(feats):].reshape(1, -1))[0][0])]))
                for c in range(k)},
            "native_health_order_best_to_worst": order,
            "native_metrics": native_metrics,
            "native_tier_benchmark_profile": benchmark_profile(df, labels, domain, order),
        }
        # ------------------------------------------------------------------- the merge
        if cfg["merged"]:
            mr = cfg["merge_health_ranks"]
            merged_native = [order[r] for r in mr]
            kept_native = [c for c in order if c not in merged_native]
            shipped = {}
            for c in kept_native:
                shipped[str(c)] = {"shipped_tier": 0, "tier_label": cfg["tier_labels"][0]}
            for c in merged_native:
                shipped[str(c)] = {"shipped_tier": 1, "tier_label": cfg["tier_labels"][1]}
            mlab = np.array([shipped[str(int(c))]["shipped_tier"] for c in labels])
            morder = health_order(df, mlab, domain)
            m_rhos, m_min_rho = ordering_rho(df, mlab, domain, morder)
            art["tier_merge"] = {
                "applied": True,
                "why": ("The native 3-way split is even by size and tops the domain on "
                        "macro-silhouette, but its 2nd- and 3rd-healthiest clusters are not "
                        "materially different in insurance coverage (0.046 IQR separation on "
                        "life_coverage_ratio, 0.064 on ci_coverage_ratio; CI inverts on the "
                        "mean, life on density). The real structure is 'adequately covered' "
                        "vs 'not'. Merging keeps the validated winner's actual partition "
                        "rather than switching to a k=2 model — no k=2 candidate reproduces "
                        "this 33.7/66.3 split, and the one flagged alternative "
                        "(B|gmm_full|2) fails the project's own 5% normal-population floor "
                        "at 3.4%."),
                "merged_health_ranks": mr,
                "native_cluster_ids_merged": merged_native,
                "native_cluster_id_kept": kept_native,
                "resulting_label_for_merged": cfg["tier_labels"][1],
                "resulting_label_for_kept": cfg["tier_labels"][0],
                "shipped_tier_map": shipped,
                "shipped_health_order_best_to_worst": morder,
                "shipped_spearman_rho_vs_health_order": m_rhos,
                "shipped_min_spearman_rho": m_min_rho,
                "shipped_metrics": metrics_for(X, mlab, 2),
                "shipped_tier_benchmark_profile": benchmark_profile(df, mlab, domain, morder),
            }
            all_labels = pd.Series(mlab, index=df.index)
            min_rho_overall = m_min_rho
        else:
            art["tier_map"] = {str(c): {"health_rank": order.index(c),
                                        "tier_label": cfg["tier_labels"][order.index(c)]}
                               for c in range(k)}
            rhos, min_rho_overall = ordering_rho(df, labels, domain, order)
            art["spearman_rho_vs_health_order"] = rhos
            all_labels = pd.Series([order.index(int(c)) for c in labels], index=df.index)
        # Either way Approach B's headline metrics are the NATIVE fit's — the merge is
        # post-hoc, and its own metrics live under tier_merge.shipped_metrics.
        art["metrics"] = native_metrics

    # ------------------------------------------------------------------ shipped-tier state
    shares = [round(100 * float((all_labels == t).mean()), 1)
              for t in range(cfg["shipped_tiers"])]
    nd = df[df["profile_type"] == "normal"]
    nd_lab = all_labels.loc[nd.index]
    pooled_normal = [round(100 * float((nd_lab == t).mean()), 2)
                     for t in range(cfg["shipped_tiers"])]
    by_band_normal = {}
    for b in range(1, 6):
        s = nd[nd["age_band"] == b]
        sh = [round(100 * float((all_labels.loc[s.index] == t).mean()), 2)
              for t in range(cfg["shipped_tiers"])]
        by_band_normal[str(b)] = {"shares_pct": sh, "min_pct": min(sh)}
    worst_band = min(by_band_normal, key=lambda b: by_band_normal[b]["min_pct"])
    art["shipped_tiers_summary"] = {
        "tier_labels": cfg["tier_labels"],
        "tier_labels_status": (
            "LOCKED. All three domains' user-facing wording is final per the brief "
            "(Section 5): Banking 'Stable'/'Building', Investment 'On Track'/'Getting "
            "Started', Insurance 'Covered'/'Not Covered'. The labels are display strings "
            "only — they are attached by health rank, so changing them changes nothing "
            "about the model."),
        "tier_definition": ("tier index 0 = healthiest, ascending to the least healthy; the "
                            "index, not the display label, is the authoritative key any "
                            "downstream (Layer 3 / scoring) rule should be written against"),
        "population_shares_pct": shares,
        "min_spearman_rho": min_rho_overall,
        "normal_population_floor": {
            "definition": ("share of NON-INJECTED ('normal' profile_type) users in the "
                           "smallest shipped tier — the project-wide 5% floor adopted July "
                           "2026, read at the level the model is fitted at"),
            "read_at": cfg["normal_floor_level"],
            "pooled_shares_pct": pooled_normal,
            "pooled_min_pct": min(pooled_normal),
            "by_band": by_band_normal,
            "worst_band": int(worst_band),
            "worst_band_min_pct": by_band_normal[worst_band]["min_pct"],
            "value_checked": (min(pooled_normal) if cfg["normal_floor_level"] == "pooled"
                              else by_band_normal[worst_band]["min_pct"]),
            "floor_pct": 100 * NORMAL_FLOOR,
        },
    }
    art["out_of_model_benchmarks"] = [
        {"variable": b[0], "operator": b[1],
         "threshold": (b[2] if not isinstance(b[2], str) else
                       {"band_nw": "per-band net-worth ladder from SYNTHETIC_IDEALS "
                                   "(1/3/6/8/10)",
                        "band_conc": "per-band concentration taper from SYNTHETIC_IDEALS",
                        "band_life": "per-band life-coverage ideal from SYNTHETIC_IDEALS "
                                     "(9/9/9/9/5)"}[b[2]]),
         "source": b[3], "strictly_out_of_model": b[4]}
        for b in ds.BENCHMARKS[domain]
    ]

    report[domain] = {"artifact": art, "scaler": scaler, "age_scaler": age_scaler,
                      "labels": all_labels}
    return art


# ======================================================================================
# Reproduction gate
# ======================================================================================
def within_tol(a, b, tol):
    """True only if both values exist AND agree — a missing value is a failed check, never a
    silently passing one."""
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def verify(domain, cfg, art, grid_row, checks):
    def rec(name, ok, got, want):
        checks.append({"domain": domain, "check": name, "passed": bool(ok),
                       "reproduced": got, "validated": want})

    m = art["metrics"]
    for name, tol in (("silhouette_macro", TOL_SILHOUETTE),
                      ("silhouette_micro", TOL_SILHOUETTE),
                      ("davies_bouldin", TOL_DB),
                      ("calinski_harabasz", TOL_CH_REL * float(grid_row["calinski_harabasz"])),
                      ("min_cluster_pct", TOL_SHARE_PP)):
        rec(name, within_tol(m[name], grid_row[name], tol), m[name], grid_row[name])

    if cfg["merged"]:
        # Ordered healthiest-first, which is how FINAL_TIER_DECISIONS.md reports `shares`.
        prof = art["model"]["native_tier_benchmark_profile"]
        native_ordered = [round(prof[r]["share_pct"], 1) for r in sorted(prof)]
        rec("native_tier_shares_pct",
            all(within_tol(a, b, TOL_SHARE_PP)
                for a, b in zip(native_ordered, cfg["expect_native_shares_pct"])),
            native_ordered, cfg["expect_native_shares_pct"])

    s = art["shipped_tiers_summary"]
    rec("shipped_tier_shares_pct",
        all(within_tol(a, b, TOL_SHARE_PP)
            for a, b in zip(s["population_shares_pct"], cfg["expect_pooled_shares_pct"])),
        s["population_shares_pct"], cfg["expect_pooled_shares_pct"])
    rec("min_spearman_rho", within_tol(s["min_spearman_rho"], cfg["expect_min_rho"], TOL_RHO),
        s["min_spearman_rho"], cfg["expect_min_rho"])
    nf = s["normal_population_floor"]
    rec("normal_population_floor_pct",
        within_tol(nf["value_checked"], cfg["expect_normal_floor_pct"], TOL_FLOOR_PP),
        nf["value_checked"], cfg["expect_normal_floor_pct"])
    rec("normal_population_floor_clears_5pct",
        nf["value_checked"] >= nf["floor_pct"], nf["value_checked"], f">= {nf['floor_pct']}")
    rec("size_floor_clears_5pct", m["min_cluster_pct"] >= 100 * MIN_CLUSTER_SHARE,
        m["min_cluster_pct"], f">= {100 * MIN_CLUSTER_SHARE}")


# ======================================================================================
# Serialization
# ======================================================================================
def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_artifacts(report, checks, df, dataset_sha):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    written = {}
    for domain, r in report.items():
        jp = os.path.join(ARTIFACT_DIR, f"{domain}_model.json")
        with open(jp, "w") as f:
            json.dump(r["artifact"], f, indent=2)
        written[f"{domain}_model.json"] = sha256(jp)

        sp = os.path.join(ARTIFACT_DIR, f"{domain}_scaler.pkl")
        with open(sp, "wb") as f:
            pickle.dump(r["scaler"], f)
        written[f"{domain}_scaler.pkl"] = sha256(sp)

        if r["age_scaler"] is not None:
            ap = os.path.join(ARTIFACT_DIR, f"{domain}_age_scaler.pkl")
            with open(ap, "wb") as f:
                pickle.dump(r["age_scaler"], f)
            written[f"{domain}_age_scaler.pkl"] = sha256(ap)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "MLAI/clustering/build_production_artifacts.py",
        "purpose": (
            "Production artifact set for the DOMAIN-SPLIT (three-model) Layer 2 "
            "architecture, locked July 2026. Supersedes the single-combined-model artifacts "
            "in the parent directory (fitted_scaler.pkl, synthetic_ideal_centroids.json, "
            "archetype_profiles_k3.csv), which are retained for historical/report reference "
            "only and must not be used for domain-split scoring."),
        "dataset": {
            "path": "MLAI/clustering/data/processed/synthetic_users.csv",
            "n_users": int(len(df)),
            "sha256": dataset_sha,
            "note": ("the current canonical dataset, with the portfolio_concentration and "
                     "risk_age_deviation corrections applied — the same file every "
                     "domain-split validation script read"),
        },
        "library_versions": lib_versions(),
        "domains": {d: {"combination": WINNERS[d]["combination"],
                        "shipped_tiers": WINNERS[d]["shipped_tiers"],
                        "tier_labels": WINNERS[d]["tier_labels"],
                        "clustering_features": SPECS[WINNERS[d]["spec"]][1]}
                    for d in WINNERS},
        "files": written,
        "reproduction_checks": {
            "total": len(checks),
            "passed": sum(1 for c in checks if c["passed"]),
            "all_passed": all(c["passed"] for c in checks),
            "evidence_sources": [
                "MLAI/clustering/results/domain_split/domain_split_grid.csv",
                "MLAI/clustering/results/domain_split/FINAL_TIER_DECISIONS.md",
            ],
            "checks": checks,
        },
        "not_included_deliberately": [
            "No scoring function, tier-threshold mapping or 0-100 conversion — Phase 3.",
            "No Layer 3 combination-detection rules — a separate later task.",
            "No change to backend/ml/scoring.py or to the parent-directory artifacts.",
        ],
    }
    mp = os.path.join(ARTIFACT_DIR, "manifest.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def lib_versions():
    import sklearn
    import scipy
    out = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
           "scipy": scipy.__version__}
    try:
        import sklearn_extra
        out["scikit-learn-extra"] = getattr(sklearn_extra, "__version__", "unknown")
    except Exception:
        pass
    try:
        import skfuzzy
        out["scikit-fuzzy"] = getattr(skfuzzy, "__version__", "unknown")
    except Exception:
        pass
    return out


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check-only", action="store_true",
                    help="refit and verify, but write no artifacts")
    args = ap.parse_args()

    df = pd.read_csv(DATA_PATH)
    dataset_sha = sha256(DATA_PATH)
    grid = pd.read_csv(GRID_CSV)

    print("=" * 100)
    print(f"PRODUCTION ARTIFACT BUILD — domain-split architecture, {len(df):,} users")
    print(f"  dataset sha256 {dataset_sha[:16]}...")
    print("=" * 100)

    # Investment ships the canonical V2 benchmark set: risk_age_deviation stays a CLUSTERING
    # feature but leaves the out-of-model benchmark list (Kruskal-Wallis eps^2 = 0.00000 — no
    # injected profile creates any variation in it). Installed before any ordering is
    # computed, since the health ranking is defined over the benchmark variables.
    install_bench(False)
    print(f"  investment benchmark set (canonical V2): {ds.BENCH_VARS['investment']}\n")

    report, checks = {}, []
    for domain, cfg in WINNERS.items():
        print(f"[{domain}] refitting {cfg['combination']} "
              f"({cfg['algorithm_display']}, k={cfg['k']}, Approach {cfg['approach']}) "
              f"on {SPECS[cfg['spec']][1]}")
        art = build_domain(df, domain, cfg, report)
        row = grid[(grid["spec"] == cfg["spec"])
                   & (grid["combination"] == cfg["combination"])]
        if row.empty:
            raise SystemExit(f"no validated grid row for {cfg['spec']} / {cfg['combination']}")
        verify(domain, cfg, art, row.iloc[0], checks)
        m = art["metrics"]
        print(f"    macroSil {m['silhouette_macro']:.4f}  microSil "
              f"{m['silhouette_micro']:.4f}  DB {m['davies_bouldin']:.4f}  "
              f"CH {m['calinski_harabasz']:.1f}  minClus {m['min_cluster_pct']:.2f}%")
        print(f"    shipped tiers {art['shipped_tiers_summary']['tier_labels']} -> "
              f"{art['shipped_tiers_summary']['population_shares_pct']}%  "
              f"minRho {art['shipped_tiers_summary']['min_spearman_rho']}  "
              f"normFloor "
              f"{art['shipped_tiers_summary']['normal_population_floor']['value_checked']}%")

    print("\n" + "=" * 100)
    print("REPRODUCTION GATE — refit vs. the committed validated values")
    print("=" * 100)
    print(f"  {'domain':<12}{'check':<34}{'reproduced':>22}{'validated':>22}   ")
    for c in checks:
        print(f"  {c['domain']:<12}{c['check']:<34}{str(c['reproduced']):>22}"
              f"{str(c['validated']):>22}   {'PASS' if c['passed'] else 'FAIL'}")
    failed = [c for c in checks if not c["passed"]]
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed")

    if failed:
        print("\n  REFIT DOES NOT REPRODUCE THE VALIDATED RESULT — nothing written.")
        for c in failed:
            print(f"    {c['domain']}/{c['check']}: got {c['reproduced']}, "
                  f"expected {c['validated']}")
        raise SystemExit(1)

    if args.check_only:
        print("\n  --check-only: verification passed, no artifacts written.")
        return

    manifest = write_artifacts(report, checks, df, dataset_sha)
    print(f"\n  wrote {len(manifest['files'])} artifacts + manifest.json -> "
          f"backend/ml/artifacts/domain_split/")
    for name in sorted(manifest["files"]):
        print(f"    {name}")


if __name__ == "__main__":
    main()
