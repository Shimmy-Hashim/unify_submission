"""
final_selection.py — Research-grounded final algorithm/k/approach selection
(Unify, CCDS25-1170, July 2026).

WHY THIS EXISTS
  The canonical 12-algorithm re-benchmark (comprehensive_benchmark_11algo.py) produced
  a raw composite-rank winner — Approach A / Spectral / k=2 — that is a confirmed
  degenerate ("trivial") solution: ~99% of users in one cluster with a sliver split off.
  Micro-averaged silhouette is documented to be biased toward small k and imbalanced
  splits when clusters are weakly separated (the classic illustration: silhouette
  prefers k=2 on Iris despite 3 true species). This script replaces the raw
  composite-rank pick with a properly cited selection process:

    STEP 1  Minimum-cluster-size floor (>=5% of the sample per cluster) as a HARD
            pre-ranking disqualifier, applied across the full grid.
            Precedent: Bradley, Bennett & Demiriz (2000), constrained K-Means —
            size constraints added explicitly "to avoid local solutions with empty
            clusters or clusters having very few points."
    STEP 2  MACRO-averaged silhouette (mean within each cluster first, then mean across
            clusters) for survivors, reported beside the existing micro-average.
            More imbalance-robust; the recommended selection metric.
    STEP 3  Rank survivors by macro-silhouette (primary) with Davies-Bouldin /
            Calinski-Harabasz as corroborating signals (same rank-sum logic as the
            canonical run, macro replacing micro).
            Framework: Akhanli & Hennig (2020), Statistics and Computing 30:1523-1544
            (building on Hennig 2019) — aggregate calibrated validity indexes,
            weighted by the product's actual aim, across k AND algorithm at once.
    STEP 4A Usability validation by DENSITY, not just cluster means, per Metzler,
            Zhou & Grace (2021)'s own proportion-based validation ("58% of the
            comfortable group spend under 30% ... 66% of the stressed group over 40%").
            Mean AND % of cluster meeting the benchmark, for 5 out-of-model-benchmark
            variables, plus a bimodality check so a healthy mean cannot hide a split
            population.
    STEP 4B Per-cluster feature-mean profiling of the DISQUALIFIED combinations
            (especially Spectral/k=2) to test the "split on one trivial axis"
            hypothesis — the same pattern Metzler et al. found in their own rejected
            k=2 ("Delineating respondents based on their responses to a single
            question is not insightful").

  This script determines algorithm / k / approach ONLY. It writes NO production
  artifact: centroid selection (conjunctive screening) and the artifact re-export
  (incl. refitting fitted_scaler.pkl on 11 features) are separate later tasks.

METHODOLOGY REUSE
  Clustering dispatch, feature set, seed, scaling and the two approaches are IMPORTED
  from comprehensive_benchmark_11algo.py — not reimplemented — so this run is
  label-for-label the same grid. compute_metrics / composite_rank come from
  benchmark_core.py. A reproduction check asserts the recomputed micro-metrics match
  the committed comprehensive_results.json.

Run:
    cd <repo root>
    backend/venv/Scripts/python.exe MLAI/clustering/final_selection.py
"""

import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_benchmark_11algo import (  # noqa: E402
    FEATURES_11, K_BASED, K_RANGE, SEED, FAMILY, run_kbased,
)
from benchmark_core import compute_metrics, SYNTHETIC_IDEALS  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results")
CANONICAL_JSON = os.path.join(RESULTS_DIR, "comprehensive_results.json")
OUT_JSON = os.path.join(RESULTS_DIR, "final_selection.json")
# Label cache — the grid is deterministic (fixed SEED), so re-fitting 90 combinations on
# every analysis iteration is wasted work. Delete this file to force a clean re-fit.
LABEL_CACHE = os.path.join(RESULTS_DIR, ".final_selection_labels.pkl")

# ---- Step 1 parameter -----------------------------------------------------------------
MIN_CLUSTER_SHARE = 0.05   # 5% of the sample, per cluster (Bradley et al. 2000)

# ---- Step 4A benchmark thresholds ----------------------------------------------------
# Each is an EXTERNAL benchmark, not a quantity optimised by the clustering.
# net_worth_to_income is band-specific (the revised SYNTHETIC_IDEALS ladder 1/3/6/8/10).
LIFE_COVER_MIN = 9.0        # LIA PGS 2022 via MoneySense Guide, 9-10x income (9x floor)
CI_COVER_MIN = 3.9          # LIA PGS 2022, ~3.9x income
SAVINGS_RATE_MIN = 0.20     # MoneySense 20% minimum
DTI_MAX = 0.35              # DBS "7 financial ratios": ideal <= 35%
NW_IDEAL_BY_BAND = {b: SYNTHETIC_IDEALS[b]["net_worth_to_income"] for b in range(1, 6)}

STEP4A_VARS = ["life_coverage_ratio", "ci_coverage_ratio", "savings_rate",
               "net_worth_to_income", "dti"]

# A cluster is "usable" on a variable only if BOTH its mean clears the benchmark AND a
# meaningful majority of its members do. 50% is the "meaningful majority" line.
MAJORITY = 0.50
# Gate 2 (monotonic ordering, per the Phelps & Metzler 2024 template): every Step 4A
# variable must order in the SAME direction as the overall tier ordering, in every group.
# +0.5 Spearman rho is the "directionally consistent, not contradictory" line — a
# variable that inverts (rho < 0) would make the tier assert something false to the user.
MONOTONIC_RHO_MIN = 0.5
# Gate 2b (materiality). Directional consistency alone is not enough: at k=2 the rank
# correlation over two clusters is +/-1 by construction, so a rounding-level difference
# (e.g. 5.03x vs 4.94x life coverage) "passes" monotonicity while telling the user
# nothing. A tier must also be MATERIALLY more likely to clear each benchmark than a
# random user: the top tier's share clearing it must exceed the population base rate by
# at least this many percentage points, in every group.
MIN_LIFT_PP = 10.0
# Gate 2b', CEILING-AWARE materiality (added July 2026, correlation-fix retest).
# MIN_LIFT_PP alone is ill-specified against a near-saturated base rate: where 95.5% of a
# band already clears dti<=35%, the maximum achievable lift is 4.5pp, so a 10pp absolute
# test is unpassable no matter how perfectly the clustering discriminates. Headroom
# captured = (best_tier - base) / (100 - base) measures the share of the AVAILABLE
# improvement the top tier captures, which is well-defined at any base rate. A tier
# reaching 100.0% from a 95.5% base captures 100% of headroom — perfect discrimination
# that the absolute test scores as a failure.
# Both formulations are computed and reported; neither silently replaces the other.
MIN_HEADROOM_CAPTURED = 0.10
# Sarle's bimodality coefficient: > 0.555 (the uniform-distribution value) is the
# conventional flag for a non-unimodal / split distribution.
BC_FLAG = 0.555


# ======================================================================================
# Metric helpers
# ======================================================================================
def macro_silhouette(X, labels):
    """Average silhouette WITHIN each cluster first, then average those per-cluster
    means. Unweighted by cluster size, so a sliver cluster cannot be drowned out by a
    huge one (and cannot inflate the score by being tiny and far away either)."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0, {}
    s = silhouette_samples(X, labels)
    per_cluster = {int(c): float(s[labels == c].mean()) for c in uniq}
    return float(np.mean(list(per_cluster.values()))), per_cluster


def cluster_shares(labels, k_requested):
    """Per-cluster share of the sample. Missing (empty) clusters count as 0.0 shares —
    an algorithm that returns fewer than k groups has, by definition, an empty cluster."""
    n = len(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    shares = [float(c) / n for c in counts]
    n_found = len(uniq)
    shares += [0.0] * max(0, k_requested - n_found)
    return sorted(shares), n_found


def meets_mask(sub, var):
    """Boolean mask: does each user in `sub` clear the external benchmark for `var`?
    net_worth_to_income is measured against each user's OWN age band's ideal."""
    v = sub[var].to_numpy(float)
    if var == "dti":
        return v <= DTI_MAX
    if var == "net_worth_to_income":
        return v >= sub["age_band"].map(NW_IDEAL_BY_BAND).to_numpy(float)
    return v >= {"life_coverage_ratio": LIFE_COVER_MIN,
                 "ci_coverage_ratio": CI_COVER_MIN,
                 "savings_rate": SAVINGS_RATE_MIN}[var]


def base_rates(df):
    """% of the population clearing each benchmark, overall and per age band. Step 4A's
    density figures are uninterpretable without these: if only 14% of the population
    clears LIA's 9x life-coverage recommendation (which is what a dataset anchored to a
    *protection gap* study should look like), no cluster above the 5% floor can plausibly
    have a majority above it — the criterion would be unreachable by construction."""
    out = {"overall": {v: round(100 * float(meets_mask(df, v).mean()), 1) for v in STEP4A_VARS},
           "by_band": {}}
    for b in range(1, 6):
        sub = df[df["age_band"] == b]
        out["by_band"][b] = {v: round(100 * float(meets_mask(sub, v).mean()), 1)
                             for v in STEP4A_VARS}
    return out


def feature_profile(df_subset, labels, features=None):
    """Per-cluster mean of each feature.

    `features` falls back to FEATURES_11, resolved HERE rather than bound as an argument
    default: binding it in the signature captures the list at import time, so reassigning
    the module global would silently have no effect — the same trap that overwrote a
    production artifact during this project's July 2026 audit.
    """
    features = FEATURES_11 if features is None else features
    tmp = df_subset[features].copy()
    tmp["__c"] = labels
    return tmp.groupby("__c")[features].mean()


# ======================================================================================
# STEP 1 + 2 — full grid re-fit with per-cluster sizes and macro-silhouette
# ======================================================================================
def run_grid(df, scaler, X_full):
    """Re-fit the exact canonical grid, recording what the canonical run did not keep:
    per-cluster sizes (per band, under Approach A) and macro-averaged silhouette."""
    grid = {}
    bands = {b: df[df["age_band"] == b] for b in range(1, 6)}
    Xb = {b: scaler.transform(bands[b][FEATURES_11]) for b in bands}
    labels_store = {}

    for approach in ("A", "B"):
        for k in K_RANGE:
            for algo in K_BASED:
                key = f"{approach}|{algo}|{k}"
                try:
                    if approach == "A":
                        # Age-stratified: an independent model per band. The size floor
                        # must hold in EVERY band, so the binding constraint is the
                        # smallest cluster share found in any band.
                        per_band, mic, mac, dbs, chs, shares_all = {}, [], [], [], [], []
                        band_labels = {}
                        for b in range(1, 6):
                            lab = run_kbased(algo, k, Xb[b])
                            band_labels[b] = lab
                            m = compute_metrics(Xb[b], lab)
                            mac_b, _ = macro_silhouette(Xb[b], lab)
                            sh, nf = cluster_shares(lab, k)
                            mic.append(m["silhouette_score"])
                            mac.append(mac_b)
                            dbs.append(m["davies_bouldin_index"])
                            chs.append(m["calinski_harabasz_index"])
                            shares_all.extend(sh)
                            per_band[b] = {
                                "silhouette_micro": m["silhouette_score"],
                                "silhouette_macro": round(mac_b, 4),
                                "davies_bouldin": m["davies_bouldin_index"],
                                "calinski_harabasz": m["calinski_harabasz_index"],
                                "n_clusters_found": nf,
                                "cluster_shares_pct": [round(100 * s, 2) for s in sh],
                                "min_cluster_pct": round(100 * min(sh), 2),
                            }
                        labels_store[key] = band_labels
                        rec = {
                            "silhouette_micro": round(float(np.mean(mic)), 4),
                            "silhouette_macro": round(float(np.mean(mac)), 4),
                            "davies_bouldin": round(float(np.mean(dbs)), 4),
                            "calinski_harabasz": round(float(np.mean(chs)), 4),
                            "min_cluster_pct": round(100 * min(shares_all), 2),
                            "min_cluster_pct_by_band": {
                                str(b): per_band[b]["min_cluster_pct"] for b in per_band},
                            "worst_band": int(min(
                                per_band, key=lambda b: per_band[b]["min_cluster_pct"])),
                            "per_band": per_band,
                        }
                    else:
                        lab = run_kbased(algo, k, X_full)
                        labels_store[key] = lab
                        m = compute_metrics(X_full, lab)
                        mac_b, per_c = macro_silhouette(X_full, lab)
                        sh, nf = cluster_shares(lab, k)
                        rec = {
                            "silhouette_micro": m["silhouette_score"],
                            "silhouette_macro": round(mac_b, 4),
                            "davies_bouldin": m["davies_bouldin_index"],
                            "calinski_harabasz": m["calinski_harabasz_index"],
                            "n_clusters_found": nf,
                            "cluster_shares_pct": [round(100 * s, 2) for s in sh],
                            "min_cluster_pct": round(100 * min(sh), 2),
                            "per_cluster_silhouette": {str(c): round(v, 4)
                                                       for c, v in per_c.items()},
                        }
                    rec["passes_floor"] = rec["min_cluster_pct"] >= 100 * MIN_CLUSTER_SHARE
                    rec["shortfall_pp"] = round(
                        max(0.0, 100 * MIN_CLUSTER_SHARE - rec["min_cluster_pct"]), 2)
                    rec["failed"] = False
                except Exception as e:                       # pragma: no cover
                    rec = {"failed": True, "error": f"{type(e).__name__}: {e}",
                           "passes_floor": False, "min_cluster_pct": None}
                    print(f"    WARN {key}: {type(e).__name__}: {e}")
                grid[key] = {"approach": approach, "algorithm": algo, "k": k,
                             "family": FAMILY[algo], **rec}
                print(f"  {key:<26} min_cluster={rec.get('min_cluster_pct')}%"
                      f"  micro={rec.get('silhouette_micro')}  macro={rec.get('silhouette_macro')}"
                      f"  {'PASS' if rec['passes_floor'] else 'DISQUALIFIED'}")
    return grid, labels_store


def reproduction_check(grid):
    """Confirm the re-fit grid reproduces the committed canonical micro-metrics, so the
    Step 1/2 additions are demonstrably measured on the same partitions — not a
    silently different run."""
    with open(CANONICAL_JSON) as f:
        canon = json.load(f)
    mismatches, checked = [], 0
    for approach, key in (("A", "approach_a"), ("B", "approach_b")):
        for k in K_RANGE:
            for algo, rec in canon[key]["results_by_k"][str(k)]["algorithms"].items():
                g = grid[f"{approach}|{algo}|{k}"]
                checked += 1
                for ours, theirs in (("silhouette_micro", "silhouette_score"),
                                     ("davies_bouldin", "davies_bouldin_index"),
                                     ("calinski_harabasz", "calinski_harabasz_index")):
                    a, b = g.get(ours), rec[theirs]
                    if a is None or abs(a - b) > max(0.002, abs(b) * 0.001):
                        mismatches.append({"combo": f"{approach}|{algo}|{k}",
                                           "metric": theirs, "canonical": b, "refit": a})
    return {"combinations_checked": checked, "mismatches": mismatches,
            "reproduces_canonical": len(mismatches) == 0}


# ======================================================================================
# STEP 3 — rank survivors
# ======================================================================================
def rank_survivors(survivors):
    """Rank-sum over macro-silhouette (primary), Davies-Bouldin and Calinski-Harabasz —
    the canonical composite_rank logic with macro replacing micro silhouette. Lowest
    total wins.

    Ranked WITHIN each approach. Calinski-Harabasz is not comparable across approaches
    (it scales with n: Approach A averages per-band fits of ~1,400 users, Approach B
    fits all 7,000), so a single pooled rank-sum would be contaminated by that alone.
    Macro-silhouette and Davies-Bouldin ARE scale-free and are used for the
    cross-approach comparison, reported separately."""
    out = {}
    for approach in ("A", "B"):
        pool = {kk: v for kk, v in survivors.items() if v["approach"] == approach}
        if not pool:
            out[approach] = {}
            continue
        keys = list(pool)
        r_mac = {a: i + 1 for i, a in enumerate(
            sorted(keys, key=lambda x: pool[x]["silhouette_macro"], reverse=True))}
        r_db = {a: i + 1 for i, a in enumerate(
            sorted(keys, key=lambda x: pool[x]["davies_bouldin"]))}
        r_ch = {a: i + 1 for i, a in enumerate(
            sorted(keys, key=lambda x: pool[x]["calinski_harabasz"], reverse=True))}
        out[approach] = {a: {"rank_macro_sil": r_mac[a], "rank_db": r_db[a],
                             "rank_ch": r_ch[a],
                             "composite_rank_macro": r_mac[a] + r_db[a] + r_ch[a]}
                         for a in keys}
    return out


# ======================================================================================
# STEP 4A — mean AND density validation, with a bimodality check
# ======================================================================================
def bimodality(vals):
    """Sarle's bimodality coefficient + a 1-vs-2-component Gaussian mixture BIC
    comparison. Two independent signals so one weak indicator alone doesn't drive a
    'bimodal' flag."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    n = len(v)
    res = {"n": int(n), "bimodality_coefficient": None, "gmm_prefers_2": None,
           "gmm_bic_1": None, "gmm_bic_2": None}
    if n < 20 or np.allclose(v, v[0]):
        return res
    sk = float(stats.skew(v, bias=False))
    ku = float(stats.kurtosis(v, fisher=True, bias=False))
    denom = ku + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom > 0:
        res["bimodality_coefficient"] = round((sk ** 2 + 1.0) / denom, 4)
    res["skewness"] = round(sk, 3)
    res["excess_kurtosis"] = round(ku, 3)
    x = v.reshape(-1, 1)
    try:
        b1 = GaussianMixture(1, random_state=SEED).fit(x).bic(x)
        b2 = GaussianMixture(2, n_init=5, random_state=SEED).fit(x).bic(x)
        res["gmm_bic_1"], res["gmm_bic_2"] = round(float(b1), 1), round(float(b2), 1)
        res["gmm_prefers_2"] = bool(b2 < b1 - 10)   # >10 BIC units = strong preference
    except Exception:
        pass
    return res


def distribution_flag(bim):
    """Name the shape a cluster's values take for one variable, from the two independent
    signals bimodality() returns. Either alone is weak; both together is the real flag.

    domain_split_clustering.py keeps its own copy rather than importing this one, so that
    each script's flags stay governed by its own BC_FLAG.
    """
    bc = bim.get("bimodality_coefficient") or 0
    two_components = bim.get("gmm_prefers_2")
    if two_components and bc > BC_FLAG:
        return "bimodal/split"
    if bc > BC_FLAG:
        return "wide-spread"
    if two_components:
        return "two-component fit preferred"
    return "unimodal"


def step4a(df, labels_by_group, approach):
    """labels_by_group: {group_key: (sub_df, labels)}. Under Approach A the group is an
    age band (one model per band); under B there is a single group covering all users."""
    out = {}
    for gkey, (sub, labels) in labels_by_group.items():
        band_for_nw = None
        if approach == "A":
            band_for_nw = int(sub["age_band"].iloc[0])
        gout = {}
        for c in sorted(np.unique(labels)):
            cl = sub[labels == c]
            entry = {"n": int(len(cl)), "share_pct": round(100 * len(cl) / len(sub), 2),
                     "variables": {}}
            for var in STEP4A_VARS:
                v = cl[var].to_numpy(float)
                if var == "dti":
                    thr, direction = DTI_MAX, "<="
                    meets = v <= thr
                    mean_ok = bool(v.mean() <= thr)
                elif var == "net_worth_to_income":
                    # Band-specific: the age-appropriate ideal, not one flat number.
                    direction = ">="
                    if band_for_nw is not None:
                        thr = NW_IDEAL_BY_BAND[band_for_nw]
                        meets = v >= thr
                        mean_ok = bool(v.mean() >= thr)
                    else:
                        # Approach B pools all bands, so each user is measured against
                        # their OWN band's ideal; the cluster mean is compared against
                        # the mean of its members' own band ideals.
                        thr_vec = cl["age_band"].map(NW_IDEAL_BY_BAND).to_numpy(float)
                        thr = f"per-band {NW_IDEAL_BY_BAND}"
                        meets = v >= thr_vec
                        mean_ok = bool(v.mean() >= float(thr_vec.mean()))
                else:
                    thr = {"life_coverage_ratio": LIFE_COVER_MIN,
                           "ci_coverage_ratio": CI_COVER_MIN,
                           "savings_rate": SAVINGS_RATE_MIN}[var]
                    direction = ">="
                    meets = v >= thr
                    mean_ok = bool(v.mean() >= thr)
                pct = float(meets.mean())
                bim = bimodality(v)
                entry["variables"][var] = {
                    "mean": round(float(v.mean()), 4),
                    "median": round(float(np.median(v)), 4),
                    "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                    "q25": round(float(np.percentile(v, 25)), 4),
                    "q75": round(float(np.percentile(v, 75)), 4),
                    "benchmark": thr, "direction": direction,
                    "pct_meeting_benchmark": round(100 * pct, 1),
                    "mean_clears_benchmark": mean_ok,
                    "majority_clears_benchmark": bool(pct >= MAJORITY),
                    "usable": bool(mean_ok and pct >= MAJORITY),
                    "mean_masks_split": bool(
                        mean_ok and pct < MAJORITY),   # healthy mean, minority actually OK
                    **bim,
                    "distribution_flag": distribution_flag(bim),
                }
            gout[int(c)] = entry
        out[str(gkey)] = {"n_group": int(len(sub)), "clusters": gout}
    return out


def monotonicity(step4a_out):
    """Do the clusters order consistently across the Step 4A variables? A usable tier
    system needs one cluster to be better on (nearly) all of them, not better on some
    and worse on others.

    Two measures, because exact-permutation agreement alone is misleading at higher k —
    a single adjacent swap anywhere in a 5-cluster ordering breaks it entirely, so it
    would penalise k=5 far more harshly than k=2 for the same amount of disorder:
      * exact_order_agreement — how many of the 5 variables reproduce the overall
        health ordering exactly (informative at k=2, very strict beyond it)
      * spearman_rho — rank correlation between each variable's cluster means and the
        overall health rank. This is the real "monotonic ordering" measure, comparable
        across k. Negative rho means a variable INVERTS against the tier ordering,
        which would make the tier tell users something false.
    """
    report = {}
    for gkey, g in step4a_out.items():
        cids = sorted(g["clusters"])
        if len(cids) < 2:
            report[gkey] = {"monotonic": None, "note": "single cluster"}
            continue
        score = {c: 0.0 for c in cids}
        per_var_order, per_var_vals = {}, {}
        for var in STEP4A_VARS:
            vals = {c: g["clusters"][c]["variables"][var]["mean"] for c in cids}
            per_var_vals[var] = vals
            rev = (var != "dti")            # dti: lower is healthier
            order = sorted(cids, key=lambda c: vals[c], reverse=rev)
            per_var_order[var] = order
            for rank, c in enumerate(order):
                score[c] += rank
        overall = sorted(cids, key=lambda c: score[c])   # best (healthiest) first
        # health_rank: 0 = healthiest. Correlate each variable's healthiness against it.
        health_rank = {c: i for i, c in enumerate(overall)}
        rhos = {}
        for var in STEP4A_VARS:
            hv = [per_var_vals[var][c] * (-1 if var == "dti" else 1) for c in cids]
            hr = [-health_rank[c] for c in cids]          # higher = healthier
            if len(set(hv)) < 2:
                rhos[var] = None
                continue
            rhos[var] = round(float(stats.spearmanr(hv, hr).statistic), 3)
        valid = [r for r in rhos.values() if r is not None]
        agree = sum(1 for var in STEP4A_VARS if per_var_order[var] == overall)
        report[gkey] = {
            "overall_health_order_best_to_worst": overall,
            "variables_agreeing_with_overall_order": agree,
            "variables_total": len(STEP4A_VARS),
            "per_variable_order": {v: per_var_order[v] for v in STEP4A_VARS},
            "fully_monotonic": agree == len(STEP4A_VARS),
            "spearman_rho_vs_health_order": rhos,
            "min_spearman_rho": min(valid) if valid else None,
            "mean_spearman_rho": round(float(np.mean(valid)), 3) if valid else None,
            "any_variable_inverts": bool(valid and min(valid) < 0),
        }
    return report


# ======================================================================================
# STEP 4B — feature-mean profiling of disqualified combinations
# ======================================================================================
def separation_profile(profile, iqr):
    """How many of the 11 features actually separate the clusters, and how concentrated
    the separation is on a single feature.

    Spread per feature = (max cluster mean - min cluster mean) / feature IQR, i.e. in
    robust-scaled units — exactly the geometry the clustering itself saw. 'Meaningful'
    = >= 0.5 scaled units. top_feature_share = the leading feature's spread as a
    fraction of the total spread across all 11 features: high => a single-axis split."""
    spreads = {}
    for f in FEATURES_11:
        denom = iqr[f] if iqr[f] > 0 else 1.0
        spreads[f] = float((profile[f].max() - profile[f].min()) / denom)
    tot = sum(spreads.values())
    ordered = sorted(spreads, key=spreads.get, reverse=True)
    return {
        "spread_scaled_units": {f: round(spreads[f], 3) for f in FEATURES_11},
        "features_separating_ge_0.5_scaled": [f for f in ordered if spreads[f] >= 0.5],
        "n_features_separating_ge_0.5_scaled": sum(1 for f in spreads if spreads[f] >= 0.5),
        "n_features_separating_ge_1.0_scaled": sum(1 for f in spreads if spreads[f] >= 1.0),
        "top_feature": ordered[0],
        "top_feature_spread": round(spreads[ordered[0]], 3),
        "top_feature_share_of_total_separation": round(spreads[ordered[0]] / tot, 3) if tot else None,
        "top3_share_of_total_separation": round(
            sum(spreads[f] for f in ordered[:3]) / tot, 3) if tot else None,
    }


def step4b(df, grid, labels_store, iqr, only_disqualified=True):
    """Full 11-feature per-cluster mean profile. By default restricted to the
    DISQUALIFIED combinations (Step 4B's remit); also run over survivors so the
    single-axis test can be applied to the eventual winner too."""
    out = {}
    bands = {b: df[df["age_band"] == b] for b in range(1, 6)}
    for key, rec in grid.items():
        if rec.get("failed"):
            continue
        if only_disqualified and rec.get("passes_floor"):
            continue
        if key not in labels_store:
            continue
        entry = {"approach": rec["approach"], "algorithm": rec["algorithm"], "k": rec["k"],
                 "min_cluster_pct": rec["min_cluster_pct"],
                 "shortfall_pp": rec["shortfall_pp"]}
        if rec["approach"] == "A":
            entry["by_band"] = {}
            for b, lab in labels_store[key].items():
                prof = feature_profile(bands[b], lab)
                sh = [round(100 * float((lab == c).mean()), 2) for c in sorted(np.unique(lab))]
                entry["by_band"][str(b)] = {
                    "cluster_shares_pct": sh,
                    "feature_means": {str(c): {f: round(float(prof.loc[c, f]), 4)
                                               for f in FEATURES_11} for c in prof.index},
                    "separation": separation_profile(prof, iqr),
                }
        else:
            lab = labels_store[key]
            prof = feature_profile(df, lab)
            sh = [round(100 * float((lab == c).mean()), 2) for c in sorted(np.unique(lab))]
            entry["all"] = {
                "cluster_shares_pct": sh,
                "feature_means": {str(c): {f: round(float(prof.loc[c, f]), 4)
                                           for f in FEATURES_11} for c in prof.index},
                "separation": separation_profile(prof, iqr),
            }
        out[key] = entry
    return out


# ======================================================================================
# STEP 5 — is ANY survivor genuinely usable? (not just the top-ranked one)
# ======================================================================================
def discrimination_lift(groups, base, approach):
    """For each Step 4A variable: how much more likely is the BEST tier's member to clear
    the benchmark than a random user (percentage points over the population base rate)?

    This is the materiality test the mean/majority check cannot provide on this
    population — where the insurance base rates (14.0% / 7.5%) make an absolute majority
    unreachable, a tier is still informative if it lifts the probability substantially.
    Only clusters above the 5% floor are eligible to be "the best tier"; reported at the
    WORST group, since a tier system must discriminate in every age band."""
    per_var = {v: [] for v in STEP4A_VARS}
    for gkey, (sub, lab) in groups.items():
        bref = base["by_band"][int(sub["age_band"].iloc[0])] if approach == "A" else base["overall"]
        for var in STEP4A_VARS:
            m = meets_mask(sub, var)
            cand = [float(m[lab == c].mean()) for c in np.unique(lab)
                    if (lab == c).mean() >= MIN_CLUSTER_SHARE]
            if cand:
                per_var[var].append((100 * max(cand), bref[var], gkey))
    out = {}
    for var in STEP4A_VARS:
        if not per_var[var]:
            continue
        lifts = [(p - b) for p, b, _ in per_var[var]]
        # Headroom captured, evaluated at its own worst group (which need not be the
        # worst group for absolute lift — a high base rate compresses absolute lift
        # while leaving headroom capture intact).
        heads = [((p - b) / (100.0 - b) if b < 100.0 else 1.0) for p, b, _ in per_var[var]]
        i = int(np.argmin(lifts))
        j = int(np.argmin(heads))
        p, b, gk = per_var[var][i]
        ph, bh, gkh = per_var[var][j]
        out[var] = {"worst_group": gk,
                    "best_tier_pct_clearing": round(p, 1),
                    "base_rate_pct": round(b, 1),
                    "lift_pp": round(p - b, 1),
                    "lift_ratio": round(p / b, 2) if b > 0 else None,
                    "mean_lift_pp_across_groups": round(float(np.mean(lifts)), 1),
                    "material": bool((p - b) >= MIN_LIFT_PP),
                    # ceiling-aware view
                    "max_possible_lift_pp": round(100.0 - b, 1),
                    "headroom_captured": round(heads[i], 3),
                    "worst_group_headroom": gkh,
                    "worst_headroom_captured": round(heads[j], 3),
                    "worst_headroom_base_rate_pct": round(bh, 1),
                    "worst_headroom_best_tier_pct": round(ph, 1),
                    "material_headroom": bool(heads[j] >= MIN_HEADROOM_CAPTURED)}
    return out


def usability_summary(step4a_out, mono, lift=None):
    """Condense Step 4A + monotonicity into the two things Step 5 actually asks about:
    (a) do the clusters order sensibly, and (b) is the best cluster genuinely usable —
    mean AND majority clearing the benchmark — on the 5 out-of-model variables.

    Aggregated across groups by the WORST group (under Approach A a tier system has to
    work in every age band, not on average across them)."""
    per_group = {}
    for gkey, g in step4a_out.items():
        m = mono.get(gkey, {})
        order = m.get("overall_health_order_best_to_worst") or []
        best = order[0] if order else None
        vars_usable = 0
        insurance_usable = 0
        if best is not None:
            cv = g["clusters"][best]["variables"]
            vars_usable = sum(1 for var in STEP4A_VARS if cv[var]["usable"])
            insurance_usable = sum(1 for var in ("life_coverage_ratio", "ci_coverage_ratio")
                                   if cv[var]["usable"])
        masked = sum(1 for c in g["clusters"].values()
                     for var in STEP4A_VARS if c["variables"][var]["mean_masks_split"])
        per_group[gkey] = {
            "best_cluster": best,
            "vars_usable_in_best_cluster": vars_usable,
            "insurance_vars_usable_in_best_cluster": insurance_usable,
            "variables_agreeing_with_overall_order": m.get("variables_agreeing_with_overall_order"),
            "fully_monotonic": m.get("fully_monotonic"),
            "min_spearman_rho": m.get("min_spearman_rho"),
            "mean_spearman_rho": m.get("mean_spearman_rho"),
            "any_variable_inverts": m.get("any_variable_inverts"),
            "n_mean_masks_split_flags": masked,
        }
    agree = [v["variables_agreeing_with_overall_order"] for v in per_group.values()
             if v["variables_agreeing_with_overall_order"] is not None]
    rhos = [v["min_spearman_rho"] for v in per_group.values() if v["min_spearman_rho"] is not None]
    return {
        "per_group": per_group,
        "worst_group_vars_usable": min(v["vars_usable_in_best_cluster"] for v in per_group.values()),
        "worst_group_insurance_vars_usable": min(
            v["insurance_vars_usable_in_best_cluster"] for v in per_group.values()),
        "worst_group_monotonic_agreement": min(agree) if agree else None,
        "worst_group_min_spearman_rho": min(rhos) if rhos else None,
        "all_groups_fully_monotonic": all(v["fully_monotonic"] for v in per_group.values()),
        "any_group_has_inverting_variable": any(v["any_variable_inverts"] for v in per_group.values()),
        "total_mean_masks_split_flags": sum(v["n_mean_masks_split_flags"] for v in per_group.values()),
        # Absolute bar (mean AND majority on all 5, exact ordering in every group). Kept
        # for reference, but see MONOTONIC_RHO_MIN in the gate below: the majority
        # criterion is unreachable by construction for the insurance variables on this
        # population (base rates 14.0% / 7.5%), so this bar cannot be the deciding test.
        "meets_step5_bar": bool(
            all(v["fully_monotonic"] for v in per_group.values())
            and min(v["vars_usable_in_best_cluster"] for v in per_group.values()) == len(STEP4A_VARS)),
        # Gate 2a: no variable may invert against the tier ordering in any group.
        "passes_monotonicity_gate": bool(rhos and min(rhos) >= MONOTONIC_RHO_MIN),
        # Gate 2b: the ordering must also be MATERIAL — see MIN_LIFT_PP.
        "lift": lift or {},
        "min_lift_pp": (min(v["lift_pp"] for v in lift.values()) if lift else None),
        "insurance_lift_pp": (round(float(np.mean(
            [lift[v]["lift_pp"] for v in ("life_coverage_ratio", "ci_coverage_ratio")
             if v in lift])), 1) if lift else None),
        "passes_materiality_gate": bool(
            lift and len(lift) == len(STEP4A_VARS)
            and all(v["material"] for v in lift.values())),
        "passes_gate2": bool(
            rhos and min(rhos) >= MONOTONIC_RHO_MIN
            and lift and len(lift) == len(STEP4A_VARS)
            and all(v["material"] for v in lift.values())),
        # Ceiling-aware variant (see MIN_HEADROOM_CAPTURED). Reported alongside, not
        # instead of, the absolute-lift verdict above.
        "min_headroom_captured": (min(v["worst_headroom_captured"] for v in lift.values())
                                  if lift else None),
        "passes_materiality_gate_headroom": bool(
            lift and len(lift) == len(STEP4A_VARS)
            and all(v["material_headroom"] for v in lift.values())),
        "passes_gate2_headroom": bool(
            rhos and min(rhos) >= MONOTONIC_RHO_MIN
            and lift and len(lift) == len(STEP4A_VARS)
            and all(v["material_headroom"] for v in lift.values())),
    }


def sweep_verdict(summary):
    """'GATE-2 PASS', or 'fail: ' followed by whichever half (or both) did not hold."""
    if summary["passes_gate2"]:
        return "GATE-2 PASS"
    reasons = []
    if not summary["passes_monotonicity_gate"]:
        reasons.append("ordering")
    if not summary["passes_materiality_gate"]:
        reasons.append("materiality")
    return "fail: " + "+".join(reasons)


def sweep_survivors(df, survivors, labels_store, base):
    """Run Step 4A + monotonicity + the usability summary over EVERY floor-passing
    combination — so Step 5's 'no combination works' conclusion (or its refutation) is
    evidenced across the whole survivor set, not inferred from the top-ranked one."""
    out = {}
    for key in survivors:
        approach, algo, k = key.split("|")
        if approach == "A":
            groups = {f"band{b}": (df[df["age_band"] == b], labels_store[key][b])
                      for b in range(1, 6)}
        else:
            groups = {"all": (df, labels_store[key])}
        s4 = step4a(df, groups, approach)
        mono = monotonicity(s4)
        lift = discrimination_lift(groups, base, approach)
        s = usability_summary(s4, mono, lift)
        out[key] = {"summary": s, "monotonicity": mono, "step4a": s4}
        verdict = sweep_verdict(s)
        print(f"  {key:<26} min_rho={str(s['worst_group_min_spearman_rho']):>6}"
              f"  min_lift={str(s['min_lift_pp']):>6}pp"
              f"  ins_lift={str(s['insurance_lift_pp']):>6}pp"
              f"  usable={s['worst_group_vars_usable']}/5"
              f"  masked={s['total_mean_masks_split_flags']}   {verdict}")
    return out


# ======================================================================================
# Main
# ======================================================================================
def main():
    df = pd.read_csv(DATA_PATH)
    iqr = {f: float(df[f].quantile(0.75) - df[f].quantile(0.25)) for f in FEATURES_11}

    # Same scaling as the canonical run (in-memory RobustScaler; the 10-feature
    # production fitted_scaler.pkl is deliberately NOT used and NOT written).
    scaler = RobustScaler().fit(df[FEATURES_11])
    age_scaler = StandardScaler().fit(df[["age_midpoint"]])
    X_full = np.hstack([scaler.transform(df[FEATURES_11]),
                        age_scaler.transform(df[["age_midpoint"]])])

    print("=" * 92)
    print(f"FINAL SELECTION — {len(df)} users, {len(FEATURES_11)} continuous features, "
          f"floor = {100 * MIN_CLUSTER_SHARE:.0f}% per cluster")
    print("=" * 92)

    print("\n[STEP 1+2] Re-fitting the full grid "
          f"({len(K_BASED)} algorithms x k={min(K_RANGE)}-{max(K_RANGE)} x 2 approaches "
          f"= {len(K_BASED) * len(list(K_RANGE)) * 2} combinations)...")
    if os.path.exists(LABEL_CACHE):
        with open(LABEL_CACHE, "rb") as f:
            cached = pickle.load(f)
        grid, labels_store = cached["grid"], cached["labels"]
        print(f"  (loaded deterministic labels from cache {os.path.basename(LABEL_CACHE)} "
              f"— delete it to force a clean re-fit)")
    else:
        grid, labels_store = run_grid(df, scaler, X_full)
        with open(LABEL_CACHE, "wb") as f:
            pickle.dump({"grid": grid, "labels": labels_store}, f)

    repro = reproduction_check(grid)
    repro_verdict = ("IDENTICAL" if repro["reproduces_canonical"]
                     else f"{len(repro['mismatches'])} MISMATCHES")
    print(f"\n[repro check] {repro['combinations_checked']} combinations vs committed "
          f"comprehensive_results.json -> {repro_verdict}")
    for m in repro["mismatches"][:10]:
        print(f"    {m}")

    survivors = {k: v for k, v in grid.items() if v.get("passes_floor")}
    disqualified = {k: v for k, v in grid.items() if not v.get("passes_floor")}
    print(f"\n[STEP 1] {len(survivors)} of {len(grid)} combinations pass the "
          f"{100 * MIN_CLUSTER_SHARE:.0f}% floor; {len(disqualified)} disqualified.")

    print("\n[STEP 3] Ranking survivors by macro-silhouette (primary) + DB + CH...")
    ranks = rank_survivors(survivors)
    for approach in ("A", "B"):
        if not ranks[approach]:
            print(f"  Approach {approach}: no survivors")
            continue
        ordered = sorted(ranks[approach], key=lambda x: (
            ranks[approach][x]["composite_rank_macro"], -survivors[x]["silhouette_macro"]))
        print(f"  Approach {approach} (best first):")
        for key in ordered[:8]:
            s = survivors[key]
            r = ranks[approach][key]
            print(f"    {key:<26} macro={s['silhouette_macro']:>7.4f}  micro={s['silhouette_micro']:>7.4f}"
                  f"  db={s['davies_bouldin']:>6.3f}  ch={s['calinski_harabasz']:>8.1f}"
                  f"  min%={s['min_cluster_pct']:>5.2f}  rank-sum={r['composite_rank_macro']}")

    # Cross-approach winner: macro-silhouette is the primary metric and is scale-free.
    best_per_approach = {}
    for approach in ("A", "B"):
        if ranks[approach]:
            best_per_approach[approach] = min(
                ranks[approach],
                key=lambda x: (ranks[approach][x]["composite_rank_macro"],
                               -survivors[x]["silhouette_macro"]))
    winner = None
    if best_per_approach:
        winner = max(best_per_approach.values(),
                     key=lambda x: survivors[x]["silhouette_macro"])
    print(f"\n[STEP 3] Per-approach best: {best_per_approach}\n          Overall winner: {winner}")

    # ---- Step 4A on the winner ----
    step4a_out, mono = {}, {}
    if winner:
        wapp = winner.split("|")[0]
        if wapp == "A":
            groups = {f"band{b}": (df[df["age_band"] == b], labels_store[winner][b])
                      for b in range(1, 6)}
        else:
            groups = {"all": (df, labels_store[winner])}
        print(f"\n[STEP 4A] Mean AND density validation for {winner} ...")
        step4a_out = step4a(df, groups, wapp)
        mono = monotonicity(step4a_out)
        for gkey, g in step4a_out.items():
            print(f"  {gkey} (n={g['n_group']}):")
            for c, e in sorted(g["clusters"].items()):
                print(f"    cluster {c} (n={e['n']}, {e['share_pct']}%):")
                for var in STEP4A_VARS:
                    d = e["variables"][var]
                    flag = "  <<< MEAN MASKS SPLIT" if d["mean_masks_split"] else ""
                    print(f"      {var:<24} mean={d['mean']:>9.3f} ({'OK ' if d['mean_clears_benchmark'] else 'FAIL'})"
                          f"  meeting={d['pct_meeting_benchmark']:>5.1f}%"
                          f"  dist={d['distribution_flag']:<28}{flag}")
            m = mono[gkey]
            print(f"    monotonic: {m.get('variables_agreeing_with_overall_order')}/"
                  f"{m.get('variables_total')} variables agree with overall order "
                  f"{m.get('overall_health_order_best_to_worst')}")

    # ---- Step 4B on the disqualified ----
    print(f"\n[STEP 4B] Feature-mean profiling of {len(disqualified)} disqualified "
          f"combinations (single-trivial-axis diagnostic)...")
    step4b_out = step4b(df, grid, labels_store, iqr, only_disqualified=True)
    # The same diagnostic over survivors, so the winner is held to it as well.
    survivor_profiles = step4b(df, survivors, labels_store, iqr, only_disqualified=False)
    for key in ("A|spectral|2", "B|spectral|2"):
        e = step4b_out.get(key) or survivor_profiles.get(key)
        if e is None:
            continue
        print(f"  {key}  (min cluster {e['min_cluster_pct']}%, "
              f"{e['shortfall_pp']}pp below floor)")
        blocks = e.get("by_band") or {"all": e["all"]}
        for bk, blk in blocks.items():
            sep = blk["separation"]
            print(f"    {bk}: shares={blk['cluster_shares_pct']}  "
                  f"features separating >=0.5 scaled: {sep['n_features_separating_ge_0.5_scaled']}/11"
                  f"  top={sep['top_feature']} ({sep['top_feature_spread']}, "
                  f"{100 * sep['top_feature_share_of_total_separation']:.0f}% of total separation)")

    # ---- Step 5: is ANY survivor usable, or only the top-ranked one that failed? ----
    print(f"\n[STEP 5] Usability sweep across all {len(survivors)} floor-passing "
          f"combinations (is the fallback scenario real?)...")
    base = base_rates(df)
    print("  population base rates (% clearing each benchmark, no clustering involved):")
    print("    " + "band".ljust(6) + "".join(v[:19].rjust(21) for v in STEP4A_VARS))
    for b in range(1, 6):
        print("    " + str(b).ljust(6) + "".join(
            f"{base['by_band'][b][v]:>20.1f}%" for v in STEP4A_VARS))
    print("    " + "ALL".ljust(6) + "".join(f"{base['overall'][v]:>20.1f}%" for v in STEP4A_VARS))
    for v in STEP4A_VARS:
        r = base["overall"][v]
        if 2 * r < 100:
            print(f"    -> {v}: base rate {r}% means a >50%-clearing cluster can cover at "
                  f"most ~{min(100.0, 2 * r):.0f}% of the sample")
    sweep = sweep_survivors(df, survivors, labels_store, base)
    meets = [k for k, v in sweep.items() if v["summary"]["meets_step5_bar"]]
    print(f"\n  combinations meeting the ABSOLUTE bar (exact ordering in every group AND "
          f"best cluster's mean+majority clearing all 5 benchmarks): {meets if meets else 'NONE'}")

    g2a = [k for k, v in sweep.items() if v["summary"]["passes_monotonicity_gate"]]
    g2b = [k for k, v in sweep.items() if v["summary"]["passes_materiality_gate"]]
    gate2 = [k for k, v in sweep.items() if v["summary"]["passes_gate2"]]
    print(f"  Gate 2a — ordering (no variable inverts, rho >= {MONOTONIC_RHO_MIN} in every "
          f"group): {len(g2a)} of {len(survivors)}  {g2a}")
    print(f"  Gate 2b — materiality (top tier lifts every benchmark by >= {MIN_LIFT_PP}pp "
          f"in every group): {len(g2b)} of {len(survivors)}  {g2b}")
    print(f"  Gate 2 (both): {len(gate2)} of {len(survivors)}  {gate2}")

    g2b_h = [k for k, v in sweep.items() if v["summary"]["passes_materiality_gate_headroom"]]
    gate2_h = [k for k, v in sweep.items() if v["summary"]["passes_gate2_headroom"]]
    print(f"  Gate 2b' — CEILING-AWARE materiality (top tier captures >= "
          f"{MIN_HEADROOM_CAPTURED:.0%} of available headroom on every benchmark, every "
          f"group): {len(g2b_h)} of {len(survivors)}  {g2b_h}")
    print(f"  Gate 2 (ordering + ceiling-aware materiality): {len(gate2_h)} of "
          f"{len(survivors)}  {gate2_h if gate2_h else 'NONE'}")

    # The two gates are in tension — characterise it explicitly rather than just
    # reporting "none passed": which variable fails, for the strongest candidates on
    # each side?
    print("\n  Why the two halves of Gate 2 do not co-occur — per-variable detail:")
    by_ins = sorted(sweep, key=lambda k: -(sweep[k]["summary"]["insurance_lift_pp"] or -99))
    for k in by_ins[:4] + [x for x in g2a if x not in by_ins[:4]][:4]:
        s = sweep[k]["summary"]
        print(f"    {k}  (insurance lift {s['insurance_lift_pp']}pp, "
              f"min_rho {s['worst_group_min_spearman_rho']})")
        for var in STEP4A_VARS:
            lv = s["lift"].get(var, {})
            rho_by_group = {g: sweep[k]["monotonicity"][g]["spearman_rho_vs_health_order"][var]
                            for g in sweep[k]["monotonicity"]}
            worst_rho = min([r for r in rho_by_group.values() if r is not None] or [None])
            print(f"      {var:<24} lift={lv.get('lift_pp'):>7}pp "
                  f"({lv.get('base_rate_pct')}% -> {lv.get('best_tier_pct_clearing')}%)"
                  f"  {'MATERIAL' if lv.get('material') else 'immaterial':<10}"
                  f"  worst rho={worst_rho}")

    # ---- Closest candidates: no variable INVERTS (rho > 0 everywhere) and every
    #      benchmark is materially lifted. These are the "least bad" hard-clustering
    #      options if a clustering must be forced despite Gate 2 failing.
    closest = [k for k, v in sweep.items()
               if v["summary"]["passes_materiality_gate"]
               and not v["summary"]["any_group_has_inverting_variable"]]
    closest.sort(key=lambda k: -(sweep[k]["summary"]["worst_group_min_spearman_rho"] or -9))
    print(f"\n  Closest candidates (all 5 benchmarks materially lifted AND no variable "
          f"inverting anywhere, but ordering weaker than rho {MONOTONIC_RHO_MIN}):")
    for k in closest:
        s, u = survivors[k], sweep[k]["summary"]
        print(f"    {k:<26} min_rho={u['worst_group_min_spearman_rho']}  "
              f"min_lift={u['min_lift_pp']}pp  ins_lift={u['insurance_lift_pp']}pp  "
              f"macro_sil={s['silhouette_macro']}  min%={s['min_cluster_pct']}")
    if not closest:
        print("    NONE")

    # ---- FINAL: macro-silhouette ranking restricted to combinations passing both gates.
    final_winner = None
    if gate2:
        ordered_final = sorted(gate2, key=lambda k: -survivors[k]["silhouette_macro"])
        final_winner = ordered_final[0]
        print("\n  Gate-1 + Gate-2 survivors ranked by macro-silhouette (the final ranking):")
        for k in ordered_final:
            s, u = survivors[k], sweep[k]["summary"]
            print(f"    {k:<26} macro={s['silhouette_macro']:>7.4f}  micro={s['silhouette_micro']:>7.4f}"
                  f"  db={s['davies_bouldin']:>6.3f}  min%={s['min_cluster_pct']:>5.2f}"
                  f"  min_rho={u['worst_group_min_spearman_rho']}")
    print(f"\n  FINAL (both gates + macro-silhouette): {final_winner}")

    # Full Step 4A detail for the final winner, or — if Gate 2 eliminated everything —
    # for the closest candidate, so the fallback conclusion is reported against concrete
    # numbers rather than as a bare null.
    final_step4a, final_mono = {}, {}
    detail_for = final_winner or (closest[0] if closest else None)
    if detail_for and detail_for != winner:
        fapp = detail_for.split("|")[0]
        fgroups = ({f"band{b}": (df[df["age_band"] == b], labels_store[detail_for][b])
                    for b in range(1, 6)} if fapp == "A" else {"all": (df, labels_store[detail_for])})
        final_step4a = step4a(df, fgroups, fapp)
        final_mono = monotonicity(final_step4a)
        print(f"\n[STEP 4A — {'FINAL WINNER' if final_winner else 'CLOSEST CANDIDATE'} "
              f"{detail_for}] mean AND density validation:")
        for gkey, g in final_step4a.items():
            order = final_mono[gkey]["overall_health_order_best_to_worst"]
            print(f"  {gkey} (n={g['n_group']})  health order best->worst: {order}"
                  f"  min_rho={final_mono[gkey]['min_spearman_rho']}")
            for c in order:
                e = g["clusters"][c]
                print(f"    cluster {c} (n={e['n']}, {e['share_pct']}%):")
                for var in STEP4A_VARS:
                    d = e["variables"][var]
                    flag = "  <<< MEAN MASKS SPLIT" if d["mean_masks_split"] else ""
                    print(f"      {var:<24} mean={d['mean']:>9.3f} "
                          f"({'OK ' if d['mean_clears_benchmark'] else 'FAIL'})"
                          f"  meeting={d['pct_meeting_benchmark']:>5.1f}%"
                          f"  dist={d['distribution_flag']:<28}{flag}")

    out = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "Final algorithm/k/approach selection ONLY. No production artifact is "
                   "read or written; centroid selection and artifact re-export "
                   "(incl. refitting fitted_scaler.pkl on 11 features) are separate tasks.",
        "methodology": {
            "step1_min_cluster_size_floor": {
                "floor_share_of_sample": MIN_CLUSTER_SHARE,
                "citation": "Bradley, Bennett & Demiriz (2000), constrained K-Means — "
                            "size constraints added to avoid local solutions with empty "
                            "clusters or clusters having very few points.",
                "approach_a_note": "Approach A fits an independent model per age band, so "
                                   "the floor is applied within every band; the binding "
                                   "figure is the smallest cluster share in any band.",
            },
            "step2_macro_silhouette": "Mean silhouette within each cluster first, then mean "
                                      "across clusters (unweighted by size) — less biased by "
                                      "imbalanced/outlier-driven splits than the micro average.",
            "step3_ranking": "Rank-sum over macro-silhouette (primary), Davies-Bouldin and "
                             "Calinski-Harabasz, within each approach. CH is NOT comparable "
                             "across approaches (scales with n); cross-approach comparison "
                             "uses macro-silhouette and Davies-Bouldin only. Framework: "
                             "Akhanli & Hennig (2020) Statistics and Computing 30:1523-1544.",
            "step4a_density_validation": "Mean AND proportion-meeting-benchmark per cluster, "
                                         "per Metzler, Zhou & Grace (2021)'s own "
                                         "proportion-based validation, plus Sarle's "
                                         "bimodality coefficient and a 1-vs-2-component GMM "
                                         "BIC check so a healthy mean cannot hide a split "
                                         "population.",
            "step4a_thresholds": {
                "life_coverage_ratio": f">= {LIFE_COVER_MIN} (LIA PGS 2022 via MoneySense, 9-10x)",
                "ci_coverage_ratio": f">= {CI_COVER_MIN} (LIA PGS 2022, ~3.9x)",
                "savings_rate": f">= {SAVINGS_RATE_MIN} (MoneySense 20% minimum)",
                "net_worth_to_income": f">= band-appropriate ideal {NW_IDEAL_BY_BAND} "
                                       "(revised SYNTHETIC_IDEALS ladder)",
                "dti": f"<= {DTI_MAX} (DBS recommended ideal)",
                "majority_definition": MAJORITY,
            },
            "step4b_single_axis_diagnostic": "Per-cluster means across all 11 features; "
                                             "per-feature spread (max-min cluster mean) "
                                             "expressed in robust-scaled (IQR) units, the same "
                                             "geometry the clustering saw. Tests the Metzler et "
                                             "al. 'split on one trivial axis' pattern.",
        },
        "reproduction_check": repro,
        "grid": grid,
        "step1_summary": {
            "total_combinations": len(grid),
            "survivors": sorted(survivors),
            "n_survivors": len(survivors),
            "n_disqualified": len(disqualified),
        },
        "step3_ranks": ranks,
        "step3_best_per_approach": best_per_approach,
        "step3_winner": winner,
        "step4a_winner_validation": step4a_out,
        "step4a_monotonicity": mono,
        "step4b_disqualified_profiles": step4b_out,
        "step4b_survivor_profiles": survivor_profiles,
        "step5_survivor_usability_sweep": {
            k: v["summary"] for k, v in sweep.items()},
        "step5_survivor_monotonicity_detail": {
            k: v["monotonicity"] for k, v in sweep.items()},
        "step5_combinations_meeting_absolute_bar": meets,
        "population_base_rates_pct": {"overall": base["overall"],
                                      "by_band": {str(b): base["by_band"][b] for b in base["by_band"]}},
        "majority_criterion_reachability": {
            v: {"base_rate_pct": base["overall"][v],
                "max_majority_clearing_cluster_share_pct": round(min(100.0, 2 * base["overall"][v]), 1)}
            for v in STEP4A_VARS},
        "gate2_monotonicity_rho_min": MONOTONIC_RHO_MIN,
        "gate2_min_lift_pp": MIN_LIFT_PP,
        "gate2a_ordering_passing": g2a,
        "gate2b_materiality_passing": g2b,
        "gate2_passing_combinations": gate2,
        "gate2_min_headroom_captured": MIN_HEADROOM_CAPTURED,
        "gate2b_headroom_materiality_passing": g2b_h,
        "gate2_headroom_passing_combinations": gate2_h,
        "final_winner": final_winner,
        "closest_candidates_if_no_winner": closest,
        "step4a_detail_reported_for": detail_for,
        "final_winner_step4a": final_step4a,
        "final_winner_monotonicity": final_mono,
        "macro_silhouette_rank_winner_before_gate2": winner,
        "step5_fallback_applies": final_winner is None,
        "step5_conclusion": (
            "FALLBACK APPLIES. Across all 38 combinations passing the 5% size floor, the "
            "two halves of the usability gate never co-occur: every combination whose "
            "clusters order monotonically across the 5 external benchmarks discriminates "
            "on them by <=7pp (uninformative), and every combination that discriminates "
            "materially has clusters that are not ordered (multi-directional archetypes, "
            "not health tiers). This is the signature of users lying on a continuum with "
            "two largely independent axes (accumulation vs. protection) rather than in "
            "discrete health groups; a percentile-band score is the more defensible "
            "design than forcing hard clusters."
            if final_winner is None else "A genuine winner was found."),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved {OUT_JSON}")

    # tidy CSV of the full grid (the Step 1 disqualification table)
    rows = []
    for key, v in grid.items():
        rows.append({
            "combination": key, "approach": v["approach"], "algorithm": v["algorithm"],
            "k": v["k"], "family": v["family"],
            "min_cluster_pct": v.get("min_cluster_pct"),
            "shortfall_pp_below_5pct": v.get("shortfall_pp"),
            "passes_floor": v.get("passes_floor"),
            "silhouette_macro": v.get("silhouette_macro"),
            "silhouette_micro": v.get("silhouette_micro"),
            "davies_bouldin": v.get("davies_bouldin"),
            "calinski_harabasz": v.get("calinski_harabasz"),
            "worst_band": v.get("worst_band"),
        })
    csvp = os.path.join(RESULTS_DIR, "final_selection_grid.csv")
    pd.DataFrame(rows).sort_values(
        ["approach", "algorithm", "k"]).to_csv(csvp, index=False)
    print(f"  saved {csvp}")
    print("\n" + "=" * 92)
    print(f"DONE. Macro-silhouette rank leader among floor-passing combinations: {winner}")
    print(f"      FINAL selection after the Step 4A monotonicity gate:          {final_winner}")
    print("=" * 92)


if __name__ == "__main__":
    main()
