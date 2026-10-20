"""
domain_split_clustering.py — Layer 2 as THREE independent domain clusterings
(Unify, CCDS25-1170, July 2026).

WHY THIS EXISTS
  The gate-based selection (final_selection.py) established, exhaustively, that ONE
  clustering across all 11 features cannot serve as a single ordered financial-health
  spectrum: accumulation and protection behave as largely independent axes, so a
  combination either orders monotonically but discriminates on nothing, or discriminates
  materially but cannot be ranked. See the brief's ARCHITECTURAL FORK section.

  This script tests the alternative that follows directly from that finding: stop forcing
  one spectrum, and run THREE separate clustering exercises — one per data source the app
  actually has (plaidAccounts -> Banking, investmentAccounts -> Investment,
  insurancePolicies -> Insurance) — each scored on its own terms. The 3-way split is
  chosen over an ML-derived 2-way accumulation/protection split because it matches both
  the app's real data architecture and the already-designed Home Phase 4 Financial Health
  Score card (Liquidity / Growth / Protection sub-scores).

  METHODOLOGY IS DELIBERATELY UNCHANGED from final_selection.py — same 5% size floor
  (Bradley et al. 2000), same macro-averaged silhouette primary metric, same rank-sum
  with Davies-Bouldin / Calinski-Harabasz (Akhanli & Hennig 2020), same monotonic-ordering
  gate (Phelps & Metzler 2024) and same mean-AND-density dual validation with ceiling-aware
  materiality. The density half follows Metzler, Zhou & Grace (2021), Quantitative Finance
  and Economics 5(3), for its own proportion-based validation of clusters — that paper's
  algorithm and data type do NOT transfer here and are not claimed to. Only the FEATURE SET
  each clustering sees changes. Shared functions are IMPORTED, not reimplemented.

  This script VALIDATES ONLY. It writes nothing outside MLAI/clustering/results/ — no
  production artifact is read or written, no Layer 3 combination rules are defined, and
  scoring.py is untouched.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/domain_split_clustering.py
"""

import itertools
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
from sklearn.cluster import HDBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_benchmark_11algo import run_kbased, FAMILY  # noqa: E402
from benchmark_core import compute_metrics, SYNTHETIC_IDEALS  # noqa: E402
from final_selection import macro_silhouette, cluster_shares, bimodality  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results")
OUT_DIR = os.path.join(RESULTS_DIR, "domain_split")
LABEL_CACHE = os.path.join(OUT_DIR, ".domain_split_labels_v2.pkl")

DOMAIN_ORDER = ["banking", "investment", "insurance"]

# ======================================================================================
# THE FEATURE SPLIT — net_worth_to_income placement is treated as an EXPERIMENTAL FACTOR,
# not an assumption.
# ======================================================================================
# The task allows three options for net_worth_to_income (Banking / both / its own
# treatment). Rather than argue it a priori, BOTH single-domain placements are run through
# the identical grid + gate process and the primary spec is chosen on the evidence; the
# "in both domains" option is additionally evaluated for its Step 5 effect. Placing it in
# both is disfavoured on methodology grounds regardless of how it scores: a shared input
# mechanically induces correlation between Banking and Investment tiers, which would
# contaminate Step 5 — whose entire purpose is to measure whether those two domains are
# independent. A shared feature must not be allowed to manufacture the answer.
SPECS = {
    "banking_core4":       ("banking", ["savings_rate", "emergency_fund_months", "dti",
                                        "healthy_debt_ratio"]),
    "banking_plus_nw":     ("banking", ["savings_rate", "emergency_fund_months", "dti",
                                        "healthy_debt_ratio", "net_worth_to_income"]),
    "investment_core3":    ("investment", ["investment_rate", "portfolio_concentration",
                                           "risk_age_deviation"]),
    "investment_plus_nw":  ("investment", ["investment_rate", "portfolio_concentration",
                                           "risk_age_deviation", "net_worth_to_income"]),
    "insurance_core3":     ("insurance", ["life_coverage_ratio", "ci_coverage_ratio",
                                          "insurance_spend_ratio"]),
}
BENCH_OF = {s: v[0] for s, v in SPECS.items()}
FEATS_OF = {s: v[1] for s, v in SPECS.items()}
ALT_SPECS = {"banking": ["banking_core4", "banking_plus_nw"],
             "investment": ["investment_core3", "investment_plus_nw"],
             "insurance": ["insurance_core3"]}

# ALGORITHM SLATE — 8 of the canonical 12 (choice documented in the report):
#   kept:    kmeans, kmedoids, gmm_full, gmm_diag, agglomerative, birch  (the requested core)
#   added:   spectral      — it produced the degenerate k=2 "winner" in the combined run; it
#                            is included specifically so the domain split can be tested for
#                            the SAME failure mode rather than assumed free of it.
#            fuzzy_cmeans  — soft-centroid family completeness (Wani 2024 taxonomy); it
#                            returned empty clusters at k>=4 before, which the size floor
#                            will catch on its own rather than by pre-exclusion.
#   dropped: archetypal    — 10/10 combinations failed the 5% floor in the combined run
#                            (down to 0.07%); with 3-5 features per domain the convex-hull
#                            degeneracy gets worse, not better. No selection value.
#            dbscan / hdbscan / umap_hdbscan — these produce no k, so they cannot enter a
#                            k-based gate/selection grid. HDBSCAN is instead run as a
#                            separate DIAGNOSTIC per domain (natural cluster count), which
#                            is the only question it can actually answer here.
K_BASED = ["kmeans", "kmedoids", "fuzzy_cmeans", "gmm_full", "gmm_diag",
           "spectral", "agglomerative", "birch"]
K_RANGE = range(2, 7)

MIN_CLUSTER_SHARE = 0.05        # Bradley, Bennett & Demiriz (2000)
MONOTONIC_RHO_MIN = 0.5         # Gate 2a, unchanged
MIN_LIFT_PP = 10.0              # Gate 2b (absolute), unchanged
MIN_HEADROOM_CAPTURED = 0.10    # Gate 2b' (ceiling-aware), unchanged
MAJORITY = 0.50
BC_FLAG = 0.555

# ======================================================================================
# PER-DOMAIN OUT-OF-SAMPLE BENCHMARK VARIABLES
# ======================================================================================
# "Out-of-sample" here means what it means everywhere else in this project: the THRESHOLD
# is an external SG benchmark (MAS/LIA/MoneySense/SingStat/DBS/MOH/AIA) the clustering
# never optimised toward. Three variables are additionally out-of-MODEL in the strict
# sense — has_investments, hospitalisation_coverage and income_protection_coverage are in
# the dataset but appear in NO domain's feature set at all (they are Layer 1 checklist
# features). Those are flagged strictly_out_of_model=True. net_worth_to_income is strictly
# out-of-model for banking_core4 and in-model for banking_plus_nw, which is itself part of
# what the two variants are being compared on.
NW_IDEAL_BY_BAND = {b: SYNTHETIC_IDEALS[b]["net_worth_to_income"] for b in range(1, 6)}
CONC_IDEAL_BY_BAND = {b: SYNTHETIC_IDEALS[b]["portfolio_concentration"] for b in range(1, 6)}
LIFE_IDEAL_BY_BAND = {b: SYNTHETIC_IDEALS[b]["life_coverage_ratio"] for b in range(1, 6)}

BENCHMARKS = {
    "banking": [
        ("savings_rate", ">=", 0.20, "MoneySense 20% minimum", False),
        ("emergency_fund_months", ">=", 6.0, "MoneySense 3-6 month fund (6 = upper end)", False),
        ("dti", "<=", 0.35, "DBS '7 financial ratios' recommended ideal <=35%", False),
        ("net_worth_to_income", ">=", "band_nw", "band-appropriate ladder 1/3/6/8/10", False),
        ("healthy_debt_ratio", ">=", 0.70, "SingStat Household Balance Sheet ~70-71% structured", False),
    ],
    "investment": [
        ("investment_rate", ">=", 0.10, "MoneySense/MAS 'invest at least 10% of income'", False),
        ("portfolio_concentration", "<=", "band_conc", "band taper 0.20/0.17/0.13/0.10/0.08", False),
        ("risk_age_deviation", "abs<=", 0.10, "within 10pp of the 100-minus-age equity target", False),
        ("has_investments", "==1", 1, "OCBC FWI 2024 88% participation (STRICTLY out-of-model)", True),
    ],
    "insurance": [
        ("life_coverage_ratio", ">=", "band_life", "LIA PGS 2022 9x (band 5: 5x, per SYNTHETIC_IDEALS)", False),
        ("ci_coverage_ratio", ">=", 3.9, "LIA PGS 2022 ~3.9x income", False),
        ("hospitalisation_coverage", "==1", 1, "MOH ~71% Integrated Shield prevalence (STRICTLY out-of-model)", True),
        ("income_protection_coverage", "==1", 1, "AIA Disability Income Survey 2022 ~53% (STRICTLY out-of-model)", True),
    ],
}
BENCH_VARS = {d: [b[0] for b in BENCHMARKS[d]] for d in BENCHMARKS}

# Life coverage uses the BAND-APPROPRIATE threshold (9/9/9/9/5), not the flat 9.0 used by
# the combined run. The brief itself documents the flat threshold as over-flagging Band 5,
# whose own age-appropriate ideal is 5.0x. Stated here so the correction is visible.

# ======================================================================================
# EXPECTED TIER COMBINATIONS — derived from the generator's ACTUAL overrides
# (generate_synthetic_data.py MISMATCH_TYPES), not from assumption.
# ======================================================================================
# "MID" means: this profile applies NO override to that domain's features, so its members
# are drawn from the ordinary copula population there and no directional expectation is
# warranted. Recording that explicitly matters — the task brief expects wealthy_exposed to
# land HIGH on Investment and protection_first to land LOW on Investment, but neither
# profile touches any investment feature, so neither expectation is supported by the data
# design. That divergence is quantified below rather than smoothed over.
EXPECTED = {
    "conservative_saver": {"banking": "HIGH", "investment": "LOW",  "insurance": "MID"},
    "over_invested":      {"banking": "LOW",  "investment": "HIGH", "insurance": "MID"},
    "protection_first":   {"banking": "LOW",  "investment": "MID",  "insurance": "HIGH"},
    "wealthy_exposed":    {"banking": "HIGH", "investment": "MID",  "insurance": "LOW"},
    "high_debt_investor": {"banking": "LOW",  "investment": "HIGH", "insurance": "MID"},
    "underinsured_saver": {"banking": "HIGH", "investment": "MID",  "insurance": "LOW"},
    "normal":             {"banking": "MID",  "investment": "MID",  "insurance": "MID"},
}
BRIEF_EXPECTED = {
    "wealthy_exposed":  {"banking": "HIGH", "investment": "HIGH", "insurance": "LOW"},
    "protection_first": {"banking": "LOW",  "investment": "LOW",  "insurance": "HIGH"},
}
OVERRIDDEN_DOMAINS = {
    "conservative_saver": ["banking", "investment"],
    "over_invested":      ["banking", "investment"],
    "protection_first":   ["banking", "insurance"],
    "wealthy_exposed":    ["banking", "insurance"],
    "high_debt_investor": ["banking", "investment"],
    "underinsured_saver": ["banking", "insurance"],
    "normal":             [],
}
PROFILE_ORDER = ["normal", "conservative_saver", "over_invested", "protection_first",
                 "wealthy_exposed", "high_debt_investor", "underinsured_saver", "outlier"]


# ======================================================================================
# Benchmark evaluation
# ======================================================================================
def benchmark_for(bd, var):
    """The (variable, operator, threshold, citation, strictly_out_of_model) row for `var`."""
    return next(b for b in BENCHMARKS[bd] if b[0] == var)


def meets_mask(sub, bd, var):
    """Boolean mask: does each user in `sub` clear this domain's benchmark for `var`?"""
    _, op, thr, _, _ = benchmark_for(bd, var)
    v = sub[var].to_numpy(float)
    if op == "<=":
        t = (sub["age_band"].map(CONC_IDEAL_BY_BAND).to_numpy(float)
             if thr == "band_conc" else float(thr))
        return v <= t
    if op == ">=":
        if thr == "band_nw":
            t = sub["age_band"].map(NW_IDEAL_BY_BAND).to_numpy(float)
        elif thr == "band_life":
            t = sub["age_band"].map(LIFE_IDEAL_BY_BAND).to_numpy(float)
        else:
            t = float(thr)
        return v >= t
    if op == "abs<=":
        return np.abs(v) <= float(thr)
    if op == "==1":
        return v >= 0.5
    raise ValueError(op)


def health_signal(sub, bd, var):
    """A scalar per user where HIGHER = healthier, used for cluster ordering."""
    _, op, _, _, _ = benchmark_for(bd, var)
    d = {">=": 1, "==1": 1, "<=": -1, "abs<=": 0}[op]
    v = sub[var].to_numpy(float)
    return -np.abs(v) if d == 0 else d * v


def base_rates(df, bd):
    out = {"overall": {v: round(100 * float(meets_mask(df, bd, v).mean()), 1)
                       for v in BENCH_VARS[bd]}, "by_band": {}}
    for b in range(1, 6):
        sub = df[df["age_band"] == b]
        out["by_band"][b] = {v: round(100 * float(meets_mask(sub, bd, v).mean()), 1)
                             for v in BENCH_VARS[bd]}
    return out


# ======================================================================================
# STEP 1 + 2 — grid, size floor, macro silhouette (per spec)
# ======================================================================================
def run_spec_grid(df, spec, scaler, X_full):
    feats = FEATS_OF[spec]
    grid, labels_store = {}, {}
    bands = {b: df[df["age_band"] == b] for b in range(1, 6)}
    Xb = {b: scaler.transform(bands[b][feats]) for b in bands}
    for approach in ("A", "B"):
        for k in K_RANGE:
            for algo in K_BASED:
                key = f"{approach}|{algo}|{k}"
                try:
                    if approach == "A":
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
                            per_band[b] = {"silhouette_micro": m["silhouette_score"],
                                           "silhouette_macro": round(mac_b, 4),
                                           "davies_bouldin": m["davies_bouldin_index"],
                                           "calinski_harabasz": m["calinski_harabasz_index"],
                                           "n_clusters_found": nf,
                                           "cluster_shares_pct": [round(100 * s, 2) for s in sh],
                                           "min_cluster_pct": round(100 * min(sh), 2)}
                        labels_store[key] = band_labels
                        rec = {"silhouette_micro": round(float(np.mean(mic)), 4),
                               "silhouette_macro": round(float(np.mean(mac)), 4),
                               "davies_bouldin": round(float(np.mean(dbs)), 4),
                               "calinski_harabasz": round(float(np.mean(chs)), 4),
                               "min_cluster_pct": round(100 * min(shares_all), 2),
                               "min_cluster_pct_by_band": {str(b): per_band[b]["min_cluster_pct"]
                                                           for b in per_band},
                               "worst_band": int(min(per_band,
                                                     key=lambda b: per_band[b]["min_cluster_pct"])),
                               "per_band": per_band}
                    else:
                        lab = run_kbased(algo, k, X_full)
                        labels_store[key] = lab
                        m = compute_metrics(X_full, lab)
                        mac_b, per_c = macro_silhouette(X_full, lab)
                        sh, nf = cluster_shares(lab, k)
                        rec = {"silhouette_micro": m["silhouette_score"],
                               "silhouette_macro": round(mac_b, 4),
                               "davies_bouldin": m["davies_bouldin_index"],
                               "calinski_harabasz": m["calinski_harabasz_index"],
                               "n_clusters_found": nf,
                               "cluster_shares_pct": [round(100 * s, 2) for s in sh],
                               "min_cluster_pct": round(100 * min(sh), 2),
                               "per_cluster_silhouette": {str(c): round(v, 4)
                                                          for c, v in per_c.items()}}
                    rec["passes_floor"] = rec["min_cluster_pct"] >= 100 * MIN_CLUSTER_SHARE
                    rec["shortfall_pp"] = round(
                        max(0.0, 100 * MIN_CLUSTER_SHARE - rec["min_cluster_pct"]), 2)
                    rec["failed"] = False
                except Exception as e:                                  # pragma: no cover
                    rec = {"failed": True, "error": f"{type(e).__name__}: {e}",
                           "passes_floor": False, "min_cluster_pct": None, "shortfall_pp": None}
                    print(f"      WARN {spec} {key}: {type(e).__name__}: {e}")
                grid[key] = {"spec": spec, "approach": approach, "algorithm": algo,
                             "k": k, "family": FAMILY[algo], **rec}
    return grid, labels_store


def rank_survivors(survivors):
    """Rank-sum over macro-silhouette (primary) + DB + CH, WITHIN each approach.
    CH is not comparable across approaches (scales with n), so the cross-approach call is
    made on macro-silhouette, which is scale-free — identical logic to final_selection."""
    out = {}
    for approach in ("A", "B"):
        pool = {kk: v for kk, v in survivors.items() if v["approach"] == approach}
        if not pool:
            out[approach] = {}
            continue
        keys = list(pool)
        r_mac = {a: i + 1 for i, a in enumerate(
            sorted(keys, key=lambda x: pool[x]["silhouette_macro"], reverse=True))}
        r_db = {a: i + 1 for i, a in enumerate(sorted(keys, key=lambda x: pool[x]["davies_bouldin"]))}
        r_ch = {a: i + 1 for i, a in enumerate(
            sorted(keys, key=lambda x: pool[x]["calinski_harabasz"], reverse=True))}
        out[approach] = {a: {"rank_macro_sil": r_mac[a], "rank_db": r_db[a], "rank_ch": r_ch[a],
                             "composite_rank_macro": r_mac[a] + r_db[a] + r_ch[a]} for a in keys}
    return out


# ======================================================================================
# STEP 3 — mean AND density validation, ordering, materiality
# ======================================================================================
def distribution_flag(bim):
    """Name the shape a cluster's values take for one variable, from the two independent
    signals bimodality() returns. Either alone is weak; both together is the real flag."""
    bc = bim.get("bimodality_coefficient") or 0
    two_components = bim.get("gmm_prefers_2")
    if two_components and bc > BC_FLAG:
        return "bimodal/split"
    if bc > BC_FLAG:
        return "wide-spread"
    if two_components:
        return "two-component fit preferred"
    return "unimodal"


def cluster_health_order(sub, labels, bd):
    cids = sorted(int(c) for c in np.unique(labels))
    score = {c: 0.0 for c in cids}
    per_var_vals = {}
    for var in BENCH_VARS[bd]:
        sig = health_signal(sub, bd, var)
        vals = {c: float(sig[labels == c].mean()) for c in cids}
        per_var_vals[var] = vals
        for rank, c in enumerate(sorted(cids, key=lambda x: vals[x], reverse=True)):
            score[c] += rank
    return sorted(cids, key=lambda c: score[c]), per_var_vals


def validate_group(sub, labels, bd, band_for_thr):
    order, per_var_vals = cluster_health_order(sub, labels, bd)
    health_rank = {c: i for i, c in enumerate(order)}
    clusters = {}
    for c in order:
        cl = sub[labels == c]
        entry = {"n": int(len(cl)), "share_pct": round(100 * len(cl) / len(sub), 2),
                 "health_rank": health_rank[c], "variables": {}}
        for var in BENCH_VARS[bd]:
            _, op, thr_spec, _, strictly_out_of_model = benchmark_for(bd, var)
            v = cl[var].to_numpy(float)
            meets = meets_mask(cl, bd, var)
            if isinstance(thr_spec, str):
                tab = {"band_nw": NW_IDEAL_BY_BAND, "band_conc": CONC_IDEAL_BY_BAND,
                       "band_life": LIFE_IDEAL_BY_BAND}[thr_spec]
                thr = tab[band_for_thr] if band_for_thr else f"per-band {tab}"
            else:
                thr = float(thr_spec)
            if op == "==1":
                # Binary coverage flag: "mean clears" is only meaningful as "most of the
                # cluster has it" — mean >= 1.0 would require a 100%-covered cluster.
                mean_ok = bool(v.mean() >= 0.5)
            elif not isinstance(thr, (int, float)):
                # A per-band threshold pooled across bands has no single value to clear.
                mean_ok = None
            elif op == "abs<=":
                mean_ok = bool(abs(v.mean()) <= thr)
            elif op == "<=":
                mean_ok = bool(v.mean() <= thr)
            else:
                mean_ok = bool(v.mean() >= thr)
            pct = float(meets.mean())
            bim = bimodality(v)
            entry["variables"][var] = {
                "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                "q25": round(float(np.percentile(v, 25)), 4),
                "q75": round(float(np.percentile(v, 75)), 4),
                "benchmark": thr, "strictly_out_of_model": strictly_out_of_model,
                "pct_meeting_benchmark": round(100 * pct, 1),
                "mean_clears_benchmark": mean_ok,
                "majority_clears_benchmark": bool(pct >= MAJORITY),
                "usable": bool(mean_ok and pct >= MAJORITY) if mean_ok is not None else None,
                "mean_masks_split": bool(mean_ok and pct < MAJORITY) if mean_ok is not None else None,
                "bimodality_coefficient": bim.get("bimodality_coefficient"),
                "gmm_prefers_2": bim.get("gmm_prefers_2"),
                "distribution_flag": distribution_flag(bim),
            }
        clusters[c] = entry
    rhos = {}
    for var in BENCH_VARS[bd]:
        hv = [per_var_vals[var][c] for c in order]
        hr = [-health_rank[c] for c in order]
        rhos[var] = (None if len(set(hv)) < 2
                     else round(float(stats.spearmanr(hv, hr).statistic), 3))
    valid = [r for r in rhos.values() if r is not None]
    return {"n_group": int(len(sub)), "health_order_best_to_worst": order, "clusters": clusters,
            "spearman_rho_vs_health_order": rhos,
            "min_spearman_rho": min(valid) if valid else None,
            "mean_spearman_rho": round(float(np.mean(valid)), 3) if valid else None,
            "any_variable_inverts": bool(valid and min(valid) < 0)}


def discrimination_lift(groups, base, bd, approach):
    per_var = {v: [] for v in BENCH_VARS[bd]}
    for gkey, (sub, lab) in groups.items():
        bref = (base["by_band"][int(sub["age_band"].iloc[0])] if approach == "A"
                else base["overall"])
        for var in BENCH_VARS[bd]:
            m = meets_mask(sub, bd, var)
            cand = [float(m[lab == c].mean()) for c in np.unique(lab)
                    if (lab == c).mean() >= MIN_CLUSTER_SHARE]
            if cand:
                per_var[var].append((100 * max(cand), bref[var], gkey))
    out = {}
    for var in BENCH_VARS[bd]:
        if not per_var[var]:
            continue
        lifts = [(p - b) for p, b, _ in per_var[var]]
        heads = [((p - b) / (100.0 - b) if b < 100.0 else 1.0) for p, b, _ in per_var[var]]
        i, j = int(np.argmin(lifts)), int(np.argmin(heads))
        p, b, gk = per_var[var][i]
        ph, bh, gkh = per_var[var][j]
        out[var] = {"worst_group": gk, "best_tier_pct_clearing": round(p, 1),
                    "base_rate_pct": round(b, 1), "lift_pp": round(p - b, 1),
                    "mean_lift_pp_across_groups": round(float(np.mean(lifts)), 1),
                    "material": bool((p - b) >= MIN_LIFT_PP),
                    "max_possible_lift_pp": round(100.0 - b, 1),
                    "worst_group_headroom": gkh,
                    "worst_headroom_captured": round(heads[j], 3),
                    "worst_headroom_base_rate_pct": round(bh, 1),
                    "worst_headroom_best_tier_pct": round(ph, 1),
                    "material_headroom": bool(heads[j] >= MIN_HEADROOM_CAPTURED)}
    return out


def groups_for(df, key, labels_store):
    if key.split("|")[0] == "A":
        return {f"band{b}": (df[df["age_band"] == b], labels_store[key][b]) for b in range(1, 6)}
    return {"all": (df, labels_store[key])}


def usability(df, key, labels_store, bd, base):
    approach = key.split("|")[0]
    groups = groups_for(df, key, labels_store)
    per_group = {}
    for gkey, (sub, lab) in groups.items():
        band = int(sub["age_band"].iloc[0]) if approach == "A" else None
        per_group[gkey] = validate_group(sub, lab, bd, band)
    lift = discrimination_lift(groups, base, bd, approach)
    rhos = [g["min_spearman_rho"] for g in per_group.values() if g["min_spearman_rho"] is not None]
    nvars = len(BENCH_VARS[bd])
    best_usable = []
    for g in per_group.values():
        cv = g["clusters"][g["health_order_best_to_worst"][0]]["variables"]
        best_usable.append(sum(1 for v in BENCH_VARS[bd] if cv[v]["usable"]))
    masked = sum(1 for g in per_group.values() for c in g["clusters"].values()
                 for v in BENCH_VARS[bd] if c["variables"][v]["mean_masks_split"])
    return {
        "per_group": per_group, "lift": lift,
        "worst_group_min_spearman_rho": min(rhos) if rhos else None,
        "mean_min_spearman_rho": round(float(np.mean(rhos)), 3) if rhos else None,
        "any_group_has_inverting_variable": any(g["any_variable_inverts"] for g in per_group.values()),
        "worst_group_vars_usable": min(best_usable), "n_benchmark_vars": nvars,
        "total_mean_masks_split_flags": masked,
        "min_lift_pp": min(v["lift_pp"] for v in lift.values()) if lift else None,
        "min_headroom_captured": (min(v["worst_headroom_captured"] for v in lift.values())
                                  if lift else None),
        "passes_monotonicity_gate": bool(rhos and min(rhos) >= MONOTONIC_RHO_MIN),
        "passes_materiality_gate": bool(lift and len(lift) == nvars
                                        and all(v["material"] for v in lift.values())),
        "passes_materiality_gate_headroom": bool(lift and len(lift) == nvars
                                                 and all(v["material_headroom"] for v in lift.values())),
        "passes_gate2_headroom": bool(rhos and min(rhos) >= MONOTONIC_RHO_MIN and lift
                                      and len(lift) == nvars
                                      and all(v["material_headroom"] for v in lift.values())),
    }


# ======================================================================================
# Tier assignment + diagnostics
# ======================================================================================
def assign_tiers(df, key, labels_store, bd):
    """Return (rank Series 0=healthiest, k found, coarse HIGH/MID/LOW Series).

    The coarse HIGH/MID/LOW collapse is an ANALYSIS-STAGE readability device only — it
    exists so domains with different k can be compared side by side in the Step 4 table.
    It is population-share based (each tier is placed in the third of the ranked population
    its own members occupy), which is a RELATIVE construction and must never be used for
    production tier assignment: the brief's Score Derivation constraint requires absolute,
    benchmark-anchored thresholds there. Exact tier ranks are reported alongside, and are
    the form any future Layer 3 rule should actually be written against."""
    rank = pd.Series(index=df.index, dtype="float")
    for gkey, (sub, lab) in groups_for(df, key, labels_store).items():
        order, _ = cluster_health_order(sub, lab, bd)
        hr = {c: i for i, c in enumerate(order)}
        rank.loc[sub.index] = [hr[int(c)] for c in lab]
    rank = rank.astype(int)
    kfound = int(rank.max()) + 1
    shares = rank.value_counts(normalize=True).sort_index()
    cum, mapping = 0.0, {}
    for t in range(kfound):
        s = float(shares.get(t, 0.0))
        mid = cum + s / 2
        mapping[t] = "HIGH" if mid < 1 / 3 else ("LOW" if mid > 2 / 3 else "MID")
        cum += s
    return rank, kfound, rank.map(mapping)


def separation_profile(df_sub, labels, feats, iqr):
    """How multidimensional is this partition? Per-feature spread of cluster means in IQR
    units; a high top_feature_share is the Metzler, Zhou & Grace (2021) 'split on one
    trivial axis' signature — the ground on which they rejected their own
    silhouette-competitive k=2."""
    prof = df_sub[feats].copy()
    prof["__c"] = labels
    prof = prof.groupby("__c")[feats].mean()
    spreads = {f: float((prof[f].max() - prof[f].min()) / (iqr[f] if iqr[f] > 0 else 1.0))
               for f in feats}
    tot = sum(spreads.values())
    top = max(spreads, key=spreads.get)
    return {"spread_iqr_units": {f: round(spreads[f], 3) for f in feats},
            "n_features_separating_ge_0.5": sum(1 for v in spreads.values() if v >= 0.5),
            "n_features": len(feats), "top_feature": top,
            "top_feature_share_of_total_separation": round(spreads[top] / tot, 3) if tot else None}


# ======================================================================================
# Main
# ======================================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print("=" * 118)
    print(f"DOMAIN-SPLIT CLUSTERING — {len(df)} users, "
          f"{len(K_BASED)} algorithms x k=2-6 x 2 approaches = "
          f"{len(K_BASED) * len(list(K_RANGE)) * 2} combinations PER SPEC, {len(SPECS)} specs")
    for s, (bd, f) in SPECS.items():
        print(f"  {s:<20} -> benchmarks:{bd:<11} ({len(f)} features): {f}")
    print("=" * 118)

    cache = {}
    if os.path.exists(LABEL_CACHE):
        with open(LABEL_CACHE, "rb") as fh:
            cache = pickle.load(fh)
        print(f"  (loaded cached labels from {os.path.basename(LABEL_CACHE)} — delete to re-fit)")

    fitted, scalers = {}, {}
    for spec, (bd, feats) in SPECS.items():
        scaler = RobustScaler().fit(df[feats])
        age_scaler = StandardScaler().fit(df[["age_midpoint"]])
        scalers[spec] = scaler
        X_full = np.hstack([scaler.transform(df[feats]),
                            age_scaler.transform(df[["age_midpoint"]])])
        if spec in cache:
            grid, labels_store = cache[spec]["grid"], cache[spec]["labels"]
        else:
            print(f"\n[STEP 1+2] fitting the {spec} grid ...")
            grid, labels_store = run_spec_grid(df, spec, scaler, X_full)
            cache[spec] = {"grid": grid, "labels": labels_store}
        fitted[spec] = {"grid": grid, "labels": labels_store}
    with open(LABEL_CACHE, "wb") as fh:
        pickle.dump(cache, fh)

    out = {"run_timestamp": datetime.now(timezone.utc).isoformat(),
           "purpose": "Validation of the 3-way domain-split (Layer 2) architecture only. "
                      "No production artifact read or written; no Layer 3 rules defined.",
           "specs": {s: {"benchmark_domain": SPECS[s][0], "features": SPECS[s][1]} for s in SPECS},
           "algorithms": K_BASED,
           "methodology_source": "final_selection.py (5% floor, macro-silhouette, "
                                 "mean+density dual check, ceiling-aware materiality)",
           "spec_results": {}}

    spec_winner = {}
    for spec, (bd, feats) in SPECS.items():
        grid, labels_store = fitted[spec]["grid"], fitted[spec]["labels"]
        base = base_rates(df, bd)
        survivors = {k: v for k, v in grid.items() if v.get("passes_floor")}
        disq = {k: v for k, v in grid.items() if not v.get("passes_floor")}

        print(f"\n{'=' * 118}\nSPEC: {spec.upper()}  (benchmarks: {bd}; "
              f"{len(survivors)}/{len(grid)} pass the 5% floor)\n{'=' * 118}")
        print("  population base rates (% clearing each benchmark, no clustering):")
        print("    " + "band".ljust(6) + "".join(v[:26].rjust(28) for v in BENCH_VARS[bd]))
        for b in range(1, 6):
            print("    " + str(b).ljust(6)
                  + "".join(f"{base['by_band'][b][v]:>27.1f}%" for v in BENCH_VARS[bd]))
        print("    " + "ALL".ljust(6)
              + "".join(f"{base['overall'][v]:>27.1f}%" for v in BENCH_VARS[bd]))

        ranks = rank_survivors(survivors)
        print(f"\n  FULL GRID (all {len(grid)} combinations):")
        print(f"    {'combination':<24}{'macroSil':>10}{'microSil':>10}{'DB':>8}{'CH':>10}"
              f"{'minClu%':>9}{'floor':>7}")
        for key in sorted(grid, key=lambda x: (x.split('|')[0], x.split('|')[1],
                                               int(x.split('|')[2]))):
            g = grid[key]
            if g.get("failed"):
                print(f"    {key:<24}  FAILED: {g['error'][:70]}")
                continue
            print(f"    {key:<24}{g['silhouette_macro']:>10.4f}{g['silhouette_micro']:>10.4f}"
                  f"{g['davies_bouldin']:>8.3f}{g['calinski_harabasz']:>10.1f}"
                  f"{g['min_cluster_pct']:>9.2f}{'PASS' if g['passes_floor'] else 'DISQ':>7}")

        sweep = {}
        print(f"\n  [STEP 2/3] usability sweep across all {len(survivors)} floor survivors:")
        for key in sorted(survivors):
            u = usability(df, key, labels_store, bd, base)
            sweep[key] = u
            verdict = ("GATE-2 PASS" if u["passes_gate2_headroom"] else
                       ("ordering-only" if u["passes_monotonicity_gate"] else
                        ("materiality-only" if u["passes_materiality_gate_headroom"] else "fail")))
            print(f"    {key:<24} macroSil={grid[key]['silhouette_macro']:>7.4f}"
                  f"  min_rho={str(u['worst_group_min_spearman_rho']):>7}"
                  f"  min_headroom={str(u['min_headroom_captured']):>6}"
                  f"  min_lift={str(u['min_lift_pp']):>7}pp"
                  f"  usable={u['worst_group_vars_usable']}/{u['n_benchmark_vars']}   {verdict}")

        gate_pass = [k for k in sweep if sweep[k]["passes_gate2_headroom"]]
        pool = gate_pass if gate_pass else sorted(survivors)
        winner = max(pool, key=lambda k: (grid[k]["silhouette_macro"],
                                          -ranks[k.split('|')[0]][k]["composite_rank_macro"]))
        spec_winner[spec] = winner
        print(f"\n  [STEP 2] combinations passing BOTH gates: {len(gate_pass)} -> {gate_pass}")
        print(f"  [STEP 2] WINNER for {spec}: {winner}  (macro-sil "
              f"{grid[winner]['silhouette_macro']}, min_cluster {grid[winner]['min_cluster_pct']}%, "
              f"{'gate-passing' if gate_pass else 'NO GATE-PASSER — best-available fallback'})")

        binding = {v: 0 for v in BENCH_VARS[bd]}
        for u in sweep.values():
            if u["lift"]:
                binding[min(u["lift"], key=lambda v: u["lift"][v]["worst_headroom_captured"])] += 1
        print("  [binding constraint] benchmark with the LOWEST headroom, counted across all "
              f"{len(sweep)} survivors: "
              + "  ".join(f"{v}={c}" for v, c in sorted(binding.items(), key=lambda x: -x[1])))
        wl = sweep[winner]["lift"]
        # NOTE: the worst-LIFT group and the worst-HEADROOM group need not be the same group
        # (a high base rate compresses absolute lift while leaving headroom capture intact),
        # so each is printed with its OWN group's base/best-tier pair — never mixed.
        print("  [winner per-benchmark materiality]  (worst group per measure, reported separately)")
        for v in BENCH_VARS[bd]:
            d = wl[v]
            print(f"      {v:<28} worst-lift grp {str(d['worst_group']):<7} "
                  f"{d['base_rate_pct']:>5.1f}% -> {d['best_tier_pct_clearing']:>5.1f}% "
                  f"(lift {d['lift_pp']:>+6.1f}pp)   |   worst-headroom grp "
                  f"{str(d['worst_group_headroom']):<7} {d['worst_headroom_base_rate_pct']:>5.1f}% -> "
                  f"{d['worst_headroom_best_tier_pct']:>5.1f}% (headroom "
                  f"{d['worst_headroom_captured']:.3f})   "
                  f"{'MATERIAL' if d['material_headroom'] else 'immaterial'}")

        iqr = {f: float(df[f].quantile(0.75) - df[f].quantile(0.25)) for f in feats}
        seps = {gk: separation_profile(sub, lab, feats, iqr)
                for gk, (sub, lab) in groups_for(df, winner, labels_store).items()}
        for gk, sp in seps.items():
            print(f"  [single-axis check] {gk}: {sp['n_features_separating_ge_0.5']}/"
                  f"{sp['n_features']} features separating >=0.5 IQR; top={sp['top_feature']} "
                  f"({100 * sp['top_feature_share_of_total_separation']:.0f}% of total separation)")

        Xn = scalers[spec].transform(df[feats])
        hdb = []
        for mcs in [50, 100, 200]:
            lab = HDBSCAN(min_cluster_size=mcs).fit_predict(Xn)
            hdb.append({"min_cluster_size": mcs,
                        "clusters_found": len(set(lab)) - (1 if -1 in lab else 0),
                        "noise_pct": round(100 * float((lab == -1).mean()), 2)})
        print("  [diagnostic] HDBSCAN natural cluster count: "
              + "  ".join(f"mcs={h['min_cluster_size']}->{h['clusters_found']}c/"
                          f"{h['noise_pct']}%noise" for h in hdb))

        out["spec_results"][spec] = {
            "benchmark_domain": bd, "features": feats,
            "benchmarks": [{"variable": var, "operator": op, "threshold": thr,
                            "citation": cite, "strictly_out_of_model": oom}
                           for var, op, thr, cite, oom in BENCHMARKS[bd]],
            "population_base_rates_pct": {
                "overall": base["overall"],
                "by_band": {str(b): base["by_band"][b] for b in base["by_band"]}},
            "grid": grid, "n_survivors": len(survivors), "n_disqualified": len(disq),
            "survivors": sorted(survivors), "step3_ranks": ranks,
            "usability_sweep": {k: {kk: vv for kk, vv in v.items() if kk != "per_group"}
                                for k, v in sweep.items()},
            "gate_passing_combinations": gate_pass, "winner": winner,
            "winner_had_gate_passer": bool(gate_pass),
            "winner_validation_detail": sweep[winner]["per_group"],
            "winner_lift_detail": wl,
            "binding_benchmark_counts_across_survivors": binding,
            "winner_single_axis_diagnostic": seps,
            "hdbscan_natural_count_diagnostic": hdb,
        }

    # ------------------------------------------------------ primary spec per domain
    print(f"\n{'=' * 118}\nnet_worth_to_income PLACEMENT — deciding between the two "
          f"Banking / Investment specs on evidence\n{'=' * 118}")
    print(f"  {'spec':<22}{'gate passers':>14}{'winner':>22}{'macroSil':>10}"
          f"{'min_rho':>9}{'headrm':>8}{'top-feature share':>19}")
    for spec in SPECS:
        r = out["spec_results"][spec]
        u = r["usability_sweep"][r["winner"]]
        sp = next(iter(r["winner_single_axis_diagnostic"].values()))
        print(f"  {spec:<22}{len(r['gate_passing_combinations']):>14}{r['winner']:>22}"
              f"{r['grid'][r['winner']]['silhouette_macro']:>10.4f}"
              f"{(u['worst_group_min_spearman_rho'] or float('nan')):>9.3f}"
              f"{(u['min_headroom_captured'] or float('nan')):>8.3f}"
              f"   {sp['top_feature']} {100 * sp['top_feature_share_of_total_separation']:.0f}%")

    # Decide on the same gate evidence used everywhere else: prefer a spec with more
    # gate-passing combinations (robustness of the result, not just one lucky config);
    # break ties on the winner's worst-case headroom, then macro-silhouette.
    def spec_evidence(s):
        r = out["spec_results"][s]
        u = r["usability_sweep"][r["winner"]]
        return (len(r["gate_passing_combinations"]), u["min_headroom_captured"] or 0,
                r["grid"][r["winner"]]["silhouette_macro"])

    primary = {dom: max(cands, key=spec_evidence) for dom, cands in ALT_SPECS.items()}
    print("\n  PRIMARY SPEC PER DOMAIN: " + "   ".join(f"{d}={primary[d]}" for d in DOMAIN_ORDER))
    print("  (rule: most gate-passing combinations, tie-broken on the winner's worst-case "
          "headroom, then macro-silhouette)")
    out["primary_spec_per_domain"] = primary
    out["winners"] = {d: spec_winner[primary[d]] for d in DOMAIN_ORDER}

    # ------------------------------------------------------ Step 3 detail (primary trio)
    print(f"\n{'=' * 118}\nSTEP 3 — WINNER VALIDATION DETAIL, primary spec per domain "
          f"(mean AND density)\n{'=' * 118}")
    for dom in DOMAIN_ORDER:
        spec = primary[dom]
        w = spec_winner[spec]
        bd = BENCH_OF[spec]
        print(f"\n  --- {dom.upper()} [{spec}]: {w} ---")
        for gkey, g in out["spec_results"][spec]["winner_validation_detail"].items():
            print(f"    {gkey} (n={g['n_group']})  health order best->worst: "
                  f"{g['health_order_best_to_worst']}  min_rho={g['min_spearman_rho']}")
            for c in g["health_order_best_to_worst"]:
                e = g["clusters"][c]
                bits = [f"{v[:20]}={e['variables'][v]['mean']:.3f}"
                        f"/{e['variables'][v]['pct_meeting_benchmark']:.0f}%"
                        for v in BENCH_VARS[bd]]
                print(f"      tier {e['health_rank']} (cluster {c}, n={e['n']}, "
                      f"{e['share_pct']}%):  " + "  ".join(bits))

    # ------------------------------------------------------ tier assignment
    tiers = {}
    for dom in DOMAIN_ORDER:
        spec = primary[dom]
        r, kf, coarse = assign_tiers(df, spec_winner[spec], fitted[spec]["labels"], BENCH_OF[spec])
        tiers[dom] = {"rank": r, "k": kf, "coarse": coarse}
        df[f"tier_{dom}"] = r
        df[f"coarse_{dom}"] = coarse

    # ------------------------------------------------------ STEP 4
    print(f"\n{'=' * 118}\nSTEP 4 — MISMATCH-PROFILE BEHAVIOUR (the decisive test; the direct "
          f"input to a future Layer 3)\n{'=' * 118}")
    print("  Tier ranks: 0 = healthiest. Coarse HIGH/MID/LOW = population-share thirds of the "
          "ranked tiers (analysis device only).")
    print("  k per domain: " + "   ".join(f"{d}={tiers[d]['k']}" for d in DOMAIN_ORDER))

    step4 = {}
    for prof in PROFILE_ORDER:
        sub = df[df["profile_type"] == prof]
        if not len(sub):
            continue
        combo = sub.apply(lambda r: "|".join(r[f"coarse_{d}"] for d in DOMAIN_ORDER), axis=1)
        modal = combo.value_counts()

        def match_pct(expectation, doms):
            if not doms:
                return None
            m = np.ones(len(sub), dtype=bool)
            for d in doms:
                m &= (sub[f"coarse_{d}"] == expectation[d]).to_numpy()
            return round(100 * float(m.mean()), 1)

        rec = {"n": int(len(sub)), "modal_combination": modal.index[0],
               "modal_pct": round(100 * modal.iloc[0] / len(sub), 1),
               "top3_combinations": {kk: round(100 * vv / len(sub), 1)
                                     for kk, vv in modal.head(3).items()},
               "mean_tier_rank": {d: round(float(sub[f"tier_{d}"].mean()), 2) for d in DOMAIN_ORDER},
               "coarse_distribution_pct": {
                   d: {kk: round(100 * vv / len(sub), 1) for kk, vv
                       in sub[f"coarse_{d}"].value_counts().items()} for d in DOMAIN_ORDER},
               "overridden_domains": OVERRIDDEN_DOMAINS.get(prof)}
        exp = EXPECTED.get(prof)
        if exp:
            rec["expected_combination"] = "|".join(exp[d] for d in DOMAIN_ORDER)
            rec["pct_matching_expected_all3"] = match_pct(exp, DOMAIN_ORDER)
            rec["pct_matching_expected_per_domain"] = {d: match_pct(exp, [d]) for d in DOMAIN_ORDER}
            rec["pct_matching_expected_overridden_domains_only"] = match_pct(
                exp, OVERRIDDEN_DOMAINS.get(prof) or [])
        if prof in BRIEF_EXPECTED:
            be = BRIEF_EXPECTED[prof]
            rec["brief_assumed_combination"] = "|".join(be[d] for d in DOMAIN_ORDER)
            rec["pct_matching_brief_assumption"] = match_pct(be, DOMAIN_ORDER)
        step4[prof] = rec

    print(f"\n  {'profile':<20}{'n':>5}  {'expected (B|I|Ins)':<20}{'modal actual':<20}"
          f"{'modal%':>8}{'exact%':>8}{'overridden%':>13}   per-domain match% B/I/Ins")
    print("  " + "-" * 122)
    for prof, r in step4.items():
        p = r.get("pct_matching_expected_per_domain")
        pdstr = ("  ".join(f"{p[d]:>3.0f}" for d in DOMAIN_ORDER) if p else "—")
        ex = r.get("pct_matching_expected_all3")
        ov = r.get("pct_matching_expected_overridden_domains_only")
        print(f"  {prof:<20}{r['n']:>5}  {r.get('expected_combination', 'n/a (by design)'):<20}"
              f"{r['modal_combination']:<20}{r['modal_pct']:>7.1f}%"
              f"{(ex if ex is not None else float('nan')):>7.1f}%"
              f"{(ov if ov is not None else float('nan')):>12.1f}%   {pdstr}")
    for prof in BRIEF_EXPECTED:
        r = step4[prof]
        print(f"  [task-brief assumption] {prof}: {r['brief_assumed_combination']} -> "
              f"{r['pct_matching_brief_assumption']}% match, vs "
              f"{r['pct_matching_expected_all3']}% for the generator-derived "
              f"{r['expected_combination']}")

    print("\n  EXACT tier-rank distribution per profile (% of that profile's users; "
          "tier 0 = healthiest) — the form a Layer 3 rule should be written against:")
    for dom in DOMAIN_ORDER:
        kd = tiers[dom]["k"]
        cmap = {t: df.loc[df[f"tier_{dom}"] == t, f"coarse_{dom}"].iloc[0] for t in range(kd)}
        print(f"\n    {dom.upper()} (k={kd}, {primary[dom]} / {spec_winner[primary[dom]]}) — "
              "coarse map: " + "  ".join(f"T{t}={cmap[t]}" for t in range(kd)))
        print(f"      {'profile':<20}" + "".join(f"{'T' + str(t):>8}" for t in range(kd))
              + f"{'mean':>8}")
        for prof in step4:
            sub = df[df["profile_type"] == prof]
            vc = sub[f"tier_{dom}"].value_counts(normalize=True)
            print(f"      {prof:<20}"
                  + "".join(f"{100 * float(vc.get(t, 0.0)):>7.1f}%" for t in range(kd))
                  + f"{sub[f'tier_{dom}'].mean():>8.2f}")
            step4[prof].setdefault("exact_tier_distribution_pct", {})[dom] = {
                int(t): round(100 * float(vc.get(t, 0.0)), 1) for t in range(kd)}

    # ------------------------------------------------------ STEP 5
    print(f"\n{'=' * 118}\nSTEP 5 — BANKING vs INVESTMENT INDEPENDENCE\n{'=' * 118}")

    def assoc(a, b, mask=None):
        x = df[a] if mask is None else df.loc[mask, a]
        y = df[b] if mask is None else df.loc[mask, b]
        ct = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(ct)[0]
        n = ct.to_numpy().sum()
        v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if min(ct.shape) > 1 else np.nan
        return {"spearman_rho": round(float(stats.spearmanr(x, y).statistic), 4),
                "cramers_v": round(v, 4), "n": int(n), "contingency": ct.to_dict()}

    normal_mask = df["profile_type"] == "normal"
    step5 = {"pairs": {}}
    for a, b in itertools.combinations(DOMAIN_ORDER, 2):
        pair = {"all_users": assoc(f"tier_{a}", f"tier_{b}"),
                "normal_only": assoc(f"tier_{a}", f"tier_{b}", normal_mask)}
        step5["pairs"][f"{a}_vs_{b}"] = pair
        au, no = pair["all_users"], pair["normal_only"]
        print(f"  {a:<11} vs {b:<11}  ALL: rho={au['spearman_rho']:+.4f} V={au['cramers_v']:.4f}"
              f"   NORMAL-ONLY: rho={no['spearman_rho']:+.4f} V={no['cramers_v']:.4f}")

    print("\n  Banking x Investment coarse-tier contingency (% of all users):")
    ct = pd.crosstab(df["coarse_banking"], df["coarse_investment"], normalize=True) * 100
    print("    " + ct.round(1).to_string().replace("\n", "\n    "))
    step5["banking_x_investment_coarse_pct"] = ct.round(2).to_dict()
    gs = float(((df["coarse_banking"] == "HIGH") & (df["coarse_investment"] == "LOW")).mean())
    ti = float(((df["coarse_banking"] == "LOW") & (df["coarse_investment"] == "HIGH")).mean())
    step5["contradictory_quadrants_pct"] = {"banking_HIGH_investment_LOW": round(100 * gs, 2),
                                            "banking_LOW_investment_HIGH": round(100 * ti, 2)}
    print(f"\n  Contradictory quadrants: Banking HIGH + Investment LOW = {100 * gs:.2f}% of users;"
          f"  Banking LOW + Investment HIGH = {100 * ti:.2f}%")
    step5["underlying_feature_correlations_normal_only"] = {
        f"{a}~{b}": round(float(df.loc[normal_mask, a].corr(df.loc[normal_mask, b])), 4)
        for a, b in [("savings_rate", "investment_rate"),
                     ("net_worth_to_income", "investment_rate"),
                     ("emergency_fund_months", "investment_rate"),
                     ("savings_rate", "life_coverage_ratio"),
                     ("investment_rate", "life_coverage_ratio")]}
    print("  underlying feature correlations (normal-profile users only): "
          + "  ".join(f"{k}={v:+.3f}" for k, v
                      in step5["underlying_feature_correlations_normal_only"].items()))

    # Step 5b — net_worth placement sensitivity across all 4 banking x investment spec pairs
    print("\n  [STEP 5b] net_worth_to_income placement sensitivity — is the independence "
          "verdict an artifact of where it sits?")
    spec_ranks = {}
    for spec in SPECS:
        r, _, c = assign_tiers(df, spec_winner[spec], fitted[spec]["labels"], BENCH_OF[spec])
        spec_ranks[spec] = r
    sens = {}
    for bs in ALT_SPECS["banking"]:
        for isp in ALT_SPECS["investment"]:
            rho = float(stats.spearmanr(spec_ranks[bs], spec_ranks[isp]).statistic)
            rho_n = float(stats.spearmanr(spec_ranks[bs][normal_mask],
                                          spec_ranks[isp][normal_mask]).statistic)
            shared = "SHARES net_worth" if ("plus_nw" in bs and "plus_nw" in isp) else ""
            sens[f"{bs} x {isp}"] = {"rho_all": round(rho, 4), "rho_normal_only": round(rho_n, 4),
                                     "shares_net_worth": bool(shared)}
            print(f"    {bs:<20} x {isp:<22} rho(all)={rho:+.4f}  "
                  f"rho(normal-only)={rho_n:+.4f}  {shared}")
    step5["net_worth_placement_sensitivity"] = sens

    # Step 5c — winner-choice stability
    print("\n  [STEP 5c] winner-choice stability — do the runner-up gate-passers agree with "
          "the chosen winner?")
    stability, alt_tiers = {}, {}
    for dom in DOMAIN_ORDER:
        spec = primary[dom]
        r0 = out["spec_results"][spec]
        gp = r0["gate_passing_combinations"] or r0["survivors"]
        top3 = sorted(gp, key=lambda k: -r0["grid"][k]["silhouette_macro"])[:3]
        alt_tiers[dom], rows = {}, []
        for key in top3:
            rr, kf, cc = assign_tiers(df, key, fitted[spec]["labels"], BENCH_OF[spec])
            alt_tiers[dom][key] = rr
            rows.append({"combination": key, "k": kf,
                         "macro_sil": r0["grid"][key]["silhouette_macro"],
                         "coarse_tier_agreement_with_winner_pct":
                             round(100 * float((cc == df[f"coarse_{dom}"]).mean()), 1),
                         "rank_spearman_with_winner":
                             round(float(stats.spearmanr(rr, df[f"tier_{dom}"]).statistic), 3)})
        stability[dom] = rows
        print(f"    {dom:<11}" + "   ".join(
            f"{r['combination']}(k={r['k']}) agree={r['coarse_tier_agreement_with_winner_pct']}%"
            f"/rho={r['rank_spearman_with_winner']}" for r in rows))
    pair_rhos = {f"{bk} x {ik}": round(float(stats.spearmanr(br, ir).statistic), 4)
                 for bk, br in alt_tiers["banking"].items()
                 for ik, ir in alt_tiers["investment"].items()}
    print(f"    Banking-vs-Investment rho across all {len(pair_rhos)} top-3 pairings: "
          f"min={min(pair_rhos.values()):+.4f}  max={max(pair_rhos.values()):+.4f}  "
          f"mean={np.mean(list(pair_rhos.values())):+.4f}")
    step5["winner_choice_stability"] = stability
    step5["banking_vs_investment_rho_across_top3_pairings"] = pair_rhos

    # ------------------------------------------------------ STEP 6
    print(f"\n{'=' * 118}\nSTEP 6 — NORMAL-POPULATION SANITY CHECK (the 65% 'normal' profiles "
          f"only, n={int(normal_mask.sum())})\n{'=' * 118}")
    step6 = {}
    ndf = df[normal_mask]
    for dom in DOMAIN_ORDER:
        spec = primary[dom]
        key, bd, feats = spec_winner[spec], BENCH_OF[spec], FEATS_OF[spec]
        share = ndf[f"tier_{dom}"].value_counts(normalize=True).sort_index()
        sils = []
        for gkey, (sub, lab) in groups_for(df, key, fitted[spec]["labels"]).items():
            m = (sub["profile_type"] == "normal").to_numpy()
            if m.sum() < 50 or len(np.unique(lab[m])) < 2:
                continue
            s, _ = macro_silhouette(scalers[spec].transform(sub.loc[m, feats]), lab[m])
            sils.append(s)
        ok, pct_by_tier = {}, {}
        for var in BENCH_VARS[bd]:
            sig = pd.Series(health_signal(ndf, bd, var), index=ndf.index)
            means = sig.groupby(ndf[f"tier_{dom}"]).mean().sort_index()
            ok[var] = round(float(stats.spearmanr(means.to_numpy(),
                                                  -means.index.to_numpy(float)).statistic), 3)
            pcts = pd.Series(meets_mask(ndf, bd, var), index=ndf.index).groupby(
                ndf[f"tier_{dom}"]).mean().sort_index()
            pct_by_tier[var] = [round(100 * float(x), 1) for x in pcts]
            # Does the DENSITY (not just the mean) also order? The dual check, on normals.
            ok[f"{var}__density_rho"] = round(float(stats.spearmanr(
                pcts.to_numpy(), -pcts.index.to_numpy(float)).statistic), 3)
        step6[dom] = {"spec": spec, "winner": key,
                      "tier_share_pct_normal_only": {int(i): round(100 * float(v), 2)
                                                     for i, v in share.items()},
                      "min_tier_share_pct": round(100 * float(share.min()), 2),
                      "macro_silhouette_normal_subset": round(float(np.mean(sils)), 4) if sils else None,
                      "benchmark_mean_ordering_rho_normal_only": {v: ok[v] for v in BENCH_VARS[bd]},
                      "benchmark_density_ordering_rho_normal_only": {
                          v: ok[f"{v}__density_rho"] for v in BENCH_VARS[bd]},
                      "pct_clearing_by_tier_normal_only": pct_by_tier,
                      "degenerate_tier_below_5pct": bool(share.min() < 0.05)}
        print(f"\n  {dom.upper()} ({spec} / {key}, k={tiers[dom]['k']}):")
        print("    tier shares among normals: "
              + "  ".join(f"T{i}={100 * v:.1f}%" for i, v in share.items())
              + f"   (min {100 * share.min():.2f}%"
              + ("  <<< DEGENERATE on normals" if share.min() < 0.05 else "") + ")")
        print("    macro-silhouette on the normal subset: "
              f"{step6[dom]['macro_silhouette_normal_subset']}")
        for var in BENCH_VARS[bd]:
            print(f"    {var:<28} mean-rho={ok[var]:+.3f}  density-rho="
                  f"{ok[f'{var}__density_rho']:+.3f}   %clearing best->worst: {pct_by_tier[var]}")

    out["step4_profile_tier_combinations"] = step4
    out["step5_domain_independence"] = step5
    out["step6_normal_population_sanity"] = step6
    out["k_by_domain"] = {d: tiers[d]["k"] for d in DOMAIN_ORDER}

    with open(os.path.join(OUT_DIR, "domain_split_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    rows = []
    for spec in SPECS:
        for key, g in fitted[spec]["grid"].items():
            if g.get("failed"):
                continue
            u = out["spec_results"][spec]["usability_sweep"].get(key, {})
            rows.append({"spec": spec, "benchmark_domain": BENCH_OF[spec], "combination": key,
                         "approach": g["approach"], "algorithm": g["algorithm"], "k": g["k"],
                         "family": g["family"], "silhouette_macro": g["silhouette_macro"],
                         "silhouette_micro": g["silhouette_micro"],
                         "davies_bouldin": g["davies_bouldin"],
                         "calinski_harabasz": g["calinski_harabasz"],
                         "min_cluster_pct": g["min_cluster_pct"], "passes_floor": g["passes_floor"],
                         "min_spearman_rho": u.get("worst_group_min_spearman_rho"),
                         "min_headroom_captured": u.get("min_headroom_captured"),
                         "min_lift_pp": u.get("min_lift_pp"),
                         "passes_gate2_headroom": u.get("passes_gate2_headroom"),
                         "is_spec_winner": key == spec_winner[spec],
                         "is_primary_spec": spec in primary.values()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "domain_split_grid.csv"), index=False)
    df[["user_id", "age_band", "profile_type"]
       + [f"tier_{d}" for d in DOMAIN_ORDER] + [f"coarse_{d}" for d in DOMAIN_ORDER]].to_csv(
        os.path.join(OUT_DIR, "domain_split_tier_assignments.csv"), index=False)
    pd.DataFrame([{"profile_type": p, **{f"{d}_mean_tier": step4[p]["mean_tier_rank"][d]
                                         for d in DOMAIN_ORDER},
                   "expected": step4[p].get("expected_combination"),
                   "modal_actual": step4[p]["modal_combination"],
                   "modal_pct": step4[p]["modal_pct"],
                   "pct_matching_expected_all3": step4[p].get("pct_matching_expected_all3"),
                   "pct_matching_expected_overridden_only":
                       step4[p].get("pct_matching_expected_overridden_domains_only")}
                  for p in step4]).to_csv(
        os.path.join(OUT_DIR, "domain_split_step4_profile_table.csv"), index=False)
    print(f"\n  saved {OUT_DIR}/domain_split_results.json")
    print(f"  saved {OUT_DIR}/domain_split_grid.csv")
    print(f"  saved {OUT_DIR}/domain_split_tier_assignments.csv")
    print(f"  saved {OUT_DIR}/domain_split_step4_profile_table.csv")
    print("\n" + "=" * 118)
    print("DONE. Primary specs + winners: "
          + "   ".join(f"{d}={primary[d]}/{spec_winner[primary[d]]}" for d in DOMAIN_ORDER))
    print("=" * 118)


if __name__ == "__main__":
    main()
