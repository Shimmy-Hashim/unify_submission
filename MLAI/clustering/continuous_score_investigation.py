"""
continuous_score_investigation.py — is a continuous 0-100 per-domain score MEANINGFUL
under the locked domain-split architecture? (Unify, CCDS25-1170, July 2026)

WHY THIS EXISTS
  The brief's "Score Derivation" section describes a distance-to-ideal-centroid -> 0-100
  mechanism built for the SINGLE COMBINED model, and is explicitly marked superseded by the
  domain split. What the domain split actually validated is a CATEGORICAL 2-tier label per
  domain, via nearest-centroid cluster membership. Nothing continuous has been validated.

  The overall Financial Health Score is meant to average three per-domain scores. That
  requires a continuous per-domain number. This script asks whether such a number carries
  real information, or whether it is manufactured precision painted onto a binary result.

WHAT IT DOES NOT DO
  Analysis only. It reads the LOCKED production artifacts at
  backend/ml/artifacts/domain_split/ and writes nothing outside
  MLAI/clustering/results/continuous_score_investigation/. No production code is touched,
  no scoring function is built, no artifact is rewritten.

THE THREE CANDIDATE FORMULATIONS (Step 1)
  (a) POP  — population min-max normalized distance to the healthy-tier centroid, computed
             inside the model's own scaled space. Included FOR COMPLETENESS ONLY: its scale
             is defined by the population's own extremes, which is the "relative ranking"
             trap the project has avoided everywhere else (brief Section 5, Score
             Derivation: "Tier labels must be anchored to absolute MAS/LIA/SingStat
             benchmarks, not ranked within the age-filtered sample"). Flagged everywhere it
             appears.
  (b) ABS  — distance to a fixed ABSOLUTE anchor built from the domain's own defining
             benchmark thresholds, in benchmark units. Deliberately does NOT use the fitted
             RobustScaler: that scaler's center_/scale_ are population medians and IQRs, so
             scoring through it would smuggle population-relativity back into a score whose
             whole point is to be absolute.
  (c) MEMB — inverse-distance membership across the model's OWN centroids: the natural
             continuous reading of the validated partition itself (and, for Investment,
             literally the fuzzy c-means membership the model already computes). Tests
             whether the clustering geometry carries continuous information, as opposed to
             information we impose on it from outside.

  A fourth quantity, BENCH, is computed as a REFERENCE CEILING rather than a candidate: the
  share of the domain's own out-of-model benchmarks the user clears. It is not a distance
  score at all; it is there to answer "how much signal is available in principle", so the
  three candidates can be judged against something rather than only against each other.

THE DECISIVE TEST (Step 2)
  A continuous score is only worth shipping if WITHIN-TIER variation carries signal. Two
  users in the same tier with scores 41 and 68 must actually differ in financial health.
  Correlating the score with the features it is computed from proves nothing (it is
  tautological), so the honest test uses each domain's STRICTLY out-of-model benchmark —
  a variable that is in the dataset but in NO domain's feature set:
        Banking     net_worth_to_income        (validation-only, excluded from clustering)
        Investment  has_investments            (Layer 1 checklist feature)
        Insurance   hospitalisation_coverage, income_protection_coverage (Layer 1)
  If within-tier score deciles show no gradient on these, within-tier variation is noise.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/continuous_score_investigation.py
"""

import os
import sys
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed/synthetic_users.csv")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "backend/ml/artifacts/domain_split")
OUT_DIR = os.path.join(REPO_ROOT,
                       "MLAI/clustering/results/continuous_score_investigation")

DOMAINS = ["banking", "investment", "insurance"]
CANDIDATES = ["POP", "ABS", "MEMB"]

# ======================================================================================
# ABSOLUTE ANCHORS for candidate (b)
# ======================================================================================
# Each feature gets an IDEAL anchor A (the benchmark) and a FLOOR anchor Z (the absolute
# worst defensible value). A user's per-feature credit is u = clip((x - Z) / (A - Z), 0, 1);
# the score is 100 * (1 - ||1 - u|| / sqrt(n_features)), i.e. Euclidean distance to the
# all-benchmarks-met point in a space where one unit = one benchmark's full span.
#
# CITATION STATE IS MIXED AND IS RECORDED PER VALUE — this is the same discipline the brief
# applies to SYNTHETIC_IDEALS, and it matters here because an "absolute" score is only as
# absolute as its anchors. `cited` marks values traceable to a named source; `reasoned`
# marks defensible-but-uncited values that the report must disclose as such.
ABS_ANCHORS = {
    "banking": {
        "savings_rate":          dict(A=0.20, Z=0.00, dir="high",
                                      A_src="MoneySense 20% minimum savings rate", cited=True,
                                      Z_src="0% — saves nothing", Z_cited=False),
        "emergency_fund_months": dict(A=6.0, Z=0.0, dir="high",
                                      A_src="MoneySense 3-6 month fund (6 = upper end)", cited=True,
                                      Z_src="0 months — no buffer at all", Z_cited=False),
        "dti":                   dict(A=0.35, Z=0.55, dir="low",
                                      A_src="DBS '7 financial ratios' recommended ideal <=35%",
                                      cited=True,
                                      Z_src="MAS TDSR ceiling 55% — the regulatory limit", Z_cited=True),
        "healthy_debt_ratio":    dict(A=0.70, Z=0.00, dir="high",
                                      A_src="SingStat Household Balance Sheet ~70-71% structured",
                                      cited=True,
                                      Z_src="0% — all debt unstructured", Z_cited=False),
    },
    "investment": {
        "investment_rate":         dict(A=0.10, Z=0.00, dir="high",
                                        A_src="MoneySense/MAS 'invest at least 10% of income'",
                                        cited=True,
                                        Z_src="0% — invests nothing", Z_cited=False),
        "portfolio_concentration": dict(A=0.33, Z=1.00, dir="low",
                                        A_src="investment_recalibration.CONC_THRESHOLD = 0.33 "
                                              "(validated flat threshold)", cited=True,
                                        Z_src="1.0 — entire portfolio in one holding", Z_cited=False),
        "risk_age_deviation":      dict(A=0.00, Z=0.50, dir="abs",
                                        A_src="structural zero of the derived quantity "
                                              "(equity_pct minus the 100-minus-age target)",
                                        cited=False,
                                        Z_src="+/-50pp from target — REASONED, no benchmark exists",
                                        Z_cited=False),
    },
    "insurance": {
        # life uses the BAND-APPROPRIATE ideal (9/9/9/9/5), same as every domain-split check.
        "life_coverage_ratio": dict(A="band_life", Z=0.0, dir="high",
                                    A_src="LIA Protection Gap Study 2022 9x income "
                                          "(band 5: 5x, per SYNTHETIC_IDEALS)", cited=True,
                                    Z_src="0x — no life cover", Z_cited=False),
        "ci_coverage_ratio":   dict(A=3.9, Z=0.0, dir="high",
                                    A_src="LIA Protection Gap Study 2022 ~3.9x income", cited=True,
                                    Z_src="0x — no CI cover", Z_cited=False),
    },
}
LIFE_IDEAL_BY_BAND = {1: 9.0, 2: 9.0, 3: 9.0, 4: 9.0, 5: 5.0}

# insurance_spend_ratio is a CLUSTERING FEATURE with NO benchmark anywhere in the project —
# it appears in no BENCHMARKS list, and SYNTHETIC_IDEALS never held a value for it. So the
# absolute anchor for Insurance can only cover 2 of its 3 clustering features. Run as an
# explicit sensitivity rather than silently dropped or silently invented.
ABS_INSURANCE_SENSITIVITY = {
    "insurance_spend_ratio": dict(A=0.10, Z=0.00, dir="high",
                                  A_src="'~10% of income on insurance' rule of thumb — "
                                        "REASONED, UNCITED, no SG benchmark exists",
                                  cited=False,
                                  Z_src="0% — spends nothing on premiums", Z_cited=False),
}

# ======================================================================================
# OUT-OF-MODEL BENCHMARK VARIABLES per domain (the canonical sets the domain split used).
# `strict` = in the dataset but in NO domain's clustering feature set -> the only
# non-tautological validity test available for a continuous score.
# ======================================================================================
BENCHMARKS = {
    "banking": [
        ("savings_rate", ">=", 0.20, False),
        ("emergency_fund_months", ">=", 6.0, False),
        ("dti", "<=", 0.35, False),
        ("net_worth_to_income", ">=", "band_nw", True),      # STRICT: excluded from clustering
        ("healthy_debt_ratio", ">=", 0.70, False),
    ],
    "investment": [   # canonical V2 set: risk_age_deviation is a feature, not a benchmark
        ("investment_rate", ">=", 0.10, False),
        ("portfolio_concentration", "<=", 0.33, False),
        ("has_investments", "==1", 1, True),                 # STRICT
    ],
    "insurance": [
        ("life_coverage_ratio", ">=", "band_life", False),
        ("ci_coverage_ratio", ">=", 3.9, False),
        ("hospitalisation_coverage", "==1", 1, True),        # STRICT
        ("income_protection_coverage", "==1", 1, True),      # STRICT
    ],
}
NW_IDEAL_BY_BAND = {1: 1.0, 2: 3.0, 3: 6.0, 4: 8.0, 5: 10.0}
STRICT_VARS = {d: [b[0] for b in BENCHMARKS[d] if b[3]] for d in BENCHMARKS}
BENCH_VARS = {d: [b[0] for b in BENCHMARKS[d]] for d in BENCHMARKS}

LAYER1_FEATURES = ["has_investments", "hospitalisation_coverage", "income_protection_coverage"]

ARCHETYPES = ["normal", "conservative_saver", "over_invested", "protection_first",
              "wealthy_exposed", "high_debt_investor", "underinsured_saver", "outlier"]


# ======================================================================================
# Benchmark helpers
# ======================================================================================
def bspec(domain, var):
    return next(b for b in BENCHMARKS[domain] if b[0] == var)


def meets_mask(sub, domain, var):
    _, op, thr, _ = bspec(domain, var)
    v = sub[var].to_numpy(float)
    if op == ">=":
        if thr == "band_nw":
            t = sub["age_band"].map(NW_IDEAL_BY_BAND).to_numpy(float)
        elif thr == "band_life":
            t = sub["age_band"].map(LIFE_IDEAL_BY_BAND).to_numpy(float)
        else:
            t = float(thr)
        return v >= t
    if op == "<=":
        return v <= float(thr)
    if op == "==1":
        return v >= 0.5
    raise ValueError(op)


def health_signal(sub, domain, var):
    """Scalar per user where HIGHER = healthier — the same construction the domain-split
    validation used for cluster ordering, reused so the checks are directly comparable."""
    d = {">=": 1, "==1": 1, "<=": -1}[bspec(domain, var)[1]]
    return d * sub[var].to_numpy(float)


# ======================================================================================
# Load the LOCKED production artifacts and replay tier assignment
# ======================================================================================
def load_models():
    models = {}
    for d in DOMAINS:
        with open(os.path.join(ARTIFACT_DIR, f"{d}_model.json")) as f:
            models[d] = json.load(f)
    with open(os.path.join(ARTIFACT_DIR, "manifest.json")) as f:
        models["_manifest"] = json.load(f)
    return models


def scale_features(df, model):
    feats = model["clustering_features"]
    c = np.asarray(model["scaler"]["center_"], float)
    s = np.asarray(model["scaler"]["scale_"], float)
    return (df[feats].to_numpy(float) - c) / s


def design_matrix(df, model):
    """Reproduces the production inference path in the artifact README: RobustScaler
    columns, then (Approach B only) standardized age_midpoint as the FINAL dimension."""
    X = scale_features(df, model)
    if "age_scaler" in model:
        a = (df[["age_midpoint"]].to_numpy(float) - model["age_scaler"]["mean_"][0]) \
            / model["age_scaler"]["scale_"][0]
        X = np.hstack([X, a])
    return X


def centroid_block(model, band=None):
    """Return (ids, centroids, health_rank_by_id, shipped_tier_by_id) for a band (Approach A)
    or the pooled model (Approach B)."""
    if "bands" in model:
        b = model["bands"][str(band)]
        ids = sorted(b["centroids_scaled"], key=int)
        cent = np.array([b["centroids_scaled"][i] for i in ids], float)
        hrank = {i: b["tier_map"][i]["health_rank"] for i in ids}
        shipped = dict(hrank)
    else:
        m = model["model"]
        ids = sorted(m["centroids_scaled"], key=int)
        cent = np.array([m["centroids_scaled"][i] for i in ids], float)
        order = [str(c) for c in m["native_health_order_best_to_worst"]]
        hrank = {i: order.index(i) for i in ids}
        shipped = {i: model["tier_merge"]["shipped_tier_map"][i]["shipped_tier"] for i in ids}
    return ids, cent, hrank, shipped


def assign_and_distance(df, model):
    """Replay nearest-centroid assignment from the artifacts alone (the Phase 3 inference
    path). Returns per-user shipped tier, native health rank, and the full distance matrix
    to every centroid, keyed healthiest-first."""
    tier = np.zeros(len(df), int)
    hrank = np.zeros(len(df), int)
    dists = {}          # health_rank -> distance array (native ranks)
    groups = ([(b, (df["age_band"] == b).to_numpy()) for b in range(1, 6)]
              if "bands" in model else [(None, np.ones(len(df), bool))])
    n_native = model["k_native"]
    for r in range(n_native):
        dists[r] = np.full(len(df), np.nan)
    for band, mask in groups:
        X = design_matrix(df[mask], model)
        ids, cent, hr, shipped = centroid_block(model, band)
        D = np.linalg.norm(X[:, None, :] - cent[None, :, :], axis=2)
        nearest = np.argmin(D, axis=1)
        tier[mask] = [shipped[ids[j]] for j in nearest]
        hrank[mask] = [hr[ids[j]] for j in nearest]
        for j, cid in enumerate(ids):
            dists[hr[cid]][mask] = D[:, j]
    return tier, hrank, dists


# ======================================================================================
# STEP 1 — the three candidate formulations
# ======================================================================================
def score_pop(df, dists, model):
    """(a) Population min-max of the distance to the healthy-tier centroid.

    RELATIVE BY CONSTRUCTION — the 0 and 100 ends are wherever this particular population's
    extremes happen to sit. Normalized WITHIN age band for Approach A (matching how the
    model is fitted) and pooled for Approach B. Reported for completeness; see the header."""
    d = dists[0].copy()
    out = np.zeros(len(df))
    groups = ([(df["age_band"] == b).to_numpy() for b in range(1, 6)]
              if "bands" in model else [np.ones(len(df), bool)])
    for mask in groups:
        v = d[mask]
        lo, hi = float(np.min(v)), float(np.max(v))
        out[mask] = 100.0 * (1.0 - (v - lo) / (hi - lo)) if hi > lo else 50.0
    return out


def abs_credit(df, anchors):
    """Per-feature absolute credit u in [0, 1]: 0 at the floor anchor, 1 at the benchmark."""
    U = {}
    for feat, a in anchors.items():
        x = df[feat].to_numpy(float)
        A = (df["age_band"].map(LIFE_IDEAL_BY_BAND).to_numpy(float)
             if a["A"] == "band_life" else float(a["A"]))
        Z = float(a["Z"])
        if a["dir"] == "abs":
            x = np.abs(x)
        U[feat] = np.clip((x - Z) / (A - Z), 0.0, 1.0)
    return U


def score_abs(df, anchors):
    """(b) Euclidean distance to the all-benchmarks-met point, in benchmark units.

    Uses NO fitted scaler and NO population statistic anywhere — center_/scale_ are
    population medians/IQRs, so scoring through them would make an 'absolute' score
    quietly relative again."""
    U = abs_credit(df, anchors)
    M = np.column_stack([U[f] for f in anchors])
    gap = np.linalg.norm(1.0 - M, axis=1) / np.sqrt(M.shape[1])
    return 100.0 * (1.0 - gap), U


def score_memb(dists, healthy_rank=0, merged_ranks=None):
    """(c) Inverse-distance membership over the model's OWN centroids.

    merged_ranks=None -> use every native centroid (native geometry).
    merged_ranks=[...] -> collapse those ranks into one reference by nearest-of-the-group,
    which is exactly how a merged tier assigns members (Insurance's shipped geometry)."""
    eps = 1e-9
    if merged_ranks is None:
        inv = {r: 1.0 / (d + eps) for r, d in dists.items()}
        tot = sum(inv.values())
        return 100.0 * inv[healthy_rank] / tot
    d_healthy = dists[healthy_rank]
    d_other = np.min(np.column_stack([dists[r] for r in merged_ranks]), axis=1)
    return 100.0 * d_other / (d_healthy + d_other + eps)


def score_bench(df, domain):
    """REFERENCE CEILING, not a candidate: share of the domain's own out-of-model
    benchmarks cleared. Not a distance score — it exists to show how much ordinal signal
    is available in principle."""
    M = np.column_stack([meets_mask(df, domain, v).astype(float) for v in BENCH_VARS[domain]])
    return 100.0 * M.mean(axis=1)


# ======================================================================================
# STEP 2 — is the granularity real or manufactured?
# ======================================================================================
def decile_gradient(score, values, is_binary, n_bins=10):
    """Bin users by score decile, then check whether an out-of-model quantity orders across
    those bins. Spearman rho over the bin means: the same ordering check tier validation
    uses, applied WITHIN a tier instead of between tiers."""
    if len(score) < n_bins * 20:
        return {"n": int(len(score)), "usable": False, "reason": "too few users to bin"}
    try:
        bins = pd.qcut(pd.Series(score), n_bins, labels=False, duplicates="drop")
    except ValueError:
        return {"n": int(len(score)), "usable": False, "reason": "score has no spread to bin"}
    nb = int(pd.Series(bins).nunique())
    if nb < 4:
        return {"n": int(len(score)), "usable": False, "n_bins_formed": nb,
                "reason": f"score collapses to {nb} distinct levels — cannot form deciles"}
    dfb = pd.DataFrame({"bin": bins, "v": values}).groupby("bin")["v"].mean()
    rho = float(stats.spearmanr(dfb.index.to_numpy(float), dfb.to_numpy()).statistic)
    lo, hi = float(dfb.iloc[0]), float(dfb.iloc[-1])
    return {"n": int(len(score)), "usable": True, "n_bins_formed": nb,
            "spearman_rho_across_bins": round(rho, 3),
            "bin_means": [round(float(x), 4) for x in dfb],
            "bottom_bin": round(lo, 4), "top_bin": round(hi, 4),
            "bottom_to_top_delta": round(hi - lo, 4),
            "relative_delta_pct": (round(100 * (hi - lo) / abs(lo), 1) if lo else None),
            "is_binary_rate": bool(is_binary)}


def within_tier_within_band(df, domain, score, tier, t, n_bins=5):
    """AGE-CONTROLLED version of the within-tier test, and the one that actually decides the
    question.

    The naive pooled-within-tier check is confounded: two of the three domains' strictly
    out-of-model benchmarks move with age by construction in this dataset —
    `net_worth_to_income` is compared against a per-band ladder (1/3/6/8/10) while its raw
    value climbs steeply with age, and `hospitalisation_coverage` / `income_protection_
    coverage` are generated from BAND-DECLINING probabilities. A score that merely tracks
    age can therefore post a large within-tier rho, positive or negative, without carrying
    any health information at all. Splitting the test into (tier x age band) cells removes
    that channel: inside one band every user faces the same threshold and the same
    generation probability, so whatever ordering survives is about financial health.

    Quintiles rather than deciles, because a cell is ~1/5 the size of a pooled tier."""
    normal = (df["profile_type"] == "normal").to_numpy()
    out = {}
    for var in BENCH_VARS[domain]:
        rhos, cells = [], {}
        for b in range(1, 6):
            m = normal & (tier == t) & (df["age_band"] == b).to_numpy()
            if m.sum() < n_bins * 20:
                cells[b] = {"n": int(m.sum()), "usable": False,
                            "reason": "cell too small to bin"}
                continue
            vals = meets_mask(df[m], domain, var).astype(float)
            if len(np.unique(vals)) < 2:
                cells[b] = {"n": int(m.sum()), "usable": False,
                            "reason": f"no variation in {var} inside this cell "
                                      f"({100 * float(vals.mean()):.1f}% clear it)"}
                continue
            g = decile_gradient(score[m], vals, True, n_bins=n_bins)
            cells[b] = g
            if g.get("usable"):
                rhos.append(g["spearman_rho_across_bins"])
        out[var] = {
            "strictly_out_of_model": var in STRICT_VARS[domain],
            "measures": "% clearing the benchmark, by within-cell score quintile",
            "per_band_cells": cells,
            "n_usable_bands": len(rhos),
            "mean_rho_across_bands": round(float(np.mean(rhos)), 3) if rhos else None,
            "min_rho_across_bands": round(min(rhos), 3) if rhos else None,
            "all_bands_positive": bool(rhos and min(rhos) > 0),
        }
    strict = [out[v] for v in STRICT_VARS[domain] if out[v]["mean_rho_across_bands"] is not None]
    inmodel = [out[v] for v in BENCH_VARS[domain]
               if v not in STRICT_VARS[domain] and out[v]["mean_rho_across_bands"] is not None]
    return {
        "per_benchmark": out,
        "strict_out_of_model_mean_rho": round(float(np.mean(
            [g["mean_rho_across_bands"] for g in strict])), 3) if strict else None,
        "strict_out_of_model_testable": bool(strict),
        "in_model_mean_rho": round(float(np.mean(
            [g["mean_rho_across_bands"] for g in inmodel])), 3) if inmodel else None,
    }


def within_tier_report(df, domain, score, tier, n_tiers=2):
    """The decisive block. For each shipped tier, among NORMAL (non-injected) users only:
    how much spread does the score have, and does that spread mean anything?"""
    normal = (df["profile_type"] == "normal").to_numpy()
    out = {"tiers": {}}
    per_tier_sd, tier_means = [], []
    for t in range(n_tiers):
        m = normal & (tier == t)
        s = score[m]
        sub = df[m]
        if len(s) < 30:
            out["tiers"][t] = {"n": int(len(s)), "usable": False}
            continue
        ent = {
            "n": int(len(s)),
            "mean": round(float(np.mean(s)), 2), "sd": round(float(np.std(s, ddof=1)), 2),
            "min": round(float(np.min(s)), 2), "max": round(float(np.max(s)), 2),
            "p5": round(float(np.percentile(s, 5)), 2),
            "q25": round(float(np.percentile(s, 25)), 2),
            "median": round(float(np.median(s)), 2),
            "q75": round(float(np.percentile(s, 75)), 2),
            "p95": round(float(np.percentile(s, 95)), 2),
            "iqr_points": round(float(np.percentile(s, 75) - np.percentile(s, 25)), 2),
            "p5_p95_span_points": round(float(np.percentile(s, 95) - np.percentile(s, 5)), 2),
            "pct_at_ceiling_ge_99.5": round(100 * float((s >= 99.5).mean()), 2),
            "pct_at_floor_le_0.5": round(100 * float((s <= 0.5).mean()), 2),
            "n_distinct_values_rounded_to_1pt": int(len(np.unique(np.round(s)))),
            "within_tier_gradients": {},
        }
        # Pooled within-tier ordering, on "% clearing the benchmark" so that band-relative
        # thresholds (net_worth ladder, band-specific life-coverage ideal) are respected.
        # Kept as a SECONDARY reading only — it is age-confounded; see the age-controlled
        # block below, which is the one the conclusions rest on.
        for var in BENCH_VARS[domain]:
            vals = meets_mask(sub, domain, var).astype(float)
            g = decile_gradient(s, vals, True)
            g["strictly_out_of_model"] = var in STRICT_VARS[domain]
            g["measures"] = "% clearing the benchmark, by within-tier score decile"
            g["point_biserial_with_score"] = (
                round(float(stats.pointbiserialr(vals > 0.5, s).statistic), 3)
                if len(np.unique(vals)) > 1 else None)
            ent["within_tier_gradients"][var] = g
        # Age contamination: a health/coverage score should not be an age proxy.
        ent["age_confound_spearman"] = round(
            float(stats.spearmanr(s, sub["age_midpoint"].to_numpy(float)).statistic), 3)
        usable_strict = [ent["within_tier_gradients"][v] for v in STRICT_VARS[domain]
                         if ent["within_tier_gradients"][v].get("usable")]
        ent["strict_out_of_model_rho_mean_POOLED_age_confounded"] = (
            round(float(np.mean([g["spearman_rho_across_bins"] for g in usable_strict])), 3)
            if usable_strict else None)
        # THE DECIDING TEST — same check, run inside (tier x age band) cells.
        ent["age_controlled"] = within_tier_within_band(df, domain, score, tier, t)
        ent["strict_out_of_model_rho_mean"] = \
            ent["age_controlled"]["strict_out_of_model_mean_rho"]
        out["tiers"][t] = ent
        per_tier_sd.append(float(np.std(s, ddof=1)))
        tier_means.append(float(np.mean(s)))
    if len(tier_means) == 2 and len(per_tier_sd) == 2:
        pooled_sd = float(np.sqrt(np.mean(np.square(per_tier_sd))))
        gap = abs(tier_means[0] - tier_means[1])
        out["between_vs_within"] = {
            "tier_mean_gap_points": round(gap, 2),
            "pooled_within_tier_sd_points": round(pooled_sd, 2),
            "cohens_d_between_tiers": round(gap / pooled_sd, 2) if pooled_sd else None,
            "within_sd_as_pct_of_gap": round(100 * pooled_sd / gap, 1) if gap else None,
            "interpretation": (
                "two point-masses — within-tier spread is small next to the tier gap"
                if pooled_sd < 0.35 * gap else
                "genuine continuous spread — within-tier variation is comparable to the gap"
                if pooled_sd > 0.7 * gap else
                "intermediate — real spread, but tier membership still dominates"),
        }
    return out


def centroid_interiority(df, model, scores):
    """STRUCTURAL diagnostic: is the healthy centroid an INTERIOR point of its own feature
    space, rather than an extreme?

    Any distance-to-centroid score is non-monotone in health wherever it is — a user with
    18x life coverage is FURTHER from a medoid at 8.55x than a user with 8.6x is, so
    "closer to the healthy centroid" stops meaning "healthier" past that point. This is a
    property of the mechanism, not of the data, and it is what the brief's superseded
    single-model Score Derivation quietly assumed away. Measured here by restricting to the
    users who are BETTER than the healthy centroid on a given feature and asking whether the
    score still orders them correctly.

    The absolute-anchor candidate is exempt by construction (it clips credit at the
    benchmark), which is why it is reported alongside as the contrast."""
    feats = model["clustering_features"]
    if "bands" in model:
        def healthy_id(b):
            tm = model["bands"][str(b)]["tier_map"]
            return next(c for c, v in tm.items() if v["health_rank"] == 0)
        per_band = {f: {b: model["bands"][str(b)]["centroids_unscaled"][healthy_id(b)][f]
                        for b in range(1, 6)} for f in feats}
        cen = {f: float(np.mean(list(per_band[f].values()))) for f in feats}
    else:
        kept = str(model["tier_merge"]["native_cluster_id_kept"][0])
        cen = {f: float(model["model"]["centroids_unscaled"][kept][f]) for f in feats}
        per_band = None
    nmask = (df["profile_type"] == "normal").to_numpy()
    nd = df[nmask]
    out = {"healthy_centroid_unscaled": {f: round(cen[f], 4) for f in feats},
           "healthy_centroid_by_band": per_band, "per_feature": {}}
    for f in feats:
        # Approach A fits a separate centroid per band, so "better than the healthy centroid"
        # is evaluated against the user's OWN band's centroid, not a pooled average.
        ref = (nd["age_band"].map(per_band[f]).to_numpy(float) if per_band
               else np.full(len(nd), cen[f]))
        x = nd[f].to_numpy(float)
        if f == "risk_age_deviation":
            better = np.abs(x) < np.abs(ref)                       # closer to target = better
            sig = -np.abs(x)
        elif f in ("dti", "portfolio_concentration"):
            better = x < ref
            sig = -x
        else:
            better = x > ref
            sig = x
        ent = {"healthy_centroid_value": round(cen[f], 4),
               "pct_of_normals_strictly_better_than_the_centroid":
                   round(100 * float(better.mean()), 1),
               "score_ordering_among_those_better_than_centroid": {}}
        for c in CANDIDATES:
            s = scores[c][nmask][better]
            ent["score_ordering_among_those_better_than_centroid"][c] = (
                round(float(stats.spearmanr(s, sig[better]).statistic), 3)
                if better.sum() >= 100 else None)
        ent["verdict"] = (
            "n/a — too few users beyond the centroid" if better.sum() < 100 else
            "NON-MONOTONE: at least one candidate penalises being better than the healthy "
            "centroid" if min(v for v in
                              ent["score_ordering_among_those_better_than_centroid"].values()
                              if v is not None) < 0 else
            "monotone for all candidates on this feature")
        out["per_feature"][f] = ent
    return out


def test_validity_diagnostics(df):
    """Can the strictly-out-of-model test actually be RUN in each domain? Measured, not
    assumed — a validity check on the validity check.

    Two generator properties determine this: (1) how strongly each strict variable moves
    with age (if it is an age proxy, any score that drifts with age scores well or badly on
    it for the wrong reason), and (2) how it correlates with the domain's own clustering
    features among normals (if it is near-zero or negative, no score built from those
    features can be expected to order it)."""
    nd = df[df["profile_type"] == "normal"]
    feats = {"banking": ["savings_rate", "emergency_fund_months", "dti", "healthy_debt_ratio"],
             "investment": ["investment_rate", "portfolio_concentration", "risk_age_deviation"],
             "insurance": ["life_coverage_ratio", "ci_coverage_ratio", "insurance_spend_ratio"]}
    out = {}
    for d in DOMAINS:
        out[d] = {}
        for v in STRICT_VARS[d]:
            cleared = meets_mask(nd, d, v).astype(float)
            by_band = {b: round(100 * float(meets_mask(nd[nd.age_band == b], d, v).mean()), 1)
                       for b in range(1, 6)}
            out[d][v] = {
                "pct_clearing_normals": round(100 * float(cleared.mean()), 1),
                "pct_clearing_by_age_band": by_band,
                "spearman_with_age_normals": round(float(stats.spearmanr(
                    nd["age_midpoint"], cleared).statistic), 3),
                "raw_value_spearman_with_age_normals": round(float(stats.spearmanr(
                    nd["age_midpoint"], nd[v]).statistic), 3),
                "pearson_with_own_domain_features_normals": {
                    f: round(float(nd[v].corr(nd[f])), 3) for f in feats[d]},
                "max_abs_corr_with_own_features": round(max(
                    abs(float(nd[v].corr(nd[f]))) for f in feats[d]), 3),
            }
    return out


def population_monotonicity(df, domain, score):
    """Whole-population (normals only) decile ordering against every benchmark — the same
    ordering check tier validation used, now applied to the continuous score."""
    nd = df[df["profile_type"] == "normal"]
    s = score[(df["profile_type"] == "normal").to_numpy()]
    out = {}
    for var in BENCH_VARS[domain]:
        vals = meets_mask(nd, domain, var).astype(float)
        g = decile_gradient(s, vals, True)
        g["strictly_out_of_model"] = var in STRICT_VARS[domain]
        g["measures"] = "% clearing the benchmark, by score decile"
        out[var] = g
    rhos = [g["spearman_rho_across_bins"] for g in out.values() if g.get("usable")]
    strict_rhos = [g["spearman_rho_across_bins"] for v, g in out.items()
                   if g.get("usable") and v in STRICT_VARS[domain]]
    return {"per_benchmark": out,
            "min_rho": round(min(rhos), 3) if rhos else None,
            "mean_rho": round(float(np.mean(rhos)), 3) if rhos else None,
            "strict_out_of_model_mean_rho": round(float(np.mean(strict_rhos)), 3)
            if strict_rhos else None,
            "all_monotone_positive": bool(rhos and min(rhos) > 0)}


# ======================================================================================
# STEP 3 — archetypes
# ======================================================================================
def cliffs_delta(a, b):
    """Non-parametric effect size, robust to the huge n here (where a p-value would flag
    trivial differences as 'significant'). |d| < 0.147 negligible, < 0.33 small."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return None
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def overlap_pct(a, b, bins=50):
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if hi <= lo:
        return 100.0
    edges = np.linspace(lo, hi, bins + 1)
    ha = np.histogram(a, edges)[0] / len(a)
    hb = np.histogram(b, edges)[0] / len(b)
    return float(100.0 * np.minimum(ha, hb).sum())


def archetype_block(df, score, tier):
    out = {"per_profile": {}}
    for p in ARCHETYPES:
        m = (df["profile_type"] == p).to_numpy()
        if m.sum() == 0:
            continue
        s = score[m]
        out["per_profile"][p] = {
            "n": int(m.sum()),
            "mean": round(float(np.mean(s)), 2), "median": round(float(np.median(s)), 2),
            "sd": round(float(np.std(s, ddof=1)), 2),
            "q25": round(float(np.percentile(s, 25)), 2),
            "q75": round(float(np.percentile(s, 75)), 2),
            "tier_shares_pct": {int(t): round(100 * float((tier[m] == t).mean()), 1)
                                for t in sorted(set(tier.tolist()))},
        }
    return out


def pair_test(df, score, tier, pa, pb, label):
    """Does the continuous score add information the TIER does not — i.e. does it separate
    two archetypes that the tier puts in the same bucket?"""
    ma = (df["profile_type"] == pa).to_numpy()
    mb = (df["profile_type"] == pb).to_numpy()
    sa, sb = score[ma], score[mb]
    ta = pd.Series(tier[ma]).value_counts(normalize=True).sort_index()
    tb = pd.Series(tier[mb]).value_counts(normalize=True).sort_index()
    same_tier = int(ta.idxmax()) == int(tb.idxmax())
    d = cliffs_delta(sa, sb)
    return {
        "pair": f"{pa} vs {pb}", "label": label,
        "n": [int(ma.sum()), int(mb.sum())],
        "modal_tier": [int(ta.idxmax()), int(tb.idxmax())],
        "modal_tier_share_pct": [round(100 * float(ta.max()), 1), round(100 * float(tb.max()), 1)],
        "land_in_same_tier": same_tier,
        "mean": [round(float(np.mean(sa)), 2), round(float(np.mean(sb)), 2)],
        "median": [round(float(np.median(sa)), 2), round(float(np.median(sb)), 2)],
        "mean_difference_points": round(float(np.mean(sb) - np.mean(sa)), 2),
        "cliffs_delta": round(d, 3) if d is not None else None,
        "effect_size_verdict": (
            None if d is None else
            "negligible" if abs(d) < 0.147 else "small" if abs(d) < 0.33 else
            "medium" if abs(d) < 0.474 else "large"),
        "distribution_overlap_pct": round(overlap_pct(sa, sb), 1),
        "mannwhitney_p": float(stats.mannwhitneyu(sa, sb, alternative="two-sided").pvalue),
    }


# ======================================================================================
# STEP 4 evidence — the alternative that needs no continuous per-domain score
# ======================================================================================
def alternative_composite(df, tiers):
    """Overall score from TIER STATE + Layer 1 checklist flags only. Not a proposal being
    implemented — evidence for whether the recommendation in Step 4 is actually viable.

    3 domain tiers (healthy = 1 point each) + 3 available Layer 1 flags. `dependants`, the
    4th Layer 1 feature, is not in the synthetic dataset, so this is 3 of 4."""
    tier_pts = sum((tiers[d] == 0).astype(int) for d in DOMAINS)
    l1_pts = sum(df[f].to_numpy(float).round().astype(int) for f in LAYER1_FEATURES)
    total = tier_pts + l1_pts                     # 0..6
    score = 100.0 * total / 6.0
    nd = df["profile_type"] == "normal"
    all_bench = [(d, v) for d in DOMAINS for v in BENCH_VARS[d]]
    grad = {}
    for d, v in all_bench:
        vals = meets_mask(df[nd], d, v).astype(float)
        key = f"{d}.{v}"
        rows = pd.DataFrame({"lv": total[nd.to_numpy()], "v": vals}).groupby("lv")["v"].mean()
        grad[key] = {"levels": [int(x) for x in rows.index],
                     "pct_clearing_by_level": [round(100 * float(x), 1) for x in rows],
                     "spearman_rho": round(float(stats.spearmanr(
                         rows.index.to_numpy(float), rows.to_numpy()).statistic), 3),
                     "strictly_out_of_model": v in STRICT_VARS[d]}
    lv = pd.Series(total[nd.to_numpy()]).value_counts(normalize=True).sort_index()
    return {
        "construction": ("1 point per domain landing in its healthy tier (Banking Stable / "
                         "Investment On Track / Insurance Covered) + 1 point per Layer 1 "
                         "checklist flag passed (has_investments, hospitalisation_coverage, "
                         "income_protection_coverage). Total 0-6, rendered as 0-100."),
        "caveat": ("Layer 1 is 3 of its 4 features here — `dependants` is not present in the "
                   "synthetic dataset. Layer 1's real thresholds are also personalised by "
                   "salary and age in production, which this flat version does not model."),
        "n_distinct_levels": 7,
        "level_distribution_pct_normals": {int(i): round(100 * float(v), 2)
                                           for i, v in lv.items()},
        "score_values": sorted({round(100 * i / 6.0, 1) for i in range(7)}),
        "benchmark_ordering_by_level_normals": grad,
        "min_rho": round(min(g["spearman_rho"] for g in grad.values()), 3),
        "mean_rho": round(float(np.mean([g["spearman_rho"] for g in grad.values()])), 3),
        "strict_out_of_model_rhos": {k: g["spearman_rho"] for k, g in grad.items()
                                     if g["strictly_out_of_model"]},
        "archetype_means": {p: round(float(score[(df["profile_type"] == p).to_numpy()].mean()), 1)
                            for p in ARCHETYPES if (df["profile_type"] == p).any()},
    }, score, total


# ======================================================================================
# Main
# ======================================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    models = load_models()
    manifest = models.pop("_manifest")

    print("=" * 116)
    print(f"CONTINUOUS-SCORE INVESTIGATION — {len(df):,} users, 3 domains x "
          f"{len(CANDIDATES)} candidate formulations")
    print("  reading the LOCKED artifacts at backend/ml/artifacts/domain_split/ "
          "(read-only; nothing is rewritten)")
    print("=" * 116)

    out = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "question": ("Is a continuous 0-100 per-domain score meaningful under the locked "
                     "domain-split architecture, or is it manufactured precision on top of "
                     "a validated 2-tier categorical result?"),
        "scope": ("Investigation only. No production scoring logic built, no production "
                  "artifact or code modified. Reads the locked domain-split artifacts."),
        "artifact_provenance": {
            "manifest_generated_utc": manifest["generated_utc"],
            "dataset_sha256": manifest["dataset"]["sha256"],
            "domains": {d: manifest["domains"][d]["combination"] for d in DOMAINS},
        },
        "candidates": {
            "POP": ("(a) population min-max of distance to the healthy-tier centroid, in the "
                    "model's own scaled space, normalized within age band (Approach A) or "
                    "pooled (Approach B). RELATIVE BY CONSTRUCTION — tested for completeness "
                    "only; shipping it would reintroduce the relative-ranking trap the "
                    "project has avoided everywhere else."),
            "ABS": ("(b) Euclidean distance to a fixed absolute anchor built from the "
                    "domain's own benchmark thresholds, in benchmark units. Uses no fitted "
                    "scaler and no population statistic."),
            "MEMB": ("(c) inverse-distance membership across the model's own centroids — the "
                     "continuous reading of the validated partition itself."),
            "BENCH": ("REFERENCE CEILING, not a candidate: share of the domain's own "
                      "out-of-model benchmarks cleared. Shows how much ordinal signal is "
                      "available in principle."),
        },
        "absolute_anchor_citations": {
            d: {f: {"ideal": a["A"], "ideal_source": a["A_src"], "ideal_cited": a["cited"],
                    "floor": a["Z"], "floor_source": a["Z_src"], "floor_cited": a["Z_cited"]}
                for f, a in ABS_ANCHORS[d].items()} for d in DOMAINS},
        "absolute_anchor_coverage_gap": (
            "Insurance's ABS anchor covers 2 of its 3 clustering features. "
            "`insurance_spend_ratio` has NO benchmark anywhere in this project — it is in no "
            "BENCHMARKS list and SYNTHETIC_IDEALS never held a value for it. A sensitivity "
            "variant adds it on a reasoned, explicitly uncited 10%-of-income anchor."),
        "domains": {},
    }

    out["test_validity_diagnostics"] = test_validity_diagnostics(df)
    print("\n  [validity of the validity check] can the strictly-out-of-model test be run "
          "at all, per domain?")
    for d in DOMAINS:
        for v, r in out["test_validity_diagnostics"][d].items():
            print(f"    {d:<11}{v:<28} clears {r['pct_clearing_normals']:>5.1f}% of normals "
                  f"(by band {list(r['pct_clearing_by_age_band'].values())})  "
                  f"rho vs age {r['spearman_with_age_normals']:+.3f}  "
                  f"max |corr| with own features {r['max_abs_corr_with_own_features']:.3f}")

    scores_csv = df[["user_id", "age_band", "profile_type"]].copy()
    tiers, all_scores = {}, {}
    normals = (df["profile_type"] == "normal").to_numpy()

    for domain in DOMAINS:
        model = models[domain]
        tier, hrank, dists = assign_and_distance(df, model)
        tiers[domain] = tier

        # sanity: does the replayed assignment reproduce the artifact's recorded shares?
        recorded = model["shipped_tiers_summary"]["population_shares_pct"]
        replayed = [round(100 * float((tier == t).mean()), 1)
                    for t in range(model["shipped_tiers"])]
        agree = all(abs(a - b) <= 0.15 for a, b in zip(recorded, replayed))
        print(f"\n{'=' * 116}\n{domain.upper()}  [{model['combination']}, "
              f"{model['algorithm_display']}, k_native={model['k_native']}, "
              f"shipped tiers {model['shipped_tiers']}]\n{'=' * 116}")
        print(f"  replayed tier shares {replayed}% vs artifact {recorded}%  "
              f"-> {'MATCH' if agree else 'MISMATCH'}")
        if not agree:
            raise SystemExit(f"{domain}: replayed assignment does not reproduce the locked "
                             f"artifact — investigation aborted rather than run on a wrong "
                             f"geometry.")

        sc = {
            "POP": score_pop(df, dists, model),
            "ABS": score_abs(df, ABS_ANCHORS[domain])[0],
            "MEMB": (score_memb(dists, 0, merged_ranks=[1, 2]) if domain == "insurance"
                     else score_memb(dists, 0, merged_ranks=[1])),
            "BENCH": score_bench(df, domain),
        }
        all_scores[domain] = sc
        for name, v in sc.items():
            scores_csv[f"{domain}_{name}"] = np.round(v, 3)
        scores_csv[f"{domain}_tier"] = tier

        dres = {
            "combination": model["combination"],
            "algorithm": model["algorithm_display"],
            "k_native": model["k_native"], "shipped_tiers": model["shipped_tiers"],
            "clustering_features": model["clustering_features"],
            "approach": model["approach"],
            "tier_labels": model["shipped_tiers_summary"]["tier_labels"],
            "replayed_tier_shares_pct": replayed,
            "artifact_tier_shares_pct": recorded,
            "assignment_reproduces_artifact": agree,
            "strictly_out_of_model_benchmarks": STRICT_VARS[domain],
            "candidates": {},
        }

        for name in CANDIDATES + ["BENCH"]:
            s = sc[name]
            wt = within_tier_report(df, domain, s, tier, model["shipped_tiers"])
            pm = population_monotonicity(df, domain, s)
            arc = archetype_block(df, s, tier)
            dres["candidates"][name] = {
                "population_mean": round(float(np.mean(s)), 2),
                "population_sd": round(float(np.std(s, ddof=1)), 2),
                "population_pct_at_ceiling": round(100 * float((s >= 99.5).mean()), 2),
                "population_pct_at_floor": round(100 * float((s <= 0.5).mean()), 2),
                "correlation_with_BENCH_reference_normals": round(float(stats.spearmanr(
                    s[normals], sc["BENCH"][normals]).statistic), 3),
                "within_tier": wt,
                "population_monotonicity_normals": pm,
                "archetypes": arc,
            }
            bvw = wt.get("between_vs_within", {})
            print(f"\n  --- {name} ---")
            print(f"    population mean {np.mean(s):6.2f}  sd {np.std(s, ddof=1):5.2f}   "
                  f"ceiling {100 * (s >= 99.5).mean():5.2f}%  floor {100 * (s <= 0.5).mean():5.2f}%")
            for t in sorted(wt["tiers"]):
                e = wt["tiers"][t]
                if not e.get("n") or e.get("usable") is False:
                    continue
                lbl = model["shipped_tiers_summary"]["tier_labels"][t]
                ac = e["age_controlled"]
                print(f"    tier {t} ({lbl:<16} n={e['n']:>4} normals): "
                      f"mean {e['mean']:6.2f}  sd {e['sd']:5.2f}  IQR {e['iqr_points']:5.2f}pt  "
                      f"p5-p95 {e['p5_p95_span_points']:5.2f}pt  age-rho "
                      f"{e['age_confound_spearman']:+.3f}")
                print(f"           within-tier ordering  |  pooled(age-confounded) "
                      f"strict-OOM {e['strict_out_of_model_rho_mean_POOLED_age_confounded']}"
                      f"   ||  AGE-CONTROLLED strict-OOM "
                      f"{ac['strict_out_of_model_mean_rho']}"
                      f"  in-model {ac['in_model_mean_rho']}"
                      f"{'' if ac['strict_out_of_model_testable'] else '  (STRICT TEST NOT RUNNABLE)'}")
            if bvw:
                print(f"    between-vs-within: gap {bvw['tier_mean_gap_points']}pt, within-SD "
                      f"{bvw['pooled_within_tier_sd_points']}pt "
                      f"({bvw['within_sd_as_pct_of_gap']}% of gap), "
                      f"d={bvw['cohens_d_between_tiers']} -> {bvw['interpretation']}")
            print(f"    population decile monotonicity (normals): min rho {pm['min_rho']}, "
                  f"mean {pm['mean_rho']}, strict-OOM {pm['strict_out_of_model_mean_rho']}")

        dres["centroid_interiority"] = centroid_interiority(df, model, sc)
        ci = dres["centroid_interiority"]
        print("\n  [structural] is the healthy centroid an INTERIOR point? "
              "(does 'closer to ideal' stop meaning 'healthier'?)")
        print(f"      {'feature':<26}{'centroid':>10}{'%normals better':>17}   "
              + "".join(f"{c:>8}" for c in CANDIDATES))
        for f, e in ci["per_feature"].items():
            o = e["score_ordering_among_those_better_than_centroid"]
            print(f"      {f:<26}{e['healthy_centroid_value']:>10.3f}"
                  f"{e['pct_of_normals_strictly_better_than_the_centroid']:>16.1f}%   "
                  + "".join(f"{(o[c] if o[c] is not None else float('nan')):>+8.3f}"
                            for c in CANDIDATES))

        # ---------------- Insurance: native-3 vs merged-2 geometry
        if domain == "insurance":
            native = score_memb(dists, 0, merged_ranks=None)
            merged = sc["MEMB"]
            wt_merged = within_tier_report(df, domain, merged, tier, 2)
            scores_csv["insurance_MEMB_native3"] = np.round(native, 3)
            wt_n = within_tier_report(df, domain, native, tier, 2)
            pm_n = population_monotonicity(df, domain, native)
            dres["insurance_native3_vs_merged2_geometry"] = {
                "why": ("The shipped 2 tiers are a post-hoc merge of 3 native K-Medoids "
                        "clusters. MEMB is the one candidate whose value depends on how many "
                        "centroids the geometry has, so it is recomputed both ways."),
                "merged2_membership": {"definition": "d(nearest of the two merged medoids) / "
                                                     "(d(Covered medoid) + that distance)"},
                "native3_membership": {"definition": "inverse-distance weight of the Covered "
                                                     "medoid across all three native medoids"},
                "spearman_between_the_two_scores_normals": round(float(stats.spearmanr(
                    native[normals], merged[normals]).statistic), 3),
                "pearson_between_the_two_scores_normals": round(float(np.corrcoef(
                    native[normals], merged[normals])[0, 1]), 3),
                "mean_abs_point_difference": round(float(np.mean(np.abs(native - merged))), 2),
                "native3_within_tier": wt_n,
                "native3_population_monotonicity_normals": pm_n,
                "native3_vs_merged2_conclusion_inputs": {
                    "note": ("both columns are the MEMB candidate — the only one whose value "
                             "depends on the centroid count — so they are directly comparable"),
                    "merged2_strict_oom_rho_by_tier": {
                        t: wt_merged["tiers"][t].get("strict_out_of_model_rho_mean")
                        for t in wt_merged["tiers"]},
                    "merged2_in_model_rho_by_tier": {
                        t: wt_merged["tiers"][t]["age_controlled"]["in_model_mean_rho"]
                        for t in wt_merged["tiers"]},
                    "native3_strict_oom_rho_by_tier": {
                        t: wt_n["tiers"][t].get("strict_out_of_model_rho_mean")
                        for t in wt_n["tiers"]},
                    "native3_in_model_rho_by_tier": {
                        t: wt_n["tiers"][t]["age_controlled"]["in_model_mean_rho"]
                        for t in wt_n["tiers"]},
                    "merged2_within_tier_sd_points": {
                        t: wt_merged["tiers"][t]["sd"] for t in wt_merged["tiers"]},
                    "native3_within_tier_sd_points": {
                        t: wt_n["tiers"][t]["sd"] for t in wt_n["tiers"]},
                    "merged2_age_confound_by_tier": {
                        t: wt_merged["tiers"][t].get("age_confound_spearman")
                        for t in wt_merged["tiers"]},
                    "native3_age_confound_by_tier": {
                        t: wt_n["tiers"][t].get("age_confound_spearman")
                        for t in wt_n["tiers"]},
                },
                "wealthy_exposed_vs_underinsured_saver": {
                    "merged2": pair_test(df, merged, tier, "wealthy_exposed",
                                         "underinsured_saver", "merged-2 geometry"),
                    "native3": pair_test(df, native, tier, "wealthy_exposed",
                                         "underinsured_saver", "native-3 geometry"),
                },
            }
            geom = dres["insurance_native3_vs_merged2_geometry"]
            print(f"\n  [native-3 vs merged-2] rho between the two MEMB scores (normals) = "
                  f"{geom['spearman_between_the_two_scores_normals']}"
                  f"; mean |diff| = {geom['mean_abs_point_difference']}pt")

            # ---------------- ABS sensitivity: add the unbenchmarked third feature
            anchors3 = dict(ABS_ANCHORS["insurance"], **ABS_INSURANCE_SENSITIVITY)
            s3 = score_abs(df, anchors3)[0]
            scores_csv["insurance_ABS_with_spend"] = np.round(s3, 3)
            dres["insurance_ABS_spend_ratio_sensitivity"] = {
                "why": ("insurance_spend_ratio is a clustering feature with no benchmark "
                        "anywhere in the project. Adding it to the absolute anchor requires "
                        "inventing a threshold, so the effect of doing so is measured rather "
                        "than assumed."),
                "anchor_used": {"ideal": 0.10, "source": ABS_INSURANCE_SENSITIVITY[
                    "insurance_spend_ratio"]["A_src"]},
                "additional_caveat": ("Treating a HIGHER premium spend as healthier is itself "
                                      "questionable — overspending on premiums is a real "
                                      "financial-health problem, so this feature is not "
                                      "monotonically 'more is better' the way coverage is."),
                "spearman_vs_2feature_version_normals": round(float(stats.spearmanr(
                    s3[normals], sc["ABS"][normals]).statistic), 3),
                "mean_abs_point_difference": round(float(np.mean(np.abs(s3 - sc["ABS"]))), 2),
                "population_monotonicity_normals": population_monotonicity(df, "insurance", s3),
            }

        out["domains"][domain] = dres

    # ================================================================== STEP 3
    print(f"\n{'=' * 116}\nSTEP 3 — ARCHETYPE BEHAVIOUR\n{'=' * 116}")
    pairs = [
        ("wealthy_exposed", "underinsured_saver", "insurance",
         "the brief's concrete case — both land 'Not Covered'; does a continuous score "
         "separate them?"),
        ("wealthy_exposed", "protection_first", "insurance",
         "opposite insurance overrides — a sanity floor the score MUST clear"),
        ("conservative_saver", "over_invested", "banking",
         "opposite banking overrides"),
        ("conservative_saver", "high_debt_investor", "investment",
         "opposite investment overrides"),
        ("over_invested", "high_debt_investor", "investment",
         "both over-invest; separable only on dti, which is not an Investment feature"),
    ]
    step3 = {}
    for pa, pb, dom, why in pairs:
        step3[f"{dom}: {pa} vs {pb}"] = {"why": why, "domain": dom, "candidates": {
            c: pair_test(df, all_scores[dom][c], tiers[dom], pa, pb, c)
            for c in CANDIDATES + ["BENCH"]}}
        print(f"\n  {dom.upper()} — {pa} vs {pb}")
        print(f"    ({why})")
        for c in CANDIDATES + ["BENCH"]:
            r = step3[f"{dom}: {pa} vs {pb}"]["candidates"][c]
            print(f"    {c:<6} means {r['mean'][0]:6.2f} vs {r['mean'][1]:6.2f} "
                  f"(diff {r['mean_difference_points']:+6.2f}pt)  "
                  f"delta={r['cliffs_delta']:+.3f} ({r['effect_size_verdict']})  "
                  f"overlap {r['distribution_overlap_pct']:5.1f}%  "
                  f"same tier: {r['land_in_same_tier']}")
    out["step3_archetype_pairs"] = step3

    print("\n  Per-archetype mean score, all domains x candidates:")
    for dom in DOMAINS:
        print(f"\n    {dom.upper()}")
        print(f"      {'profile':<22}" + "".join(f"{c:>10}" for c in CANDIDATES + ["BENCH"])
              + f"{'tier0%':>9}")
        for p in ARCHETYPES:
            m = (df["profile_type"] == p).to_numpy()
            if not m.any():
                continue
            t0 = 100 * float((tiers[dom][m] == 0).mean())
            print(f"      {p:<22}"
                  + "".join(f"{np.mean(all_scores[dom][c][m]):>10.2f}"
                            for c in CANDIDATES + ["BENCH"])
                  + f"{t0:>8.1f}%")

    # ================================================================== STEP 4 evidence
    print(f"\n{'=' * 116}\nSTEP 4 EVIDENCE — the tier-state + Layer-1 alternative\n{'=' * 116}")
    alt, alt_score, alt_total = alternative_composite(df, tiers)
    scores_csv["alt_tier_plus_layer1_points"] = alt_total
    scores_csv["alt_tier_plus_layer1_score"] = np.round(alt_score, 1)
    out["step4_alternative_tier_plus_layer1"] = alt
    print(f"  levels 0-6 -> {alt['score_values']}")
    print(f"  distribution among normals: {alt['level_distribution_pct_normals']}")
    print(f"  ordering across levels vs all 12 domain benchmarks (normals): "
          f"min rho {alt['min_rho']}, mean {alt['mean_rho']}")
    print(f"  strictly out-of-model rhos: {alt['strict_out_of_model_rhos']}")
    print(f"  archetype means: {alt['archetype_means']}")

    # ================================================================== figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 4, figsize=(19, 11))
        for i, dom in enumerate(DOMAINS):
            labels = out["domains"][dom]["tier_labels"]
            for j, c in enumerate(CANDIDATES + ["BENCH"]):
                ax = axes[i][j]
                s = all_scores[dom][c]
                bins = np.linspace(0, 100, 51)
                for t, col in [(0, "#2f7d5a"), (1, "#b8562f")]:
                    ax.hist(s[normals & (tiers[dom] == t)], bins=bins, alpha=0.62,
                            color=col, label=f"{labels[t]} (tier {t})")
                ax.set_title(f"{dom} — {c}"
                             + ("  [RELATIVE]" if c == "POP" else
                                "  [reference]" if c == "BENCH" else ""), fontsize=10)
                ax.set_xlabel("score (0-100)", fontsize=8)
                ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel("normal users", fontsize=8)
                ax.legend(fontsize=7)
        fig.suptitle("Candidate continuous per-domain scores — distribution among the 4,550 "
                     "non-injected ('normal') users, split by shipped tier\n"
                     "Two separated humps = the score is re-stating the tier. Overlapping, "
                     "wide humps = genuine within-tier granularity.", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(OUT_DIR, "candidate_score_distributions.png"), dpi=130)
        plt.close(fig)
        print(f"\n  saved {OUT_DIR}/candidate_score_distributions.png")
    except Exception as e:                                          # pragma: no cover
        print(f"\n  (figure skipped: {type(e).__name__}: {e})")

    # ================================================================== outputs
    scores_csv.to_csv(os.path.join(OUT_DIR, "per_user_candidate_scores.csv"), index=False)
    with open(os.path.join(OUT_DIR, "continuous_score_investigation.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    summary_rows = []
    for dom in DOMAINS:
        for c in CANDIDATES + ["BENCH"]:
            r = out["domains"][dom]["candidates"][c]
            wt, b = r["within_tier"], r["within_tier"].get("between_vs_within", {})
            for t in sorted(wt["tiers"]):
                e = wt["tiers"][t]
                if e.get("usable") is False:
                    continue
                summary_rows.append({
                    "domain": dom, "candidate": c, "tier": t,
                    "tier_label": out["domains"][dom]["tier_labels"][t],
                    "n_normals": e["n"], "mean": e["mean"], "sd": e["sd"],
                    "iqr_points": e["iqr_points"], "p5_p95_span_points": e["p5_p95_span_points"],
                    "pct_at_ceiling": e["pct_at_ceiling_ge_99.5"],
                    "pct_at_floor": e["pct_at_floor_le_0.5"],
                    "n_distinct_values_1pt": e["n_distinct_values_rounded_to_1pt"],
                    "strict_oom_within_tier_rho_AGE_CONTROLLED":
                        e["strict_out_of_model_rho_mean"],
                    "strict_oom_within_tier_rho_pooled_confounded":
                        e["strict_out_of_model_rho_mean_POOLED_age_confounded"],
                    "in_model_within_tier_rho_AGE_CONTROLLED":
                        e["age_controlled"]["in_model_mean_rho"],
                    "age_confound_rho": e["age_confound_spearman"],
                    "tier_gap_points": b.get("tier_mean_gap_points"),
                    "within_sd_pct_of_gap": b.get("within_sd_as_pct_of_gap"),
                    "cohens_d": b.get("cohens_d_between_tiers"),
                    "pop_monotonicity_min_rho": r["population_monotonicity_normals"]["min_rho"],
                    "pop_monotonicity_strict_rho":
                        r["population_monotonicity_normals"]["strict_out_of_model_mean_rho"],
                })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "within_tier_summary.csv"), index=False)

    print(f"\n  saved {OUT_DIR}/continuous_score_investigation.json")
    print(f"  saved {OUT_DIR}/within_tier_summary.csv")
    print(f"  saved {OUT_DIR}/per_user_candidate_scores.csv")
    print("=" * 116)


if __name__ == "__main__":
    main()
