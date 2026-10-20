"""
adjacent_tier_distinctness.py — is every tier a real tier? (Unify, CCDS25-1170, July 2026)

WHY THIS EXISTS
  Insurance's technical winner (B|kmedoids|3) was found by domain_split_followup.py to be a
  2-tier structure wearing a 3-tier label: tiers 1 and 2 sat 0.046-0.064 IQR units apart on
  life/CI coverage (noise), while tier 0 vs the rest was a 1.2+ IQR gap. That diagnostic was
  never applied to Banking or Investment, and B|gmm_full|2 was flagged as the possibly-more-
  honest Insurance model but never formally adopted. This script applies the SAME diagnostic
  uniformly to every gate-passing combination in all three domains, so the final winner in
  each domain is chosen because its tiers are substantively distinct — not only because it
  topped the composite rank.

THE DIAGNOSTIC (defined once, applied identically everywhere)
  Tiers are ordered best-to-worst on the domain's composite health ranking (assign_tiers,
  unchanged). For every ADJACENT pair (t, t+1) — not just first-vs-last — two separations are
  computed, each as |mean_t - mean_t+1| / IQR_population(feature), maximised over its set:

    GEOMETRIC separation  — over the domain's CLUSTERING FEATURES.
        "Are these two tiers actually different in the space the model clustered in?"
    HEALTH separation     — over the domain's OUT-OF-MODEL BENCHMARK VARIABLES.
        "Do these two tiers mean anything different for the user's financial health?"

  Both must clear 0.20 IQR units. The two are reported separately because they can and do
  disagree, and the disagreement IS the Insurance finding: B|kmedoids|3's T1-vs-T2 clears
  geometrically (0.328, carried entirely by insurance_spend_ratio — a clustering feature that
  is deliberately NOT a benchmark variable, being non-monotonic as a health signal) while
  failing on health (0.153). A split that is real in feature space but meaningless in health
  terms is exactly the failure mode a tier label must not hide.

  A third check runs alongside, REPORTED SEPARATELY rather than folded into the distinctness
  verdict: DIRECTIONAL CONSISTENCY — does the healthier-ranked tier of the pair actually score
  better on every benchmark variable, on the mean health signal and on the density (% of
  members clearing the benchmark)? Insurance's T1/T2 inverts on life-coverage density (0.3%
  vs 5.6% clearing), which no separation magnitude would reveal. This is kept out of the
  verdict deliberately: separation and monotonicity are different failures, an inversion is
  already partly priced into the min-rho gate column, and collapsing them would make a
  well-separated tier system with one non-monotonic variable look like a collapsed one.

  Verdict per adjacent pair (distinctness only):
    DISTINCT      both separations >= 0.20
    NOT DISTINCT  either separation < 0.20
  Combination verdict = its worst adjacent pair. Sensitivity at 0.15 is reported alongside,
  since the task allows 0.15-0.20 and no result should rest on the choice within that band.

  APPROACH A IS READ PER BAND. Under Approach A the model is fitted per age band, so its
  tiers only exist per band and a pooled tier mean averages five different models — the same
  correction risk_age_final.py applied to the normal-population floor. Every Approach A
  combination is therefore evaluated in all 5 bands and bound by its WORST band; Approach B
  (fitted once on the pooled population) is evaluated pooled, which is its fitted level.

SCOPE. Diagnostic only. Reads the cached deterministic labels written by
domain_split_clustering.py and risk_age_diagnostic.py; refits nothing, changes no gate, and
writes nothing outside MLAI/clustering/results/. No production artifact, no Layer 3.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/adjacent_tier_distinctness.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import (  # noqa: E402
    DATA_PATH, OUT_DIR, LABEL_CACHE, FEATS_OF, groups_for, assign_tiers,
    meets_mask, health_signal, cluster_health_order, base_rates, usability,
)
from benchmark_core import compute_metrics  # noqa: E402
from final_selection import macro_silhouette, cluster_shares  # noqa: E402
from risk_age_diagnostic import (  # noqa: E402
    CACHE as RAD_CACHE, INV_FEATS_FULL, NORMAL_FLOOR, install_bench,
)

RESULTS_JSON = os.path.join(OUT_DIR, "domain_split_results.json")
RAD_FINAL_JSON = os.path.join(OUT_DIR, "risk_age_final.json")
RAD_DIAG_JSON = os.path.join(OUT_DIR, "risk_age_diagnostic.json")
OUT_JSON = os.path.join(OUT_DIR, "adjacent_tier_distinctness.json")
OUT_CSV = os.path.join(OUT_DIR, "adjacent_tier_distinctness_ranked.csv")

DIST_MIN = 0.20          # the cutoff, fixed before any result was looked at
DIST_MIN_ALT = 0.15      # reported as sensitivity only
INV_VARIANT = "V2_bench_drop_only"   # Investment's canonical spec (risk_age_final.py)

# The locked outcome (UNIFY_PROJECT_BRIEF.md Section 5, "FINAL ALGORITHM/TIER DECISIONS").
# Recorded here so the saved evidence file identifies which row of each ranked table was
# actually adopted, rather than leaving that to be reconstructed from prose.
FINAL_DECISIONS = {
    "banking": {"combination": "A|kmeans|2", "shipped_tiers": 2, "relabelled": False},
    "investment": {"combination": "A|fuzzy_cmeans|2", "shipped_tiers": 2, "relabelled": False},
    "insurance": {"combination": "B|kmedoids3merged|2", "shipped_tiers": 2, "relabelled": True,
                  "derived_from": "B|kmedoids|3", "merge": "health tiers 1 and 2"},
}


# ======================================================================================
# The diagnostic
# ======================================================================================
def pair_stats(sub, lab, feats, bvars, bd, band_for_thr, iqr):
    """Adjacent-tier separation within ONE fitted group (a band for A, everything for B)."""
    order, _ = cluster_health_order(sub, lab, bd)
    hr = {int(c): i for i, c in enumerate(order)}
    rank = np.array([hr[int(c)] for c in lab])
    kf = len(order)
    cols = list(dict.fromkeys(feats + bvars))

    means, sig, dens, shares = {}, {}, {}, {}
    for t in range(kf):
        m = rank == t
        s = sub[m]
        shares[t] = round(100 * float(m.mean()), 2)
        means[t] = {c: float(s[c].mean()) for c in cols}
        sig[t] = {v: float(np.mean(health_signal(s, bd, v))) for v in bvars}
        dens[t] = {v: 100 * float(meets_mask(s, bd, v).mean()) for v in bvars}

    pairs = []
    for t in range(kf - 1):
        geo = {f: abs(means[t][f] - means[t + 1][f]) / (iqr[f] or 1.0) for f in feats}
        hea = {v: abs(means[t][v] - means[t + 1][v]) / (iqr[v] or 1.0) for v in bvars}
        mean_inv = [v for v in bvars if sig[t][v] < sig[t + 1][v]]
        dens_inv = [v for v in bvars if dens[t][v] < dens[t + 1][v]]
        gmax, hmax = max(geo.values()), max(hea.values())
        verdict = "DISTINCT" if (gmax >= DIST_MIN and hmax >= DIST_MIN) else "NOT DISTINCT"
        pairs.append({
            "pair": f"T{t}|T{t+1}", "verdict": verdict,
            "geometric_sep_iqr": round(gmax, 3), "health_sep_iqr": round(hmax, 3),
            "geometric_by_feature": {f: round(v, 3) for f, v in geo.items()},
            "health_by_variable": {v: round(x, 3) for v, x in hea.items()},
            "mean_inverting_vars": mean_inv, "density_inverting_vars": dens_inv,
            "passes_at_0.20": bool(gmax >= 0.20 and hmax >= 0.20),
            "passes_at_0.15": bool(gmax >= 0.15 and hmax >= 0.15),
            "tier_shares_pct": [shares[t], shares[t + 1]]})
    return {"k_found": kf, "tier_shares_pct": [shares[t] for t in range(kf)],
            "tier_means": {t: {c: round(v, 4) for c, v in means[t].items()} for t in range(kf)},
            "pct_clearing": {t: {v: round(x, 1) for v, x in dens[t].items()} for t in range(kf)},
            "adjacent_pairs": pairs}


def distinctness(df, key, labels_store, feats, bvars, bd, iqr):
    """Per fitted group, then bound by the worst group (worst band for A)."""
    approach = key.split("|")[0]
    groups = {}
    for gkey, (sub, lab) in groups_for(df, key, labels_store).items():
        band = int(sub["age_band"].iloc[0]) if approach == "A" else None
        groups[gkey] = pair_stats(sub, lab, feats, bvars, bd, band, iqr)
    all_pairs = [(g, p) for g, gg in groups.items() for p in gg["adjacent_pairs"]]
    worst = min(all_pairs, key=lambda gp: (gp[1]["passes_at_0.20"],
                                           min(gp[1]["geometric_sep_iqr"],
                                               gp[1]["health_sep_iqr"])))
    n_fail = sum(1 for _, p in all_pairs if not p["passes_at_0.20"])
    n_inv = sum(1 for _, p in all_pairs if p["mean_inverting_vars"])
    verdict = "NOT DISTINCT" if n_fail else "DISTINCT"
    # How many tiers actually survive: within each fitted group, every failing adjacent pair
    # collapses two labelled tiers into one. The binding figure is the worst group.
    eff = min(gg["k_found"] - sum(1 for p in gg["adjacent_pairs"] if not p["passes_at_0.20"])
              for gg in groups.values())
    return {"fitted_level": "per_band" if approach == "A" else "pooled",
            "groups": groups, "n_adjacent_pairs": len(all_pairs),
            "n_pairs_failing_0.20": n_fail,
            "n_pairs_failing_0.15": sum(1 for _, p in all_pairs if not p["passes_at_0.15"]),
            "n_pairs_with_mean_inversion": n_inv,
            "n_pairs_with_density_inversion": sum(1 for _, p in all_pairs
                                                  if p["density_inverting_vars"]),
            "worst_pair": {"group": worst[0], **worst[1]},
            "min_geometric_sep_iqr": round(min(p["geometric_sep_iqr"] for _, p in all_pairs), 3),
            "min_health_sep_iqr": round(min(p["health_sep_iqr"] for _, p in all_pairs), 3),
            "verdict": verdict, "effective_tiers": int(eff)}


def per_band_normal(df, key, labels, bd):
    """The project's adopted normal-population 5% floor, read at the fitted level."""
    rank, kf, _ = assign_tiers(df, key, labels, bd)
    nd = df.assign(tier=rank)
    nd = nd[nd.profile_type == "normal"]
    pooled = [float((nd.tier == t).mean()) for t in range(kf)]
    by_band = {b: min(float((nd[nd.age_band == b].tier == t).mean()) for t in range(kf))
               for b in range(1, 6)}
    per_band_ok = min(by_band.values()) >= NORMAL_FLOOR
    pooled_ok = min(pooled) >= NORMAL_FLOOR
    return {"pooled_shares_pct": [round(100 * x, 2) for x in pooled],
            "pooled_min_pct": round(100 * min(pooled), 2),
            "by_band_min_pct": {b: round(100 * v, 2) for b, v in by_band.items()},
            "worst_band_min_pct": round(100 * min(by_band.values()), 2),
            "passes_pooled": bool(pooled_ok), "passes_per_band": bool(per_band_ok),
            "passes_at_fitted_level": bool(per_band_ok if key.split("|")[0] == "A" else pooled_ok)}


# ======================================================================================
# Reporting
# ======================================================================================
def table(domain, rows, note=""):
    print("\n" + "=" * 132)
    print(f"{domain.upper()} — every gate-passing combination, ranked by macro-silhouette")
    if note:
        print(note)
    print("=" * 132)
    print(f"  {'#':>2} {'combination':<22}{'appr':>5}{'k':>3}{'macroSil':>10}{'minRho':>8}"
          f"{'headrm':>8}{'liftPP':>8}{'minClus%':>9}{'bal':>6}{'normFloor':>11}"
          f"  {'distinctness':<13}{'>=.20':>7}{'effK':>5}{'inv':>4}{'ageV':>7}{'<5%':>6}")
    print("  " + "-" * 130)
    for i, r in enumerate(rows, 1):
        d, nf, ac = r["distinct"], r["normal_floor"], r["age_confound"]
        flag = "OK " if d["verdict"] == "DISTINCT" else "XX "
        nfs = (f"{nf['worst_band_min_pct']:.1f}%b" if r["approach"] == "A"
               else f"{nf['pooled_min_pct']:.1f}%p")
        nfs += "" if nf["passes_at_fitted_level"] else "!"
        pairs_ok = f"{d['n_adjacent_pairs'] - d['n_pairs_failing_0.20']}/{d['n_adjacent_pairs']}"
        thin_cells = f"{ac['cells_below_5pct_of_band_normals']}/{ac['n_cells']}"
        print(f"  {i:>2} {r['key']:<22}{r['approach']:>5}{r['k']:>3}{r['macro_sil']:>10.4f}"
              f"{r['min_rho']:>8.3f}{r['headroom']:>8.3f}{r['lift_pp']:>8.1f}"
              f"{r['min_cluster_pct']:>9.2f}{r['balance']:>6.2f}{nfs:>11}"
              f"  {flag}{d['verdict']:<10}{pairs_ok:>7}"
              f"{d['effective_tiers']:>5}{d['n_pairs_with_mean_inversion']:>4}"
              f"{ac['cramers_v_age_vs_tier']:>7.2f}{thin_cells:>6}")
    print("  normFloor: share of NORMAL-profile users in the smallest tier, read at the model's")
    print("  fitted level (b = worst age band for Approach A, p = pooled for B); '!' = below the")
    print("  project's adopted 5% normal-population floor.  effK = tiers surviving the collapse")
    print("  of every sub-0.20 adjacent pair.  inv = adjacent pairs where a benchmark variable")
    print("  moves the wrong way (reported separately from distinctness, see module docstring).")
    print("  ageV = Cramer's V between age band and tier among normals (1.00 = the tier system")
    print("  IS the age band); <5% = band x tier cells holding under 5% of that band's normals.")
    print("\n  worst adjacent pair per combination (geometric = clustering features, "
          "health = benchmark variables):")
    print(f"    {'combination':<22}{'worst pair':<14}{'geom':>7}{'health':>8}"
          f"{'@0.15':>7}  inverting benchmark variables")
    for r in rows:
        d, w = r["distinct"], r["distinct"]["worst_pair"]
        inv = ", ".join(w["mean_inverting_vars"]) or "-"
        dinv = ", ".join(v for v in w["density_inverting_vars"]
                         if v not in w["mean_inverting_vars"])
        if dinv:
            inv += f"  (density only: {dinv})"
        print(f"    {r['key']:<22}{w['group'] + ' ' + w['pair']:<14}"
              f"{w['geometric_sep_iqr']:>7.3f}{w['health_sep_iqr']:>8.3f}"
              f"{'pass' if w['passes_at_0.15'] else 'FAIL':>7}  {inv}")
        if d["n_adjacent_pairs"] > 1:
            print("      all pairs: " + "; ".join(
                f"{g}/{p['pair']} g={p['geometric_sep_iqr']:.2f} h={p['health_sep_iqr']:.3f} "
                f"{p['verdict'][:4]}"
                for g, gg in d["groups"].items() for p in gg["adjacent_pairs"]))

    heavy = [r for r in rows if r["age_confound"]["cramers_v_age_vs_tier"] >= 0.5]
    if heavy:
        print(f"\n  AGE CONFOUND — the {len(heavy)} combination(s) with Cramer's V >= 0.50. "
              "Rows are age bands, columns")
        print("  tiers (best -> worst), cells are the share of that band's NORMAL users. A tier "
              "system that")
        print("  is mostly age is not a health tier system, however cleanly separated its tiers "
              "are:")
        for r in heavy:
            print(f"    {r['key']}  (V={r['age_confound']['cramers_v_age_vs_tier']:.2f}, "
                  f"{r['age_confound']['cells_below_5pct_of_band_normals']}/"
                  f"{r['age_confound']['n_cells']} cells under 5%)")
            t = pd.DataFrame(r["age_confound"]["row_normalised"])
            print("      " + t.to_string().replace("\n", "\n      "))


def build_rows(df, spec_keys, grid, labels, usab, feats, bvars, bd, iqr):
    rows = []
    for key in spec_keys:
        g = grid[key]
        u = usab[key]
        d = distinctness(df, key, labels, feats, bvars, bd, iqr)
        rank, kf, _ = assign_tiers(df, key, labels, bd)
        shares = [float((rank == t).mean()) for t in range(kf)]
        rows.append({
            "key": key, "approach": key.split("|")[0], "algorithm": key.split("|")[1],
            "k": g["k"], "macro_sil": g["silhouette_macro"], "micro_sil": g["silhouette_micro"],
            "davies_bouldin": g["davies_bouldin"], "calinski_harabasz": g["calinski_harabasz"],
            "min_rho": u["worst_group_min_spearman_rho"],
            "headroom": u["min_headroom_captured"], "lift_pp": u["min_lift_pp"],
            "min_cluster_pct": g["min_cluster_pct"],
            "tier_shares_pct": [round(100 * s, 2) for s in shares],
            "balance": round(min(shares) / max(shares), 3),
            "normal_floor": per_band_normal(df, key, labels, bd),
            "age_confound": age_confound(df, key, labels, bd),
            "distinct": d})
    return sorted(rows, key=lambda r: -r["macro_sil"])


def age_confound(df, key, labels, bd):
    """How much of this tier system is just age? Cramer's V between age band and tier among
    NORMAL-profile users only, plus the count of band x tier cells holding under 5% of that
    band's normals. Added because adjacent-tier distinctness alone cannot see this: an age
    partition separates cleanly on savings/emergency fund precisely BECAUSE those features
    trend with age, so it scores as 'distinct' while carrying no health information at all.
    Approach A is immune by construction (fitted within band) and is reported anyway."""
    rank, kf, _ = assign_tiers(df, key, labels, bd)
    nd = df.assign(tier=rank)
    nd = nd[nd.profile_type == "normal"]
    tab = pd.crosstab(nd.age_band, nd.tier)
    n = int(tab.to_numpy().sum())
    exp = np.outer(tab.sum(1), tab.sum(0)) / n
    chi2 = float((((tab.to_numpy() - exp) ** 2) / np.where(exp > 0, exp, 1)).sum())
    v = float(np.sqrt((chi2 / n) / max(1, min(tab.shape) - 1)))
    row = pd.crosstab(nd.age_band, nd.tier, normalize="index")
    return {"cramers_v_age_vs_tier": round(v, 3),
            "cells_below_5pct_of_band_normals": int((row < 0.05).to_numpy().sum()),
            "n_cells": int(row.size),
            "max_single_tier_share_in_a_band": round(float(row.to_numpy().max()), 3),
            "row_normalised": row.round(3).to_dict()}


def merge_labels(df, key, labels, bd, merge_ranks):
    """Collapse adjacent tiers (given by health-rank index) of an Approach B partition into
    one, returning a fresh flat label array. This is the "ship the honest structure that is
    actually there" option: if a k=3 model's tiers 1 and 2 are not distinct, the 2-tier
    system it really encodes is not necessarily the same partition any k=2 model finds."""
    lab = np.asarray(labels[key], dtype=int)
    order, _ = cluster_health_order(df, lab, bd)
    keep = [int(c) for c in order]
    new = lab.copy()
    for r in merge_ranks[1:]:
        new[new == keep[r]] = keep[merge_ranks[0]]
    remap = {c: i for i, c in enumerate(sorted(set(new.tolist())))}
    return np.array([remap[c] for c in new])


def crosstab_k2_vs_k3(df, k2, k3, labels, bd, feats, bvars, iqr):
    """Where does the third tier come from? Cross-tab the k=2 winner's tiers against a
    genuinely-3-tier candidate's, then report how distinct the resulting split actually is."""
    r2, _, _ = assign_tiers(df, k2, labels, bd)
    r3, k3f, _ = assign_tiers(df, k3, labels, bd)
    tab = pd.crosstab(r2, r3, normalize="index").round(3)
    counts = pd.crosstab(r2, r3)
    # For the k=2 tier that gets split, how far apart are its two halves?
    split_of = {}
    for t2 in sorted(set(r2)):
        sub = df[r2 == t2]
        sr3 = r3[r2 == t2]
        present = [t for t in sorted(set(sr3)) if (sr3 == t).mean() >= 0.05]
        if len(present) < 2:
            continue
        a, b = present[0], present[-1]
        sa, sb = sub[sr3 == a], sub[sr3 == b]
        geo = max(abs(sa[f].mean() - sb[f].mean()) / (iqr[f] or 1.0) for f in feats)
        hea = max(abs(sa[v].mean() - sb[v].mean()) / (iqr[v] or 1.0) for v in bvars)
        split_of[int(t2)] = {"k3_tiers": [int(a), int(b)],
                             "shares_pct": [round(100 * float((sr3 == a).mean()), 1),
                                            round(100 * float((sr3 == b).mean()), 1)],
                             "geometric_sep_iqr": round(geo, 3),
                             "health_sep_iqr": round(hea, 3),
                             "distinct": bool(geo >= DIST_MIN and hea >= DIST_MIN)}
    return {"k2": k2, "k3": k3, "row_normalised": tab.to_dict(),
            "counts": counts.to_dict(), "split_analysis": split_of}


def csv_rows(domain, rows, row_type="gate_passer", final_decision=None):
    """One row per combination, carrying EVERY column reported in the ranked tables. This
    file is the evidentiary record behind the final decisions — it must stand alone, so
    nothing that appeared in a reported table is left out of it."""
    out = []
    for i, r in enumerate(rows, 1):
        d = r["distinct"]
        w = d["worst_pair"]
        out.append({
            "domain": domain, "rank_by_macro_silhouette": (i if row_type == "gate_passer"
                                                           else None),
            "row_type": row_type,
            "is_final_winner": bool(final_decision == r["key"]),
            "combination": r["key"], "approach": r["approach"],
            "algorithm": r["algorithm"], "k": r["k"], "macro_silhouette": r["macro_sil"],
            "micro_silhouette": r["micro_sil"], "davies_bouldin": r["davies_bouldin"],
            "calinski_harabasz": r["calinski_harabasz"],
            "min_ordering_rho": r["min_rho"], "min_headroom_captured": r["headroom"],
            "min_lift_pp": r["lift_pp"],
            "tier_shares_pct": "/".join(f"{s:.1f}" for s in r["tier_shares_pct"]),
            "min_cluster_pct": r["min_cluster_pct"],
            "balance_min_over_max": r["balance"],
            "normal_floor_pooled_min_pct": r["normal_floor"]["pooled_min_pct"],
            "normal_floor_worst_band_min_pct": r["normal_floor"]["worst_band_min_pct"],
            "normal_floor_passes_at_fitted_level": r["normal_floor"]["passes_at_fitted_level"],
            "distinctness_verdict": d["verdict"],
            "adjacent_pairs_total": d["n_adjacent_pairs"],
            "adjacent_pairs_failing_0.20": d["n_pairs_failing_0.20"],
            "adjacent_pairs_failing_0.15": d["n_pairs_failing_0.15"],
            "adjacent_pairs_with_mean_inversion": d["n_pairs_with_mean_inversion"],
            "adjacent_pairs_with_density_inversion": d["n_pairs_with_density_inversion"],
            "worst_pair": f"{w['group']}/{w['pair']}",
            "worst_geometric_sep_iqr": w["geometric_sep_iqr"],
            "worst_health_sep_iqr": w["health_sep_iqr"],
            "worst_pair_mean_inversions": ";".join(w["mean_inverting_vars"]) or "",
            "worst_pair_density_inversions": ";".join(w["density_inverting_vars"]) or "",
            "effective_tiers": d["effective_tiers"],
            "cramers_v_age_vs_tier": r["age_confound"]["cramers_v_age_vs_tier"],
            "band_x_tier_cells_below_5pct": r["age_confound"]["cells_below_5pct_of_band_normals"],
            "band_x_tier_cells": r["age_confound"]["n_cells"]})
    return out


# ======================================================================================
def main():
    df = pd.read_csv(DATA_PATH)
    with open(LABEL_CACHE, "rb") as f:
        dcache = pickle.load(f)
    with open(RESULTS_JSON) as f:
        res = json.load(f)
    with open(RAD_CACHE, "rb") as f:
        rcache = pickle.load(f)
    with open(RAD_FINAL_JSON) as f:
        rfinal = json.load(f)

    print("=" * 132)
    print("ADJACENT-TIER DISTINCTNESS — all three domains, every gate-passing combination")
    print("=" * 132)
    print(f"  cutoff {DIST_MIN} IQR units on BOTH the geometric (clustering-feature) and the")
    print(f"  health (benchmark-variable) separation of every adjacent tier pair; {DIST_MIN_ALT}")
    print("  reported as sensitivity. Approach A read PER BAND (its fitted level), B pooled.")
    print("  Population IQR is computed once on all 7,000 users and reused everywhere.")

    out = {"cutoff_iqr": DIST_MIN, "sensitivity_cutoff_iqr": DIST_MIN_ALT,
           "final_decisions": FINAL_DECISIONS, "domains": {}}
    csv_all = []

    # ---------------------------------------------------------------- BANKING + INSURANCE
    for domain, spec in (("banking", "banking_core4"), ("insurance", "insurance_core3")):
        sr = res["spec_results"][spec]
        feats, bvars = FEATS_OF[spec], ds.BENCH_VARS[domain]
        iqr = {c: float(df[c].quantile(0.75) - df[c].quantile(0.25))
               for c in dict.fromkeys(feats + bvars)}
        rows = build_rows(df, sr["gate_passing_combinations"], sr["grid"],
                          dcache[spec]["labels"], sr["usability_sweep"], feats, bvars,
                          domain, iqr)
        note = (f"  clustering features: {feats}\n"
                f"  benchmark variables: {bvars}\n"
                f"  current designated winner: {sr['winner']}")
        table(domain, rows, note)
        out["domains"][domain] = {"spec": spec, "features": feats, "benchmark_variables": bvars,
                                  "current_winner": sr["winner"], "combinations": rows}
        csv_all += csv_rows(domain, rows,
                            final_decision=FINAL_DECISIONS[domain]["combination"])

    # ---------------------------------------------------------------- INVESTMENT
    # Canonical spec is risk_age_final.py's V2: risk_age_deviation kept as a clustering
    # feature, dropped from the benchmark set. Labels come from that run's cached grid.
    grid, labels = rcache["rad100"]
    install_bench(False)
    for s in ("rad_diag_rad100",):
        ds.SPECS[s] = ("investment", list(INV_FEATS_FULL))
        ds.FEATS_OF[s] = list(INV_FEATS_FULL)
        ds.BENCH_OF[s] = "investment"
    bvars = ds.BENCH_VARS["investment"]
    feats = INV_FEATS_FULL
    iqr = {c: float(df[c].quantile(0.75) - df[c].quantile(0.25))
           for c in dict.fromkeys(feats + bvars)}
    base = base_rates(df, "investment")
    gate = rfinal["variants"][INV_VARIANT]["gate_passers"]
    usab = {k: usability(df, k, labels, "investment", base) for k in gate}
    rows = build_rows(df, gate, grid, labels, usab, feats, bvars, "investment", iqr)
    note = (f"  clustering features: {feats}\n"
            f"  benchmark variables: {bvars}  (risk_age_deviation dropped — no injected "
            f"profile varies on it)\n"
            f"  current designated winner: A|fuzzy_cmeans|2  "
            f"(the only gate-passer also clearing the per-band normal floor)")
    table("investment", rows, note)
    out["domains"]["investment"] = {"spec": f"{INV_VARIANT} / rad100",
                                    "features": feats, "benchmark_variables": bvars,
                                    "current_winner": "A|fuzzy_cmeans|2",
                                    "combinations": rows}
    csv_all += csv_rows("investment", rows,
                        final_decision=FINAL_DECISIONS["investment"]["combination"])

    # ------------------------------------------------- Insurance: the merged-tier option
    # B|kmedoids|3's T1 and T2 are not distinct. Merging them is a THIRD option alongside
    # "keep the 3-tier label" and "switch to a k=2 model" — and it is not equivalent to
    # either, because no k=2 model reproduces its 33.7/66.3 split.
    print("\n" + "=" * 132)
    print("INSURANCE — the merged-tier option: B|kmedoids|3 with its two non-distinct tiers "
          "collapsed")
    print("=" * 132)
    ispec = "insurance_core3"
    ifeats, ibvars = FEATS_OF[ispec], ds.BENCH_VARS["insurance"]
    iiqr = {c: float(df[c].quantile(0.75) - df[c].quantile(0.25))
            for c in dict.fromkeys(ifeats + ibvars)}
    ilabels = dict(dcache[ispec]["labels"])
    igrid = dict(res["spec_results"][ispec]["grid"])
    sc = RobustScaler().fit(df[ifeats])
    asc = StandardScaler().fit(df[["age_midpoint"]])
    X_ins = np.hstack([sc.transform(df[ifeats]), asc.transform(df[["age_midpoint"]])])
    mkey = "B|kmedoids3merged|2"
    ilabels[mkey] = merge_labels(df, "B|kmedoids|3", ilabels, "insurance", [1, 2])
    m = compute_metrics(X_ins, ilabels[mkey])
    mac, _ = macro_silhouette(X_ins, ilabels[mkey])
    sh, _ = cluster_shares(ilabels[mkey], 2)
    igrid[mkey] = {"spec": ispec, "approach": "B", "algorithm": "kmedoids3merged", "k": 2,
                   "silhouette_micro": m["silhouette_score"], "silhouette_macro": round(mac, 4),
                   "davies_bouldin": m["davies_bouldin_index"],
                   "calinski_harabasz": m["calinski_harabasz_index"],
                   "min_cluster_pct": round(100 * min(sh), 2)}
    ibase = base_rates(df, "insurance")
    iusab = {mkey: usability(df, mkey, ilabels, "insurance", ibase)}
    mrows = build_rows(df, [mkey], igrid, ilabels, iusab, ifeats, ibvars, "insurance", iiqr)
    mr = mrows[0]
    orig = next(r for r in out["domains"]["insurance"]["combinations"]
                if r["key"] == "B|kmedoids|3")
    print(f"  {mkey}: k=2, macro-sil {mr['macro_sil']:.4f} "
          f"(vs {orig['macro_sil']:.4f} for the 3-tier original it is derived from)")
    print(f"    tier shares {mr['tier_shares_pct']}   min cluster {mr['min_cluster_pct']:.2f}%   "
          f"balance {mr['balance']:.2f}")
    print(f"    gate: min rho {mr['min_rho']:.3f}   headroom {mr['headroom']:.3f}   "
          f"lift {mr['lift_pp']:.1f}pp   -> "
          f"{'PASSES both gates' if (mr['min_rho'] >= 0.5 and mr['headroom'] >= 0.10) else 'FAILS'}")
    print(f"    normal-population floor (pooled, its fitted level): "
          f"{mr['normal_floor']['pooled_min_pct']:.2f}%  "
          f"{'PASS' if mr['normal_floor']['passes_pooled'] else 'FAIL'}")
    w = mr["distinct"]["worst_pair"]
    print(f"    adjacent-tier distinctness: {mr['distinct']['verdict']}  "
          f"(geometric {w['geometric_sep_iqr']:.3f}, health {w['health_sep_iqr']:.3f})")
    print("    by benchmark variable: " + "  ".join(
        f"{v}={x}" for v, x in w["health_by_variable"].items()))
    print("    % clearing each benchmark by tier: " + "  ".join(
        f"{v}={[mr['distinct']['groups']['all']['pct_clearing'][t][v] for t in (0, 1)]}"
        for v in ibvars))
    out["domains"]["insurance"]["merged_tier_candidate"] = mr
    csv_all += csv_rows("insurance", mrows, row_type="constructed_merge",
                        final_decision=FINAL_DECISIONS["insurance"]["combination"])

    # ---------------------------------------------------- where does a third tier come from?
    print("\n" + "=" * 132)
    print("WHERE WOULD A THIRD TIER COME FROM? k=2 winner cross-tabbed against the best "
          "genuinely-3-tier candidate")
    print("=" * 132)
    xt = {}
    for domain, k2, k3, spec_labels, feats2, bvars2 in (
            ("banking", "A|kmeans|2", "A|fuzzy_cmeans|3", dcache["banking_core4"]["labels"],
             FEATS_OF["banking_core4"], ds.BENCH_VARS["banking"]),
            ("banking", "A|kmeans|2", "B|kmeans|3", dcache["banking_core4"]["labels"],
             FEATS_OF["banking_core4"], ds.BENCH_VARS["banking"]),
            ("insurance", "B|agglomerative|2", "B|kmedoids|3", ilabels, ifeats, ibvars)):
        # Population IQR over this pairing's own feature+benchmark union — the same formula
        # used for every other IQR in this script, so it is derived here rather than passed
        # in for some rows and computed for others.
        iqr2 = {c: float(df[c].quantile(0.75) - df[c].quantile(0.25))
                for c in dict.fromkeys(feats2 + bvars2)}
        ct = crosstab_k2_vs_k3(df, k2, k3, spec_labels, domain, feats2, bvars2, iqr2)
        xt[f"{domain}:{k2} x {k3}"] = ct
        print(f"\n  {domain}: {k2} (rows) x {k3} (cols), row-normalised over all 7,000 users")
        print("    " + pd.DataFrame(ct["row_normalised"]).to_string().replace("\n", "\n    "))
        for t2, s in ct["split_analysis"].items():
            print(f"    k=2 tier {t2} is split into k=3 tiers {s['k3_tiers']} "
                  f"({s['shares_pct'][0]}% / {s['shares_pct'][1]}% of it) — geometric "
                  f"{s['geometric_sep_iqr']:.3f}, health {s['health_sep_iqr']:.3f} -> "
                  f"{'A REAL SPLIT' if s['distinct'] else 'not distinct'}")
        if not ct["split_analysis"]:
            print("    no k=2 tier is split into two materially-sized k=3 tiers")
    out["third_tier_origin_crosstabs"] = xt

    # ---------------------------------------------------------------- head-to-head
    print("\n" + "=" * 132)
    print("HEAD-TO-HEAD — the specific comparisons this task was raised to settle")
    print("=" * 132)
    ins = {r["key"]: r for r in out["domains"]["insurance"]["combinations"]}
    bnk = {r["key"]: r for r in out["domains"]["banking"]["combinations"]}
    for label, r in (("insurance technical winner", ins.get("B|kmedoids|3")),
                     ("insurance k=2 alternative", ins.get("B|gmm_full|2")),
                     ("insurance k=2 alternative", ins.get("B|agglomerative|2")),
                     ("banking current winner", bnk.get("A|kmeans|2"))):
        if r is None:
            continue
        d = r["distinct"]
        print(f"\n  {label}: {r['key']}  (macro-sil {r['macro_sil']:.4f}, k={r['k']}, "
              f"verdict {d['verdict']})")
        for g, gg in d["groups"].items():
            print(f"    {g}: shares {gg['tier_shares_pct']}")
            for p in gg["adjacent_pairs"]:
                print(f"      {p['pair']}  geometric {p['geometric_sep_iqr']:.3f}  "
                      f"health {p['health_sep_iqr']:.3f}  -> {p['verdict']}")
                print("        by benchmark variable: " + "  ".join(
                    f"{v}={x}" for v, x in p["health_by_variable"].items()))
                print("        by clustering feature: " + "  ".join(
                    f"{v}={x}" for v, x in p["geometric_by_feature"].items()))

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    pd.DataFrame(csv_all).to_csv(OUT_CSV, index=False)
    print(f"\n  saved {OUT_JSON}")
    print(f"  saved {OUT_CSV}")


if __name__ == "__main__":
    main()
