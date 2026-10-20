"""
risk_age_final.py — final Investment selection under the PER-BAND normal-population floor,
plus a cost check against the already-validated Banking and Insurance winners.

WHY THIS EXISTS
  risk_age_followup.py found that the pooled normal-population floor can be passed by a
  partition whose tiers mean different things in different age bands. Under Approach A the
  clustering is FITTED per band, so its tiers only exist per band — a pooled tier share
  averages a "tier 0" that is a 9% injected-profile-only cluster in bands 1/2/5 with a
  "tier 0" that is a 50% broad cluster in bands 3/4, and reports a healthy 23.78%. Applying
  the project's own adopted normal-population 5% floor AT THE LEVEL THE MODEL WAS FITTED is
  the honest reading of that check, not a new gate: for Approach A that is per band, for
  Approach B the pooled figure already is the fitted level (its per-band figures are
  reported anyway, since age is one of its input features and can dominate).

  This script therefore:
    1. Re-screens every gate-passing combination in all five variants against the per-band
       normal floor, so no variant is privileged.
    2. Reports full detail on whatever survives.
    3. Re-checks Banking's (A|kmeans|2) and Insurance's (B|kmedoids|3) already-validated
       winners under the same stricter reading, so adopting it is not silently paid for by
       the two domains that are already finished.

Reads cached labels only; refits nothing; writes nothing outside results/.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/risk_age_final.py
"""

import json
import os
import pickle
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# SPECS / FEATS_OF / BENCH_OF are reached through `ds.` rather than imported by name: the
# spec registrations below mutate domain_split_clustering's own tables, and going through
# the module makes that cross-module write visible at the call site.
import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import (  # noqa: E402
    DATA_PATH, OUT_DIR, LABEL_CACHE, groups_for, assign_tiers,
    meets_mask, health_signal, separation_profile, base_rates, usability, PROFILE_ORDER,
)
from risk_age_diagnostic import (  # noqa: E402
    CACHE, INV_FEATS_FULL, INV_FEATS_NORAD, NORMAL_FLOOR, install_bench, recompute_rad,
)

DIAG_JSON = os.path.join(OUT_DIR, "risk_age_diagnostic.json")
OUT_JSON = os.path.join(OUT_DIR, "risk_age_final.json")


def per_band_normal(df, key, labels, bd):
    rank, kf, _ = assign_tiers(df, key, labels, bd)
    d = df.assign(tier=rank)
    nd = d[d.profile_type == "normal"]
    pooled = [float((nd.tier == t).mean()) for t in range(kf)]
    by_band = {}
    for b in range(1, 6):
        s = nd[nd.age_band == b]
        sh = [float((s.tier == t).mean()) for t in range(kf)]
        by_band[b] = {"shares_pct": [round(100 * x, 2) for x in sh],
                      "min_pct": round(100 * min(sh), 2)}
    worst = min(by_band, key=lambda b: by_band[b]["min_pct"])
    return {"k_found": kf,
            "pooled_shares_pct": [round(100 * x, 2) for x in pooled],
            "pooled_min_pct": round(100 * min(pooled), 2),
            "by_band": by_band, "worst_band": worst,
            "worst_band_min_pct": by_band[worst]["min_pct"],
            "passes_pooled": bool(min(pooled) >= NORMAL_FLOOR),
            "passes_per_band": bool(by_band[worst]["min_pct"] >= 100 * NORMAL_FLOOR)}


def detail(df, key, labels, base, feats, bvars):
    g_u = usability(df, key, labels, "investment", base)
    rank, kf, _ = assign_tiers(df, key, labels, "investment")
    d = df.assign(tier=rank)
    nd = d[d.profile_type == "normal"]
    iqr = {f: float(df[f].quantile(0.75) - df[f].quantile(0.25)) for f in feats}
    seps = {gk: separation_profile(sub, lab, feats, iqr)
            for gk, (sub, lab) in groups_for(df, key, labels).items()}
    ordering = {}
    for var in bvars:
        sig = pd.Series(health_signal(nd, "investment", var), index=nd.index)
        means = sig.groupby(nd.tier).mean().sort_index()
        pcts = pd.Series(meets_mask(nd, "investment", var), index=nd.index).groupby(
            nd.tier).mean().sort_index()
        ordering[var] = {
            "mean_rho": round(float(stats.spearmanr(
                means.to_numpy(), -means.index.to_numpy(float)).statistic), 3),
            "density_rho": round(float(stats.spearmanr(
                pcts.to_numpy(), -pcts.index.to_numpy(float)).statistic), 3),
            "pct_clearing_by_tier": [round(100 * float(x), 1) for x in pcts]}
    arche, comp = {}, {}
    for p in PROFILE_ORDER:
        s = d[d.profile_type == p]
        vc = s.tier.value_counts(normalize=True)
        arche[p] = {"n": int(len(s)),
                    "tier_pct": {int(t): round(100 * float(vc.get(t, 0.0)), 1) for t in range(kf)},
                    "normalised_mean_tier": round(float(s.tier.mean()) / max(1, kf - 1), 3)}
    for t in range(kf):
        sub = d[d.tier == t]
        comp[t] = {"n": int(len(sub)),
                   "pct_injected": round(100 * float((sub.profile_type != "normal").mean()), 1),
                   "pct_of_normals": round(100 * float((nd.tier == t).mean()), 2)}
    return {"usability": {k: g_u[k] for k in
                          ("worst_group_min_spearman_rho", "min_headroom_captured", "min_lift_pp",
                           "passes_monotonicity_gate", "passes_materiality_gate_headroom",
                           "passes_gate2_headroom")},
            "lift": g_u["lift"], "per_group": g_u["per_group"],
            "separation": seps, "normal_ordering": ordering,
            "archetype_placement": arche, "tier_composition": comp}


def main():
    df0 = pd.read_csv(DATA_PATH)
    with open(CACHE, "rb") as f:
        cached = pickle.load(f)
    with open(DIAG_JSON) as f:
        diag = json.load(f)
    pop_t = {int(b): diag["step1_formula_calibration"]["per_band"][str(b)]["realized_median"]
             for b in range(1, 6)}
    dfs = {"rad100": df0, "radPop": recompute_rad(df0, pop_t), "noRad": df0}
    featsets = {"rad100": INV_FEATS_FULL, "radPop": INV_FEATS_FULL, "noRad": INV_FEATS_NORAD}
    for gname, feats in featsets.items():
        spec = f"rad_diag_{gname}"
        ds.SPECS[spec] = ("investment", list(feats))
        ds.FEATS_OF[spec] = list(feats)
        ds.BENCH_OF[spec] = "investment"

    VARIANTS = {"V0_baseline": ("rad100", True), "V1_formula_fix_only": ("radPop", True),
                "V2_bench_drop_only": ("rad100", False), "V3_drop_entirely": ("noRad", False),
                "V4_both_fixes": ("radPop", False)}

    print("=" * 118)
    print("1. EVERY GATE-PASSING COMBINATION, RE-SCREENED AGAINST THE PER-BAND NORMAL FLOOR")
    print("=" * 118)
    print("  'pooled' = share of ALL normal users in the smallest tier (the check as applied so")
    print("  far).  'worst band' = the same check inside the single worst age band, i.e. at the")
    print("  level Approach A's tiers actually exist. Both are reported for every candidate.\n")
    out = {"variants": {}}
    for v, (gname, inc) in VARIANTS.items():
        df = dfs[gname]
        grid, labels = cached[gname]
        install_bench(inc)
        base = base_rates(df, "investment")
        gate = diag["step3_4_variants"][v]["gate_passers"]
        print(f"  {v}  (features {featsets[gname]}, rad in benchmarks: {inc})")
        print(f"    {'combination':<22}{'macroSil':>10}{'minRho':>8}{'headrm':>8}"
              f"{'pooled min':>12}{'worst band':>12}   per-band minima")
        recs = {}
        for key in gate:
            pb = per_band_normal(df, key, labels, "investment")
            per_combo = diag["step3_4_variants"][v]["per_combination"][key]
            recs[key] = {"macro_sil": grid[key]["silhouette_macro"], **pb}
            print(f"    {key:<22}{grid[key]['silhouette_macro']:>10.4f}"
                  f"{per_combo['min_rho']:>8.3f}{per_combo['min_headroom']:>8.3f}"
                  f"{pb['pooled_min_pct']:>11.2f}%{pb['worst_band_min_pct']:>11.2f}%   "
                  + " ".join(f"{pb['by_band'][b]['min_pct']:.1f}" for b in range(1, 6))
                  + ("   PASS" if pb["passes_per_band"] else ""))
        surv = [k for k in recs if recs[k]["passes_per_band"]]
        print(f"    -> {len(surv)} of {len(gate)} gate-passers also clear the PER-BAND normal "
              f"floor: {surv if surv else 'NONE'}\n")
        out["variants"][v] = {"features": featsets[gname], "rad_in_benchmarks": inc,
                              "gate_passers": gate, "per_band_screen": recs,
                              "passing_gate_and_per_band_floor": surv}

    # ------------------------------------------------------------------ winner detail
    print("=" * 118)
    print("2. DETAIL ON EVERY COMBINATION THAT CLEARS GATES + THE PER-BAND FLOOR")
    print("=" * 118)
    out["detail"] = {}
    for v, (gname, inc) in VARIANTS.items():
        surv = out["variants"][v]["passing_gate_and_per_band_floor"]
        if not surv:
            continue
        df = dfs[gname]
        grid, labels = cached[gname]
        install_bench(inc)
        base = base_rates(df, "investment")
        bvars = ds.BENCH_VARS["investment"]
        feats = featsets[gname]
        for key in sorted(surv, key=lambda k: -grid[k]["silhouette_macro"]):
            g = grid[key]
            dt = detail(df, key, labels, base, feats, bvars)
            pb = out["variants"][v]["per_band_screen"][key]
            out["detail"][f"{v}|{key}"] = {"metrics": {m: g[m] for m in
                                                      ("silhouette_macro", "silhouette_micro",
                                                       "davies_bouldin", "calinski_harabasz",
                                                       "min_cluster_pct", "k")},
                                           "per_band_normal": pb, **dt}
            print(f"\n  --- {v}  {key}  (k={g['k']}, macro-sil {g['silhouette_macro']:.4f}, "
                  f"micro {g['silhouette_micro']:.4f}, DB {g['davies_bouldin']:.3f}, "
                  f"min cluster {g['min_cluster_pct']:.2f}%)")
            u = dt["usability"]
            print(f"      gate: rho {u['worst_group_min_spearman_rho']:.3f}  headroom "
                  f"{u['min_headroom_captured']:.3f}  lift {u['min_lift_pp']:.1f}pp  -> "
                  f"{'PASS' if u['passes_gate2_headroom'] else 'FAIL'}")
            print(f"      normal tiers pooled {pb['pooled_shares_pct']} (min "
                  f"{pb['pooled_min_pct']}%);  per-band minima "
                  + " ".join(f"b{b}={pb['by_band'][b]['min_pct']:.1f}%" for b in range(1, 6)))
            for var, dd in dt["lift"].items():
                print(f"      {var:<28} base {dd['base_rate_pct']:>5.1f}% -> best tier "
                      f"{dd['best_tier_pct_clearing']:>5.1f}%  (lift {dd['lift_pp']:>+6.1f}pp, "
                      f"headroom {dd['worst_headroom_captured']:.3f})  "
                      f"{'MATERIAL' if dd['material_headroom'] else 'immaterial'}")
            print("      ordering on NORMALS only:")
            for var, o in dt["normal_ordering"].items():
                print(f"        {var:<28} mean-rho={o['mean_rho']:+.3f}  "
                      f"density-rho={o['density_rho']:+.3f}   %clearing best->worst: "
                      f"{o['pct_clearing_by_tier']}")
            print("      tier profile per group (mean / % clearing), best -> worst:")
            for gk, gg in dt["per_group"].items():
                print(f"        {gk} (n={gg['n_group']}) min_rho={gg['min_spearman_rho']}")
                for c in gg["health_order_best_to_worst"]:
                    e = gg["clusters"][c]
                    bits = [f"{bvar[:20]}={e['variables'][bvar]['mean']:.3f}"
                            f"/{e['variables'][bvar]['pct_meeting_benchmark']:.0f}%"
                            for bvar in bvars]
                    print(f"          tier {e['health_rank']} (n={e['n']}, {e['share_pct']}%): "
                          + "  ".join(bits))
            print("      separation (IQR units of cluster-mean spread):")
            for gk, sp in dt["separation"].items():
                print(f"        {gk}: {sp['n_features_separating_ge_0.5']}/{sp['n_features']} "
                      f">=0.5 IQR; top={sp['top_feature']} "
                      f"({100 * sp['top_feature_share_of_total_separation']:.0f}%); "
                      f"{sp['spread_iqr_units']}")
            print("      tier composition:  " + "  ".join(
                f"T{t}(n={c['n']}, {c['pct_injected']}% injected, {c['pct_of_normals']}% of normals)"
                for t, c in dt["tier_composition"].items()))
            print("      archetype normalised mean tier (0=best, 1=worst):")
            print("        " + "  ".join(f"{p}={dt['archetype_placement'][p]['normalised_mean_tier']}"
                                         for p in PROFILE_ORDER))

    # ------------------------------------------------------------------ 3. cost check
    print("\n" + "=" * 118)
    print("3. DOES THE PER-BAND READING COST ANYTHING ALREADY VALIDATED?")
    print("=" * 118)
    with open(LABEL_CACHE, "rb") as f:
        dcache = pickle.load(f)
    with open(os.path.join(OUT_DIR, "domain_split_results.json")) as f:
        prior = json.load(f)
    cost = {}
    for dom, spec in (("banking", "banking_core4"), ("insurance", "insurance_core3")):
        key = prior["spec_results"][spec]["winner"]
        pb = per_band_normal(df0, key, dcache[spec]["labels"], dom)
        cost[dom] = {"winner": key, **pb}
        print(f"  {dom:<11}{key:<20} pooled min {pb['pooled_min_pct']:>6.2f}%   "
              f"per-band minima "
              + " ".join(f"b{b}={pb['by_band'][b]['min_pct']:.1f}%" for b in range(1, 6))
              + f"   worst band {pb['worst_band']} at {pb['worst_band_min_pct']:.2f}%  "
              + ("PASS" if pb["passes_per_band"] else "<<< WOULD NOW FAIL"))
    out["already_validated_domains_under_per_band_floor"] = cost
    print("  NOTE: the per-band reading is the correct level for APPROACH A (fitted per band).")
    print("  Insurance's winner is Approach B — fitted once on the pooled population, so the")
    print("  pooled figure (30.55%) is its fitted level and it remains validated. Its per-band")
    print("  zeros are nonetheless a real, separate finding: see section 4.")

    # ------------------------------------------------------------------ 4. age segregation
    print("\n" + "=" * 118)
    print("4. TIER x AGE-BAND CROSS-TABS (normal-profile users only) — how much of each tier "
          "system is really age?")
    print("=" * 118)
    xt = {}
    install_bench(False)
    inv_labels = cached["rad100"][1]
    for label, d, key, labels, bd in (
        ("investment_chosen_A|fuzzy_cmeans|2", dfs["rad100"], "A|fuzzy_cmeans|2",
         inv_labels, "investment"),
        ("investment_runnerup_B|fuzzy_cmeans|3", dfs["rad100"], "B|fuzzy_cmeans|3",
         inv_labels, "investment"),
        ("investment_higher_sil_A|kmeans|3", dfs["rad100"], "A|kmeans|3",
         inv_labels, "investment"),
        ("banking_validated_A|kmeans|2", df0, prior["spec_results"]["banking_core4"]["winner"],
         dcache["banking_core4"]["labels"], "banking"),
        ("insurance_validated_B|kmedoids|3", df0,
         prior["spec_results"]["insurance_core3"]["winner"],
         dcache["insurance_core3"]["labels"], "insurance"),
    ):
        rank, kf, _ = assign_tiers(d, key, labels, bd)
        nd = d.assign(tier=rank)
        nd = nd[nd.profile_type == "normal"]
        tab = pd.crosstab(nd.age_band, nd.tier, normalize="index").round(3)
        zeros = int((tab < 0.05).to_numpy().sum())
        xt[label] = {"crosstab_row_normalised": tab.to_dict(),
                     "cells_below_5pct": zeros, "n_cells": int(tab.size)}
        print(f"\n  {label}  ({zeros} of {tab.size} band x tier cells hold <5% of that band's "
              f"normals)")
        print("    " + tab.to_string().replace("\n", "\n    "))
    out["tier_by_age_band_crosstabs"] = xt

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  saved {OUT_JSON}")


if __name__ == "__main__":
    main()
