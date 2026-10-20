"""
risk_age_followup.py — three checks the risk_age_deviation diagnostic raised, on the
chosen fix (V2: risk_age_deviation dropped from Investment's BENCHMARK list, kept as a
clustering feature). Reads the cached grids written by risk_age_diagnostic.py; refits
nothing and writes nothing outside results/.

  1. STRICTER NORMAL-POPULATION FLOOR, PER BAND. The adopted floor ("no tier holds <5% of
     normal-profile users") was applied to the whole population. Under Approach A the tiers
     are fitted per band, so a band-level tier could sit below 5% of that band's normals
     while the pooled figure looks healthy. Recomputed per band for every candidate that
     passed both gates.
  2. WHAT DOES "TIER 0" MEAN IN EACH BAND? A|kmeans|3's best tier is a ~9% high-investment
     cluster in bands 1/2/5 but a ~50% moderate cluster in bands 3/4 — i.e. the tier label
     does not denote the same kind of investor at every age. Quantified rather than assumed.
  3. HEAD-TO-HEAD vs THE NAMED NEAR-MISS. Full tier/archetype/ordering detail for
     B|fuzzy_cmeans|3 (Approach B, k=3) alongside the selected A|kmeans|3, since the
     selection rule is macro-silhouette-first and the runner-up has materially more
     materiality headroom and one consistent tier definition across ages.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/risk_age_followup.py
"""

import json
import os
import pickle
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import (  # noqa: E402
    DATA_PATH, OUT_DIR, groups_for, assign_tiers, meets_mask, health_signal,
    separation_profile, base_rates, usability, PROFILE_ORDER,
)
from risk_age_diagnostic import (  # noqa: E402
    CACHE, INV_FEATS_FULL, NORMAL_FLOOR, install_bench, RAD,
)

CANDIDATES = ["A|kmeans|3", "B|fuzzy_cmeans|3", "A|fuzzy_cmeans|2", "A|fuzzy_cmeans|6"]
OUT_JSON = os.path.join(OUT_DIR, "risk_age_followup.json")


def main():
    df = pd.read_csv(DATA_PATH)
    with open(CACHE, "rb") as f:
        grid, labels = pickle.load(f)["rad100"]
    spec = "rad_diag_rad100"
    ds.SPECS[spec] = ("investment", list(INV_FEATS_FULL))
    ds.FEATS_OF[spec] = list(INV_FEATS_FULL)
    ds.BENCH_OF[spec] = "investment"
    install_bench(False)          # the chosen fix: rad is a feature, not a benchmark
    base = base_rates(df, "investment")
    bvars = ds.BENCH_VARS["investment"]
    normal = df.profile_type == "normal"
    out = {"benchmark_vars": bvars, "population_base_rates_pct": base["overall"]}

    # ---------------------------------------------------------------- 1. per-band floor
    print("=" * 118)
    print("1. NORMAL-POPULATION FLOOR, RECOMPUTED PER BAND (the pooled figure can hide a "
          "band-level degenerate tier)")
    print("=" * 118)
    print(f"  {'candidate':<20}{'pooled min':>12}   per-band minimum share of that band's normals")
    perband = {}
    for key in CANDIDATES:
        rank, kf, _ = assign_tiers(df, key, labels, "investment")
        d = df.assign(tier=rank)
        pooled = min(float((d.loc[normal, "tier"] == t).mean()) for t in range(kf))
        rows = {}
        for b in range(1, 6):
            m = normal & (d.age_band == b)
            sh = [float((d.loc[m, "tier"] == t).mean()) for t in range(kf)]
            rows[b] = {"shares_pct": [round(100 * s, 2) for s in sh],
                       "min_pct": round(100 * min(sh), 2)}
        worst = min(rows, key=lambda b: rows[b]["min_pct"])
        perband[key] = {"k_found": kf, "pooled_min_pct": round(100 * pooled, 2),
                        "by_band": rows, "worst_band": worst,
                        "worst_band_min_pct": rows[worst]["min_pct"],
                        "passes_per_band_floor": bool(rows[worst]["min_pct"] >= 100 * NORMAL_FLOOR)}
        p = perband[key]
        print(f"  {key:<20}{p['pooled_min_pct']:>11.2f}%   "
              + "  ".join(f"b{b}:{rows[b]['min_pct']:.1f}%" for b in range(1, 6))
              + f"   -> worst band {worst} at {p['worst_band_min_pct']:.2f}% "
              + ("PASS" if p["passes_per_band_floor"] else "<<< FAILS PER-BAND FLOOR"))
    out["per_band_normal_floor"] = perband

    # ---------------------------------------------------------------- 2. tier semantics
    print("\n" + "=" * 118)
    print("2. DOES 'TIER 0' MEAN THE SAME THING IN EVERY BAND? (Approach A fits each band "
          "separately, so it need not)")
    print("=" * 118)
    sem = {}
    for key in [k for k in CANDIDATES if k.startswith("A|")]:
        u = usability(df, key, labels, "investment", base)
        sem[key] = {}
        print(f"\n  {key}:")
        print(f"    {'band':<7}{'best-tier share':>17}{'best-tier inv_rate':>21}"
              f"{'best-tier conc':>17}   what the best tier is")
        for gk, g in u["per_group"].items():
            c = g["health_order_best_to_worst"][0]
            e = g["clusters"][c]
            ir = e["variables"]["investment_rate"]["mean"]
            pc = e["variables"]["portfolio_concentration"]["mean"]
            if e["share_pct"] < 20 and ir > 0.25:
                kind = "small ultra-high-investment cluster"
            elif e["share_pct"] > 35:
                kind = "broad moderate-investment cluster"
            else:
                kind = "mid-sized cluster"
            sem[key][gk] = {"share_pct": e["share_pct"], "investment_rate_mean": ir,
                            "portfolio_concentration_mean": pc, "n": e["n"],
                            "characterisation": kind}
            print(f"    {gk:<7}{e['share_pct']:>16.2f}%{ir:>21.3f}{pc:>17.3f}   {kind}")
        # Collect the shares BEFORE adding "_consistency": it lives in the same dict as the
        # per-band entries, so this comprehension must not see it.
        shares = [v["share_pct"] for v in sem[key].values()]
        sem[key]["_consistency"] = {"best_tier_share_range_pp": round(max(shares) - min(shares), 1),
                                    "consistent": bool(max(shares) - min(shares) < 20)}
        print(f"    -> best-tier share ranges {min(shares):.1f}%-{max(shares):.1f}% across bands "
              f"(spread {max(shares) - min(shares):.1f}pp)"
              + ("" if sem[key]["_consistency"]["consistent"]
                 else "  <<< the tier label does not denote one archetype across ages"))
    out["tier_semantics_by_band"] = sem

    # ---------------------------------------------------------------- 3. head-to-head
    print("\n" + "=" * 118)
    print("3. HEAD-TO-HEAD: selected A|kmeans|3 vs the named near-miss B|fuzzy_cmeans|3")
    print("=" * 118)
    iqr = {f: float(df[f].quantile(0.75) - df[f].quantile(0.25)) for f in INV_FEATS_FULL}
    h2h = {}
    for key in ["A|kmeans|3", "B|fuzzy_cmeans|3"]:
        g, u = grid[key], usability(df, key, labels, "investment", base)
        rank, kf, _ = assign_tiers(df, key, labels, "investment")
        d = df.assign(tier=rank)
        nd = d[normal]
        share = nd.tier.value_counts(normalize=True).sort_index()
        seps = {gk: separation_profile(sub, lab, INV_FEATS_FULL, iqr)
                for gk, (sub, lab) in groups_for(df, key, labels).items()}
        rad_spreads = [s["spread_iqr_units"][RAD] for s in seps.values()]
        top_shares = [s["top_feature_share_of_total_separation"] for s in seps.values()]
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
        arche = {}
        for p in PROFILE_ORDER:
            s = d[d.profile_type == p]
            vc = s.tier.value_counts(normalize=True)
            arche[p] = {"n": int(len(s)),
                        "tier_pct": {int(t): round(100 * float(vc.get(t, 0.0)), 1)
                                     for t in range(kf)},
                        "normalised_mean_tier": round(float(s.tier.mean()) / max(1, kf - 1), 3)}
        comp = {t: {"n": int((d.tier == t).sum()),
                    "pct_injected": round(100 * float(
                        (d.loc[d.tier == t, "profile_type"] != "normal").mean()), 1)}
                for t in range(kf)}
        h2h[key] = {
            "approach": key.split("|")[0], "k": kf,
            "macro_silhouette": g["silhouette_macro"], "micro_silhouette": g["silhouette_micro"],
            "davies_bouldin": g["davies_bouldin"], "calinski_harabasz": g["calinski_harabasz"],
            "min_cluster_pct": g["min_cluster_pct"],
            "min_rho": u["worst_group_min_spearman_rho"],
            "min_headroom_captured": u["min_headroom_captured"], "min_lift_pp": u["min_lift_pp"],
            "passes_gate": u["passes_gate2_headroom"],
            "normal_tier_share_pct": {int(i): round(100 * float(v), 2) for i, v in share.items()},
            "normal_min_tier_pct": round(100 * float(share.min()), 2),
            "per_band_normal_min_pct": perband[key]["worst_band_min_pct"],
            "normal_ordering": ordering,
            "top_feature_share_of_separation_range": [round(min(top_shares), 3),
                                                      round(max(top_shares), 3)],
            "rad_separation_iqr_units_range": [round(min(rad_spreads), 3),
                                               round(max(rad_spreads), 3)],
            "n_features_separating": [s["n_features_separating_ge_0.5"] for s in seps.values()],
            "archetype_placement": arche, "tier_composition": comp,
            "tier_composition_max_injected_pct": max(c["pct_injected"] for c in comp.values()),
        }

    a, b = h2h["A|kmeans|3"], h2h["B|fuzzy_cmeans|3"]
    print(f"  {'metric':<42}{'A|kmeans|3':>18}{'B|fuzzy_cmeans|3':>20}")
    for row_label, field in [
            ("approach", "approach"), ("k", "k"),
            ("macro-silhouette (selection metric)", "macro_silhouette"),
            ("micro-silhouette", "micro_silhouette"),
            ("Davies-Bouldin (lower better)", "davies_bouldin"),
            ("min cluster % (whole population)", "min_cluster_pct"),
            ("min Spearman rho (ordering gate)", "min_rho"),
            ("min headroom captured (materiality)", "min_headroom_captured"),
            ("min lift pp", "min_lift_pp"),
            ("min normal tier % (pooled)", "normal_min_tier_pct"),
            ("min normal tier % (worst band)", "per_band_normal_min_pct"),
            ("most injected-heavy tier, % injected", "tier_composition_max_injected_pct"),
            ("top-feature share of separation", "top_feature_share_of_separation_range"),
            ("risk_age_deviation separation (IQR units)", "rad_separation_iqr_units_range"),
            ("features separating >=0.5 IQR (of 3)", "n_features_separating")]:
        print(f"  {row_label:<42}{str(a[field]):>18}{str(b[field]):>20}")

    for key in ["A|kmeans|3", "B|fuzzy_cmeans|3"]:
        r = h2h[key]
        print(f"\n  {key} — ordering on the NORMAL population only:")
        for v, o in r["normal_ordering"].items():
            print(f"    {v:<28} mean-rho={o['mean_rho']:+.3f}  density-rho={o['density_rho']:+.3f}"
                  f"   %clearing best->worst: {o['pct_clearing_by_tier']}")
        print(f"  {key} — normal tier shares: "
              + "  ".join(f"T{t}={v}%" for t, v in r["normal_tier_share_pct"].items()))
        print(f"  {key} — tier composition: "
              + "  ".join(f"T{t}(n={c['n']}, {c['pct_injected']}% injected)"
                          for t, c in r["tier_composition"].items()))
        print(f"  {key} — archetype normalised mean tier (0=best, 1=worst):")
        print("    " + "  ".join(f"{p}={r['archetype_placement'][p]['normalised_mean_tier']}"
                                 for p in PROFILE_ORDER))
    out["head_to_head"] = h2h

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  saved {OUT_JSON}")


if __name__ == "__main__":
    main()
