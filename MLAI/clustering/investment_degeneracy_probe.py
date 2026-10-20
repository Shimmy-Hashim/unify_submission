"""
investment_degeneracy_probe.py — why is the Investment domain STILL degenerate on the
normal population after the portfolio_concentration recalibration?

The recalibration (investment_recalibration.py) fixed what it targeted: gate passers went
1 -> 4, portfolio_concentration stopped being the binding constraint (56/60 -> 5/60), and
it now orders +1.000 on means / +0.800 on density among normal users. But the smallest
normal-population tier went 0.33% -> 0.15% — the degeneracy did NOT move. That means it was
never caused by portfolio_concentration, and the honest conclusion depends on identifying
what does cause it.

This probe answers three questions, read-only:
  1. Is the degeneracy specific to the selected winner, or present in every gate-passer?
  2. What is the degenerate tier actually made of?
  3. Is there any combination — gate-passing or not — that is non-degenerate on normals?

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/investment_degeneracy_probe.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import (  # noqa: E402
    DATA_PATH, OUT_DIR, FEATS_OF, run_spec_grid, usability, assign_tiers, base_rates,
)
from investment_recalibration import (  # noqa: E402
    SPEC, CONC_THRESHOLD, set_concentration_benchmark,
)

NORMAL_FLOOR = 0.05


def main():
    df = pd.read_csv(DATA_PATH)
    feats = FEATS_OF[SPEC]
    normal = (df["profile_type"] == "normal").to_numpy()
    set_concentration_benchmark(CONC_THRESHOLD)

    scaler = RobustScaler().fit(df[feats])
    age_scaler = StandardScaler().fit(df[["age_midpoint"]])
    X_full = np.hstack([scaler.transform(df[feats]),
                        age_scaler.transform(df[["age_midpoint"]])])
    grid, labels = run_spec_grid(df, SPEC, scaler, X_full)
    base = base_rates(df, "investment")
    survivors = {k: v for k, v in grid.items() if v.get("passes_floor")}
    sweep = {k: usability(df, k, labels, "investment", base) for k in sorted(survivors)}
    gate = [k for k in sweep if sweep[k]["passes_gate2_headroom"]]
    out = {}

    # ---------------------------------------------------------------- 2. what IS the tier?
    print("=" * 118)
    print("2. WHAT IS THE DEGENERATE TIER MADE OF?  (winner B|kmedoids|4, tier 1)")
    print("=" * 118)
    rank, k4, _ = assign_tiers(df, "B|kmedoids|4", labels, "investment")
    t1 = rank == 1
    comp = df.loc[t1.to_numpy(), "profile_type"].value_counts()
    print(f"  tier 1 has {int(t1.sum())} members overall, "
          f"{int((t1.to_numpy() & normal).sum())} of them normal-profile:")
    for p, c in comp.items():
        print(f"    {p:<20}{c:>5}  ({100 * c / t1.sum():>5.1f}% of the tier;  "
              f"{100 * c / (df.profile_type == p).sum():>5.1f}% of that profile)")
    out["degenerate_tier_composition"] = {p: int(c) for p, c in comp.items()}

    print("\n  investment_rate distributions — is the injected mode reachable by a normal user?")
    rows = {}
    for p in ["normal", "over_invested", "high_debt_investor"]:
        s = df.loc[df.profile_type == p, "investment_rate"]
        rows[p] = {"n": int(len(s)), "p50": round(float(s.median()), 4),
                   "p95": round(float(s.quantile(.95)), 4),
                   "p99": round(float(s.quantile(.99)), 4), "max": round(float(s.max()), 4),
                   "p01": round(float(s.quantile(.01)), 4), "min": round(float(s.min()), 4)}
        r = rows[p]
        print(f"    {p:<20} n={r['n']:>5}  min={r['min']:.3f}  p50={r['p50']:.3f}  "
              f"p95={r['p95']:.3f}  p99={r['p99']:.3f}  max={r['max']:.3f}")
    # The decisive comparison: how reachable is the degenerate tier's entry level for a
    # normal user, versus for the two profiles whose overrides create that mode?
    thr = df.loc[rank == 1, "investment_rate"].min()
    injected = df.profile_type.isin(["over_invested", "high_debt_investor"])
    pct_normal_reaching = 100 * float((df.loc[normal, "investment_rate"] >= thr).mean())
    pct_injected_reaching = 100 * float((df.loc[injected, "investment_rate"] >= thr).mean())
    print(f"    tier-1 members have investment_rate >= {thr:.3f}; only "
          f"{pct_normal_reaching:.2f}% of normal users reach that, vs "
          f"{pct_injected_reaching:.1f}% of the two injected high-investment profiles.")
    out["investment_rate_distributions"] = rows
    out["tier1_investment_rate_floor"] = round(float(thr), 4)

    # ---------------------------------------------------------------- 1 + 3. sweep
    print("\n" + "=" * 118)
    print("1 + 3. NORMAL-POPULATION TIER SHARES ACROSS EVERY FLOOR-SURVIVING COMBINATION")
    print("=" * 118)
    print(f"  A combination is 'normal-degenerate' if any tier holds <{NORMAL_FLOOR:.0%} of the "
          f"{int(normal.sum()):,} normal-profile users — i.e. a tier a typical real user "
          "could never land in.")
    recs = []
    for key in sorted(survivors):
        r, kf, _ = assign_tiers(df, key, labels, "investment")
        shares = [float((r[normal] == t).mean()) for t in range(kf)]
        recs.append({"combination": key, "k": grid[key]["k"], "k_found": kf,
                     "macro_sil": grid[key]["silhouette_macro"],
                     "min_normal_tier_pct": round(100 * min(shares), 2),
                     "normal_tier_shares_pct": [round(100 * s, 2) for s in shares],
                     "passes_normal_floor": bool(min(shares) >= NORMAL_FLOOR),
                     "passes_gates": key in gate,
                     "min_rho": sweep[key]["worst_group_min_spearman_rho"],
                     "min_headroom": sweep[key]["min_headroom_captured"]})
    probe = pd.DataFrame(recs)
    print(f"\n  Of {len(probe)} floor survivors: {int(probe.passes_normal_floor.sum())} are "
          f"non-degenerate on normals; {int(probe.passes_gates.sum())} pass the usability gates; "
          f"{int((probe.passes_normal_floor & probe.passes_gates).sum())} do BOTH.")

    print(f"\n  The {int(probe.passes_gates.sum())} gate-passing combinations:")
    print(f"    {'combination':<24}{'k':>3}{'macroSil':>10}{'minRho':>8}{'headrm':>8}"
          f"{'min normal tier':>17}   normal tier shares %")
    for _, r in probe[probe.passes_gates].sort_values("macro_sil", ascending=False).iterrows():
        flag = "" if r.passes_normal_floor else "   <<< normal-degenerate"
        print(f"    {r.combination:<24}{r.k:>3}{r.macro_sil:>10.4f}{r.min_rho:>8.3f}"
              f"{r.min_headroom:>8.3f}{r.min_normal_tier_pct:>16.2f}%   "
              f"{r.normal_tier_shares_pct}{flag}")

    both = probe[probe.passes_normal_floor & probe.passes_gates]
    print("\n  Combinations passing BOTH the usability gates AND the normal-population "
          f"floor: {len(both)}")
    if len(both):
        print(both[["combination", "k", "macro_sil", "min_rho", "min_headroom",
                    "min_normal_tier_pct"]].to_string(index=False))

    non_degenerate = probe[probe.passes_normal_floor]
    print(f"\n  Best {min(8, len(non_degenerate))} normal-non-degenerate combinations by "
          "macro-silhouette (regardless of gate):")
    print(non_degenerate.sort_values("macro_sil", ascending=False).head(8)[
        ["combination", "k", "macro_sil", "min_rho", "min_headroom",
         "min_normal_tier_pct", "passes_gates"]].to_string(index=False))

    # Why do the non-degenerate ones fail the gate?
    print("\n  Why the normal-non-degenerate combinations fail the usability gate "
          "(per-benchmark, top 6 by macro-silhouette):")
    for key in non_degenerate.sort_values("macro_sil", ascending=False).head(6)["combination"]:
        u = sweep[key]
        bits = []
        for v in ds.BENCH_VARS["investment"]:
            lv = u["lift"][v]
            bits.append(f"{v[:12]}: rho ok, head={lv['worst_headroom_captured']:.3f}"
                        f"{'' if lv['material_headroom'] else ' FAIL'}")
        print(f"    {key:<24} min_rho={u['worst_group_min_spearman_rho']:.3f}"
              f"{'' if u['passes_monotonicity_gate'] else ' (ORDERING FAIL)'}   "
              + " | ".join(bits))

    out["per_combination"] = recs
    out["summary"] = {
        "n_survivors": len(probe), "n_gate_passers": int(probe.passes_gates.sum()),
        "n_normal_non_degenerate": int(probe.passes_normal_floor.sum()),
        "n_passing_both": int((probe.passes_normal_floor & probe.passes_gates).sum()),
        "normal_floor": NORMAL_FLOOR,
    }
    json_path = os.path.join(OUT_DIR, "investment_degeneracy_probe.json")
    csv_path = os.path.join(OUT_DIR, "investment_degeneracy_probe.csv")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    probe.to_csv(csv_path, index=False)
    print(f"\n  saved {json_path}")
    print(f"  saved {csv_path}")


if __name__ == "__main__":
    main()
