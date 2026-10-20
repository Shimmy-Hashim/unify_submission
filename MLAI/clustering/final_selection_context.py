"""
final_selection_context.py — Population base rates + discrimination lift, the context
Step 4A's density check cannot be read without.

WHY THIS EXISTS
  Step 4A asks whether a "meaningful majority" of a cluster's members clear each
  benchmark. That criterion is only interpretable against the POPULATION base rate: if
  only ~10% of the synthetic population clears LIA's 9x life-coverage recommendation
  (which is what a dataset anchored to a *protection gap* study should look like), then
  no cluster larger than the 5% floor can plausibly have a majority above it — the bar
  would be unreachable by construction, not by clustering failure.

  So this reports, per band and overall:
    * base rate: % of the population meeting each Step 4A benchmark
    * the maximum share of the population that could form a "majority-clearing" cluster
    * for each floor-passing combination: the best cluster's LIFT over the base rate,
      i.e. does the clustering discriminate on that benchmark at all?

  Lift is the honest version of the usability question: a tier system is informative if
  its top tier is materially more likely to clear a benchmark than a random user, even
  when no tier reaches an absolute majority.

Run (after final_selection.py, which writes the label cache this reads):
    backend/venv/Scripts/python.exe MLAI/clustering/final_selection_context.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# meets_mask is imported rather than reimplemented: this script and final_selection.py
# must apply the SAME benchmark thresholds by construction, not by two copies agreeing.
from final_selection import (  # noqa: E402
    DATA_PATH, RESULTS_DIR, LABEL_CACHE, STEP4A_VARS, MAJORITY,
    MIN_CLUSTER_SHARE, meets_mask,
)

OUT_JSON = os.path.join(RESULTS_DIR, "final_selection_context.json")

INSURANCE_VARS = ("life_coverage_ratio", "ci_coverage_ratio")


def insurance_lift(rec):
    """Mean lift across the two insurance benchmarks — the app's actual purpose, and so
    the primary ranking key below."""
    return float(np.mean([rec[v]["mean_lift_pp_across_groups"]
                          for v in INSURANCE_VARS if v in rec]))


def lift_cell(rec, var):
    """One right-aligned lift column, or 'n/a' where the benchmark produced no groups."""
    return f"{rec[var]['mean_lift_pp_across_groups']:>8.1f}" if var in rec else "     n/a"


def main():
    df = pd.read_csv(DATA_PATH)
    with open(LABEL_CACHE, "rb") as f:
        cached = pickle.load(f)
    grid, labels = cached["grid"], cached["labels"]

    # ---- base rates -------------------------------------------------------------------
    print("=" * 92)
    print("POPULATION BASE RATES — % meeting each Step 4A benchmark (no clustering involved)")
    print("=" * 92)
    base = {"overall": {}, "by_band": {}}
    for var in STEP4A_VARS:
        base["overall"][var] = round(100 * float(meets_mask(df, var).mean()), 1)
    for b in range(1, 6):
        sub = df[df["age_band"] == b]
        base["by_band"][str(b)] = {var: round(100 * float(meets_mask(sub, var).mean()), 1)
                                   for var in STEP4A_VARS}
    print("  " + "band".ljust(8) + "".join(v[:20].rjust(22) for v in STEP4A_VARS))
    for b in range(1, 6):
        print("  " + f"{b}".ljust(8) + "".join(
            f"{base['by_band'][str(b)][v]:>21.1f}%" for v in STEP4A_VARS))
    print("  " + "ALL".ljust(8) + "".join(f"{base['overall'][v]:>21.1f}%" for v in STEP4A_VARS))

    print("\n  Reachability of the 'majority clears benchmark' criterion:")
    for var in STEP4A_VARS:
        r = base["overall"][var]
        # If x% of users clear a benchmark, a cluster with a >50% majority clearing it can
        # cover at most ~2x% of the population (all clearers + an equal number of others).
        print(f"    {var:<24} base rate {r:>5.1f}%  -> largest possible majority-clearing "
              f"cluster ~{min(100.0, 2 * r):>5.1f}% of the sample"
              f"{'   (BELOW/near the 5% floor)' if 2 * r < 12 else ''}")

    # ---- discrimination lift per floor-passing combination ---------------------------
    print("\n" + "=" * 92)
    print("DISCRIMINATION LIFT — best cluster's % clearing each benchmark vs base rate")
    print("=" * 92)
    survivors = [k for k, v in grid.items() if v.get("passes_floor")]
    lift_out = {}
    for key in survivors:
        approach = key.split("|")[0]
        groups = ({f"band{b}": (df[df["age_band"] == b], labels[key][b]) for b in range(1, 6)}
                  if approach == "A" else {"all": (df, labels[key])})
        per_var_best = {var: [] for var in STEP4A_VARS}
        for gkey, (sub, lab) in groups.items():
            bref = (base["by_band"][str(int(sub["age_band"].iloc[0]))]
                    if approach == "A" else base["overall"])
            for var in STEP4A_VARS:
                m = meets_mask(sub, var)
                # best cluster on THIS benchmark, restricted to clusters above the floor
                cand = [float(m[lab == c].mean()) for c in np.unique(lab)
                        if (lab == c).mean() >= MIN_CLUSTER_SHARE]
                if cand:
                    per_var_best[var].append((100 * max(cand), bref[var]))
        rec = {}
        for var in STEP4A_VARS:
            if not per_var_best[var]:
                continue
            # worst group again: a tier system must discriminate in every band
            lifts = [p - b for p, b in per_var_best[var]]
            worst_pct, worst_base = per_var_best[var][int(np.argmin(lifts))]
            rec[var] = {
                "best_cluster_pct_worst_group": round(worst_pct, 1),
                "base_rate_worst_group": round(worst_base, 1),
                "lift_pp_worst_group": round(worst_pct - worst_base, 1),
                "lift_ratio_worst_group": (round(worst_pct / worst_base, 2)
                                           if worst_base > 0 else None),
                "mean_lift_pp_across_groups": round(float(np.mean(lifts)), 1),
            }
        lift_out[key] = rec

    # rank by mean insurance lift, then by overall mean lift
    def score(key):
        rec = lift_out[key]
        all_lift = np.mean([rec[v]["mean_lift_pp_across_groups"] for v in rec])
        return -insurance_lift(rec), -all_lift

    ordered = sorted(lift_out, key=score)
    print("  (lift in percentage points; 'worst group' = the age band where the "
          "combination discriminates least)\n")
    print("  " + "combination".ljust(24) + "life_lift  ci_lift  sav_lift   nw_lift  dti_lift"
          "   | mean_ins_lift")
    for key in ordered:
        rec = lift_out[key]
        cells = "".join(lift_cell(rec, v) for v in
                        ("life_coverage_ratio", "ci_coverage_ratio", "savings_rate",
                         "net_worth_to_income", "dti"))
        print("  " + key.ljust(24) + cells + f"   | {insurance_lift(rec):>8.1f}")

    out = {"base_rates_pct": base,
           "majority_criterion_reachability": {
               v: {"base_rate_pct": base["overall"][v],
                   "max_majority_clearing_cluster_share_pct":
                       round(min(100.0, 2 * base["overall"][v]), 1),
                   "reachable_above_5pct_floor": bool(2 * base["overall"][v] >= 5.0)}
               for v in STEP4A_VARS},
           "majority_threshold": MAJORITY,
           "discrimination_lift_by_combination": lift_out,
           "ranked_by_insurance_lift": ordered}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved {OUT_JSON}")


if __name__ == "__main__":
    main()
