"""
compare_correlation_retest.py — Old vs. new side-by-side comparison for the
accumulation<->protection correlation fix (July 2026).

Evaluates BOTH datasets (pre-fix and post-fix) under BOTH materiality formulations
(absolute percentage-point lift, and the ceiling-aware headroom-captured variant added
during this retest), because a threshold artifact was found mid-analysis:

  A|kmedoids|6 on the NEW data passes the ordering gate and materially lifts 4 of 5
  benchmarks, failing only on dti at +4.5pp — but band 5's dti base rate is already
  95.5%, so 4.5pp IS the maximum achievable lift, and the top tier reaches 100.0%. The
  absolute test scores perfect discrimination as a failure.

Running the ceiling-aware gate on the OLD data too is what separates "the correlation fix
produced a new candidate" from "my threshold was hiding a candidate that was always
there". That distinction decides whether the architectural fork is resolved, so it is
computed rather than argued.

Writes results/correlation_retest_comparison.json and prints the full comparison.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/compare_correlation_retest.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_selection as fs  # noqa: E402
from final_selection import (  # noqa: E402
    STEP4A_VARS, MONOTONIC_RHO_MIN, MIN_LIFT_PP, MIN_HEADROOM_CAPTURED,
    step4a, monotonicity, discrimination_lift, usability_summary, base_rates,
    run_grid, feature_profile, separation_profile, FEATURES_11,
)
from sklearn.preprocessing import RobustScaler, StandardScaler  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO_ROOT, "MLAI/clustering/results")
PROC = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed")

DATASETS = {
    "old": os.path.join(PROC, "synthetic_users_pre_correlation_fix.csv"),
    "new": os.path.join(PROC, "synthetic_users.csv"),
}
# Reuse each run's already-computed label cache where available (deterministic, same
# seed) so this comparison cannot drift from the runs it is comparing.
CACHES = {
    "old": os.path.join(RES, ".final_selection_labels.pkl"),
    "new": os.path.join(RES, "new_results", ".final_selection_labels.pkl"),
}
OUT = os.path.join(RES, "correlation_retest_comparison.json")

ACC = ["savings_rate", "net_worth_to_income", "investment_rate"]
PRO = ["life_coverage_ratio", "ci_coverage_ratio", "insurance_spend_ratio"]

# The copula's configured block mean per era — the parameter that was changed, as opposed
# to the realized correlation measured off the data below. Reported in one place and
# reused by both the printout and the payload so the two can't state different numbers.
COPULA_BLOCK_MEAN = {"old": 0.182, "new": 0.290}

RULE = "=" * 100


def banner(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


def pair(old, new, field):
    """Format one metric as the "old -> new" cell used throughout the side-by-side table."""
    return f"{old[field]} -> {new[field]}"


def cross_domain_corr(df):
    vals = [float(np.corrcoef(df[a], df[p])[0, 1]) for a in ACC for p in PRO]
    return float(np.mean(vals))


def evaluate(tag):
    """Re-derive one era's full usability sweep from its dataset and its label cache.

    ⚠️ The cache is trusted to belong to DATASETS[tag]; nothing verifies the pairing, and
    both eras have 7,000 rows so a mismatch cannot be caught by shape. CACHES["old"] is
    results/.final_selection_labels.pkl, which is ALSO final_selection.py's default
    LABEL_CACHE — and that script fits against the CURRENT synthetic_users.csv. Running it
    standalone therefore overwrites the "old" cache with new-era labels, after which this
    comparison would silently pair new labels with the old dataframe and report a wrong
    "old" column with no error. See the module docstring's note on why the old-data column
    is the load-bearing half of this comparison.
    """
    df = pd.read_csv(DATASETS[tag])
    cache = CACHES[tag]
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            cached = pickle.load(f)
        grid, labels = cached["grid"], cached["labels"]
        print(f"  [{tag}] loaded {len(grid)} combinations from label cache")
    else:
        scaler = RobustScaler().fit(df[FEATURES_11])
        age_scaler = StandardScaler().fit(df[["age_midpoint"]])
        X_full = np.hstack([scaler.transform(df[FEATURES_11]),
                            age_scaler.transform(df[["age_midpoint"]])])
        print(f"  [{tag}] no cache — re-fitting the grid")
        grid, labels = run_grid(df, scaler, X_full)
        with open(cache, "wb") as f:
            pickle.dump({"grid": grid, "labels": labels}, f)

    base = base_rates(df)
    survivors = {k: v for k, v in grid.items() if v.get("passes_floor")}
    sweep = {}
    for key in survivors:
        approach = key.split("|")[0]
        groups = ({f"band{b}": (df[df["age_band"] == b], labels[key][b]) for b in range(1, 6)}
                  if approach == "A" else {"all": (df, labels[key])})
        s4 = step4a(df, groups, approach)
        mono = monotonicity(s4)
        lift = discrimination_lift(groups, base, approach)
        sweep[key] = usability_summary(s4, mono, lift)
    return {"df": df, "grid": grid, "labels": labels, "base": base,
            "survivors": survivors, "sweep": sweep}


def main():
    print(RULE)
    print("CORRELATION-FIX RETEST — OLD vs NEW, both materiality formulations")
    print(RULE)
    runs = {}
    for tag in ("old", "new"):
        print(f"\nEvaluating {tag} dataset...")
        runs[tag] = evaluate(tag)

    # ---- realized correlation ---------------------------------------------------------
    banner("REALIZED accumulation<->protection correlation (block mean of 9 pairs)")
    corr = {}
    for tag in ("old", "new"):
        df = runs[tag]["df"]
        corr[tag] = {"all_users": round(cross_domain_corr(df), 3),
                     "normal_profile_only": round(
                         cross_domain_corr(df[df.profile_type == "normal"]), 3)}
        print(f"  {tag:<4} all users {corr[tag]['all_users']:+.3f}   "
              f"normal-profile only {corr[tag]['normal_profile_only']:+.3f}")
    print(f"  copula parameter block mean: {COPULA_BLOCK_MEAN['old']:.3f} (old) -> "
          f"{COPULA_BLOCK_MEAN['new']:.3f} (new)")
    print(f"  DELTA all users {corr['new']['all_users'] - corr['old']['all_users']:+.3f}   "
          f"normal-only {corr['new']['normal_profile_only'] - corr['old']['normal_profile_only']:+.3f}")

    # ---- Step 1 floor grid comparison -------------------------------------------------
    banner("STEP 1 — minimum-size floor: which combinations change status?")
    all_keys = sorted(set(runs["old"]["grid"]) | set(runs["new"]["grid"]))
    floor_rows = []
    for k in all_keys:
        old, new = runs["old"]["grid"].get(k, {}), runs["new"]["grid"].get(k, {})
        floor_rows.append({"combination": k, "old_min_pct": old.get("min_cluster_pct"),
                           "new_min_pct": new.get("min_cluster_pct"),
                           "old_passes": old.get("passes_floor"),
                           "new_passes": new.get("passes_floor")})
    # Derived from floor_rows rather than accumulated alongside it, so the printed flips
    # and the serialized ones cannot disagree about what counts as a flip.
    flips = [r for r in floor_rows if r["old_passes"] != r["new_passes"]]
    print(f"  survivors: OLD {len(runs['old']['survivors'])}/90  ->  "
          f"NEW {len(runs['new']['survivors'])}/90")
    print(f"  status flips: {len(flips)}")
    for r in flips:
        print(f"    {r['combination']:<26} {r['old_min_pct']:>6}% -> {r['new_min_pct']:>6}%   "
              f"{'DISQUALIFIED -> PASS' if r['new_passes'] else 'PASS -> DISQUALIFIED'}")

    # ---- Gate comparison, both formulations ------------------------------------------
    banner("STEP 5 — ordering vs materiality, under BOTH formulations")
    gates = {}
    for tag in ("old", "new"):
        sw = runs[tag]["sweep"]
        gates[tag] = {
            "gate2a_ordering": sorted(k for k, v in sw.items() if v["passes_monotonicity_gate"]),
            "gate2b_absolute": sorted(k for k, v in sw.items() if v["passes_materiality_gate"]),
            "gate2_both_absolute": sorted(k for k, v in sw.items() if v["passes_gate2"]),
            "gate2b_headroom": sorted(k for k, v in sw.items()
                                      if v["passes_materiality_gate_headroom"]),
            "gate2_both_headroom": sorted(k for k, v in sw.items() if v["passes_gate2_headroom"]),
        }
        print(f"\n  --- {tag.upper()} dataset ({len(sw)} survivors) ---")
        print(f"    2a ordering (rho>={MONOTONIC_RHO_MIN} everywhere)      : "
              f"{len(gates[tag]['gate2a_ordering']):>2}  {gates[tag]['gate2a_ordering']}")
        print(f"    2b materiality, ABSOLUTE (>={MIN_LIFT_PP:.0f}pp)          : "
              f"{len(gates[tag]['gate2b_absolute']):>2}  {gates[tag]['gate2b_absolute']}")
        print(f"    => BOTH (absolute)                             : "
              f"{len(gates[tag]['gate2_both_absolute']):>2}  "
              f"{gates[tag]['gate2_both_absolute'] or 'NONE'}")
        print(f"    2b' materiality, CEILING-AWARE (>={MIN_HEADROOM_CAPTURED:.0%} headroom): "
              f"{len(gates[tag]['gate2b_headroom']):>2}  {gates[tag]['gate2b_headroom']}")
        print(f"    => BOTH (ceiling-aware)                        : "
              f"{len(gates[tag]['gate2_both_headroom']):>2}  "
              f"{gates[tag]['gate2_both_headroom'] or 'NONE'}")

    # ---- side-by-side numbers for every combination surviving in both ----------------
    banner("KEY METRICS SIDE BY SIDE (combinations surviving the floor in both runs)")
    both = sorted(set(runs["old"]["sweep"]) & set(runs["new"]["sweep"]))
    print(f"  {'combination':<24}{'rho_min':>16}{'min_lift_pp':>18}"
          f"{'min_headroom':>18}{'ins_lift_pp':>18}")
    print(f"  {'':<24}{'old -> new':>16}{'old -> new':>18}{'old -> new':>18}{'old -> new':>18}")
    rows = []
    for k in both:
        old, new = runs["old"]["sweep"][k], runs["new"]["sweep"][k]
        rows.append({
            "combination": k,
            "old_rho_min": old["worst_group_min_spearman_rho"],
            "new_rho_min": new["worst_group_min_spearman_rho"],
            "old_min_lift_pp": old["min_lift_pp"], "new_min_lift_pp": new["min_lift_pp"],
            "old_min_headroom": old["min_headroom_captured"],
            "new_min_headroom": new["min_headroom_captured"],
            "old_insurance_lift_pp": old["insurance_lift_pp"],
            "new_insurance_lift_pp": new["insurance_lift_pp"],
            "old_passes_gate2_headroom": old["passes_gate2_headroom"],
            "new_passes_gate2_headroom": new["passes_gate2_headroom"],
        })
        print(f"  {k:<24}{pair(old, new, 'worst_group_min_spearman_rho'):>16}"
              f"{pair(old, new, 'min_lift_pp'):>18}"
              f"{pair(old, new, 'min_headroom_captured'):>18}"
              f"{pair(old, new, 'insurance_lift_pp'):>18}")

    # ---- single-trivial-axis diagnostic on each run's macro-sil leader ---------------
    banner("STEP 4B — single-trivial-axis diagnostic on each run's top combination")
    axis = {}
    for tag in ("old", "new"):
        surv = runs[tag]["survivors"]
        leader = max(surv, key=lambda x: surv[x]["silhouette_macro"])
        df, labels = runs[tag]["df"], runs[tag]["labels"]
        iqr = {f: float(df[f].quantile(0.75) - df[f].quantile(0.25)) for f in FEATURES_11}
        approach = leader.split("|")[0]
        blocks = {}
        if approach == "A":
            for b in range(1, 6):
                sub = df[df["age_band"] == b]
                prof = feature_profile(sub, labels[leader][b])
                blocks[f"band{b}"] = separation_profile(prof, iqr)
        else:
            prof = feature_profile(df, labels[leader])
            blocks["all"] = separation_profile(prof, iqr)
        axis[tag] = {"leader": leader, "blocks": blocks}
        print(f"\n  {tag.upper()} leader: {leader}  (macro-sil {surv[leader]['silhouette_macro']})")
        for bk, s in blocks.items():
            print(f"    {bk:<8} features separating >=0.5 IQR: "
                  f"{s['n_features_separating_ge_0.5_scaled']}/11   top={s['top_feature']} "
                  f"({s['top_feature_spread']:.2f} IQR, "
                  f"{100 * s['top_feature_share_of_total_separation']:.0f}% of total separation)")

    payload = {
        "purpose": "Old vs new comparison for the accumulation<->protection correlation fix. "
                   "Comparison and reporting only — no promotion, no artifact export.",
        "copula_block_mean": COPULA_BLOCK_MEAN,
        "realized_correlation": corr,
        "step1_floor": {
            "old_survivors": len(runs["old"]["survivors"]),
            "new_survivors": len(runs["new"]["survivors"]),
            "status_flips": flips,
            "full_grid": floor_rows,
        },
        "gates": gates,
        "gate_thresholds": {"monotonic_rho_min": MONOTONIC_RHO_MIN,
                            "min_lift_pp": MIN_LIFT_PP,
                            "min_headroom_captured": MIN_HEADROOM_CAPTURED},
        "side_by_side": rows,
        "step4b_axis_diagnostic": axis,
        "per_combination_detail": {
            "old": runs["old"]["sweep"], "new": runs["new"]["sweep"]},
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  saved {OUT}")


if __name__ == "__main__":
    main()
