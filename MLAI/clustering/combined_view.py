"""
combined_view.py — One combined geometry + usability landscape across all floor-passing
combinations (Unify, CCDS25-1170, July 2026).

WHY THIS EXISTS
  The selection process so far produced two SEPARATE rankings: a geometric one
  (macro-silhouette, which picked A|spectral|2 then A|birch|2 — both confirmed unusable)
  and a usability one (ordering + materiality, which picked A|kmedoids|6 — never checked
  for its silhouette). Nobody had looked at both axes at once, so a middle-ground
  combination — mediocre on each metric but decent on both — could have been sitting in
  the gap between the two rankings, invisible to either ranking alone.

  This script joins the two views into one table and searches the interior of the
  landscape rather than its extremes.

  READS ONLY — consumes the already-computed outputs of final_selection.py (via its
  results JSON) plus its label cache for the axis diagnostic. No clustering is re-fitted,
  no thresholds are changed, nothing is written outside results/. That keeps this view
  provably consistent with the runs it summarises.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/combined_view.py            # new data
    backend/venv/Scripts/python.exe MLAI/clustering/combined_view.py --which old # pre-fix
"""

import argparse
import itertools
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_selection import (  # noqa: E402
    STEP4A_VARS, MONOTONIC_RHO_MIN, MIN_HEADROOM_CAPTURED, FEATURES_11,
    feature_profile, separation_profile, step4a, monotonicity,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO_ROOT, "MLAI/clustering/results")
PROC = os.path.join(REPO_ROOT, "MLAI/clustering/data/processed")

SOURCES = {
    "new": {"json": os.path.join(RES, "new_results/final_selection.json"),
            "cache": os.path.join(RES, "new_results/.final_selection_labels.pkl"),
            "data": os.path.join(PROC, "synthetic_users.csv")},
    "old": {"json": os.path.join(RES, "old_results/final_selection.json"),
            "cache": os.path.join(RES, ".final_selection_labels.pkl"),
            "data": os.path.join(PROC, "synthetic_users_pre_correlation_fix.csv")},
}
# The old run's JSON predates the ceiling-aware metric; its headroom figures live in the
# retest comparison instead. Same labels, same data — just computed later.
COMPARISON_JSON = os.path.join(RES, "correlation_retest_comparison.json")

# "Reasonably decent" bands for the sweet-spot search. Deliberately LOOSER than the
# strict gates — the point is to find middle ground the gates would reject, not to
# re-apply them. The silhouette floor follows the brief's own stated norm for this data
# (~0.13-0.20 is normal for 11-dimensional cross-domain synthetic data; the Phase-1
# banking-only 0.558 is not a comparable reference point).
SIL_DECENT = 0.15
RHO_DECENT = 0.30            # positive and consistent, below the strict 0.5 gate
HEADROOM_DECENT = 0.08       # just below the strict 0.10 gate


def load(which):
    src = SOURCES[which]
    with open(src["json"]) as f:
        j = json.load(f)
    grid = j["grid"]
    sweep = j["step5_survivor_usability_sweep"]
    # Backfill ceiling-aware headroom for the old run, whose JSON predates the metric.
    if which == "old" or any("min_headroom_captured" not in v for v in sweep.values()):
        with open(COMPARISON_JSON) as f:
            comp = json.load(f)
        detail = comp["per_combination_detail"][which]
        for k, v in sweep.items():
            if k in detail:
                v["min_headroom_captured"] = detail[k].get("min_headroom_captured")
                v.setdefault("passes_materiality_gate_headroom",
                             detail[k].get("passes_materiality_gate_headroom"))
                v.setdefault("passes_gate2_headroom", detail[k].get("passes_gate2_headroom"))
    return grid, sweep, src


def build_table(grid, sweep):
    rows = []
    for key, u in sweep.items():
        g = grid[key]
        approach, algo, k = key.split("|")
        rows.append({
            "combination": key, "approach": approach, "algorithm": algo, "k": int(k),
            "macro_sil": g["silhouette_macro"],
            "micro_sil": g["silhouette_micro"],
            "davies_bouldin": g["davies_bouldin"],
            "min_cluster_pct": g["min_cluster_pct"],
            "rho_min": u["worst_group_min_spearman_rho"],
            "headroom_min": u.get("min_headroom_captured"),
            "min_lift_pp": u["min_lift_pp"],
            "insurance_lift_pp": u["insurance_lift_pp"],
            "passes_ordering": u["passes_monotonicity_gate"],
            "passes_materiality_headroom": u.get("passes_materiality_gate_headroom"),
            "passes_both": u.get("passes_gate2_headroom"),
        })
    df = pd.DataFrame(rows)
    # Joint rank: how good is a combination on BOTH axes at once? Percentile-rank each
    # axis within the survivor pool, then take the MINIMUM of the two — a combination
    # scores well only if it is decent on both, so excelling at one cannot compensate.
    # (Deliberately non-compensatory, same reasoning as the conjunctive-screening logic
    # used for centroid selection.)
    df["pr_sil"] = df["macro_sil"].rank(pct=True)
    df["pr_use"] = df[["rho_min", "headroom_min"]].rank(pct=True).mean(axis=1)
    df["joint_min_pr"] = df[["pr_sil", "pr_use"]].min(axis=1)
    return df.sort_values("joint_min_pr", ascending=False).reset_index(drop=True)


def axis_diag(df_data, labels, key, iqr):
    approach = key.split("|")[0]
    out = {}
    if approach == "A":
        for b in range(1, 6):
            sub = df_data[df_data["age_band"] == b]
            out[f"band{b}"] = separation_profile(feature_profile(sub, labels[key][b]), iqr)
    else:
        out["all"] = separation_profile(feature_profile(df_data, labels[key]), iqr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="new", choices=["new", "old"])
    args = ap.parse_args()

    grid, sweep, src = load(args.which)
    df = build_table(grid, sweep)
    # The three "decent" bands, evaluated once. Used both to flag rows in the full table
    # and to select the sweet spot below, so the two can never drift apart.
    decent = ((df["macro_sil"] >= SIL_DECENT)
              & (df["rho_min"].fillna(-9) >= RHO_DECENT)
              & (df["headroom_min"].fillna(0) >= HEADROOM_DECENT))
    label = "POST-CORRELATION-FIX (new)" if args.which == "new" else "PRE-FIX (old)"

    print("=" * 118)
    print(f"COMBINED GEOMETRY + USABILITY LANDSCAPE — {label} — {len(df)} floor-passing combinations")
    print("=" * 118)
    print(f"  strict gates:      ordering rho >= {MONOTONIC_RHO_MIN}   "
          f"materiality headroom >= {MIN_HEADROOM_CAPTURED}")
    print(f"  'decent' bands:    macro_sil >= {SIL_DECENT}   rho >= {RHO_DECENT}   "
          f"headroom >= {HEADROOM_DECENT}   (looser than the gates, on purpose)")

    # ---------------------------------------------------------------- full table
    print(f"\n{'=' * 118}\nSTEP 1 — full table, sorted by JOINT strength "
          f"(min of the two percentile ranks; non-compensatory)\n{'=' * 118}")
    print(f"  {'combination':<24}{'macro_sil':>10}{'rho_min':>9}{'headrm':>8}"
          f"{'minclu%':>9}{'insLift':>9}{'jointPR':>9}  flags")
    print("  " + "-" * 114)
    for i, r in df.iterrows():
        flags = []
        if r["passes_both"]:
            flags.append("BOTH-GATES")
        else:
            if r["passes_ordering"]:
                flags.append("ord")
            if r["passes_materiality_headroom"]:
                flags.append("mat")
        if decent.iloc[i]:
            flags.append("<<< DECENT-BOTH")
        print(f"  {r['combination']:<24}{r['macro_sil']:>10.4f}"
              f"{(r['rho_min'] if r['rho_min'] is not None else float('nan')):>9.3f}"
              f"{(r['headroom_min'] if r['headroom_min'] is not None else float('nan')):>8.3f}"
              f"{r['min_cluster_pct']:>9.2f}{r['insurance_lift_pp']:>9.1f}"
              f"{r['joint_min_pr']:>9.2f}  {' '.join(flags)}")

    # ---------------------------------------------------------------- sweet spot search
    print(f"\n{'=' * 118}\nSTEP 2 — sweet-spot search: decent on BOTH, top scorer on neither\n{'=' * 118}")
    sweet = df[decent]
    if len(sweet):
        print(sweet[["combination", "macro_sil", "rho_min", "headroom_min",
                     "min_cluster_pct", "insurance_lift_pp"]].to_string(index=False))
    else:
        print("  NONE — no combination clears all three 'decent' bands simultaneously.")
        # How close does anything get? Report the best on each pairing.
        print("\n  Closest approaches to the middle ground (relaxing one band at a time):")
        for name, mask in [
            (f"macro_sil>={SIL_DECENT} AND rho>={RHO_DECENT} (headroom free)",
             (df["macro_sil"] >= SIL_DECENT) & (df["rho_min"].fillna(-9) >= RHO_DECENT)),
            (f"macro_sil>={SIL_DECENT} AND headroom>={HEADROOM_DECENT} (rho free)",
             (df["macro_sil"] >= SIL_DECENT) & (df["headroom_min"].fillna(0) >= HEADROOM_DECENT)),
            (f"rho>={RHO_DECENT} AND headroom>={HEADROOM_DECENT} (silhouette free)",
             (df["rho_min"].fillna(-9) >= RHO_DECENT) & (df["headroom_min"].fillna(0) >= HEADROOM_DECENT)),
        ]:
            sub = df[mask]
            print(f"\n   [{name}] -> {len(sub)} match")
            if len(sub):
                print(sub[["combination", "macro_sil", "rho_min", "headroom_min",
                           "min_cluster_pct"]].to_string(index=False))

        # Best achievable "worst axis" — quantifies the size of the gap.
        print("\n  Best JOINT performers regardless of bands (top 6 by min-percentile-rank):")
        print(df.head(6)[["combination", "macro_sil", "rho_min", "headroom_min",
                          "pr_sil", "pr_use", "joint_min_pr"]].to_string(index=False))

    # ---------------------------------------------------------------- named candidates
    print(f"\n{'=' * 118}\nSTEP 3 — the specifically-requested combinations\n{'=' * 118}")
    named = {
        "usability near-misses (pre-A|kmedoids|6 candidates)":
            ["B|gmm_full|5", "B|gmm_diag|6", "B|gmm_diag|5", "B|gmm_diag|4",
             "B|agglomerative|6", "B|birch|5", "A|gmm_diag|5", "A|gmm_diag|4"],
        "the both-gates candidate": ["A|kmedoids|6"],
        "top silhouette scorers at k=2 (never separately usability-checked)":
            ["A|spectral|2", "A|agglomerative|2", "A|birch|2", "A|kmeans|2", "B|birch|2"],
    }
    idx = df.set_index("combination")
    for group, keys in named.items():
        print(f"\n  --- {group} ---")
        print(f"  {'combination':<24}{'macro_sil':>10}{'micro_sil':>10}{'rho_min':>9}"
              f"{'headrm':>8}{'insLift':>9}{'minclu%':>9}")
        for k in keys:
            if k not in idx.index:
                print(f"  {k:<24}{'— did not pass the size floor —':>55}")
                continue
            r = idx.loc[k]
            print(f"  {k:<24}{r['macro_sil']:>10.4f}{r['micro_sil']:>10.4f}"
                  f"{(r['rho_min'] if r['rho_min'] is not None else float('nan')):>9.3f}"
                  f"{(r['headroom_min'] if r['headroom_min'] is not None else float('nan')):>8.3f}"
                  f"{r['insurance_lift_pp']:>9.1f}{r['min_cluster_pct']:>9.2f}")

    # Do agglomerative/birch at k=2 fail the SAME way spectral does, or differently?
    print(f"\n{'=' * 118}\nSTEP 3b — do the k=2 silhouette leaders fail identically, or differently?\n{'=' * 118}")
    df_data = pd.read_csv(src["data"])
    with open(src["cache"], "rb") as f:
        labels = pickle.load(f)["labels"]
    iqr = {f: float(df_data[f].quantile(0.75) - df_data[f].quantile(0.25)) for f in FEATURES_11}
    k2 = ["A|spectral|2", "A|agglomerative|2", "A|birch|2", "A|kmeans|2"]
    agree = {}
    for key in k2:
        if key not in labels:
            continue
        ad = axis_diag(df_data, labels, key, iqr)
        b1 = ad["band1"]
        print(f"\n  {key}")
        print(f"    band1: {b1['n_features_separating_ge_0.5_scaled']}/11 features >=0.5 IQR   "
              f"top={b1['top_feature']} ({b1['top_feature_spread']:.2f} IQR, "
              f"{100 * b1['top_feature_share_of_total_separation']:.0f}% of separation)")
        ins = {f: b1["spread_scaled_units"][f] for f in
               ("life_coverage_ratio", "ci_coverage_ratio", "insurance_spend_ratio")}
        print("    band1 insurance separation (IQR units): "
              + "  ".join(f"{var}={v:.3f}" for var, v in ins.items()))
        agree[key] = labels[key][1]
    # Are the partitions literally the same clustering?
    print("\n  Pairwise agreement of the band-1 partitions (are these the same split?):")
    for a, b in itertools.combinations([k for k in k2 if k in agree], 2):
        ari = adjusted_rand_score(agree[a], agree[b])
        print(f"    ARI({a:<20}, {b:<20}) = {ari:.4f}"
              + ("   IDENTICAL partition" if ari > 0.999 else ""))

    out = os.path.join(RES, f"combined_view_{args.which}.csv")
    df.to_csv(out, index=False)
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
