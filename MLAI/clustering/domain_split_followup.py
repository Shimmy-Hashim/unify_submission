"""
domain_split_followup.py — three targeted follow-ups the main domain-split run raised.

READS ONLY the cached deterministic labels written by domain_split_clustering.py; no
clustering is re-fitted with different settings, nothing outside results/ is written.

  1. BANKING TIER STABILITY. The chosen Banking winner (A|kmeans|2) agrees with its
     runner-up gate-passers on only ~29-30% of users. Is the Banking tier system itself
     unstable, or is that just a k=2-vs-k=3 comparison artifact? Pairwise ARI + rank
     correlation across all 9 Banking gate-passers, grouped by k.
  2. STEP 5 UNDER A k-MATCHED COMPARISON. Banking's winner is k=2 and Investment's is
     k=5; a 2x5 rank correlation is mechanically attenuated. Recompute Banking-vs-
     Investment association across every Banking gate-passer, so the independence verdict
     is not read off one k=2 result.
  3. ARE INSURANCE'S THREE TIERS REALLY THREE? Tier 1 and tier 2 have nearly identical
     life/CI coverage means; full per-cluster feature means (incl. insurance_spend_ratio,
     which is a clustering feature but deliberately not a benchmark variable because it is
     non-monotonic as a health signal) show what actually separates them.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/domain_split_followup.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_split_clustering import (  # noqa: E402
    DATA_PATH, OUT_DIR, LABEL_CACHE, SPECS, BENCH_OF, FEATS_OF, BENCH_VARS,
    DOMAIN_ORDER, PROFILE_ORDER, EXPECTED, OVERRIDDEN_DOMAINS,
    assign_tiers, groups_for, meets_mask,
)

RESULTS_JSON = os.path.join(OUT_DIR, "domain_split_results.json")


def flat_labels(df, key, labels_store):
    """One label per user, made globally unique across bands for Approach A."""
    out = pd.Series(index=df.index, dtype="object")
    for gkey, (sub, lab) in groups_for(df, key, labels_store).items():
        out.loc[sub.index] = [f"{gkey}_{c}" for c in lab]
    return out.to_numpy()


def short_label(key, width=6):
    """'A|agglomerative|4' -> 'agglom|4', short enough for a matrix header."""
    _, algo, k = key.split("|")
    return f"{algo[:width]}|{k}"


def main():
    df = pd.read_csv(DATA_PATH)
    with open(LABEL_CACHE, "rb") as f:
        cache = pickle.load(f)
    with open(RESULTS_JSON) as f:
        res = json.load(f)

    primary = res["primary_spec_per_domain"]
    winners = res["winners"]
    out = {}

    # ---------------------------------------------------------------- 1. Banking stability
    bank_spec = primary["banking"]
    bank_res = res["spec_results"][bank_spec]
    bank_gp = sorted(bank_res["gate_passing_combinations"],
                     key=lambda k: -bank_res["grid"][k]["silhouette_macro"])
    print("=" * 110)
    print(f"1. BANKING TIER STABILITY — all {len(bank_gp)} gate-passing combinations "
          f"of {bank_spec}")
    print("=" * 110)
    ranks, labs = {}, {}
    for key in bank_gp:
        r, kf, _ = assign_tiers(df, key, cache[bank_spec]["labels"], BENCH_OF[bank_spec])
        ranks[key] = r
        labs[key] = flat_labels(df, key, cache[bank_spec]["labels"])
    print(f"  {'combination':<24}{'k':>4}{'macroSil':>10}{'minRho':>9}{'headroom':>10}"
          f"{'T0 share%':>11}")
    for key in bank_gp:
        g = bank_res["grid"][key]
        u = bank_res["usability_sweep"][key]
        print(f"  {key:<24}{g['k']:>4}{g['silhouette_macro']:>10.4f}"
              f"{u['worst_group_min_spearman_rho']:>9.3f}{u['min_headroom_captured']:>10.3f}"
              f"{100 * float((ranks[key] == 0).mean()):>10.1f}%")

    # Both measures are symmetric, so each pair is computed once and then read off twice:
    # into the printed matrix (ARI above the diagonal, rank correlation below) and into out.
    ari = {(a, b): float(adjusted_rand_score(labs[a], labs[b]))
           for i, a in enumerate(bank_gp) for b in bank_gp[i + 1:]}
    rank_rho = {(a, b): float(stats.spearmanr(ranks[a], ranks[b]).statistic)
                for i, a in enumerate(bank_gp) for b in bank_gp[i + 1:]}
    print("\n  Pairwise agreement (upper = adjusted Rand on raw partitions; "
          "lower = Spearman on tier RANKS):")
    print("    " + "".join(f"{short_label(k):>14}" for k in bank_gp))
    for i, a in enumerate(bank_gp):
        row = f"  {short_label(a):<14}"
        for j, b in enumerate(bank_gp):
            if i == j:
                row += f"{'—':>14}"
            elif j > i:
                row += f"{ari[(a, b)]:>14.3f}"
            else:
                row += f"{rank_rho[(b, a)]:>14.3f}"
        print(row)
    out["banking_stability"] = {
        "gate_passers": bank_gp,
        "pairwise_ari": {f"{a} x {b}": round(v, 4) for (a, b), v in ari.items()},
        "pairwise_rank_spearman": {f"{a} x {b}": round(v, 4) for (a, b), v in rank_rho.items()},
    }

    # Does the Step 4 profile signal survive the winner choice? Mean banking tier rank,
    # normalised to 0-1 so k=2 and k=3/5/6 systems are directly comparable.
    print("\n  Does the Step 4 Banking signal survive the winner choice? "
          "(mean tier rank, normalised 0=healthiest .. 1=worst)")
    print(f"    {'profile':<20}" + "".join(f"{short_label(k, 5):>12}" for k in bank_gp))
    norm_tbl = {}
    for prof in PROFILE_ORDER:
        m = df["profile_type"] == prof
        vals = []
        for key in bank_gp:
            kf = int(ranks[key].max())
            vals.append(float(ranks[key][m].mean()) / max(1, kf))
        norm_tbl[prof] = {k: round(v, 3) for k, v in zip(bank_gp, vals)}
        print(f"    {prof:<20}" + "".join(f"{v:>12.3f}" for v in vals))
    out["banking_normalised_mean_tier_by_profile"] = norm_tbl
    spread = {p: round(max(v.values()) - min(v.values()), 3) for p, v in norm_tbl.items()}
    print(f"    spread (max-min across the {len(bank_gp)} combinations): "
          + "  ".join(f"{p[:14]}={s}" for p, s in spread.items()))
    out["banking_normalised_mean_tier_spread"] = spread

    # ---------------------------------------------------------------- 2. k-matched Step 5
    inv_spec = primary["investment"]
    inv_res = res["spec_results"][inv_spec]
    inv_gp = sorted(inv_res["gate_passing_combinations"] or inv_res["survivors"],
                    key=lambda k: -inv_res["grid"][k]["silhouette_macro"])[:5]
    print("\n" + "=" * 110)
    print("2. STEP 5 UNDER EVERY BANKING GATE-PASSER — is the near-zero Banking/Investment "
          "association a k=2 artifact?")
    print("=" * 110)
    iranks = {}
    for key in inv_gp:
        r, _, _ = assign_tiers(df, key, cache[inv_spec]["labels"], BENCH_OF[inv_spec])
        iranks[key] = r
    normal = df["profile_type"] == "normal"
    print(f"  Investment side: {inv_gp}")
    print(f"\n  {'banking combination':<24}{'k':>4}"
          + "".join(f"{'rho vs ' + short_label(k):>22}" for k in inv_gp))
    grid_rho = {}
    for bk in bank_gp:
        kb = int(ranks[bk].max()) + 1
        cells = []
        for ik in inv_gp:
            ra = float(stats.spearmanr(ranks[bk], iranks[ik]).statistic)
            rn = float(stats.spearmanr(ranks[bk][normal], iranks[ik][normal]).statistic)
            grid_rho[f"{bk} x {ik}"] = {"rho_all": round(ra, 4), "rho_normal": round(rn, 4)}
            cells.append(f"{ra:+.3f}/{rn:+.3f}")
        print(f"  {bk:<24}{kb:>4}" + "".join(f"{c:>22}" for c in cells))
    allr = [v["rho_all"] for v in grid_rho.values()]
    normr = [v["rho_normal"] for v in grid_rho.values()]
    print(f"\n  across all {len(grid_rho)} pairings —  rho(all users):   "
          f"min={min(allr):+.3f}  max={max(allr):+.3f}  mean={np.mean(allr):+.3f}")
    print(f"                                     rho(normal only): "
          f"min={min(normr):+.3f}  max={max(normr):+.3f}  mean={np.mean(normr):+.3f}")
    out["step5_kmatched_grid"] = grid_rho
    out["step5_kmatched_summary"] = {
        "n_pairings": len(grid_rho),
        "rho_all_min": round(min(allr), 4), "rho_all_max": round(max(allr), 4),
        "rho_all_mean": round(float(np.mean(allr)), 4),
        "rho_normal_min": round(min(normr), 4), "rho_normal_max": round(max(normr), 4),
        "rho_normal_mean": round(float(np.mean(normr)), 4)}

    # ---------------------------------------------------------------- 3. Insurance tiers
    print("\n" + "=" * 110)
    print("3. ARE INSURANCE'S THREE TIERS REALLY THREE? Full per-cluster feature means")
    print("=" * 110)
    ins_spec = primary["insurance"]
    ins_bd = BENCH_OF[ins_spec]
    skey = winners["insurance"]
    feats = FEATS_OF[ins_spec]
    r, kf, _ = assign_tiers(df, skey, cache[ins_spec]["labels"], ins_bd)
    # Clustering features first, then any benchmark variable not already among them;
    # dict.fromkeys de-duplicates while keeping that order.
    show = list(dict.fromkeys(feats + BENCH_VARS[ins_bd]))
    print(f"  {ins_spec} / {skey}, k={kf}")
    print(f"    {'tier':<7}{'share':>8}" + "".join(f"{c[:22]:>24}" for c in show))
    ins_tbl = {}
    for t in range(kf):
        m = r == t
        row = {c: round(float(df.loc[m, c].mean()), 4) for c in show}
        ins_tbl[t] = {"share_pct": round(100 * float(m.mean()), 2), "means": row}
        print(f"    T{t:<6}{100 * float(m.mean()):>7.1f}%"
              + "".join(f"{row[c]:>24.4f}" for c in show))
    # pairwise separation between adjacent tiers, in IQR units
    iqr = {c: float(df[c].quantile(0.75) - df[c].quantile(0.25)) for c in show}
    print("\n    Adjacent-tier separation in IQR units (|mean difference| / feature IQR):")
    seps = {}
    for t in range(kf - 1):
        d = {c: round(abs(ins_tbl[t]["means"][c] - ins_tbl[t + 1]["means"][c]) /
                      (iqr[c] or 1.0), 3) for c in show}
        seps[f"T{t}_vs_T{t + 1}"] = d
        print(f"    T{t} vs T{t + 1}: " + "  ".join(f"{c[:18]}={v}" for c, v in d.items()))
    out["insurance_tier_profile"] = ins_tbl
    out["insurance_adjacent_tier_separation_iqr"] = seps

    # And what the two lower tiers actually mean for a user: % clearing each benchmark
    print("\n    % of each tier clearing each insurance benchmark:")
    pct_clearing = {}
    for var in BENCH_VARS[ins_bd]:
        pcts = [round(100 * float(meets_mask(df[r == t], ins_bd, var).mean()), 1)
                for t in range(kf)]
        print(f"      {var:<28} {pcts}")
        pct_clearing[var] = pcts
    out["insurance_pct_clearing_by_tier"] = pct_clearing

    p = os.path.join(OUT_DIR, "domain_split_followup.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  saved {p}")


if __name__ == "__main__":
    main()
