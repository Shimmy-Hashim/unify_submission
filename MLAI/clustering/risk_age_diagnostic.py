"""
risk_age_diagnostic.py — is `risk_age_deviation`'s failure the FORMULA, the missing
ARCHETYPE, or both? (Unify, CCDS25-1170, July 2026)

WHY THIS EXISTS
  The portfolio_concentration recalibration (investment_recalibration.py) fixed what it
  targeted but did not validate the Investment domain. Two blockers remain; this script
  addresses the second one, which was identified but not yet diagnosed:
  `risk_age_deviation` is the binding (lowest-headroom) constraint in 48 of 60
  floor-surviving combinations, yet its distribution looks identical across every profile
  type. Two genuinely different causes were hypothesised, with different fixes:

    (a) THE FORMULA. The age-equity target ("100 minus age", standardized across the
        generator and scoring.py earlier this session for internal consistency but never
        re-examined for calibration) may simply not describe how this population invests,
        making deviation-from-target a measurement of the target's own displacement rather
        than of any user behaviour.
    (b) THE MISSING ARCHETYPE. No injected mismatch profile may create age-inappropriate
        risk-taking at all, in which case "no signal" persists under ANY formula, because
        nothing in the data design produces the variation the feature is meant to capture.

  Both can be true simultaneously. This script tests each independently, then runs the
  identical gate machinery under five variants so the fix is chosen on evidence.

  METHODOLOGY UNCHANGED from domain_split_clustering.py / investment_recalibration.py —
  same 8 algorithms, k=2-6, both approaches, 5% size floor, macro-silhouette + DB/CH
  rank-sum, mean-AND-density dual validation with ceiling-aware materiality, plus the
  normal-population 5% floor adopted project-wide. portfolio_concentration is held at its
  validated <= 0.33. Everything is IMPORTED; only the target formula and the
  feature/benchmark membership of risk_age_deviation change.

  VALIDATION ONLY. Writes nothing outside MLAI/clustering/results/. No production
  artifact and no generator file is modified; the dataset is NOT regenerated (the derived
  column is recomputed in memory from the CSV's own `equity_pct`, which is what makes the
  formula question answerable without regeneration).

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/risk_age_diagnostic.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import (  # noqa: E402
    DATA_PATH, OUT_DIR, K_BASED, K_RANGE, MIN_CLUSTER_SHARE, MONOTONIC_RHO_MIN,
    MIN_HEADROOM_CAPTURED, run_spec_grid, rank_survivors, usability, assign_tiers,
    groups_for, separation_profile, base_rates, meets_mask, health_signal,
    PROFILE_ORDER,
)
from investment_recalibration import CONC_THRESHOLD, set_concentration_benchmark  # noqa: E402
from generate_synthetic_data import (  # noqa: E402
    AGE_MIDPOINTS, EQUITY_TARGETS, MARGINALS, MISMATCH_TYPES, OUTLIER_EXTREMES,
    CORR_MODERATE, COPULA_FEATURES,
)

NORMAL_FLOOR = 0.05
RAD = "risk_age_deviation"
RAD_TOL = 0.10           # the benchmark's tolerance band, unchanged: |deviation| <= 10pp
INV_FEATS_FULL = ["investment_rate", "portfolio_concentration", RAD]
INV_FEATS_NORAD = ["investment_rate", "portfolio_concentration"]
CACHE = os.path.join(OUT_DIR, ".risk_age_diagnostic_grids.pkl")
OUT_JSON = os.path.join(OUT_DIR, "risk_age_diagnostic.json")

# Candidate target formulas. All are "BASE minus age" over the band midpoint except
# `population_median`, which anchors on the data itself and is therefore run as a
# DIAGNOSTIC BOUND only — see the note printed in Step 1. Fixed here, before any gate runs.
BASE_CANDIDATES = [85, 90, 95, 100, 105, 110, 120]
# Annotations for the two bases that are not merely hypothetical.
BASE_NOTES = {100: "   <- current", 110: "   <- pre-July-2026 scoring.py"}


def targets_from_base(base):
    return {b: min(1.0, max(0.0, (base - AGE_MIDPOINTS[b]) / 100.0)) for b in range(1, 6)}


def recompute_rad(df, targets):
    """Recompute the derived deviation column from the CSV's own equity_pct. No
    regeneration: the copula draws (equity_pct) are untouched, only the target subtracted
    from them changes."""
    out = df.copy()
    out[RAD] = out["equity_pct"] - out["age_band"].map(targets).to_numpy(float)
    return out


def eta_sq(df, feat, group="profile_type"):
    """Share of a feature's total variance explained by profile membership (one-way R^2).
    0 = the injected profiles create no variation in this feature at all."""
    v = df[feat].to_numpy(float)
    grand = v.mean()
    sst = float(((v - grand) ** 2).sum())
    ssb = float(sum(len(g) * (g[feat].mean() - grand) ** 2 for _, g in df.groupby(group)))
    return round(ssb / sst, 5) if sst > 0 else None


def kruskal_eps_sq(df, feat, group="profile_type"):
    groups = [g[feat].to_numpy(float) for _, g in df.groupby(group)]
    H = float(stats.kruskal(*groups).statistic)
    n, k = len(df), len(groups)
    return round(H, 1), round(max(0.0, (H - k + 1) / (n - k)), 5)


# ======================================================================================
# STEP 1 — is "100 minus age" calibrated to THIS population?
# ======================================================================================
def step1(df):
    print("=" * 118)
    print("STEP 1 — IS THE '100 MINUS AGE' TARGET CALIBRATED TO THIS POPULATION?")
    print("=" * 118)
    print("  The target is compared against the generator's `equity_pct` marginal, which is the")
    print("  ONLY thing risk_age_deviation is computed from (rad = equity_pct - target).")
    print("  `implied base` = what BASE would place the target exactly on that band's median")
    print("  equity holding, i.e. base = 100*median + age_midpoint.\n")

    nrm = df[df.profile_type == "normal"]
    print(f"  {'band':<6}{'age':>5}{'target':>9}{'gen mean':>10}{'p25':>8}{'median':>8}{'p75':>8}"
          f"{'gap(t-med)':>12}{'implied base':>14}{'% |dev|<=.10':>14}{'% above t':>11}")
    rows = {}
    for b in range(1, 6):
        s = nrm.loc[nrm.age_band == b, "equity_pct"]
        t = EQUITY_TARGETS[b]
        med = float(s.median())
        gen_mean_param = MARGINALS["equity_pct"][b][0]
        dev = s - t
        rows[b] = {
            "age_midpoint": AGE_MIDPOINTS[b], "target_100_minus_age": t,
            "generator_marginal_mean_param": gen_mean_param,
            "realized_mean": round(float(s.mean()), 4), "realized_median": round(med, 4),
            "p25": round(float(s.quantile(.25)), 4), "p75": round(float(s.quantile(.75)), 4),
            "gap_target_minus_median_pp": round(100 * (t - med), 1),
            "implied_base_from_median": round(100 * med + AGE_MIDPOINTS[b], 1),
            "pct_within_tolerance": round(100 * float((dev.abs() <= RAD_TOL).mean()), 1),
            "pct_above_target": round(100 * float((s > t).mean()), 1),
            "mean_signed_deviation": round(float(dev.mean()), 4),
        }
        r = rows[b]
        print(f"  {b:<6}{AGE_MIDPOINTS[b]:>5}{t:>9.2f}{r['realized_mean']:>10.3f}{r['p25']:>8.3f}"
              f"{r['realized_median']:>8.3f}{r['p75']:>8.3f}{r['gap_target_minus_median_pp']:>11.1f}pp"
              f"{r['implied_base_from_median']:>14.1f}{r['pct_within_tolerance']:>13.1f}%"
              f"{r['pct_above_target']:>10.1f}%")

    implied = [rows[b]["implied_base_from_median"] for b in range(1, 6)]
    print(f"\n  implied base by band: {implied}  -> spread {max(implied) - min(implied):.1f} "
          f"percentage points")
    # Slope check: the population's own age-equity glidepath vs the formula's fixed 10pp/decade.
    x = np.array([AGE_MIDPOINTS[b] for b in range(1, 6)], float)
    y = np.array([rows[b]["realized_median"] for b in range(1, 6)], float)
    slope, intercept = np.polyfit(x, y, 1)
    print(f"  population glidepath (OLS on band medians): equity = {intercept:.3f} "
          f"{slope:+.4f} * age   ->  {abs(slope) * 1000:.1f}pp per decade, "
          f"intercept implies a base of {100 * intercept:.1f}")
    print(f"  '100 minus age' glidepath:                  equity = 1.000 -0.0100 * age   "
          f"->  10.0pp per decade, base 100")
    print("  -> the formula is not merely displaced (as portfolio_concentration's ladder was);")
    print("     its SLOPE also disagrees with the population's, so no single 'BASE minus age'")
    print("     value can centre it in all five bands at once.")

    print(f"\n  Base sweep — % of NORMAL users inside the +/-{RAD_TOL:.2f} tolerance band, "
          f"and the resulting benchmark base rate:")
    print(f"    {'base':<8}" + "".join(f"{'band' + str(b):>9}" for b in range(1, 6))
          + f"{'ALL normals':>13}{'ALL users':>11}{'|mean dev|':>12}")
    sweep = {}
    for base in BASE_CANDIDATES:
        tg = targets_from_base(base)
        dn = recompute_rad(nrm, tg)[RAD]
        da = recompute_rad(df, tg)[RAD]
        per_band = [round(100 * float((dn[nrm.age_band == b].abs() <= RAD_TOL).mean()), 1)
                    for b in range(1, 6)]
        sweep[base] = {"targets": tg, "pct_within_by_band_normals": per_band,
                       "pct_within_all_normals": round(100 * float((dn.abs() <= RAD_TOL).mean()), 1),
                       "pct_within_all_users": round(100 * float((da.abs() <= RAD_TOL).mean()), 1),
                       "abs_mean_deviation_normals": round(float(abs(dn.mean())), 4)}
        s = sweep[base]
        mark = BASE_NOTES.get(base, "")
        print(f"    {base:<8}" + "".join(f"{v:>8.1f}%" for v in per_band)
              + f"{s['pct_within_all_normals']:>12.1f}%{s['pct_within_all_users']:>10.1f}%"
              + f"{s['abs_mean_deviation_normals']:>12.4f}{mark}")

    pop_t = {b: round(rows[b]["realized_median"], 4) for b in range(1, 6)}
    dn = recompute_rad(nrm, pop_t)[RAD]
    da = recompute_rad(df, pop_t)[RAD]
    pop_row = {"targets": pop_t,
               "pct_within_by_band_normals": [
                   round(100 * float((dn[nrm.age_band == b].abs() <= RAD_TOL).mean()), 1)
                   for b in range(1, 6)],
               "pct_within_all_normals": round(100 * float((dn.abs() <= RAD_TOL).mean()), 1),
               "pct_within_all_users": round(100 * float((da.abs() <= RAD_TOL).mean()), 1),
               "abs_mean_deviation_normals": round(float(abs(dn.mean())), 4)}
    print(f"    {'pop-med':<8}" + "".join(f"{v:>8.1f}%" for v in pop_row["pct_within_by_band_normals"])
          + f"{pop_row['pct_within_all_normals']:>12.1f}%{pop_row['pct_within_all_users']:>10.1f}%"
          + f"{pop_row['abs_mean_deviation_normals']:>12.4f}   <- per-band population median "
            f"(RELATIVE anchor: diagnostic bound only, see note)")
    print("\n  NOTE on the population-median row: anchoring the target on the sample's own")
    print("  median is a RELATIVE benchmark, which the brief's Score Derivation rule forbids")
    print("  for tier anchoring ('absolute MAS/LIA/SingStat benchmarks, not ranked within the")
    print("  age-filtered sample'). It is computed here only to bound what PERFECT centring")
    print("  could buy — not proposed as a fix. Same reasoning that rejected a percentile")
    print("  anchor for portfolio_concentration.")

    print("\n  Internal-documentation inconsistency found (worth recording):")
    print("    generate_synthetic_data.py FEATURE_SOURCES['equity_pct'] states the marginal is")
    print(f"    the \"Standard (100-age) equity rule\" — but its own mean parameters "
          f"({[MARGINALS['equity_pct'][b][0] for b in range(1, 6)]})")
    print(f"    sit below the 100-minus-age targets ({[EQUITY_TARGETS[b] for b in range(1, 6)]}) "
          f"in bands 1-4 and only")
    print("    coincide at band 5. The marginal was drawn on a shallower glidepath than the rule")
    print("    it cites, so the generator disagrees with itself, independently of scoring.py.")
    return {"per_band": rows, "base_sweep": sweep, "population_median_anchor": pop_row,
            "population_glidepath": {"slope_per_year": round(float(slope), 5),
                                     "pp_per_decade": round(float(abs(slope) * 1000), 1),
                                     "intercept": round(float(intercept), 4),
                                     "implied_base": round(float(100 * intercept), 1)},
            "implied_base_by_band": implied,
            "implied_base_spread_pp": round(max(implied) - min(implied), 1)}


# ======================================================================================
# STEP 2 — does ANY injected profile create age-inappropriate risk-taking?
# ======================================================================================
def step2(df):
    print("\n" + "=" * 118)
    print("STEP 2 — DOES ANY INJECTED PROFILE CREATE REAL VARIATION IN AGE-APPROPRIATE RISK?")
    print("=" * 118)
    print("  Read directly out of the generator's own override tables, not assumed.\n")

    print(f"  {'profile':<20}{'# overrides':>12}   overridden features")
    touch = {}
    for t, spec in MISMATCH_TYPES.items():
        feats = sorted(spec)
        touch[t] = {"overrides": feats,
                    "touches_equity_pct": "equity_pct" in feats,
                    "touches_risk_age_deviation": RAD in feats}
        print(f"  {t:<20}{len(feats):>12}   {', '.join(feats)}")
    print(f"\n  OUTLIER_EXTREMES features ({len(OUTLIER_EXTREMES)}): "
          f"{', '.join(sorted(OUTLIER_EXTREMES))}")
    eq_mismatch = [t for t, v in touch.items() if v["touches_equity_pct"]]
    rad_mismatch = [t for t, v in touch.items() if v["touches_risk_age_deviation"]]
    eq_outlier = "equity_pct" in OUTLIER_EXTREMES
    print(f"\n  profiles overriding `equity_pct`:          {eq_mismatch or 'NONE'}")
    print(f"  profiles overriding `risk_age_deviation`:  {rad_mismatch or 'NONE'}")
    print(f"  `equity_pct` in the outlier extremes:      {'YES' if eq_outlier else 'NO'}")
    print("  -> equity_pct is generated ONLY by the copula. No mismatch profile and not even the")
    print("     outlier mechanism can move it, so no user in this dataset is age-inappropriately")
    print("     invested by construction. Contrast: investment_rate is overridden by 3 of 6")
    print("     profiles AND is an outlier feature.")

    print(f"\n  Copula row for equity_pct (index {COPULA_FEATURES.index('equity_pct')}) — how "
          f"much structure it shares with anything else:")
    ei = COPULA_FEATURES.index("equity_pct")
    pairs = sorted(((abs(CORR_MODERATE[ei][j]), COPULA_FEATURES[j], CORR_MODERATE[ei][j])
                    for j in range(len(COPULA_FEATURES)) if j != ei), reverse=True)
    print("    " + "  ".join(f"{n}={v:+.2f}" for _, n, v in pairs))
    print(f"    max |rho| to any other variable: {pairs[0][0]:.2f} ({pairs[0][1]}) — the weakest "
          f"row in the matrix.")

    print("\n  Realized per-profile behaviour (the empirical counterpart of the above):")
    print(f"    {'profile':<20}{'n':>6}{'rad mean':>10}{'rad sd':>9}{'|rad| mean':>12}"
          f"{'% within .10':>14}{'equity mean':>13}{'inv_rate mean':>15}")
    prof = {}
    for p in PROFILE_ORDER:
        s = df[df.profile_type == p]
        if not len(s):
            continue
        prof[p] = {"n": int(len(s)), "rad_mean": round(float(s[RAD].mean()), 4),
                   "rad_sd": round(float(s[RAD].std()), 4),
                   "abs_rad_mean": round(float(s[RAD].abs().mean()), 4),
                   "pct_within_tolerance": round(100 * float((s[RAD].abs() <= RAD_TOL).mean()), 1),
                   "equity_pct_mean": round(float(s.equity_pct.mean()), 4),
                   "investment_rate_mean": round(float(s.investment_rate.mean()), 4),
                   "portfolio_concentration_mean": round(float(s.portfolio_concentration.mean()), 4)}
        r = prof[p]
        print(f"    {p:<20}{r['n']:>6}{r['rad_mean']:>10.4f}{r['rad_sd']:>9.4f}"
              f"{r['abs_rad_mean']:>12.4f}{r['pct_within_tolerance']:>13.1f}%"
              f"{r['equity_pct_mean']:>13.4f}{r['investment_rate_mean']:>15.4f}")

    print("\n  Variance attributable to profile membership (one-way R^2) — the direct test of "
          "'does the data design create variation here?':")
    var = {}
    for f in ["investment_rate", "portfolio_concentration", RAD, "equity_pct"]:
        H, eps = kruskal_eps_sq(df, f)
        var[f] = {"eta_squared": eta_sq(df, f), "kruskal_H": H, "epsilon_squared": eps}
        print(f"    {f:<28} eta^2={var[f]['eta_squared']:.5f}   Kruskal-Wallis H={H:>9.1f}   "
              f"epsilon^2={eps:.5f}")
    print("    -> investment_rate's variation is largely profile-driven; risk_age_deviation's is")
    print("       essentially not at all. Any non-zero value for rad is the mechanical knock-on of")
    print("       profile membership being mildly age-correlated, not of injected risk behaviour.")

    print("\n  Correlation of risk_age_deviation with the other two Investment features:")
    cors = {}
    for label, sub in (("all users", df), ("normals only", df[df.profile_type == "normal"])):
        cors[label] = {}
        for f in INV_FEATS_NORAD:
            pr = float(stats.pearsonr(sub[RAD], sub[f]).statistic)
            sp = float(stats.spearmanr(sub[RAD], sub[f]).statistic)
            cors[label][f] = {"pearson": round(pr, 4), "spearman": round(sp, 4)}
            print(f"    {label:<14} rad vs {f:<26} pearson={pr:+.3f}  spearman={sp:+.3f}")
        # The benchmark's actual health signal is -|rad|, not rad.
        for f in INV_FEATS_NORAD:
            pr = float(stats.pearsonr(-sub[RAD].abs(), sub[f]).statistic)
            cors[label][f"-|rad| vs {f}"] = round(pr, 4)
            print(f"    {label:<14} -|rad| vs {f:<24} pearson={pr:+.3f}")

    print("\n  Two-sidedness check — the benchmark is |rad| <= 0.10, so its health signal is")
    print("  -|rad|, which is NON-MONOTONE in the clustered feature (both tails are 'unhealthy').")
    print(f"    correlation between rad and its own health signal -|rad|: "
          f"{float(stats.pearsonr(df[RAD], -df[RAD].abs()).statistic):+.3f}")
    print("    Project precedent for exactly this: domain_split_clustering.py already excludes")
    print("    `insurance_spend_ratio` from Insurance's BENCHMARK list (while keeping it as a")
    print("    clustering feature) on the stated ground that it is 'non-monotonic as a health")
    print("    signal'. risk_age_deviation is the same category and was not given the same")
    print("    treatment.")
    return {"mismatch_overrides": touch,
            "profiles_overriding_equity_pct": eq_mismatch,
            "profiles_overriding_risk_age_deviation": rad_mismatch,
            "equity_pct_in_outlier_extremes": eq_outlier,
            "outlier_features": sorted(OUTLIER_EXTREMES),
            "equity_pct_copula_row": {n: round(float(v), 2) for _, n, v in pairs},
            "per_profile": prof, "variance_explained_by_profile": var,
            "correlations": cors,
            "rad_vs_health_signal_pearson": round(
                float(stats.pearsonr(df[RAD], -df[RAD].abs()).statistic), 4)}


# ======================================================================================
# STEPS 3+4 — variant grid: fix candidates run through the identical gate machinery
# ======================================================================================
def install_bench(include_rad):
    """Rebuild BENCHMARKS['investment'] (and the derived BENCH_VARS) in place."""
    rows = [("investment_rate", ">=", 0.10, "MoneySense/MAS 'invest at least 10% of income'", False),
            ("portfolio_concentration", "<=", float(CONC_THRESHOLD),
             "validated flat <=0.33 (investment_recalibration.py)", False)]
    if include_rad:
        rows.append((RAD, "abs<=", RAD_TOL,
                     "within 10pp of the age-equity target", False))
    rows.append(("has_investments", "==1", 1,
                 "OCBC FWI 2024 88% participation (STRICTLY out-of-model)", True))
    ds.BENCHMARKS["investment"] = rows
    ds.BENCH_VARS["investment"] = [r[0] for r in rows]


def fit_geometry(df, feats, spec_name):
    """Register a spec on domain_split_clustering and fit its full grid, unchanged."""
    ds.SPECS[spec_name] = ("investment", list(feats))
    ds.FEATS_OF[spec_name] = list(feats)
    ds.BENCH_OF[spec_name] = "investment"
    scaler = RobustScaler().fit(df[feats])
    age_scaler = StandardScaler().fit(df[["age_midpoint"]])
    X_full = np.hstack([scaler.transform(df[feats]), age_scaler.transform(df[["age_midpoint"]])])
    return run_spec_grid(df, spec_name, scaler, X_full)


def evaluate(df, grid, labels, include_rad):
    install_bench(include_rad)
    base = base_rates(df, "investment")
    survivors = {k: v for k, v in grid.items() if v.get("passes_floor")}
    ranks = rank_survivors(survivors)
    sweep = {k: usability(df, k, labels, "investment", base) for k in sorted(survivors)}
    gate = [k for k in sweep if sweep[k]["passes_gate2_headroom"]]
    normal = (df.profile_type == "normal").to_numpy()
    nrec = {}
    for k in sorted(survivors):
        r, kf, _ = assign_tiers(df, k, labels, "investment")
        sh = [float((r[normal] == t).mean()) for t in range(kf)]
        nrec[k] = {"k_found": kf, "shares_pct": [round(100 * s, 2) for s in sh],
                   "min_pct": round(100 * min(sh), 2), "passes": bool(min(sh) >= NORMAL_FLOOR)}
    both = [k for k in gate if nrec[k]["passes"]]
    pool = both or gate or sorted(survivors)
    winner = max(pool, key=lambda k: (grid[k]["silhouette_macro"],
                                      -ranks[k.split('|')[0]][k]["composite_rank_macro"]))
    binding = {v: 0 for v in ds.BENCH_VARS["investment"]}
    for u in sweep.values():
        if u["lift"]:
            binding[min(u["lift"], key=lambda v: u["lift"][v]["worst_headroom_captured"])] += 1
    return {"base": base, "survivors": survivors, "ranks": ranks, "sweep": sweep, "gate": gate,
            "normal": nrec, "both": both, "winner": winner, "binding": binding,
            "winner_selected_from": ("gate+normal-floor" if both else
                                     "gate only (none clears the normal floor)" if gate else
                                     "best available (no gate passer)")}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df0 = pd.read_csv(DATA_PATH)
    rad_drift = float((df0.equity_pct - df0.age_band.map(EQUITY_TARGETS) - df0[RAD]).abs().max())
    assert rad_drift < 1e-3, \
        "CSV's risk_age_deviation must equal equity_pct - EQUITY_TARGETS (100-minus-age)"
    print("=" * 118)
    print(f"risk_age_deviation DIAGNOSTIC — {len(df0)} users, "
          f"{int((df0.profile_type == 'normal').sum())} normal-profile")
    print(f"  portfolio_concentration held at its validated <= {CONC_THRESHOLD}; "
          f"tolerance band |rad| <= {RAD_TOL}")
    print("=" * 118)

    s1 = step1(df0)
    s2 = step2(df0)

    # ---------------------------------------------------------------- variants
    pop_t = {b: s1["per_band"][b]["realized_median"] for b in range(1, 6)}
    df_pop = recompute_rad(df0, pop_t)

    print("\n" + "=" * 118)
    print("STEP 3/4 — FIVE VARIANTS THROUGH THE IDENTICAL GATE (3 distinct geometries)")
    print("=" * 118)
    print("  V0  baseline           feats [ir, pc, rad(100-age)]   benchmarks incl. rad")
    print("  V1  formula fix only   feats [ir, pc, rad(pop-med)]   benchmarks incl. rad")
    print("  V2  bench-drop only    feats [ir, pc, rad(100-age)]   benchmarks EXCL. rad")
    print("  V3  drop entirely      feats [ir, pc]                 benchmarks EXCL. rad")
    print("  V4  both fixes         feats [ir, pc, rad(pop-med)]   benchmarks EXCL. rad")
    print("  V1/V4 exist to test hypothesis (a) in isolation: if the formula were the problem,")
    print("  re-centring it must change the outcome. V2/V3 test hypothesis (b)'s honest options.\n")

    geoms = {"rad100": (df0, INV_FEATS_FULL), "radPop": (df_pop, INV_FEATS_FULL),
             "noRad": (df0, INV_FEATS_NORAD)}
    cached = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            cached = pickle.load(f)
    fitted = {}
    for name, (d, feats) in geoms.items():
        spec = f"rad_diag_{name}"
        if name in cached:
            print(f"  [{name}] reusing cached grid ({len(cached[name][0])} combinations)")
            ds.SPECS[spec] = ("investment", list(feats))
            ds.FEATS_OF[spec] = list(feats)
            ds.BENCH_OF[spec] = "investment"
            fitted[name] = cached[name]
        else:
            print(f"  [{name}] fitting {len(K_BASED) * len(list(K_RANGE)) * 2} combinations on "
                  f"{feats} ...")
            fitted[name] = cached[name] = fit_geometry(d, feats, spec)
    with open(CACHE, "wb") as f:
        pickle.dump(cached, f)

    VARIANTS = {
        "V0_baseline":           ("rad100", True),
        "V1_formula_fix_only":   ("radPop", True),
        "V2_bench_drop_only":    ("rad100", False),
        "V3_drop_entirely":      ("noRad", False),
        "V4_both_fixes":         ("radPop", False),
    }
    res = {}
    for vname, (gname, inc) in VARIANTS.items():
        d = geoms[gname][0]
        g, l = fitted[gname]
        res[vname] = evaluate(d, g, l, inc)
        res[vname]["geometry"] = gname
        res[vname]["includes_rad_benchmark"] = inc
        res[vname]["features"] = geoms[gname][1]

    print(f"\n  {'variant':<24}{'geometry':<10}{'radBench':>9}{'surv':>6}{'gate':>6}"
          f"{'normOK':>8}{'BOTH':>6}   winner (min normal tier)")
    for v, r in res.items():
        w = r["winner"]
        print(f"  {v:<24}{r['geometry']:<10}{str(r['includes_rad_benchmark']):>9}"
              f"{len(r['survivors']):>6}{len(r['gate']):>6}"
              f"{sum(1 for x in r['normal'].values() if x['passes']):>8}{len(r['both']):>6}   "
              f"{w} ({r['normal'][w]['min_pct']:.2f}%)"
              f"{'' if r['normal'][w]['passes'] else '  <<< DEGENERATE ON NORMALS'}")

    print("\n  Binding (lowest-headroom) benchmark across each variant's survivors:")
    for v, r in res.items():
        print(f"    {v:<24}" + "  ".join(f"{k}={c}" for k, c in
                                         sorted(r["binding"].items(), key=lambda x: -x[1])))

    print("\n  Gate-passing combinations per variant:")
    for v, r in res.items():
        print(f"    {v:<24}{len(r['gate']):>3}  {r['gate']}")

    # ------------------------------------------------ the named near-miss candidate
    print(f"\n  The near-miss candidate B|fuzzy_cmeans|3, tracked across variants "
          f"(gate needs rho>={MONOTONIC_RHO_MIN}, headroom>={MIN_HEADROOM_CAPTURED}):")
    for v, r in res.items():
        k = "B|fuzzy_cmeans|3"
        if k not in r["sweep"]:
            print(f"    {v:<24} not a floor survivor")
            continue
        u, n = r["sweep"][k], r["normal"][k]
        print(f"    {v:<24} rho={u['worst_group_min_spearman_rho']:.3f}  "
              f"headroom={u['min_headroom_captured']:.3f}  lift={u['min_lift_pp']:>5.1f}pp  "
              f"gate={'PASS' if u['passes_gate2_headroom'] else 'fail'}  "
              f"minNormalTier={n['min_pct']:.2f}%  "
              f"{'both OK' if u['passes_gate2_headroom'] and n['passes'] else ''}")
        if u["lift"]:
            for var, d in u["lift"].items():
                print(f"        {var:<28} base {d['base_rate_pct']:>5.1f}% -> best tier "
                      f"{d['worst_headroom_best_tier_pct']:>5.1f}%  headroom "
                      f"{d['worst_headroom_captured']:.3f}"
                      f"{'' if d['material_headroom'] else '  FAIL'}")

    return df0, df_pop, geoms, fitted, res, s1, s2


def report_winner(df, key, grid, labels, r, feats, tag):
    """Full detail on one variant's winner: tiers, ordering on normals, archetypes."""
    print("\n" + "=" * 118)
    print(f"STEP 4 — WINNER DETAIL: {tag}  ->  {key}")
    print("=" * 118)
    g, u = grid[key], r["sweep"][key]
    print(f"  macro-sil {g['silhouette_macro']:.4f}  micro-sil {g['silhouette_micro']:.4f}  "
          f"DB {g['davies_bouldin']:.3f}  CH {g['calinski_harabasz']:.1f}  "
          f"min cluster {g['min_cluster_pct']:.2f}%  k={g['k']}")
    print(f"  gate: min rho {u['worst_group_min_spearman_rho']:.3f}  "
          f"min headroom {u['min_headroom_captured']:.3f}  min lift {u['min_lift_pp']:.1f}pp  "
          f"-> {'PASS' if u['passes_gate2_headroom'] else 'FAIL'}")
    print("\n  per-benchmark materiality:")
    for var, d in u["lift"].items():
        print(f"    {var:<28} base {d['base_rate_pct']:>5.1f}% -> best tier "
              f"{d['best_tier_pct_clearing']:>5.1f}%  (lift {d['lift_pp']:>+6.1f}pp, headroom "
              f"{d['worst_headroom_captured']:.3f})  "
              f"{'MATERIAL' if d['material_headroom'] else 'immaterial'}")

    iqr = {f: float(df[f].quantile(0.75) - df[f].quantile(0.25)) for f in feats}
    for gk, (sub, lab) in groups_for(df, key, labels).items():
        sp = separation_profile(sub, lab, feats, iqr)
        print(f"  single-axis check {gk}: {sp['n_features_separating_ge_0.5']}/{sp['n_features']} "
              f"features separating >=0.5 IQR; top={sp['top_feature']} "
              f"({100 * sp['top_feature_share_of_total_separation']:.0f}% of separation); "
              f"spreads {sp['spread_iqr_units']}")

    print("\n  tier profile (mean / % clearing), best -> worst:")
    for gkey, gg in u["per_group"].items():
        print(f"    {gkey} (n={gg['n_group']})  order {gg['health_order_best_to_worst']}  "
              f"min_rho={gg['min_spearman_rho']}")
        for c in gg["health_order_best_to_worst"]:
            e = gg["clusters"][c]
            bits = [f"{v[:20]}={e['variables'][v]['mean']:.3f}"
                    f"/{e['variables'][v]['pct_meeting_benchmark']:.0f}%"
                    for v in ds.BENCH_VARS["investment"]]
            print(f"      tier {e['health_rank']} (cluster {c}, n={e['n']}, {e['share_pct']}%):  "
                  + "  ".join(bits))

    rank, kf, _ = assign_tiers(df, key, labels, "investment")
    d = df.copy()
    d["tier_investment"] = rank
    nd = d[d.profile_type == "normal"]
    share = nd["tier_investment"].value_counts(normalize=True).sort_index()
    print(f"\n  NORMAL-POPULATION CHECK (n={len(nd)}): "
          + "  ".join(f"T{i}={100 * v:.2f}%" for i, v in share.items())
          + f"   min {100 * share.min():.2f}%  "
          + ("<<< DEGENERATE" if share.min() < NORMAL_FLOOR else "-- non-degenerate"))
    normal_order = {}
    for var in ds.BENCH_VARS["investment"]:
        sig = pd.Series(health_signal(nd, "investment", var), index=nd.index)
        means = sig.groupby(nd["tier_investment"]).mean().sort_index()
        pcts = pd.Series(meets_mask(nd, "investment", var), index=nd.index).groupby(
            nd["tier_investment"]).mean().sort_index()
        mrho = round(float(stats.spearmanr(means.to_numpy(), -means.index.to_numpy(float)).statistic), 3)
        drho = round(float(stats.spearmanr(pcts.to_numpy(), -pcts.index.to_numpy(float)).statistic), 3)
        normal_order[var] = {"mean_rho": mrho, "density_rho": drho,
                             "pct_clearing_by_tier": [round(100 * float(x), 1) for x in pcts]}
        print(f"    {var:<28} mean-rho={mrho:+.3f}  density-rho={drho:+.3f}   "
              f"%clearing best->worst: {normal_order[var]['pct_clearing_by_tier']}")

    print("\n  archetype placement (tier distribution per profile):")
    print(f"    {'profile':<20}{'n':>6}" + "".join(f"{'T' + str(t):>8}" for t in range(kf))
          + f"{'mean':>8}{'norm':>7}")
    prof = {}
    for p in PROFILE_ORDER:
        s = d[d.profile_type == p]
        if not len(s):
            continue
        vc = s["tier_investment"].value_counts(normalize=True)
        nm = float(s["tier_investment"].mean()) / max(1, kf - 1)
        prof[p] = {"n": int(len(s)),
                   "tier_distribution_pct": {int(t): round(100 * float(vc.get(t, 0.0)), 1)
                                             for t in range(kf)},
                   "mean_tier": round(float(s["tier_investment"].mean()), 2),
                   "normalised_mean_tier": round(nm, 3)}
        print(f"    {p:<20}{len(s):>6}"
              + "".join(f"{100 * float(vc.get(t, 0.0)):>7.1f}%" for t in range(kf))
              + f"{s['tier_investment'].mean():>8.2f}{nm:>7.3f}")

    comp = {}
    for t in range(kf):
        sub = d[d.tier_investment == t]
        vc = sub.profile_type.value_counts()
        comp[t] = {"n": int(len(sub)),
                   "pct_injected": round(100 * float((sub.profile_type != "normal").mean()), 1),
                   "top": {p: int(c) for p, c in vc.head(4).items()}}
        print(f"    tier {t}: n={len(sub)}, {comp[t]['pct_injected']}% injected/outlier, "
              f"top profiles {comp[t]['top']}")
    return {"combination": key, "metrics": {m: g[m] for m in
                                            ("silhouette_macro", "silhouette_micro",
                                             "davies_bouldin", "calinski_harabasz",
                                             "min_cluster_pct", "k")},
            "gate": {kk: u[kk] for kk in ("worst_group_min_spearman_rho", "min_headroom_captured",
                                          "min_lift_pp", "passes_monotonicity_gate",
                                          "passes_materiality_gate_headroom",
                                          "passes_gate2_headroom")},
            "lift": u["lift"], "per_group": u["per_group"],
            "normal_tier_share_pct": {int(i): round(100 * float(v), 2) for i, v in share.items()},
            "normal_min_tier_share_pct": round(100 * float(share.min()), 2),
            "normal_non_degenerate": bool(share.min() >= NORMAL_FLOOR),
            "normal_benchmark_ordering": normal_order,
            "archetype_placement": prof, "tier_composition": comp}


if __name__ == "__main__":
    df0, df_pop, geoms, fitted, res, s1, s2 = main()

    # Full detail on the chosen fix (V2 — reasoning in the report) plus the baseline and
    # the formula-only variant, so the comparison is visible rather than asserted.
    detail = {}
    for v in ("V0_baseline", "V1_formula_fix_only", "V2_bench_drop_only", "V3_drop_entirely"):
        r = res[v]
        gname = r["geometry"]
        d = geoms[gname][0]
        g, l = fitted[gname]
        install_bench(r["includes_rad_benchmark"])
        detail[v] = report_winner(d, r["winner"], g, l, r, geoms[gname][1], v)

    out = {
        "purpose": "Diagnose whether risk_age_deviation's lack of signal is the 100-minus-age "
                   "FORMULA, the missing age-inappropriate-risk ARCHETYPE, or both. Validation "
                   "only — no regeneration, no production artifact, no generator edit.",
        "config": {"n_users": int(len(df0)), "tolerance_band": RAD_TOL,
                   "portfolio_concentration_benchmark": CONC_THRESHOLD,
                   "normal_population_floor": NORMAL_FLOOR,
                   "monotonic_rho_min": MONOTONIC_RHO_MIN,
                   "min_headroom_captured": MIN_HEADROOM_CAPTURED},
        "step1_formula_calibration": s1,
        "step2_archetype_check": s2,
        "step3_4_variants": {
            v: {"geometry": r["geometry"], "features": r["features"],
                "includes_rad_benchmark": r["includes_rad_benchmark"],
                "n_survivors": len(r["survivors"]), "n_gate_passers": len(r["gate"]),
                "gate_passers": r["gate"],
                "n_normal_non_degenerate": sum(1 for x in r["normal"].values() if x["passes"]),
                "n_passing_both": len(r["both"]), "passing_both": r["both"],
                "winner": r["winner"], "winner_selected_from": r["winner_selected_from"],
                "binding_benchmark_counts": r["binding"],
                "base_rates": r["base"]["overall"],
                "per_combination": {k: {"macro_sil": r["survivors"][k]["silhouette_macro"],
                                        "min_rho": r["sweep"][k]["worst_group_min_spearman_rho"],
                                        "min_headroom": r["sweep"][k]["min_headroom_captured"],
                                        "passes_gate": r["sweep"][k]["passes_gate2_headroom"],
                                        "min_normal_tier_pct": r["normal"][k]["min_pct"],
                                        "passes_normal_floor": r["normal"][k]["passes"]}
                                    for k in sorted(r["survivors"])}}
            for v, r in res.items()},
        "winner_detail": detail,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    rows = []
    for v, r in res.items():
        for k in sorted(r["survivors"]):
            rows.append({"variant": v, "combination": k,
                         "macro_sil": r["survivors"][k]["silhouette_macro"],
                         "min_cluster_pct": r["survivors"][k]["min_cluster_pct"],
                         "min_rho": r["sweep"][k]["worst_group_min_spearman_rho"],
                         "min_headroom": r["sweep"][k]["min_headroom_captured"],
                         "min_lift_pp": r["sweep"][k]["min_lift_pp"],
                         "passes_gate": r["sweep"][k]["passes_gate2_headroom"],
                         "min_normal_tier_pct": r["normal"][k]["min_pct"],
                         "passes_normal_floor": r["normal"][k]["passes"],
                         "is_winner": k == r["winner"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "risk_age_diagnostic_grid.csv"), index=False)
    print(f"\n  saved {OUT_JSON}")
    print(f"  saved {os.path.join(OUT_DIR, 'risk_age_diagnostic_grid.csv')}")
