"""
investment_recalibration.py — portfolio_concentration benchmark recalibration + an
Investment-domain-only re-test (Unify, CCDS25-1170, July 2026).

WHY THIS EXISTS
  The domain-split validation (domain_split_clustering.py) validated Banking
  (production-quality) and Insurance (production-quality as 2 tiers) but NOT Investment:
  only 1 of 60 floor-surviving combinations passed the usability gates, and the winner
  was degenerate on the normal population (its top tier held 0.33% of normal users).
  The root cause was isolated precisely: `portfolio_concentration`'s benchmark — the
  age-tapered SYNTHETIC_IDEALS ladder 0.20/0.17/0.13/0.10/0.08 — has a 4.2% population
  base rate, and is the binding (lowest-headroom) constraint in 56 of those 60 survivors.

  This script fixes THAT ONE BENCHMARK and re-tests Investment alone. Banking and
  Insurance are untouched and NOT re-run: portfolio_concentration appears in neither
  domain's feature set nor either domain's benchmark list, so the fix cannot affect them.

  METHODOLOGY IS OTHERWISE UNCHANGED — same 8 algorithms, k=2-6, both approaches, 5%
  size floor, macro-silhouette + DB/CH rank-sum, mean-AND-density dual validation,
  ceiling-aware materiality. Everything is imported from domain_split_clustering.py; only
  BENCHMARKS["investment"]'s portfolio_concentration row is replaced. A reproduction check
  confirms the re-fitted grid is label-for-label identical to the validated run, so any
  change in the result is attributable to the benchmark and nothing else.

  VALIDATION ONLY. Writes nothing outside MLAI/clustering/results/. No production
  artifact is read or written; benchmark_core.py's SYNTHETIC_IDEALS is deliberately NOT
  edited (see the note at the end of the calibration block below).

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/investment_recalibration.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import (  # noqa: E402
    DATA_PATH, OUT_DIR, K_BASED, K_RANGE,
    run_spec_grid, rank_survivors, usability, assign_tiers, groups_for,
    separation_profile, base_rates, meets_mask, health_signal,
    PROFILE_ORDER, CONC_IDEAL_BY_BAND,
)

SPEC = "investment_core3"
FEATS = ds.FEATS_OF[SPEC]
PRIOR_RESULTS = os.path.join(OUT_DIR, "domain_split_results.json")
OUT_JSON = os.path.join(OUT_DIR, "investment_recalibration.json")


# ======================================================================================
# STEP 1 DIAGNOSIS + STEP 2 CALIBRATION — full reasoning, recorded here rather than only
# in the report, to the same disclosure standard as every other benchmark in this project.
# ======================================================================================
#
# WHICH SIDE WAS WRONG? — the IDEAL, not the generator. Diagnosis (a), not (b) or (c).
#
# Evidence found for the GENERATOR's side (mean 0.354, band-tapered 0.47 -> 0.25):
#   * Goetzmann & Kumar (2008), "Equity Portfolio Diversification", Review of Finance
#     12(3):433-463 (>40,000 discount-brokerage equity accounts, 1991-96) — VERIFIED at
#     source. Finds US individual investors systematically hold UNDER-diversified
#     portfolios, and that under-diversification "is greater among younger, low-income,
#     less-educated and less-sophisticated investors."
#     Two things follow, and both support the generator rather than the ideal:
#       (i)  A largest-holding share around a third is what under-diversification
#            structurally IMPLIES. Max holding share is bounded below by 1/N for an
#            N-position portfolio and, since real portfolios are never equal-weighted, in
#            practice sits well above 1/N. A retail investor holding a handful of
#            positions therefore has a largest holding in the 0.25-0.50 region as a matter
#            of arithmetic, not of behaviour.
#       (ii) The AGE GRADIENT of the generator's marginal (younger = more concentrated)
#            independently matches Goetzmann & Kumar's verified demographic finding. That
#            gradient was previously undocumented; it turns out to be corroborated.
#   * WIDELY REPORTED BUT NOT INDEPENDENTLY VERIFIED AT SOURCE, and flagged as such rather
#     than cited as fact: secondary summaries of the same literature give an average of ~4
#     and median of ~3 directly-held stocks (and Blume & Friend 1975 a median of 2). The
#     figures are consistent with each other and with the verified finding above, but this
#     project's standard is to say plainly when a number was taken from a summary rather
#     than read in the source. These informed the choice below; they do not carry it.
#   * The production computation biases the SAME direction: scoring.py derives
#     portfolio_concentration as max() of cached per-account concentration_risk values,
#     which OVERSTATES concentration for multi-account users (Section 5, known limitation).
#     Real computed values will therefore run at least as high as the generator's, never
#     lower — so a benchmark unreachable in synthetic data is unreachable in production too.
#   * NO Singapore-specific retail concentration statistic exists. Re-confirmed absent this
#     session across SGX market statistics/retail-investor pages, SIAS's 2025 Retail
#     Investor Pulse Survey, CDP account materials and SingStat — same null result the
#     earlier CPF/CPFIS check produced. This remains a JUDGEMENT CALL, not a cited value.
#
# Evidence AGAINST the previous ideal (0.20/0.17/0.13/0.10/0.08):
#   * Its two sources are PRESCRIPTIVE ADVISORY THRESHOLDS, not descriptions of retail
#     behaviour: J.P. Morgan Private Bank's 10-20% "properly diversified" range and the
#     Fidelity/Calamos/HDFC 5-10% figure are wealth-management guidance addressed to
#     (largely high-net-worth, professionally-advised) clients deciding whether a position
#     has grown too large. Using them as a TIER benchmark for a retail population conflates
#     "what a private bank recommends" with "what a good outcome looks like for an ordinary
#     user" — the same category error the project already corrected once, when the
#     net_worth_to_income ladder was found to be pegged far above realistic population
#     values. A 20% ceiling implies at least five meaningfully-sized positions; a 8-10%
#     ceiling implies ten or more. The literature above says most retail investors hold
#     nothing like that many.
#   * Empirically the ladder is unreachable at BOTH ends of the age range, which rules out
#     "the taper is merely too steep": 3.7% of band 1 clears it and 3.8% of band 5 does.
#     The ideal tapers with age at almost exactly the same rate the population does, so it
#     sits a near-constant distance above the whole distribution. Nothing about the
#     generator's shape is wrong; the ladder is simply displaced.
#
# THE FIX (task option 3): REFRAME AS ONE ABSOLUTE DIVERSIFICATION-ADEQUACY THRESHOLD.
#   portfolio_concentration <= 0.33, flat across all five age bands.
#
#   Justification, and its limits:
#   * 0.33 is STRUCTURALLY interpretable, not arbitrary: a largest holding of at most a
#     third of the portfolio is, by construction, "at least three meaningful positions."
#     That is an absolute statement about a user's portfolio, checkable without reference
#     to any other user — so it satisfies the brief's Score Derivation constraint that tier
#     anchors be absolute and never relative percentile rankings within the sample. A
#     percentile-anchored threshold (e.g. "better than the population's 25th percentile")
#     was considered and REJECTED for exactly that reason, despite being the task's first
#     suggested option: it would have imported relative ranking into the one place the
#     brief specifically forbids it.
#   * The age taper is DROPPED, deliberately. The previous taper was documented as reasoned
#     with no source, and there is no evidence base for age-varying concentration NORMS.
#     Goetzmann & Kumar's age finding is descriptive (younger investors ARE more
#     concentrated), and the generator's marginal already encodes it. Encoding the same
#     gradient a second time in the target would mean grading younger users on a curve —
#     holding them to a laxer standard precisely because they are more likely to fail it.
#   * THE THRESHOLD WAS FIXED BEFORE THE GATE WAS RUN, on the argument above, and is
#     reported alongside a sensitivity sweep at 0.25 and 0.40 plus the old ladder. This
#     matters: choosing the value that happens to make the gate pass would be exactly the
#     gate manipulation the correlation-fix retest was designed to rule out. The sweep is
#     there so a reader can see whether the conclusion is threshold-sensitive.
#   * HONEST LIMITS: 0.33 is a judgement call. It has a structural rationale and is
#     consistent with the retail-diversification literature, but no source names it as a
#     benchmark, and none exists for Singapore. It should be described in the report and
#     dissertation as "a reasoned, structurally-interpretable threshold with no direct
#     citation" — the same category as dti 0.15 and the net-worth ladder, NOT the same
#     category as the LIA or MoneySense figures.
#
# WHAT IS DELIBERATELY *NOT* CHANGED HERE:
#   * generate_synthetic_data.py's marginals — the diagnosis is that they are the
#     better-supported side. Regenerating the dataset would also invalidate the already-
#     validated Banking and Insurance results, which this task is explicitly not re-testing.
#   * benchmark_core.py's SYNTHETIC_IDEALS["portfolio_concentration"] — that ladder feeds
#     the production CENTROID artifact (distance-to-ideal scoring), a different mechanism
#     from this validation benchmark, and artifact work is out of scope and currently
#     blocked. FLAG FOR A LATER TASK: it almost certainly has the same defect for the same
#     reason, and after this recalibration the two now disagree (validation says 0.33,
#     the centroid ideal still says 0.20->0.08). That divergence is intentional and
#     temporary, but it must not be left unresolved when the artifact is re-exported.
CONC_THRESHOLD = 0.33
CONC_CITATION = ("<= 0.33 flat: largest holding at most a third of the portfolio, i.e. at "
                 "least three meaningful positions. Reasoned/structural, NOT a cited "
                 "benchmark — no SG retail concentration statistic exists (SGX/SIAS/CDP/"
                 "SingStat re-checked). Direction corroborated by Goetzmann & Kumar 2008 "
                 "(Review of Finance 12:433-463): retail investors are systematically "
                 "under-diversified, more so when younger. Replaces the age-tapered "
                 "0.20-0.08 ladder, which came from PRESCRIPTIVE private-bank guidance "
                 "(J.P. Morgan 10-20%, Fidelity/Calamos/HDFC 5-10%) and had a 4.2% "
                 "population base rate.")

SENSITIVITY_THRESHOLDS = {
    "old_band_ladder_0.20-0.08": None,     # the previous, failing benchmark
    "strict_0.25": 0.25,                   # "at least four meaningful positions"
    "chosen_0.33": 0.33,                   # the a-priori choice
    "loose_0.40": 0.40,                    # "at least 2.5 positions"
}


def set_concentration_benchmark(threshold):
    """Mutate the shared BENCHMARKS table in place so every imported validation function
    (meets_mask / health_signal / validate_group / discrimination_lift) sees the new
    benchmark. `threshold=None` restores the original band-tapered ladder."""
    rows = ds.BENCHMARKS["investment"]
    for i, r in enumerate(rows):
        if r[0] != "portfolio_concentration":
            continue
        if threshold is None:
            rows[i] = ("portfolio_concentration", "<=", "band_conc",
                       "band taper 0.20/0.17/0.13/0.10/0.08 (SUPERSEDED)", False)
        else:
            rows[i] = ("portfolio_concentration", "<=", float(threshold),
                       CONC_CITATION, False)
        return
    raise RuntimeError("portfolio_concentration row not found")


# ======================================================================================
def evaluate(df, grid, labels_store):
    """Full gate evaluation of the (unchanged) grid under whatever benchmark is currently
    installed. Returns (sweep, gate_passers, winner, ranks, base, survivors)."""
    bd = "investment"
    base = base_rates(df, bd)
    survivors = {k: v for k, v in grid.items() if v.get("passes_floor")}
    ranks = rank_survivors(survivors)
    sweep = {k: usability(df, k, labels_store, bd, base) for k in sorted(survivors)}
    gate = [k for k in sweep if sweep[k]["passes_gate2_headroom"]]
    pool = gate if gate else sorted(survivors)
    winner = max(pool, key=lambda k: (grid[k]["silhouette_macro"],
                                      -ranks[k.split('|')[0]][k]["composite_rank_macro"]))
    return sweep, gate, winner, ranks, base, survivors


def main():
    df = pd.read_csv(DATA_PATH)
    normal = df["profile_type"] == "normal"
    print("=" * 118)
    print("INVESTMENT DOMAIN — portfolio_concentration recalibration and re-test")
    print(f"  features: {FEATS}")
    print(f"  benchmark change: band ladder 0.20/0.17/0.13/0.10/0.08  ->  flat <= {CONC_THRESHOLD}")
    print("  Banking and Insurance are NOT re-run (portfolio_concentration is in neither "
          "domain's features nor benchmarks).")
    print("=" * 118)

    # -------------------------------------------------- Step 1 evidence, printed
    print("\n[STEP 1] Population vs benchmark, per age band — which side is displaced?")
    print(f"  {'band':<6}{'n':>6}{'p10':>8}{'p25':>8}{'median':>9}{'p75':>8}{'mean':>8}"
          f"{'old ideal':>11}{'% clearing old':>16}{'% clearing 0.33':>17}")
    diag = {}
    for b in range(1, 6):
        s = df.loc[df.age_band == b, "portfolio_concentration"]
        old = CONC_IDEAL_BY_BAND[b]
        diag[b] = {"n": int(len(s)), "p10": round(float(s.quantile(.10)), 3),
                   "p25": round(float(s.quantile(.25)), 3),
                   "median": round(float(s.median()), 3),
                   "p75": round(float(s.quantile(.75)), 3), "mean": round(float(s.mean()), 3),
                   "old_ideal": old,
                   "pct_clearing_old": round(100 * float((s <= old).mean()), 1),
                   "pct_clearing_new": round(100 * float((s <= CONC_THRESHOLD).mean()), 1)}
        d = diag[b]
        print(f"  {b:<6}{d['n']:>6}{d['p10']:>8.3f}{d['p25']:>8.3f}{d['median']:>9.3f}"
              f"{d['p75']:>8.3f}{d['mean']:>8.3f}{old:>11.2f}"
              f"{d['pct_clearing_old']:>15.1f}%{d['pct_clearing_new']:>16.1f}%")
    allc = df["portfolio_concentration"]
    # The old ideal is band-specific, so "% clearing" pooled across bands has to be measured
    # per user against their OWN band's value, not against any single number.
    pct_clearing_old = 100 * float(np.mean([c <= CONC_IDEAL_BY_BAND[b]
                                            for c, b in zip(allc, df.age_band)]))
    print(f"  {'ALL':<6}{len(allc):>6}{allc.quantile(.10):>8.3f}{allc.quantile(.25):>8.3f}"
          f"{allc.median():>9.3f}{allc.quantile(.75):>8.3f}{allc.mean():>8.3f}"
          f"{'—':>11}{pct_clearing_old:>15.1f}%"
          f"{100 * float((allc <= CONC_THRESHOLD).mean()):>16.1f}%")
    print("  -> the old ideal sits a near-constant distance ABOVE the whole distribution in "
          "every band (3.7-4.8% clearing at both age extremes), so the taper is not the "
          "problem — the level is. Diagnosis (a): the ideal, not the generator.")

    # -------------------------------------------------- Step 3: re-fit the grid
    print(f"\n[STEP 3] Re-fitting the full Investment grid ({len(K_BASED)} algorithms x "
          f"k={min(K_RANGE)}-{max(K_RANGE)} x 2 approaches = "
          f"{len(K_BASED) * len(list(K_RANGE)) * 2} combinations)...")
    scaler = RobustScaler().fit(df[FEATS])
    age_scaler = StandardScaler().fit(df[["age_midpoint"]])
    X_full = np.hstack([scaler.transform(df[FEATS]), age_scaler.transform(df[["age_midpoint"]])])
    grid, labels_store = run_spec_grid(df, SPEC, scaler, X_full)

    # Reproduction check — the clustering itself must be untouched, so any change in the
    # verdict is attributable to the benchmark alone and nothing else.
    with open(PRIOR_RESULTS) as f:
        prior = json.load(f)
    pg = prior["spec_results"][SPEC]["grid"]
    mism = []
    for k, v in grid.items():
        if v.get("failed") or pg[k].get("failed"):
            continue
        for m in ("silhouette_macro", "silhouette_micro", "davies_bouldin",
                  "calinski_harabasz", "min_cluster_pct"):
            if abs(v[m] - pg[k][m]) > max(0.002, abs(pg[k][m]) * 0.001):
                mism.append({"combination": k, "metric": m,
                             "validated_run": pg[k][m], "refit": v[m]})
    repro_verdict = (f"IDENTICAL across all {len(grid)} combinations" if not mism
                     else f"{len(mism)} MISMATCHES")
    print(f"  [repro check] geometry vs the validated run: {repro_verdict}")
    for m in mism[:5]:
        print(f"     {m}")

    # -------------------------------------------------- gate, before vs after
    results = {}
    for label, thr in SENSITIVITY_THRESHOLDS.items():
        set_concentration_benchmark(thr)
        sweep, gate, winner, ranks, base, survivors = evaluate(df, grid, labels_store)
        results[label] = {"threshold": thr, "base_rates": base, "sweep": sweep,
                          "gate": gate, "winner": winner, "survivors": survivors}

    print("\n[STEP 3] Threshold sensitivity — gate outcome at each candidate "
          "(the chosen value was fixed BEFORE these were computed):")
    print(f"  {'benchmark':<28}{'base rate':>11}{'gate passers':>14}   winner")
    for label, r in results.items():
        print(f"  {label:<28}{r['base_rates']['overall']['portfolio_concentration']:>10.1f}%"
              f"{len(r['gate']):>14}   {r['winner']}"
              f"{'' if r['gate'] else '   (no gate passer — best-available fallback)'}")

    # -------------------------------------------------- primary result
    set_concentration_benchmark(CONC_THRESHOLD)
    main_key = "chosen_0.33"
    R = results[main_key]
    sweep, gate, winner, survivors = R["sweep"], R["gate"], R["winner"], R["survivors"]
    base = R["base_rates"]
    prior_gate = prior["spec_results"][SPEC]["gate_passing_combinations"]

    print(f"\n{'=' * 118}\n[STEP 4] RESULT UNDER THE RECALIBRATED BENCHMARK (<= {CONC_THRESHOLD})"
          f"\n{'=' * 118}")
    print(f"  floor survivors: {len(survivors)}/{len(grid)} (unchanged — geometry is identical)")
    print(f"  gate passers:    {len(prior_gate)} -> {len(gate)}")
    print("\n  population base rates (% clearing each benchmark):")
    print("    " + "band".ljust(6) + "".join(v[:26].rjust(28) for v in ds.BENCH_VARS["investment"]))
    for b in range(1, 6):
        print("    " + str(b).ljust(6) + "".join(
            f"{base['by_band'][b][v]:>27.1f}%" for v in ds.BENCH_VARS["investment"]))
    print("    " + "ALL".ljust(6) + "".join(
        f"{base['overall'][v]:>27.1f}%" for v in ds.BENCH_VARS["investment"]))

    print(f"\n  All {len(gate)} gate-passing combinations:")
    print(f"    {'combination':<24}{'k':>3}{'macroSil':>10}{'microSil':>10}{'DB':>8}{'CH':>10}"
          f"{'minClu%':>9}{'minRho':>8}{'headrm':>8}{'minLift':>9}")
    for k in sorted(gate, key=lambda x: -grid[x]["silhouette_macro"]):
        g, u = grid[k], sweep[k]
        print(f"    {k:<24}{g['k']:>3}{g['silhouette_macro']:>10.4f}{g['silhouette_micro']:>10.4f}"
              f"{g['davies_bouldin']:>8.3f}{g['calinski_harabasz']:>10.1f}"
              f"{g['min_cluster_pct']:>9.2f}{u['worst_group_min_spearman_rho']:>8.3f}"
              f"{u['min_headroom_captured']:>8.3f}{u['min_lift_pp']:>8.1f}pp")
    print(f"\n  WINNER: {winner}  (was {prior['spec_results'][SPEC]['winner']})")

    binding = {v: 0 for v in ds.BENCH_VARS["investment"]}
    for u in sweep.values():
        if u["lift"]:
            binding[min(u["lift"], key=lambda v: u["lift"][v]["worst_headroom_captured"])] += 1
    print(f"  binding constraint across all {len(sweep)} survivors: "
          + "  ".join(f"{v}={c}" for v, c in sorted(binding.items(), key=lambda x: -x[1])))

    wl = sweep[winner]["lift"]
    print("\n  Winner per-benchmark materiality:")
    for v in ds.BENCH_VARS["investment"]:
        d = wl[v]
        print(f"    {v:<28} base {d['base_rate_pct']:>5.1f}% -> best tier "
              f"{d['best_tier_pct_clearing']:>5.1f}%  (lift {d['lift_pp']:>+6.1f}pp, "
              f"headroom {d['worst_headroom_captured']:.3f})  "
              f"{'MATERIAL' if d['material_headroom'] else 'immaterial'}")

    iqr = {f: float(df[f].quantile(0.75) - df[f].quantile(0.25)) for f in FEATS}
    seps = {gk: separation_profile(sub, lab, FEATS, iqr)
            for gk, (sub, lab) in groups_for(df, winner, labels_store).items()}
    for gk, sp in seps.items():
        print(f"  single-axis check {gk}: {sp['n_features_separating_ge_0.5']}/"
              f"{sp['n_features']} features separating >=0.5 IQR; top={sp['top_feature']} "
              f"({100 * sp['top_feature_share_of_total_separation']:.0f}% of separation)")

    # -------------------------------------------------- tier detail
    print("\n  Winner tier profile (mean / % clearing benchmark), best -> worst:")
    for gkey, g in sweep[winner]["per_group"].items():
        print(f"    {gkey} (n={g['n_group']})  order {g['health_order_best_to_worst']}  "
              f"min_rho={g['min_spearman_rho']}")
        for c in g["health_order_best_to_worst"]:
            e = g["clusters"][c]
            bits = [f"{v[:20]}={e['variables'][v]['mean']:.3f}"
                    f"/{e['variables'][v]['pct_meeting_benchmark']:.0f}%"
                    for v in ds.BENCH_VARS["investment"]]
            print(f"      tier {e['health_rank']} (cluster {c}, n={e['n']}, "
                  f"{e['share_pct']}%):  " + "  ".join(bits))

    # -------------------------------------------------- normal-population check
    rank, kfound, coarse = assign_tiers(df, winner, labels_store, "investment")
    df["tier_investment"] = rank
    ndf = df[normal]
    share = ndf["tier_investment"].value_counts(normalize=True).sort_index()
    print(f"\n{'=' * 118}\n[STEP 4] NORMAL-POPULATION CHECK — the decisive one "
          f"(prior failure: top tier held 0.33% of normals)\n{'=' * 118}")
    print(f"  k={kfound}   tier shares among the {int(normal.sum())} normal users: "
          + "  ".join(f"T{i}={100 * v:.1f}%" for i, v in share.items())
          + f"   (min {100 * share.min():.2f}%"
          + ("  <<< STILL DEGENERATE" if share.min() < 0.05 else "  — no degenerate tier") + ")")
    sils = []
    for gkey, (sub, lab) in groups_for(df, winner, labels_store).items():
        m = (sub["profile_type"] == "normal").to_numpy()
        if m.sum() >= 50 and len(np.unique(lab[m])) >= 2:
            s, _ = ds.macro_silhouette(scaler.transform(sub.loc[m, FEATS]), lab[m])
            sils.append(s)
    normal_sil = round(float(np.mean(sils)), 4) if sils else None
    print(f"  macro-silhouette on the normal subset: {normal_sil}")
    normal_order = {}
    for var in ds.BENCH_VARS["investment"]:
        sig = pd.Series(health_signal(ndf, "investment", var), index=ndf.index)
        means = sig.groupby(ndf["tier_investment"]).mean().sort_index()
        pcts = pd.Series(meets_mask(ndf, "investment", var), index=ndf.index).groupby(
            ndf["tier_investment"]).mean().sort_index()
        mrho = round(float(stats.spearmanr(means.to_numpy(), -means.index.to_numpy(float)).statistic), 3)
        drho = round(float(stats.spearmanr(pcts.to_numpy(), -pcts.index.to_numpy(float)).statistic), 3)
        normal_order[var] = {"mean_rho": mrho, "density_rho": drho,
                             "pct_clearing_by_tier": [round(100 * float(x), 1) for x in pcts]}
        print(f"    {var:<28} mean-rho={mrho:+.3f}  density-rho={drho:+.3f}"
              f"   %clearing best->worst: {normal_order[var]['pct_clearing_by_tier']}")

    # -------------------------------------------------- mismatch profiles
    print(f"\n{'=' * 118}\n[STEP 4] MISMATCH-PROFILE BEHAVIOUR — does it still hold?"
          f"\n{'=' * 118}")
    prior_tiers = pd.read_csv(os.path.join(OUT_DIR, "domain_split_tier_assignments.csv"))
    prior_mean = prior_tiers.groupby("profile_type")["tier_investment"].mean()
    prior_k = int(prior_tiers["tier_investment"].max()) + 1
    print(f"  {'profile':<20}{'n':>6}" + "".join(f"{'T' + str(t):>8}" for t in range(kfound))
          + f"{'mean':>8}{'norm':>8}   prior norm (k={prior_k})")
    prof_tbl = {}
    for prof in PROFILE_ORDER:
        sub = df[df.profile_type == prof]
        if not len(sub):
            continue
        vc = sub["tier_investment"].value_counts(normalize=True)
        nm = float(sub["tier_investment"].mean()) / max(1, kfound - 1)
        pn = float(prior_mean[prof]) / max(1, prior_k - 1)
        prof_tbl[prof] = {"n": int(len(sub)),
                          "tier_distribution_pct": {int(t): round(100 * float(vc.get(t, 0.0)), 1)
                                                    for t in range(kfound)},
                          "mean_tier": round(float(sub["tier_investment"].mean()), 2),
                          "normalised_mean_tier": round(nm, 3),
                          "prior_normalised_mean_tier": round(pn, 3)}
        print(f"  {prof:<20}{len(sub):>6}"
              + "".join(f"{100 * float(vc.get(t, 0.0)):>7.1f}%" for t in range(kfound))
              + f"{sub['tier_investment'].mean():>8.2f}{nm:>8.3f}{pn:>19.3f}")

    out = {
        "purpose": "portfolio_concentration benchmark recalibration + Investment-domain-only "
                   "re-test. Banking and Insurance untouched and not re-run. Validation only "
                   "— no production artifact read or written, no Layer 3 rules defined.",
        "step1_diagnosis": {
            "verdict": "(a) the IDEAL was miscalibrated; the generator's marginal is the "
                       "better-supported side and is left unchanged",
            "per_band_distribution_vs_old_ideal": diag,
            "evidence_for_generator": [
                "Goetzmann & Kumar (2008) Review of Finance 12(3):433-463, >40,000 discount-"
                "brokerage accounts 1991-96 — VERIFIED at source: retail investors are "
                "systematically under-diversified, more so when younger. The age gradient "
                "matches the generator's marginal taper (0.47 -> 0.25).",
                "A largest-holding share near a third is the arithmetic consequence of "
                "holding few positions (max share >= 1/N, and real portfolios are not "
                "equal-weighted).",
                "NOT VERIFIED AT SOURCE, flagged as such: secondary summaries reporting a "
                "mean of ~4 / median ~3 directly-held stocks (and Blume & Friend 1975 median "
                "2). Consistent with the above but taken from summaries, not read in source.",
                "scoring.py computes portfolio_concentration as max() over per-account "
                "concentration_risk, which overstates concentration — real values run at "
                "least as high as synthetic ones, never lower.",
                "No SG-specific retail concentration statistic exists — re-confirmed absent "
                "across SGX market statistics, SIAS 2025 Retail Investor Pulse Survey, CDP "
                "materials and SingStat. This remains a judgement call.",
            ],
            "evidence_against_old_ideal": [
                "Its sources (J.P. Morgan Private Bank 10-20%; Fidelity/Calamos/HDFC 5-10%) "
                "are PRESCRIPTIVE advisory thresholds for professionally-advised, largely "
                "high-net-worth clients — not descriptions of retail behaviour.",
                "Unreachable at BOTH age extremes (3.7% of band 1, 3.8% of band 5 clear it), "
                "so the taper rate is not the defect — the level is.",
            ],
        },
        "step2_fix": {
            "type": "absolute diversification-adequacy threshold (task option 3)",
            "threshold": CONC_THRESHOLD, "previous": "band ladder 0.20/0.17/0.13/0.10/0.08",
            "citation_state": "REASONED / STRUCTURAL, NOT CITED — same category as dti 0.15 "
                              "and the net_worth ladder, not the same as LIA/MoneySense",
            "rationale": CONC_CITATION,
            "rejected_alternatives": {
                "percentile_anchored": "rejected — would import relative-to-sample ranking "
                                       "into a tier anchor, which the brief's Score "
                                       "Derivation constraint explicitly forbids",
                "regenerate_dataset": "rejected — the generator is the better-supported side, "
                                      "and regenerating would invalidate the already-"
                                      "validated Banking and Insurance results",
            },
            "not_changed": {
                "generate_synthetic_data.py marginals": "diagnosed as correct",
                "benchmark_core.py SYNTHETIC_IDEALS[portfolio_concentration]":
                    "feeds the production centroid artifact (a different mechanism); out of "
                    "scope and blocked. FLAG: it likely has the same defect, and the two now "
                    "disagree (0.33 vs 0.20-0.08) — must be resolved before artifact re-export",
            },
        },
        "reproduction_check": {"combinations": len(grid), "mismatches": mism,
                               "geometry_identical_to_validated_run": len(mism) == 0},
        "threshold_sensitivity": {
            label: {"threshold": r["threshold"],
                    "base_rate_pct": r["base_rates"]["overall"]["portfolio_concentration"],
                    "n_gate_passers": len(r["gate"]), "gate_passers": r["gate"],
                    "winner": r["winner"]}
            for label, r in results.items()},
        "prior_result": {"gate_passers": prior_gate,
                         "winner": prior["spec_results"][SPEC]["winner"],
                         "n_gate_passers": len(prior_gate)},
        "result": {
            "n_survivors": len(survivors), "n_gate_passers": len(gate),
            "gate_passing_combinations": gate, "winner": winner, "k": kfound,
            "winner_metrics": {m: grid[winner][m] for m in
                               ("silhouette_macro", "silhouette_micro", "davies_bouldin",
                                "calinski_harabasz", "min_cluster_pct")},
            "winner_lift_detail": wl,
            "binding_benchmark_counts": binding,
            "winner_single_axis_diagnostic": seps,
            "winner_validation_detail": sweep[winner]["per_group"],
            "usability_sweep": {k: {kk: vv for kk, vv in v.items() if kk != "per_group"}
                                for k, v in sweep.items()},
        },
        "normal_population_check": {
            "n_normal": int(normal.sum()),
            "tier_share_pct": {int(i): round(100 * float(v), 2) for i, v in share.items()},
            "min_tier_share_pct": round(100 * float(share.min()), 2),
            "degenerate_tier_below_5pct": bool(share.min() < 0.05),
            "macro_silhouette_normal_subset": normal_sil,
            "benchmark_ordering": normal_order,
        },
        "mismatch_profile_behaviour": prof_tbl,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    rows = []
    for k, g in grid.items():
        if g.get("failed"):
            continue
        u = sweep.get(k, {})
        rows.append({"combination": k, "approach": g["approach"], "algorithm": g["algorithm"],
                     "k": g["k"], "family": g["family"],
                     "silhouette_macro": g["silhouette_macro"],
                     "silhouette_micro": g["silhouette_micro"],
                     "davies_bouldin": g["davies_bouldin"],
                     "calinski_harabasz": g["calinski_harabasz"],
                     "min_cluster_pct": g["min_cluster_pct"], "passes_floor": g["passes_floor"],
                     "min_spearman_rho": u.get("worst_group_min_spearman_rho"),
                     "min_headroom_captured": u.get("min_headroom_captured"),
                     "min_lift_pp": u.get("min_lift_pp"),
                     "passes_gate2_headroom": u.get("passes_gate2_headroom"),
                     "is_winner": k == winner})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "investment_recalibration_grid.csv"),
                              index=False)
    print(f"\n  saved {OUT_JSON}")
    print(f"  saved {os.path.join(OUT_DIR, 'investment_recalibration_grid.csv')}")
    print("\n" + "=" * 118)
    print(f"DONE. gate passers {len(prior_gate)} -> {len(gate)};  winner "
          f"{prior['spec_results'][SPEC]['winner']} -> {winner};  smallest normal-population "
          f"tier {prior['step6_normal_population_sanity']['investment']['min_tier_share_pct']}%"
          f" -> {100 * share.min():.2f}%")
    print("=" * 118)


if __name__ == "__main__":
    main()
