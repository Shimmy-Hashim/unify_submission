"""
overall_score_validation.py — validate the LOCKED overall-score formula (Option B: equal
domain thirds, Layer 1 folded into Insurance's share) against the synthetic population.
(Unify, CCDS25-1170, July 2026)

WHY THIS EXISTS
  continuous_score_investigation.py established that continuous per-domain scores do not
  hold up (Insurance inverts), and recommended a discrete tier + Layer 1 composite. The
  version it actually validated was a NAIVE flat equal-bit sum: 3 domain tiers + 3 Layer 1
  flags, every bit worth the same, which handed Insurance 3 of 6 bits (50% of the score)
  because both surviving Layer 1 flags are insurance-related.

  A different formula was then locked for equal domain weighting, and has never been
  checked against data. This script checks it, using the same method.

THE LOCKED FORMULA (Option B)
    banking_score    = 100 if banking_tier    == "Stable"   else 0
    investment_score = 100 if investment_tier == "On Track" else 0
    insurance_third  = mean(100 * [insurance_tier == "Covered",
                                   income_protection_coverage,
                                   hospitalisation_coverage])
    overall_score    = (banking_score + investment_score + insurance_third) / 3

  `has_investments` is DROPPED (redundant with the Investment tier). See the LAYER1 note
  printed at run time — this drop is NOT reflected in backend/ml/scoring.py.

A METHODOLOGICAL CORRECTION THIS SCRIPT MAKES, AND THE PRIOR ONE DID NOT
  The prior investigation reported that the naive composite "orders every strictly
  out-of-model variable at 0.991-1.000, including the two Insurance flags no continuous
  candidate could". Those four variables are strictly out-of-model with respect to the three
  CLUSTERING feature sets — which is what made them the right test for a continuous
  distance score. They are NOT all out-of-model with respect to a COMPOSITE that consumes
  them directly:

    naive     inputs: banking tier, investment tier, insurance tier, has_investments,
                      hospitalisation_coverage, income_protection_coverage
              -> 3 of the 4 test variables are its own inputs. Only
                 net_worth_to_income is a genuine test.
    Option B  inputs: banking tier, investment tier, insurance tier,
                      hospitalisation_coverage, income_protection_coverage
              -> 2 of the 4 are its own inputs. net_worth_to_income AND
                 has_investments are genuine tests.

  All four are reported, for the direct comparison the task asks for, but each is labelled
  TAUTOLOGICAL or INDEPENDENT per formula and the headline verdict rests only on the
  independent ones. Dropping has_investments makes Option B strictly MORE testable than the
  naive version, which is a point in its favour that only shows up once this distinction is
  made.

Analysis only. Reads the locked artifacts read-only; writes only to
MLAI/clustering/results/overall_score_validation/.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/overall_score_validation.py
"""

import os
import sys
import json
import itertools
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continuous_score_investigation import (            # noqa: E402
    DATA_PATH, DOMAINS, ARCHETYPES, BENCH_VARS, STRICT_VARS,
    load_models, assign_and_distance, meets_mask,
)

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results/overall_score_validation")
SCORING_PY = os.path.join(REPO_ROOT, "backend/ml/scoring.py")

# Layer 1 inputs actually used by Option B (has_investments dropped).
OPTION_B_L1 = ["income_protection_coverage", "hospitalisation_coverage"]
NAIVE_L1 = ["has_investments", "hospitalisation_coverage", "income_protection_coverage"]

# The four variables the naive version was checked against, kept verbatim for comparability.
TEST_VARS = [("banking", "net_worth_to_income"),
             ("investment", "has_investments"),
             ("insurance", "hospitalisation_coverage"),
             ("insurance", "income_protection_coverage")]

# The naive score is points-out-of-6, so its levels are fixed by the formula, not enumerated.
NAIVE_LEVELS = [round(100 * i / 6, 4) for i in range(7)]


def band_of(score):
    """Score -> the historical absolute tier band (SYNTHETIC_IDEALS.TIER_THRESHOLDS).

    The mechanism those bands belonged to is superseded, but they are the only banding this
    project has ever written down, so they are the fairest available stand-in for "does the
    choice of formula change a real user-facing outcome, not just a number"."""
    return np.select([score <= 39, score <= 69], ["At Risk", "Building"], "Stable")


def check_layer1_constant():
    """Confirm (or refute) the task's premise that scoring.py already reflects the
    has_investments drop. Read as text — this script imports nothing from backend/."""
    with open(SCORING_PY, encoding="utf-8") as f:
        src = f.read()
    start = src.index("LAYER1_CHECKLIST_FEATURES = [")
    block = src[start:src.index("]", start) + 1]
    listed = [ln.split('"')[1] for ln in block.splitlines() if '"' in ln]
    return {
        "scoring_py_LAYER1_CHECKLIST_FEATURES": listed,
        "option_b_layer1_inputs": OPTION_B_L1,
        "has_investments_still_listed_in_scoring_py": "has_investments" in listed,
        "matches_option_b": set(listed) == set(OPTION_B_L1),
        "verdict": (
            "MATCH" if set(listed) == set(OPTION_B_L1) else
            "MISMATCH — scoring.py has not been updated for the has_investments drop"),
        "extra_in_scoring_py": sorted(set(listed) - set(OPTION_B_L1)),
    }


# ======================================================================================
# The two formulas
# ======================================================================================
def option_b(df, tiers):
    banking = 100.0 * (tiers["banking"] == 0)
    investment = 100.0 * (tiers["investment"] == 0)
    ins_bits = np.column_stack([
        (tiers["insurance"] == 0).astype(float),
        df["income_protection_coverage"].to_numpy(float).round(),
        df["hospitalisation_coverage"].to_numpy(float).round(),
    ])
    insurance_third = 100.0 * ins_bits.mean(axis=1)
    return ((banking + investment + insurance_third) / 3.0,
            {"banking": banking, "investment": investment,
             "insurance_third": insurance_third, "insurance_bits": ins_bits})


def naive(df, tiers):
    """The flat equal-bit version validated in continuous_score_investigation.py, rebuilt
    here identically so the head-to-head is like for like."""
    tier_pts = sum((tiers[d] == 0).astype(int) for d in DOMAINS)
    l1_pts = sum(df[f].to_numpy(float).round().astype(int) for f in NAIVE_L1)
    total = tier_pts + l1_pts
    return 100.0 * total / 6.0, total


# ======================================================================================
# Checks
# ======================================================================================
def enumerate_levels():
    """Every value the formula can take, derived from the formula rather than from the
    data, so an unreachable level is visible as unreachable rather than absent."""
    combos = []
    for b, i, c, p, h in itertools.product([0, 1], repeat=5):
        third = 100.0 * (c + p + h) / 3.0
        combos.append({
            "banking_stable": b, "investment_on_track": i,
            "insurance_covered": c, "income_protection": p, "hospitalisation": h,
            "insurance_third": round(third, 2),
            "overall": round((100.0 * b + 100.0 * i + third) / 3.0, 4)})
    return combos, sorted({c["overall"] for c in combos})


def distribution(score, label, expected_levels=None):
    s = pd.Series(np.round(score, 4))
    vc = s.value_counts().sort_index()
    pct = (100 * vc / len(s)).round(2)
    out = {"formula": label, "n": int(len(s)),
           "n_distinct_values_observed": int(vc.size),
           "observed_levels": [float(x) for x in vc.index],
           "counts": {float(k): int(v) for k, v in vc.items()},
           "pct": {float(k): float(v) for k, v in pct.items()},
           "mean": round(float(s.mean()), 2), "median": round(float(s.median()), 2),
           "sd": round(float(s.std(ddof=1)), 2),
           "modal_level": float(vc.idxmax()), "modal_pct": float(pct.max()),
           "top2_levels_pct": float(pct.nlargest(2).sum()),
           "top3_levels_pct": float(pct.nlargest(3).sum()),
           "gini_style_concentration": round(float((pct / 100).pow(2).sum()), 4)}
    if expected_levels is not None:
        obs = set(np.round(out["observed_levels"], 4))
        exp = set(np.round(expected_levels, 4))
        out["n_distinct_values_possible"] = len(exp)
        out["unreachable_levels"] = sorted(float(x) for x in exp - obs)
        out["levels_below_1pct"] = [float(k) for k, v in pct.items() if v < 1.0]
    return out


def ordering(df, score, label, inputs_of_formula):
    """Spearman of '% clearing the benchmark' across the formula's own levels, among the
    non-injected population — the identical check the naive version was reported on."""
    nd = df["profile_type"] == "normal"
    lvl = pd.Series(np.round(score, 4))[nd.to_numpy()]
    out = {}
    for dom in DOMAINS:
        for var in BENCH_VARS[dom]:
            vals = pd.Series(meets_mask(df[nd], dom, var).astype(float)).to_numpy()
            clearing = pd.DataFrame({"lv": lvl.to_numpy(), "v": vals}).groupby("lv")["v"].mean()
            taut = var in inputs_of_formula
            out[f"{dom}.{var}"] = {
                "spearman_rho": round(float(stats.spearmanr(
                    clearing.index.to_numpy(float), clearing.to_numpy()).statistic), 3),
                "pct_clearing_by_level": [round(100 * float(x), 1) for x in clearing],
                "levels": [float(x) for x in clearing.index],
                "n_levels_populated": int(clearing.size),
                "strictly_out_of_model_vs_clustering": var in STRICT_VARS[dom],
                "is_an_input_to_this_formula": bool(taut),
                "test_status": "TAUTOLOGICAL (formula input)" if taut else "INDEPENDENT",
            }
    rows = list(out.values())
    indep = [r for r in rows if not r["is_an_input_to_this_formula"]]
    key4 = {f"{d}.{v}": out[f"{d}.{v}"] for d, v in TEST_VARS}
    key4_indep = [r for r in key4.values() if not r["is_an_input_to_this_formula"]]
    return {
        "formula": label,
        "per_benchmark": out,
        "all_12": {"min_rho": round(min(r["spearman_rho"] for r in rows), 3),
                   "mean_rho": round(float(np.mean([r["spearman_rho"] for r in rows])), 3)},
        "independent_only": {
            "n": len(indep),
            "variables": [k for k, r in out.items() if not r["is_an_input_to_this_formula"]],
            "min_rho": round(min(r["spearman_rho"] for r in indep), 3) if indep else None,
            "mean_rho": round(float(np.mean([r["spearman_rho"] for r in indep])), 3)
            if indep else None},
        "the_four_variables_the_naive_version_was_reported_on": {
            k: {"rho": r["spearman_rho"], "test_status": r["test_status"]}
            for k, r in key4.items()},
        "of_those_four_independent_only": {
            "n": len(key4_indep),
            "min_rho": round(min(r["spearman_rho"] for r in key4_indep), 3)
            if key4_indep else None},
    }


def variance_decomposition(score, parts):
    """Realised influence of each additive term, not its nominal weight. For an additive
    score, Cov(term, total) / Var(total) sums to exactly 1 — so this reads directly as
    'what fraction of the score's variation this input actually drives', which is the
    question 'equal domain thirds' is really making a claim about."""
    v = float(np.var(score, ddof=1))
    return {k: round(float(np.cov(t, score, ddof=1)[0, 1]) / v, 4) for k, t in parts.items()}


def archetypes(df, scores):
    out = {}
    for p in ARCHETYPES:
        m = (df["profile_type"] == p).to_numpy()
        if not m.any():
            continue
        out[p] = {"n": int(m.sum())}
        for label, s in scores.items():
            out[p][label] = {"mean": round(float(np.mean(s[m])), 2),
                             "median": round(float(np.median(s[m])), 2),
                             "sd": round(float(np.std(s[m], ddof=1)), 2)}
    for label in scores:
        ranked = sorted(out, key=lambda p: -out[p][label]["mean"])
        for r, p in enumerate(ranked):
            out[p][label]["rank_among_profiles"] = r + 1
    return out


def level_aliasing(sb, sn, insurance_third):
    """32 input combinations collapse onto 10 Option B values, so a level can be reached by
    structurally opposite users. Measured, because it is what makes the insurance ordering
    check zigzag: the same number means different things.

    Reported for both formulas — the naive version aliases too, but its levels are literally
    'how many of 6 boxes are ticked', so an insurance box is worth the same as any other and
    the insurance clearing rate still climbs with the level. Like for like: BOTH are scored
    against the SAME insurance_third, so the comparison is about how each formula positions
    that quantity, not about two different definitions of "insurance component"."""
    out = {}
    for label, s in [("option_b", sb), ("naive", sn)]:
        rows = {}
        for lv in sorted(set(np.round(s, 4))):
            m = np.isclose(s, lv)
            comp = pd.Series(np.round(insurance_third[m], 2)).value_counts(normalize=True)
            rows[float(round(lv, 2))] = {
                "n": int(m.sum()),
                "pct_of_population": round(100 * float(m.mean()), 2),
                "insurance_component_distribution_pct": {
                    float(k): round(100 * float(v), 1) for k, v in comp.sort_index().items()},
                "pct_with_zero_insurance_component": round(
                    100 * float((insurance_third[m] == 0).mean()), 1),
                "pct_with_full_insurance_component": round(
                    100 * float((insurance_third[m] == 100).mean()), 1),
                "n_distinct_insurance_states_at_this_level": int(comp.size),
            }
        mixed = [lv for lv, r in rows.items()
                 if r["pct_with_zero_insurance_component"] > 5
                 and r["pct_with_full_insurance_component"] > 5]
        out[label] = {
            "per_level": rows,
            "levels_holding_both_zero_and_full_insurance_over_5pct_each": mixed,
            "n_such_levels": len(mixed),
            "pct_of_population_on_such_levels": round(
                sum(rows[lv]["pct_of_population"] for lv in mixed), 2),
        }
    return out


def protection_gap(df, sb, sn, tiers):
    """Does the formula still surface a protection gap?

    The brief's stated reason for disqualifying the original cluster-derived centroid was
    that it would have told a wealthy, underinsured user they were 'close to ideal' —
    strong banking numbers outweighing weak coverage in a compensatory sum. Option B hands
    Banking a full third and compresses all three insurance indicators into one third, so
    that exact failure mode has to be re-tested, not assumed absent.

    Two populations, defined structurally rather than by profile label:
      total_gap  — Banking Stable AND all three insurance indicators zero (the worst
                   possible protection state combined with the best banking state)
      rule1      — Layer 3 Rule 1 as locked: Banking Stable AND Insurance Not Covered"""
    nd = (df["profile_type"] == "normal").to_numpy()
    ip = df["income_protection_coverage"].to_numpy(float).round()
    hp = df["hospitalisation_coverage"].to_numpy(float).round()
    total_gap = (tiers["banking"] == 0) & (tiers["insurance"] == 1) & (ip == 0) & (hp == 0)
    rule1 = (tiers["banking"] == 0) & (tiers["insurance"] == 1)

    groups = [
        ("total_protection_gap", total_gap,
         "Banking Stable + Insurance Not Covered + no income protection + no hospitalisation"),
        ("layer3_rule1_wealth_without_protection", rule1,
         "Banking Stable + Insurance Not Covered (Layer 3 Rule 1 as locked)"),
    ]
    # The three archetypes are the label-based counterpart to the two structural groups
    # above: same measurements, so that "defined by behaviour" and "defined by profile tag"
    # can be read off against each other.
    groups += [(f"archetype:{p}", (df["profile_type"] == p).to_numpy(), f"profile_type == {p}")
               for p in ("wealthy_exposed", "underinsured_saver", "protection_first")]

    def compare_to_normals(s, m, med_n):
        return {"mean": round(float(np.mean(s[m])), 2),
                "median": round(float(np.median(s[m])), 2),
                "gap_below_normal_median_points": round(med_n - float(np.median(s[m])), 2),
                "pct_scoring_at_or_above_normal_median": round(
                    100 * float((s[m] >= med_n).mean()), 2),
                "pct_scoring_at_or_above_50": round(100 * float((s[m] >= 50).mean()), 2)}

    scores = [("option_b", sb), ("naive", sn)]
    normal_median = {label: float(np.median(s[nd])) for label, s in scores}
    out = {"normal_population_median": {label: round(normal_median[label], 2)
                                        for label, _ in scores},
           "normal_population_mean": {label: round(float(np.mean(s[nd])), 2)
                                      for label, s in scores},
           "groups": {}}
    for name, m, why in groups:
        out["groups"][name] = {"definition": why, "n": int(m.sum()),
                               "pct_of_population": round(100 * float(m.mean()), 2)}
        for label, s in scores:
            out["groups"][name][label] = compare_to_normals(s, m, normal_median[label])
    return out


def head_to_head(df, sb, sn):
    nd = (df["profile_type"] == "normal").to_numpy()
    bb, bn = band_of(sb), band_of(sn)
    xt = pd.crosstab(pd.Series(bn, name="naive_band"), pd.Series(bb, name="optionB_band"))
    moved = float((bb != bn).mean())
    # Do the two formulas disagree about which of two users is healthier? Sampled pairs,
    # excluding pairs the two agree are tied — a discordance rate is the outcome-level
    # question, where a correlation is only a shape question.
    rng = np.random.default_rng(42)
    idx = np.arange(len(sb))
    i, j = rng.choice(idx, 400_000), rng.choice(idx, 400_000)
    ok = i != j
    i, j = i[ok], j[ok]
    db, dn = np.sign(sb[i] - sb[j]), np.sign(sn[i] - sn[j])
    both_rank = (db != 0) & (dn != 0)
    return {
        "spearman": round(float(stats.spearmanr(sb, sn).statistic), 4),
        "pearson": round(float(np.corrcoef(sb, sn)[0, 1]), 4),
        "kendall_tau_b": round(float(stats.kendalltau(sb, sn).statistic), 4),
        "mean_absolute_difference_points": round(float(np.mean(np.abs(sb - sn))), 2),
        "median_absolute_difference_points": round(float(np.median(np.abs(sb - sn))), 2),
        "max_absolute_difference_points": round(float(np.max(np.abs(sb - sn))), 2),
        "pct_users_identical_score": round(100 * float((np.abs(sb - sn) < 1e-9).mean()), 2),
        "pct_users_differing_by_over_10_points": round(
            100 * float((np.abs(sb - sn) > 10).mean()), 2),
        "mean_option_b_minus_naive": round(float(np.mean(sb - sn)), 2),
        "pairwise_ordering": {
            "definition": ("400k random user pairs; of the pairs BOTH formulas rank "
                           "strictly, how often do they disagree about who is healthier"),
            "n_pairs_both_rank": int(both_rank.sum()),
            "pct_of_pairs_both_rank": round(100 * float(both_rank.mean()), 2),
            "discordant_pct": round(100 * float((db[both_rank] != dn[both_rank]).mean()), 2),
            "pct_pairs_tied_under_naive_but_ranked_by_optionB": round(
                100 * float(((dn == 0) & (db != 0)).mean()), 2),
            "pct_pairs_tied_under_optionB_but_ranked_by_naive": round(
                100 * float(((db == 0) & (dn != 0)).mean()), 2)},
        "outcome_band_shift": {
            "bands_used": "0-39 At Risk / 40-69 Building / 70-100 Stable "
                          "(the project's historical absolute thresholds — the only banding "
                          "ever written down; the mechanism they belonged to is superseded)",
            "pct_users_changing_band": round(100 * moved, 2),
            "crosstab_counts": xt.to_dict(),
            "band_shares_naive_pct": {k: round(100 * v, 2) for k, v in
                                      pd.Series(bn).value_counts(normalize=True).items()},
            "band_shares_optionB_pct": {k: round(100 * v, 2) for k, v in
                                        pd.Series(bb).value_counts(normalize=True).items()},
            "pct_users_changing_band_normals_only": round(
                100 * float((bb[nd] != bn[nd]).mean()), 2)},
    }


# ======================================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    models = load_models()
    manifest = models.pop("_manifest")

    print("=" * 112)
    print(f"OVERALL-SCORE VALIDATION — Option B (equal domain thirds), {len(df):,} users")
    print("=" * 112)

    l1 = check_layer1_constant()
    print(f"\n[Layer 1 constant] scoring.py LAYER1_CHECKLIST_FEATURES = "
          f"{l1['scoring_py_LAYER1_CHECKLIST_FEATURES']}")
    print(f"                   Option B actually uses  = {l1['option_b_layer1_inputs']}")
    print(f"                   -> {l1['verdict']}")
    if l1["extra_in_scoring_py"]:
        print(f"                   extra in scoring.py: {l1['extra_in_scoring_py']}")

    # ------------------------------------------------------------------ STEP 1: replay
    tiers = {}
    print()
    for d in DOMAINS:
        t, _, _ = assign_and_distance(df, models[d])
        tiers[d] = t
        rec = models[d]["shipped_tiers_summary"]["population_shares_pct"]
        rep = [round(100 * float((t == k).mean()), 1) for k in range(models[d]["shipped_tiers"])]
        ok = all(abs(a - b) <= 0.15 for a, b in zip(rec, rep))
        print(f"  [replay] {d:<11}{models[d]['combination']:<20} tiers {rep}% vs artifact "
              f"{rec}%  -> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{d}: replay does not reproduce the locked artifact — aborting")

    sb, parts = option_b(df, tiers)
    sn, naive_total = naive(df, tiers)
    normals = (df["profile_type"] == "normal").to_numpy()

    out = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": ("Validate the locked Option B overall-score formula (equal domain "
                    "thirds, Layer 1 folded into Insurance's share) against the 7,000-user "
                    "synthetic population, head to head with the naive flat equal-bit "
                    "version validated in continuous_score_investigation.py."),
        "scope": "Analysis only. No production code or artifact modified.",
        "artifact_provenance": {
            "manifest_generated_utc": manifest["generated_utc"],
            "dataset_sha256": manifest["dataset"]["sha256"],
            "domains": {d: manifest["domains"][d]["combination"] for d in DOMAINS}},
        "formulas": {
            "option_b": ("(100*[banking==Stable] + 100*[investment==On Track] + "
                         "mean(100*[insurance==Covered], 100*income_protection, "
                         "100*hospitalisation)) / 3"),
            "naive": ("100 * (count of: banking==Stable, investment==On Track, "
                      "insurance==Covered, has_investments, hospitalisation_coverage, "
                      "income_protection_coverage) / 6")},
        "nominal_weights": {
            "option_b": {"banking": "1/3 = 33.3%", "investment": "1/3 = 33.3%",
                         "insurance_tier": "1/9 = 11.1%",
                         "income_protection": "1/9 = 11.1%",
                         "hospitalisation": "1/9 = 11.1%"},
            "naive": {"banking": "1/6 = 16.7%", "investment": "1/6 = 16.7%",
                      "insurance_tier": "1/6 = 16.7%", "has_investments": "1/6 = 16.7%",
                      "hospitalisation": "1/6 = 16.7%", "income_protection": "1/6 = 16.7%"}},
        "layer1_constant_check": l1,
    }

    # ------------------------------------------------------------------ STEP 2: levels
    combos, levels = enumerate_levels()
    dist_b = distribution(sb, "option_b", levels)
    dist_n = distribution(sn, "naive", NAIVE_LEVELS)
    out["step2_distribution"] = {
        "all_32_input_combinations": combos,
        "distinct_overall_values_possible": {
            "n": len(levels), "values": [round(v, 2) for v in levels],
            "note": ("5 binary inputs = 32 combinations, but banking and investment are "
                     "interchangeable in the sum, so they collapse onto 10 distinct "
                     "overall values (vs the naive version's 7).")},
        "option_b": dist_b, "naive": dist_n,
        # The prior investigation quoted the naive spread on NORMALS ONLY
        # (1.5/7.9/23.8/30.4/22.6/11.1/2.8). Repeated on the same basis so the comparison
        # the task asks for is like for like, rather than all-users vs normals-only.
        "normals_only": {
            "option_b": distribution(sb[normals], "option_b_normals", levels),
            "naive": distribution(sn[normals], "naive_normals", NAIVE_LEVELS)},
        "insurance_third_distribution_pct": {
            float(k): round(100 * float(v), 2) for k, v in
            pd.Series(np.round(parts["insurance_third"], 2)).value_counts(
                normalize=True).sort_index().items()},
    }

    print(f"\n{'=' * 112}\nSTEP 2 — DISTRIBUTION / DEGENERACY\n{'=' * 112}")
    print(f"  Option B can take {len(levels)} distinct values "
          f"(32 input combinations -> 10 values): {[round(v, 1) for v in levels]}")
    third_pct = out["step2_distribution"]["insurance_third_distribution_pct"]
    print(f"  insurance_third alone: {sorted(third_pct)}")
    print(f"\n  {'level':>8}{'Option B %':>13}{'n':>8}   |  {'level':>8}{'naive %':>10}{'n':>8}")
    for k in sorted(set(dist_b["pct"]) | set(dist_n["pct"])):
        print(f"  {k:>8.1f}{dist_b['pct'].get(k, 0):>12.2f}%"
              f"{dist_b['counts'].get(k, 0):>8}   |  {k:>8.1f}"
              f"{dist_n['pct'].get(k, 0):>9.2f}%{dist_n['counts'].get(k, 0):>8}")
    print(f"\n  Option B: {dist_b['n_distinct_values_observed']}/"
          f"{dist_b['n_distinct_values_possible']} levels reached, modal "
          f"{dist_b['modal_level']:.1f} at {dist_b['modal_pct']:.2f}%, "
          f"top-2 {dist_b['top2_levels_pct']:.1f}%, top-3 {dist_b['top3_levels_pct']:.1f}%, "
          f"unreachable {dist_b['unreachable_levels']}")
    print(f"  naive   : {dist_n['n_distinct_values_observed']}/"
          f"{dist_n['n_distinct_values_possible']} levels reached, modal "
          f"{dist_n['modal_level']:.1f} at {dist_n['modal_pct']:.2f}%, "
          f"top-2 {dist_n['top2_levels_pct']:.1f}%, top-3 {dist_n['top3_levels_pct']:.1f}%")

    # ------------------------------------------------------------------ STEP 3: ordering
    inputs_b = {"hospitalisation_coverage", "income_protection_coverage"}
    inputs_n = {"hospitalisation_coverage", "income_protection_coverage", "has_investments"}
    ord_b = ordering(df, sb, "option_b", inputs_b)
    ord_n = ordering(df, sn, "naive", inputs_n)
    out["step3_ordering"] = {"option_b": ord_b, "naive": ord_n}

    print(f"\n{'=' * 112}\nSTEP 3 — ORDERING vs OUT-OF-MODEL BENCHMARKS (normals only)\n"
          f"{'=' * 112}")
    print("  The four variables the naive version was reported on (0.991-1.000):")
    print(f"    {'variable':<40}{'Option B':>10}{'naive':>10}   status under Option B / naive")
    for d, v in TEST_VARS:
        k = f"{d}.{v}"
        rb = ord_b["per_benchmark"][k]
        rn = ord_n["per_benchmark"][k]
        print(f"    {k:<40}{rb['spearman_rho']:>10.3f}{rn['spearman_rho']:>10.3f}   "
              f"{rb['test_status']} / {rn['test_status']}")
    print(f"\n  All 12 domain benchmarks —  Option B: min {ord_b['all_12']['min_rho']}, "
          f"mean {ord_b['all_12']['mean_rho']}   |   naive: min {ord_n['all_12']['min_rho']}, "
          f"mean {ord_n['all_12']['mean_rho']}")
    print(f"  INDEPENDENT only     —  Option B: {ord_b['independent_only']['n']} vars, "
          f"min {ord_b['independent_only']['min_rho']}, "
          f"mean {ord_b['independent_only']['mean_rho']}   |   naive: "
          f"{ord_n['independent_only']['n']} vars, min {ord_n['independent_only']['min_rho']}, "
          f"mean {ord_n['independent_only']['mean_rho']}")

    # ---------------------------------------------------------- realised weighting
    parts_b = {"banking": parts["banking"] / 3.0,
               "investment": parts["investment"] / 3.0,
               "insurance_tier": 100.0 * parts["insurance_bits"][:, 0] / 9.0,
               "income_protection": 100.0 * parts["insurance_bits"][:, 1] / 9.0,
               "hospitalisation": 100.0 * parts["insurance_bits"][:, 2] / 9.0}
    parts_n = {"banking": 100.0 * (tiers["banking"] == 0) / 6.0,
               "investment": 100.0 * (tiers["investment"] == 0) / 6.0,
               "insurance_tier": 100.0 * (tiers["insurance"] == 0) / 6.0,
               "has_investments": 100.0 * df["has_investments"].to_numpy(float).round() / 6.0,
               "hospitalisation": 100.0 * df["hospitalisation_coverage"].to_numpy(
                   float).round() / 6.0,
               "income_protection": 100.0 * df["income_protection_coverage"].to_numpy(
                   float).round() / 6.0}
    vd_b, vd_n = variance_decomposition(sb, parts_b), variance_decomposition(sn, parts_n)
    out["realised_influence_variance_share"] = {
        "definition": ("Cov(term, total)/Var(total) — sums to 1 for an additive score. "
                       "This is the term's ACTUAL share of the score's variation, which is "
                       "what 'equal domain thirds' is a claim about; the nominal weight is "
                       "only equal to it when the binary inputs have equal variance."),
        "option_b": vd_b, "naive": vd_n,
        "option_b_by_domain": {
            "banking": round(vd_b["banking"], 4), "investment": round(vd_b["investment"], 4),
            "insurance_total": round(vd_b["insurance_tier"] + vd_b["income_protection"]
                                     + vd_b["hospitalisation"], 4)},
        "naive_by_domain": {
            "banking": round(vd_n["banking"], 4), "investment": round(
                vd_n["investment"] + vd_n["has_investments"], 4),
            "insurance_total": round(vd_n["insurance_tier"] + vd_n["hospitalisation"]
                                     + vd_n["income_protection"], 4)},
    }
    print("\n  Realised influence (share of score variance, not nominal weight):")
    influence = out["realised_influence_variance_share"]
    for label, key in [("Option B", "option_b_by_domain"), ("naive", "naive_by_domain")]:
        by_domain = influence[key]
        print(f"    {label:<10}by domain: banking {100 * by_domain['banking']:.1f}%"
              f"  investment {100 * by_domain['investment']:.1f}%"
              f"  insurance {100 * by_domain['insurance_total']:.1f}%")

    # ------------------------------------------------------------------ STEP 4: archetypes
    arc = archetypes(df, {"option_b": sb, "naive": sn})
    out["step4_archetypes"] = arc
    print(f"\n{'=' * 112}\nSTEP 4 — ARCHETYPES\n{'=' * 112}")
    print(f"  {'profile':<22}{'n':>6}{'Option B':>11}{'rank':>6}{'naive':>10}{'rank':>6}"
          f"{'delta':>9}")
    for p in sorted(arc, key=lambda x: -arc[x]["option_b"]["mean"]):
        a = arc[p]
        print(f"  {p:<22}{a['n']:>6}{a['option_b']['mean']:>11.2f}"
              f"{a['option_b']['rank_among_profiles']:>6}{a['naive']['mean']:>10.2f}"
              f"{a['naive']['rank_among_profiles']:>6}"
              f"{a['option_b']['mean'] - a['naive']['mean']:>+9.2f}")

    alias = level_aliasing(sb, sn, parts["insurance_third"])
    out["step3b_level_aliasing"] = alias
    print("\n  Level aliasing — what the insurance component looks like at each level:")
    print(f"    {'level':>8}{'n':>7}{'% zero ins':>13}{'% full ins':>13}   Option B")
    for lv, r in alias["option_b"]["per_level"].items():
        print(f"    {lv:>8.1f}{r['n']:>7}{r['pct_with_zero_insurance_component']:>12.1f}%"
              f"{r['pct_with_full_insurance_component']:>12.1f}%")
    print(f"    -> {alias['option_b']['n_such_levels']} levels hold both zero-insurance and "
          f"full-insurance users at >5% each, covering "
          f"{alias['option_b']['pct_of_population_on_such_levels']}% of the population "
          f"(naive: {alias['naive']['n_such_levels']} levels, "
          f"{alias['naive']['pct_of_population_on_such_levels']}%)")

    pg = protection_gap(df, sb, sn, tiers)
    out["step4b_protection_gap"] = pg
    print("\n  Protection-gap check — does strong banking wash out zero coverage?")
    print(f"    normal-population median: Option B {pg['normal_population_median']['option_b']}"
          f"  |  naive {pg['normal_population_median']['naive']}")
    print(f"    {'group':<44}{'n':>6}{'B mean':>9}{'B >=normMed':>13}"
          f"{'naive mean':>12}{'n >=normMed':>13}")
    for g, r in pg["groups"].items():
        print(f"    {g:<44}{r['n']:>6}{r['option_b']['mean']:>9.2f}"
              f"{r['option_b']['pct_scoring_at_or_above_normal_median']:>12.1f}%"
              f"{r['naive']['mean']:>12.2f}"
              f"{r['naive']['pct_scoring_at_or_above_normal_median']:>12.1f}%")

    # ------------------------------------------------------------------ STEP 5: head to head
    h2h = head_to_head(df, sb, sn)
    out["step5_head_to_head"] = h2h
    print(f"\n{'=' * 112}\nSTEP 5 — HEAD TO HEAD\n{'=' * 112}")
    print(f"  Spearman {h2h['spearman']}   Pearson {h2h['pearson']}   "
          f"Kendall tau-b {h2h['kendall_tau_b']}")
    print(f"  mean |diff| {h2h['mean_absolute_difference_points']}pt   median "
          f"{h2h['median_absolute_difference_points']}pt   max "
          f"{h2h['max_absolute_difference_points']}pt   "
          f"identical for {h2h['pct_users_identical_score']}% of users   "
          f">10pt apart for {h2h['pct_users_differing_by_over_10_points']}%")
    po = h2h["pairwise_ordering"]
    print(f"  pairwise: of the {po['pct_of_pairs_both_rank']}% of pairs both formulas rank "
          f"strictly, they disagree on {po['discordant_pct']}%")
    print(f"            naive ties that Option B breaks: "
          f"{po['pct_pairs_tied_under_naive_but_ranked_by_optionB']}% of pairs;  "
          f"Option B ties naive breaks: "
          f"{po['pct_pairs_tied_under_optionB_but_ranked_by_naive']}%")
    ob = h2h["outcome_band_shift"]
    print(f"  outcome bands (0-39/40-69/70-100): {ob['pct_users_changing_band']}% of users "
          f"change band ({ob['pct_users_changing_band_normals_only']}% of normals)")
    print(f"            naive {ob['band_shares_naive_pct']}  ->  "
          f"Option B {ob['band_shares_optionB_pct']}")

    # ------------------------------------------------------------------ outputs
    per_user = df[["user_id", "age_band", "profile_type"]].copy()
    for d in DOMAINS:
        per_user[f"tier_{d}"] = tiers[d]
    per_user["income_protection_coverage"] = df["income_protection_coverage"]
    per_user["hospitalisation_coverage"] = df["hospitalisation_coverage"]
    per_user["has_investments"] = df["has_investments"]
    per_user["insurance_third"] = np.round(parts["insurance_third"], 2)
    per_user["option_b_score"] = np.round(sb, 2)
    per_user["naive_score"] = np.round(sn, 2)
    per_user["naive_points_0_6"] = naive_total
    per_user["option_b_band"] = band_of(sb)
    per_user["naive_band"] = band_of(sn)
    per_user.to_csv(os.path.join(OUT_DIR, "per_user_overall_scores.csv"), index=False)

    # The four reported-on variables lead, then the rest of the 12 benchmarks in domain order.
    key_vars = [f"{d}.{v}" for d, v in TEST_VARS]
    all_vars = key_vars + [f"{d}.{v}" for d in DOMAINS for v in BENCH_VARS[d]
                           if f"{d}.{v}" not in key_vars]
    pd.DataFrame([{"variable": k,
                   "option_b_rho": ord_b["per_benchmark"][k]["spearman_rho"],
                   "option_b_status": ord_b["per_benchmark"][k]["test_status"],
                   "naive_rho": ord_n["per_benchmark"][k]["spearman_rho"],
                   "naive_status": ord_n["per_benchmark"][k]["test_status"]}
                  for k in all_vars]).to_csv(
        os.path.join(OUT_DIR, "ordering_comparison.csv"), index=False)

    pd.DataFrame([{"level": k, "option_b_pct": dist_b["pct"].get(k, 0.0),
                   "option_b_n": dist_b["counts"].get(k, 0),
                   "naive_pct": dist_n["pct"].get(k, 0.0),
                   "naive_n": dist_n["counts"].get(k, 0)}
                  for k in sorted(set(list(dist_b["pct"]) + list(dist_n["pct"])))]).to_csv(
        os.path.join(OUT_DIR, "level_distribution.csv"), index=False)

    with open(os.path.join(OUT_DIR, "overall_score_validation.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    for n in ("overall_score_validation.json", "per_user_overall_scores.csv",
              "ordering_comparison.csv", "level_distribution.csv"):
        print(f"\n  saved {OUT_DIR}/{n} "
              f"({os.path.getsize(os.path.join(OUT_DIR, n)) / 1024:.0f} KB)")
    print("=" * 112)


if __name__ == "__main__":
    main()
