"""
gated_score_validation.py — validate INSURANCE-AS-A-GATE overall-score formulas
(Unify, CCDS25-1170, July 2026).

WHY THIS EXISTS
  overall_score_validation.py found that Option B (equal domain thirds) lets strong
  Banking/Investment scores mask zero insurance coverage: the "Banking Stable + zero on all
  three insurance indicators" group (n=1,005) scores a mean of 45.80 against the normal
  population's 42.68 — 3.1 points ABOVE the typical user. That is the same compensatory
  failure the brief already disqualified the original cluster-derived centroid for.

  The brief already states the principle the fix should implement:
      "Critical gaps (e.g. no hospitalisation cover) surface as warnings regardless of
       overall score."
  No gate/cap mechanism has ever been designed or tested. This script proposes candidates
  and validates them on the identical suite used for Option B and the naive version, so all
  five formulas are directly comparable.

  It assumes no formula in advance. Candidate (a) and (b) are the shapes the task named;
  candidate (c) exists because (a) turns out to have a specific, measurable weakness (see
  the STRUCTURAL NOTE below) that only shows up once it is built.

STRUCTURAL NOTE THAT DRIVES CANDIDATE (c)
  base_score = mean(banking_score, investment_score) where both are 0/100, so the base has
  only THREE possible values: 0, 50, 100. Any formula that only caps or scales that base
  inherits its coarseness. Candidate (a) as specified reaches 5 distinct scores across the
  whole population; the naive version reaches 7 and Option B reaches 10. Candidate (c)
  therefore gates Option B's score rather than the two-domain base — keeping Option B's
  granularity while removing its compensatory behaviour.

Analysis only. Reads the locked artifacts read-only; writes only to
MLAI/clustering/results/gated_score_validation/.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/gated_score_validation.py
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
    DATA_PATH, DOMAINS, ARCHETYPES, load_models, assign_and_distance,
)
from overall_score_validation import (                  # noqa: E402
    option_b, naive, distribution, ordering, band_of, enumerate_levels,
    NAIVE_LEVELS, TEST_VARS, level_aliasing,
)

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results/gated_score_validation")

# ======================================================================================
# THE CEILING LADDER — grounding, stated honestly
# ======================================================================================
# The ONLY score banding this project has ever written down is SYNTHETIC_IDEALS'
# TIER_THRESHOLDS: 0-39 At Risk / 40-69 Building / 70-100 Stable. The mechanism those bands
# belonged to (distance-to-centroid) is superseded, but the boundaries themselves are the
# only non-arbitrary numbers available, so the ceilings are taken from them rather than
# invented:
#     39  = the top of At Risk   (a user capped here can never read as "Building")
#     69  = the top of Building  (a user capped here can never read as "Stable")
#     100 = no cap
#
# ⚠️ WHAT IS AND IS NOT GROUNDED — the same honesty standard applied to SYNTHETIC_IDEALS:
#   GROUNDED:     the ceiling VALUES (39, 69, 100) are the project's own band boundaries.
#   GROUNDED:     "zero insurance coverage must never read as Stable" follows directly from
#                 the brief's stated principle about critical gaps.
#   JUDGEMENT CALL, NO INDEPENDENT GROUNDING: the MAPPING from how many of the three
#                 indicators are missing onto which ceiling applies. "0 of 3 costs you two
#                 bands, 1 of 3 costs you one band, 2 of 3 costs you nothing" is a
#                 defensible reading of proportionality, not a benchmark. It is therefore
#                 run at three harshness settings and the sensitivity reported, rather than
#                 presented as the answer.
CEILING_LADDERS = {
    # indicators cleared (0..3) -> ceiling
    "lenient":     {0: 69.0, 1: 69.0, 2: 100.0, 3: 100.0},
    "band_ladder": {0: 39.0, 1: 69.0, 2: 100.0, 3: 100.0},   # primary
    "strict":      {0: 39.0, 1: 39.0, 2: 69.0, 3: 100.0},
}
PRIMARY_LADDER = "band_ladder"

# Candidate (b)'s gate factor. The 0.5 floor is a PURE JUDGEMENT CALL with no independent
# grounding anywhere in this project — it says "zero insurance halves your score", which no
# benchmark, brief statement or prior result supports. Flagged here and in the output rather
# than quietly adopted; it is the weakest-grounded number in this script.
MULT_FLOOR = 0.5


def ceiling_for(n_cleared, ladder):
    tab = CEILING_LADDERS[ladder]
    return np.array([tab[n] for n in n_cleared], float)


def ladder_of(name):
    """The ladder a candidate was built with, recovered from its name: the ceiling
    candidates are keyed "a_hard_ceiling[band_ladder]" and so on. The multiplicative gate
    has no ladder and returns None."""
    return name.split("[")[1][:-1] if "[" in name else None


# ======================================================================================
# The candidates
# ======================================================================================
def build_candidates(df, tiers):
    """Returns {name: (score, meta)}. insurance_third is the SAME quantity as Option B's —
    mean of the three 0/100 indicators — used as a gate signal rather than an addend."""
    banking = 100.0 * (tiers["banking"] == 0)
    investment = 100.0 * (tiers["investment"] == 0)
    bits = np.column_stack([
        (tiers["insurance"] == 0).astype(float),
        df["income_protection_coverage"].to_numpy(float).round(),
        df["hospitalisation_coverage"].to_numpy(float).round(),
    ])
    n_cleared = bits.sum(axis=1).astype(int)
    third = 100.0 * bits.mean(axis=1)
    base = (banking + investment) / 2.0                 # 0 / 50 / 100 — only 3 values
    ob, _ = option_b(df, tiers)

    out = {}
    for ladder in CEILING_LADDERS:
        ceil = ceiling_for(n_cleared, ladder)
        out[f"a_hard_ceiling[{ladder}]"] = (np.minimum(base, ceil), {
            "shape": "min(mean(banking, investment), ceiling(insurance_third))",
            "ladder": ladder, "gates_what": "the two-domain base (3 possible values)"})
        out[f"c_optionb_capped[{ladder}]"] = (np.minimum(ob, ceil), {
            "shape": "min(option_b_score, ceiling(insurance_third))",
            "ladder": ladder,
            "gates_what": "Option B's own score (10 possible values before capping)"})
    out["b_multiplicative_gate"] = (base * (MULT_FLOOR + (1 - MULT_FLOOR) * third / 100.0), {
        "shape": f"mean(banking, investment) * ({MULT_FLOOR} + "
                 f"{1 - MULT_FLOOR} * insurance_third/100)",
        "ladder": None, "gates_what": "the two-domain base, multiplicatively",
        "grounding_warning": (f"the {MULT_FLOOR} floor is a pure judgement call with no "
                              f"independent grounding — see the module docstring")})
    return out, {"banking": banking, "investment": investment, "bits": bits,
                 "n_cleared": n_cleared, "third": third, "base": base, "option_b": ob}


def enumerate_candidate_levels(name, ladder):
    """Every value the formula can take, from the formula rather than the data, so an
    unreachable level shows as unreachable rather than merely absent."""
    vals = set()
    for b, i, c, p, h in itertools.product([0, 1], repeat=5):
        nc = c + p + h
        third = 100.0 * nc / 3.0
        base = 100.0 * (b + i) / 2.0
        ob = (100.0 * b + 100.0 * i + third) / 3.0
        if name.startswith("a_"):
            vals.add(round(min(base, CEILING_LADDERS[ladder][nc]), 4))
        elif name.startswith("c_"):
            vals.add(round(min(ob, CEILING_LADDERS[ladder][nc]), 4))
        else:
            vals.add(round(base * (MULT_FLOOR + (1 - MULT_FLOOR) * third / 100.0), 4))
    return sorted(vals)


# ======================================================================================
# The key check + the overcorrection check
# ======================================================================================
def protection_gap_check(df, tiers, scores):
    """THE specific failure being fixed: does 'Banking Stable + zero on all three insurance
    indicators' now sit BELOW the normal-population mean rather than above it?"""
    nd = (df["profile_type"] == "normal").to_numpy()
    ip = df["income_protection_coverage"].to_numpy(float).round()
    hp = df["hospitalisation_coverage"].to_numpy(float).round()
    gap = (tiers["banking"] == 0) & (tiers["insurance"] == 1) & (ip == 0) & (hp == 0)
    out = {"group_definition": ("Banking Stable + Insurance Not Covered + no income "
                               "protection + no hospitalisation"),
           "n": int(gap.sum()), "pct_of_population": round(100 * float(gap.mean()), 2),
           "per_formula": {}}
    for name, s in scores.items():
        nm, nmed = float(np.mean(s[nd])), float(np.median(s[nd]))
        gm, gmed = float(np.mean(s[gap])), float(np.median(s[gap]))
        out["per_formula"][name] = {
            "group_mean": round(gm, 2), "normal_mean": round(nm, 2),
            "group_minus_normal_mean": round(gm - nm, 2),
            "group_median": round(gmed, 2), "normal_median": round(nmed, 2),
            "group_minus_normal_median": round(gmed - nmed, 2),
            "pct_of_group_at_or_above_normal_mean": round(
                100 * float((s[gap] >= nm).mean()), 2),
            "pct_of_group_reaching_stable_band_70plus": round(
                100 * float((s[gap] >= 70).mean()), 2),
            "PASSES_below_normal_mean": bool(gm < nm),
        }
    return out


def overcorrection_check(df, scores, comp):
    """The opposite failure: has Insurance become the ONLY thing that matters?

    Three readings:
      1. realised influence — share of score variance attributable to each domain;
      2. compression — among users whose score is set by the cap, does banking/investment
         still order them at all (Spearman ~0 means their effort is invisible);
      3. headroom — among users with FULL insurance, does the score still separate
         banking/investment states."""
    nd = (df["profile_type"] == "normal").to_numpy()
    out = {}
    for name, s in scores.items():
        # What the cap acts ON differs by candidate: (a) caps the two-domain base, (c) caps
        # Option B's score, and naive / option_b / the multiplicative gate cap nothing.
        if name.startswith("a_"):
            uncapped_source = comp["base"]
            capped = uncapped_source > ceiling_for(comp["n_cleared"], ladder_of(name))
        elif name.startswith("c_"):
            uncapped_source = comp["option_b"]
            capped = uncapped_source > ceiling_for(comp["n_cleared"], ladder_of(name))
        else:
            uncapped_source = comp["base"]
            capped = np.zeros(len(s), bool)
        full = comp["n_cleared"] == 3
        ent = {
            "pct_of_users_score_set_by_the_cap": round(100 * float(capped.mean()), 2),
            "among_capped_spearman_score_vs_uncapped_base": (
                round(float(stats.spearmanr(s[capped], uncapped_source[capped]).statistic), 3)
                if capped.sum() > 50 and len(np.unique(s[capped])) > 1 else None),
            "among_capped_n_distinct_scores": int(len(np.unique(np.round(s[capped], 4))))
            if capped.sum() else 0,
            "among_full_insurance_spearman_score_vs_base": round(float(stats.spearmanr(
                s[full], comp["base"][full]).statistic), 3) if full.sum() > 50 else None,
            "among_full_insurance_n_distinct_scores": int(
                len(np.unique(np.round(s[full], 4)))),
            "spearman_score_vs_insurance_third_normals": round(float(stats.spearmanr(
                s[nd], comp["third"][nd]).statistic), 3),
            "spearman_score_vs_banking_normals": round(float(stats.spearmanr(
                s[nd], comp["banking"][nd]).statistic), 3),
            "spearman_score_vs_investment_normals": round(float(stats.spearmanr(
                s[nd], comp["investment"][nd]).statistic), 3),
        }
        if name == "b_multiplicative_gate":
            # Multiplicative -> not additively decomposable; report the correlation view only.
            ent["variance_decomposition"] = "n/a — not an additive score"
        else:
            ent["variance_decomposition"] = "n/a — min() is not additive; see correlations"
        ent["insurance_dominance_flag"] = bool(
            abs(ent["spearman_score_vs_insurance_third_normals"])
            > 1.35 * max(abs(ent["spearman_score_vs_banking_normals"]),
                         abs(ent["spearman_score_vs_investment_normals"])))
        out[name] = ent
    return out


def head_to_head(a, b, name_a, name_b):
    ba, bb = band_of(a), band_of(b)
    moved = ba != bb
    # Directional sanity: relative to the reference formula, do users with WORSE insurance
    # move down and users with BETTER insurance move up?
    return {
        "pair": f"{name_a} vs {name_b}",
        "spearman": round(float(stats.spearmanr(a, b).statistic), 4),
        "pearson": round(float(np.corrcoef(a, b)[0, 1]), 4),
        "kendall_tau_b": round(float(stats.kendalltau(a, b).statistic), 4),
        "mean_absolute_difference_points": round(float(np.mean(np.abs(a - b))), 2),
        "pct_users_changing_band": round(100 * float(moved.mean()), 2),
        "band_shares_a_pct": {k: round(100 * v, 2) for k, v in
                              pd.Series(ba).value_counts(normalize=True).items()},
        "band_shares_b_pct": {k: round(100 * v, 2) for k, v in
                              pd.Series(bb).value_counts(normalize=True).items()},
        "crosstab_counts": pd.crosstab(pd.Series(bb, name=name_b),
                                       pd.Series(ba, name=name_a)).to_dict(),
    }


def directional_check(comp, new, ref, name_new, name_ref):
    """Do band changes make sense? Uncovered users should move DOWN relative to the
    reference formula, covered users should not be penalised."""
    order = {"At Risk": 0, "Building": 1, "Stable": 2}
    dn = np.array([order[x] for x in band_of(new)])
    dr = np.array([order[x] for x in band_of(ref)])
    delta = dn - dr
    out = {"comparison": f"{name_new} relative to {name_ref}",
           "by_insurance_indicators_cleared": {}}
    for nc in range(4):
        m = comp["n_cleared"] == nc
        out["by_insurance_indicators_cleared"][nc] = {
            "n": int(m.sum()),
            "pct_moving_down_a_band": round(100 * float((delta[m] < 0).mean()), 2),
            "pct_unchanged": round(100 * float((delta[m] == 0).mean()), 2),
            "pct_moving_up_a_band": round(100 * float((delta[m] > 0).mean()), 2),
            "mean_band_change": round(float(delta[m].mean()), 3),
        }
    zero, full = comp["n_cleared"] == 0, comp["n_cleared"] == 3
    out["verdict_inputs"] = {
        "zero_insurance_mean_band_change": round(float(delta[zero].mean()), 3),
        "full_insurance_mean_band_change": round(float(delta[full].mean()), 3),
        "direction_is_correct": bool(delta[zero].mean() < 0 <= delta[full].mean()),
        "monotone_in_indicators_cleared": bool(all(
            delta[comp["n_cleared"] == i].mean() <= delta[comp["n_cleared"] == i + 1].mean()
            for i in range(3))),
    }
    return out


# ======================================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    models = load_models()
    manifest = models.pop("_manifest")

    print("=" * 116)
    print(f"GATED OVERALL-SCORE VALIDATION — Insurance as a non-compensatory gate, "
          f"{len(df):,} users")
    print("=" * 116)

    tiers = {}
    for d in DOMAINS:
        t, _, _ = assign_and_distance(df, models[d])
        tiers[d] = t
        rec = models[d]["shipped_tiers_summary"]["population_shares_pct"]
        rep = [round(100 * float((t == k).mean()), 1)
               for k in range(models[d]["shipped_tiers"])]
        ok = all(abs(x - y) <= 0.15 for x, y in zip(rec, rep))
        print(f"  [replay] {d:<11}{models[d]['combination']:<20} {rep}% vs artifact {rec}%"
              f"  -> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{d}: replay does not reproduce the locked artifact — aborting")

    cands, comp = build_candidates(df, tiers)
    nv, _ = naive(df, tiers)
    # comp["option_b"] is the same option_b(df, tiers) call build_candidates already made.
    scores = {"naive": nv, "option_b": comp["option_b"],
              **{k: v[0] for k, v in cands.items()}}
    normals = (df["profile_type"] == "normal").to_numpy()

    print("\n  insurance indicators cleared (0-3), population shares: "
          + "  ".join(f"{i}:{100 * float((comp['n_cleared'] == i).mean()):.1f}%"
                      for i in range(4)))
    print(f"  base = mean(banking, investment) takes "
          f"{len(np.unique(comp['base']))} values: {sorted(np.unique(comp['base']))}")

    out = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": ("Propose and validate INSURANCE-AS-A-GATE overall-score formulas, "
                    "fixing the compensatory failure overall_score_validation.py found in "
                    "Option B. Same validation suite, so naive / Option B / every gate "
                    "candidate are directly comparable."),
        "scope": "Analysis only. No production code or artifact modified.",
        "artifact_provenance": {
            "manifest_generated_utc": manifest["generated_utc"],
            "dataset_sha256": manifest["dataset"]["sha256"],
            "domains": {d: manifest["domains"][d]["combination"] for d in DOMAINS}},
        "insurance_third_definition": ("mean of three 0/100 indicators — "
                                       "insurance_tier == Covered, "
                                       "income_protection_coverage, "
                                       "hospitalisation_coverage — used ONLY as a gate "
                                       "signal, never as an addend"),
        "ceiling_ladders": CEILING_LADDERS,
        "ceiling_grounding": {
            "values_39_69_100": ("the project's own historical TIER_THRESHOLDS "
                                 "(0-39 At Risk / 40-69 Building / 70-100 Stable). The "
                                 "mechanism they belonged to is superseded but the "
                                 "boundaries are the only banding this project has ever "
                                 "written down."),
            "zero_coverage_must_not_read_as_stable": ("follows directly from the brief's "
                                                      "stated principle that critical gaps "
                                                      "surface regardless of overall score"),
            "JUDGEMENT_CALL_no_independent_grounding": (
                "the MAPPING from indicators-missing to which ceiling applies. Run at three "
                "harshness settings (lenient / band_ladder / strict) and reported as a "
                "sensitivity rather than presented as the answer."),
            "multiplicative_floor_0.5": ("PURE JUDGEMENT CALL. Nothing in this project "
                                         "supports 'zero insurance halves your score'. The "
                                         "weakest-grounded number here."),
        },
        "candidate_definitions": {k: v[1] for k, v in cands.items()},
        "population_shares_indicators_cleared_pct": {
            i: round(100 * float((comp["n_cleared"] == i).mean()), 2) for i in range(4)},
    }

    # ------------------------------------------------------------------ STEP 3a: levels
    print(f"\n{'=' * 116}\nDISTRIBUTION\n{'=' * 116}")
    dists = {}
    _, option_b_levels = enumerate_levels()
    for name, s in scores.items():
        if name == "naive":
            exp = NAIVE_LEVELS
        elif name == "option_b":
            exp = option_b_levels
        else:
            exp = enumerate_candidate_levels(name, ladder_of(name))
        dists[name] = distribution(s, name, exp)
        dists[name + "__normals"] = distribution(s[normals], name + "_normals", exp)
    out["step3_distribution"] = dists
    print(f"  {'formula':<32}{'levels':>8}{'reached':>9}{'modal%':>9}{'top3%':>8}"
          f"{'mean':>8}{'sd':>7}")
    for name in scores:
        d = dists[name]
        print(f"  {name:<32}{d.get('n_distinct_values_possible', 0):>8}"
              f"{d['n_distinct_values_observed']:>9}{d['modal_pct']:>9.2f}"
              f"{d['top3_levels_pct']:>8.1f}{d['mean']:>8.2f}{d['sd']:>7.2f}")

    # ------------------------------------------------------------------ STEP 3b: ordering
    inputs_gate = {"hospitalisation_coverage", "income_protection_coverage"}
    inputs_naive = inputs_gate | {"has_investments"}
    ords = {}
    for name, s in scores.items():
        ords[name] = ordering(df, s, name,
                              inputs_naive if name == "naive" else inputs_gate)
    out["step3_ordering"] = ords
    print(f"\n{'=' * 116}\nORDERING (normals only)\n{'=' * 116}")
    print("  The four variables reported for every formula so far "
          "(status is per-formula; see the prior report's correction):")
    print(f"  {'formula':<32}" + "".join(f"{v[:22]:>24}" for _, v in TEST_VARS))
    for name in scores:
        print(f"  {name:<32}" + "".join(
            f"{ords[name]['per_benchmark'][f'{d}.{v}']['spearman_rho']:>24.3f}"
            for d, v in TEST_VARS))
    print(f"\n  {'formula':<32}{'all-12 min':>12}{'all-12 mean':>13}"
          f"{'indep min':>11}{'indep mean':>12}{'n indep':>9}")
    for name in scores:
        o = ords[name]
        print(f"  {name:<32}{o['all_12']['min_rho']:>12.3f}{o['all_12']['mean_rho']:>13.3f}"
              f"{o['independent_only']['min_rho']:>11.3f}"
              f"{o['independent_only']['mean_rho']:>12.3f}"
              f"{o['independent_only']['n']:>9}")

    # ------------------------------------------------------------------ STEP 3c: archetypes
    arc = {}
    for p in ARCHETYPES:
        m = (df["profile_type"] == p).to_numpy()
        if not m.any():
            continue
        arc[p] = {"n": int(m.sum()),
                  **{name: {"mean": round(float(np.mean(s[m])), 2),
                            "median": round(float(np.median(s[m])), 2)}
                     for name, s in scores.items()}}
    for name in scores:
        for r, p in enumerate(sorted(arc, key=lambda x: -arc[x][name]["mean"])):
            arc[p][name]["rank"] = r + 1
    out["step3_archetypes"] = arc
    print(f"\n{'=' * 116}\nARCHETYPES — mean score (rank in brackets)\n{'=' * 116}")
    show = ["naive", "option_b", f"a_hard_ceiling[{PRIMARY_LADDER}]",
            "b_multiplicative_gate", f"c_optionb_capped[{PRIMARY_LADDER}]"]
    print(f"  {'profile':<22}{'n':>6}" + "".join(f"{n.split('[')[0][:16]:>18}" for n in show))
    for p in sorted(arc, key=lambda x: -arc[x][show[-1]]["mean"]):
        print(f"  {p:<22}{arc[p]['n']:>6}"
              + "".join(f"{arc[p][n]['mean']:>13.2f}({arc[p][n]['rank']})" for n in show))

    # ------------------------------------------------------------------ STEP 3d: KEY CHECK
    pg = protection_gap_check(df, tiers, scores)
    out["step3_KEY_CHECK_protection_gap"] = pg
    print(f"\n{'=' * 116}\nTHE KEY CHECK — 'Banking Stable + zero insurance coverage' "
          f"(n={pg['n']}, {pg['pct_of_population']}% of users)\n{'=' * 116}")
    print(f"  {'formula':<32}{'group mean':>12}{'normal mean':>13}{'difference':>12}"
          f"{'%>=norm mean':>14}{'% reaching 70+':>16}   verdict")
    for name in scores:
        r = pg["per_formula"][name]
        print(f"  {name:<32}{r['group_mean']:>12.2f}{r['normal_mean']:>13.2f}"
              f"{r['group_minus_normal_mean']:>+12.2f}"
              f"{r['pct_of_group_at_or_above_normal_mean']:>13.1f}%"
              f"{r['pct_of_group_reaching_stable_band_70plus']:>15.1f}%   "
              f"{'PASS' if r['PASSES_below_normal_mean'] else 'FAIL'}")

    # ------------------------------------------------------------------ overcorrection
    oc = overcorrection_check(df, scores, comp)
    out["step5_overcorrection_check"] = oc
    print(f"\n{'=' * 116}\nOVERCORRECTION CHECK — has Insurance become the ONLY thing that "
          f"matters?\n{'=' * 116}")
    print(f"  {'formula':<32}{'rho:ins':>9}{'rho:bank':>10}{'rho:invest':>12}"
          f"{'% capped':>10}{'capped rho vs base':>20}{'lvls|full ins':>15}")
    for name in scores:
        r = oc[name]
        cr = r["among_capped_spearman_score_vs_uncapped_base"]
        print(f"  {name:<32}{r['spearman_score_vs_insurance_third_normals']:>9.3f}"
              f"{r['spearman_score_vs_banking_normals']:>10.3f}"
              f"{r['spearman_score_vs_investment_normals']:>12.3f}"
              f"{r['pct_of_users_score_set_by_the_cap']:>9.1f}%"
              f"{(cr if cr is not None else float('nan')):>20.3f}"
              f"{r['among_full_insurance_n_distinct_scores']:>15}"
              + ("   <<< INSURANCE-DOMINATED" if r["insurance_dominance_flag"] else ""))

    # ------------------------------------------------------------------ STEP 4: head to head
    winner_key = f"c_optionb_capped[{PRIMARY_LADDER}]"
    h2h, direction = {}, {}
    for ref in ["naive", "option_b"]:
        for cand in [f"a_hard_ceiling[{PRIMARY_LADDER}]", "b_multiplicative_gate", winner_key]:
            h2h[f"{cand} vs {ref}"] = head_to_head(scores[ref], scores[cand], ref, cand)
            direction[f"{cand} vs {ref}"] = directional_check(
                comp, scores[cand], scores[ref], cand, ref)
    out["step4_head_to_head"] = h2h
    out["step4_directional_check"] = direction
    print(f"\n{'=' * 116}\nSTEP 4 — HEAD TO HEAD\n{'=' * 116}")
    print(f"  {'comparison':<52}{'spearman':>10}{'mean|d|':>9}{'band change':>13}"
          f"{'0-ins dband':>13}{'3-ins dband':>13}  dir")
    for k in h2h:
        r, d = h2h[k], direction[k]["verdict_inputs"]
        print(f"  {k:<52}{r['spearman']:>10.3f}{r['mean_absolute_difference_points']:>9.2f}"
              f"{r['pct_users_changing_band']:>12.1f}%"
              f"{d['zero_insurance_mean_band_change']:>+13.3f}"
              f"{d['full_insurance_mean_band_change']:>+13.3f}"
              f"  {'OK' if d['direction_is_correct'] else 'WRONG'}")

    # ------------------------------------------------------------------ aliasing (winner)
    out["step4_level_aliasing_winner"] = level_aliasing(
        scores[winner_key], scores[winner_key], comp["third"])["option_b"]
    al = out["step4_level_aliasing_winner"]
    print(f"\n  [aliasing, {winner_key}] {al['n_such_levels']} levels hold both "
          f"zero-insurance and full-insurance users at >5% each, covering "
          f"{al['pct_of_population_on_such_levels']}% of the population "
          f"(Option B: 2 levels / 26.81%; naive: 0 / 0%)")

    # ------------------------------------------------------------------ outputs
    per_user = df[["user_id", "age_band", "profile_type"]].copy()
    for d in DOMAINS:
        per_user[f"tier_{d}"] = tiers[d]
    per_user["income_protection_coverage"] = df["income_protection_coverage"]
    per_user["hospitalisation_coverage"] = df["hospitalisation_coverage"]
    per_user["insurance_indicators_cleared"] = comp["n_cleared"]
    per_user["insurance_third"] = np.round(comp["third"], 2)
    per_user["base_bank_invest"] = comp["base"]
    # Per-user output carries only the THREE formulas that remain decision-relevant: the
    # locked winner plus the two comparators it is judged against. The other six candidates
    # were an experimental sweep — their full results live in gated_score_validation.json
    # and candidate_summary.csv, which keep every formula. Trimmed July 2026 (the untrimmed
    # nine-formula file was 1,130 KB); re-add names here if a sweep column is needed again.
    # The __band columns are dropped too: a band is a pure function of the score
    # (0-39 At Risk / 40-69 Building / 70-100 Stable, band_of()), so storing it is
    # redundant and cost ~170 KB.
    per_user_formulas = ["naive", "option_b", f"c_optionb_capped[{PRIMARY_LADDER}]"]
    for name in per_user_formulas:
        per_user[name] = np.round(scores[name], 2)
    per_user.to_csv(os.path.join(OUT_DIR, "per_user_gated_scores.csv"), index=False)

    rows = []
    for name in scores:
        d, o, r, c = dists[name], ords[name], pg["per_formula"][name], oc[name]
        rows.append({
            "formula": name,
            "levels_possible": d.get("n_distinct_values_possible"),
            "levels_reached": d["n_distinct_values_observed"],
            "modal_pct": d["modal_pct"], "top3_pct": d["top3_levels_pct"],
            "mean": d["mean"], "sd": d["sd"],
            "ordering_indep_min_rho": o["independent_only"]["min_rho"],
            "ordering_indep_mean_rho": o["independent_only"]["mean_rho"],
            "ordering_all12_min_rho": o["all_12"]["min_rho"],
            "protection_gap_group_mean": r["group_mean"],
            "protection_gap_minus_normal_mean": r["group_minus_normal_mean"],
            "protection_gap_PASSES": r["PASSES_below_normal_mean"],
            "protection_first_rank": arc["protection_first"][name]["rank"],
            "protection_first_mean": arc["protection_first"][name]["mean"],
            "wealthy_exposed_rank": arc["wealthy_exposed"][name]["rank"],
            "underinsured_saver_rank": arc["underinsured_saver"][name]["rank"],
            "rho_vs_insurance": c["spearman_score_vs_insurance_third_normals"],
            "rho_vs_banking": c["spearman_score_vs_banking_normals"],
            "rho_vs_investment": c["spearman_score_vs_investment_normals"],
            "pct_score_set_by_cap": c["pct_of_users_score_set_by_the_cap"],
            "insurance_dominated": c["insurance_dominance_flag"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "candidate_summary.csv"), index=False)

    with open(os.path.join(OUT_DIR, "gated_score_validation.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    for n in ("gated_score_validation.json", "per_user_gated_scores.csv",
              "candidate_summary.csv"):
        print(f"\n  saved {OUT_DIR}/{n} "
              f"({os.path.getsize(os.path.join(OUT_DIR, n)) / 1024:.0f} KB)")
    print("=" * 116)


if __name__ == "__main__":
    main()
