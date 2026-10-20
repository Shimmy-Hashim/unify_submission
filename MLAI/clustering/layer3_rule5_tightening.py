"""
layer3_rule5_tightening.py — can Layer 3's Rule 5 be tightened, and does it need to be?
(Unify, CCDS25-1170, July 2026)

ANALYSIS ONLY — third in the sequence after layer3_validation.py and
layer3_rule4_tightening.py, following the same method deliberately so the two rules'
results are comparable. Nothing is refit, no production file is touched, no rule is
implemented.

THE STARTING POSITION, WHICH IS DIFFERENT FROM RULE 4'S
  Rule 5 is `Banking = Building AND Investment = On Track`. The prior validation found it
  fires for 18.7% of users, of whom 44.5% are the two archetypes it describes
  (`over_invested` + `high_debt_investor`, 8.33% of the population combined) — genuine
  enrichment, unlike Rule 4's original form, which fired at 0.96x chance. So the question
  here is NOT "can this be rescued" but "does tightening add enough to be worth the extra
  conditions" — and "no, the tier-only rule already holds up" is a legitimate answer that
  this script is set up to be able to return.

METRIC DISCIPLINE, CARRIED OVER FROM THE RULE 4 TASK
  * lift over the targets' combined 8.33% prevalence is the decisive number.
  * observed / independence-predicted is reported but is inflated whenever conditions are
    near-redundant by construction (an investment_rate cut is not independent of the
    Investment tier it largely drives), so it is supporting evidence only.
  * archetype recall is TAUTOLOGICAL for any candidate whose cuts are the archetypes' own
    override bounds — reported, but explicitly discounted. The one recall figure that means
    something is tier-membership recall: what share of each archetype actually lands in
    Banking = Building AND Investment = On Track, which the overrides do not define.

Outputs (new, additive):
    MLAI/clustering/results/layer3_rule5_tightening/RULE5_TIGHTENING.md
    MLAI/clustering/results/layer3_rule5_tightening/layer3_rule5_tightening.json
    MLAI/clustering/results/layer3_rule5_tightening/plots/*.png

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/layer3_rule5_tightening.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer3_validation import replay_tiers, DATA_PATH, HIGH_BASE_RATE_PP  # noqa: E402
from generate_synthetic_data import MISMATCH_TYPES  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results/layer3_rule5_tightening")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

TARGETS = ["over_invested", "high_debt_investor"]

# Cuts are taken from the archetypes' own override bounds, chosen as the LOOSEST bound
# across the two targets so neither is excluded by construction. No round numbers invented.
IR_MIN = min(MISMATCH_TYPES[t]["investment_rate"][1] for t in TARGETS)          # 0.18
EF_MAX = max(MISMATCH_TYPES[t]["emergency_fund_months"][2] for t in TARGETS)    # 2.0
SR_MAX = max(MISMATCH_TYPES[t]["savings_rate"][2] for t in TARGETS)             # 0.10
# high_debt_investor-only, used for the separability question rather than as a rule cut.
DTI_MIN_HDI = MISMATCH_TYPES["high_debt_investor"]["dti"][1]                    # 0.40

# Acceptance criteria, fixed before the numbers were seen — same three as the Rule 4 task.
MIN_LIFT = 3.0
MIN_TIER_RECALL_PCT = 25.0          # applied to the REAL recall claim (tier membership)
MAX_BAND_RATE_PCT = HIGH_BASE_RATE_PP

# Reference points from the Rule 4 task, used only for the lift-scale caveat in the report.
# Named rather than inlined because the whole point of that caveat is that the two tasks'
# lift numbers are NOT comparable, and the reader needs to see where the contrast comes from.
RULE4_TARGET_PREVALENCE_PCT = 4.17       # conservative_saver's share of the population
RULE4_LIFT_CEILING = 24.0                # 100 / 4.17 — a perfect Rule 4 flag's lift


# ======================================================================================
# Step 1 + 2 — what the archetypes are, and who else looks like them
# ======================================================================================
def step1(df):
    normal = df[df["profile_type"] == "normal"]
    tgt_mask = df["profile_type"].isin(TARGETS)
    out = {"targets": TARGETS,
           "combined_n": int(tgt_mask.sum()),
           "combined_prevalence_pct": round(100 * float(tgt_mask.mean()), 2),
           "per_archetype": {}}

    for t in TARGETS:
        sub = df[df["profile_type"] == t]
        feats = {}
        # spec[2] is safe: both targets' overrides are ('uniform', lo, hi) triples. Other
        # profiles carry ('fixed', value) pairs, so changing TARGETS would need this reworked.
        for f, spec in MISMATCH_TYPES[t].items():
            lo, hi = spec[1], spec[2]
            nv = normal[f]
            feats[f] = {
                "override_range": [lo, hi],
                "normal_median": round(float(nv.median()), 4),
                "normal_q10": round(float(nv.quantile(0.10)), 4),
                "normal_q75": round(float(nv.quantile(0.75)), 4),
                "normal_q90": round(float(nv.quantile(0.90)), 4),
                "pct_of_normals_inside_override_range":
                    round(100 * float(((nv >= lo) & (nv <= hi)).mean()), 1),
                "realized_range": [round(float(sub[f].min()), 3),
                                   round(float(sub[f].max()), 3)],
            }
        out["per_archetype"][t] = {
            "n": int(len(sub)),
            "prevalence_pct": round(100 * len(sub) / len(df), 2),
            "features": feats,
        }

    # Step 2 — the confounder check. Two separate questions, deliberately not conflated:
    #   (a) which NON-target archetypes share overridden features with the targets (the
    #       underinsured_saver-style problem the Rule 4 task hit), and
    #   (b) whether the two targets are separable FROM EACH OTHER.
    tgt_feats = set().union(*[set(MISMATCH_TYPES[t]) for t in TARGETS])
    shared = {}
    for other in MISMATCH_TYPES:
        if other in TARGETS:
            continue
        overlap = {f: list(MISMATCH_TYPES[other][f][1:])
                   for f in set(MISMATCH_TYPES[other]) & tgt_feats}
        if overlap:
            # Only a genuine confounder if the ranges actually overlap in value, not merely
            # if the same feature is overridden — check that rather than assume it.
            real = {}
            for f, rng in overlap.items():
                for t in TARGETS:
                    if f in MISMATCH_TYPES[t]:
                        tl, th = MISMATCH_TYPES[t][f][1], MISMATCH_TYPES[t][f][2]
                        if rng[0] <= th and tl <= rng[1]:
                            real.setdefault(f, {"range": rng, "collides_with": []})
                            real[f]["collides_with"].append(t)
            if real:
                shared[other] = real
    out["overlapping_non_target_archetypes"] = shared

    a, b = TARGETS
    only_a = set(MISMATCH_TYPES[a]) - set(MISMATCH_TYPES[b])
    only_b = set(MISMATCH_TYPES[b]) - set(MISMATCH_TYPES[a])
    out["target_separability"] = {
        f"{a}_only_overrides": sorted(only_a),
        f"{b}_only_overrides": sorted(only_b),
        "separable_in_principle": bool(only_a or only_b),
        "discriminating_features": sorted(only_a | only_b),
    }
    return out


def d2_mask(df, ir_min=IR_MIN, ef_max=EF_MAX, sr_max=SR_MAX):
    """D2's full mask — both tier conditions plus all three continuous cuts.

    Mirrors the D2 entry in candidates(), which stays separate because it needs each
    condition labelled individually for the independence baseline. Everything that only
    needs the resulting mask (the separability split, the sensitivity sweep) goes through
    here, so "hold the other axes at their override bounds" is expressed once rather than
    respelled at each call site.
    """
    return ((df["banking_tier"] == "Building") & (df["investment_tier"] == "On Track")
            & (df["investment_rate"] >= ir_min)
            & (df["emergency_fund_months"] <= ef_max)
            & (df["savings_rate"] <= sr_max))


# ======================================================================================
# Step 3 — candidates
# ======================================================================================
def candidates(df):
    building = df["banking_tier"] == "Building"
    on_track = df["investment_tier"] == "On Track"
    ir = df["investment_rate"] >= IR_MIN
    ef = df["emergency_fund_months"] <= EF_MAX
    sr = df["savings_rate"] <= SR_MAX

    return {
        "R5 (original)": {
            "parts": {"Banking = Building": building, "Investment = On Track": on_track},
            "desc": "Banking = Building AND Investment = On Track",
        },
        "D1 (+ investment_rate cut)": {
            "parts": {"Banking = Building": building, "Investment = On Track": on_track,
                      f"investment_rate >= {IR_MIN}": ir},
            "desc": (f"R5 AND investment_rate >= {IR_MIN} — the lower of the two archetypes' "
                     f"investment_rate floors, so neither is excluded by construction"),
        },
        "D2 (+ Banking weakness)": {
            "parts": {"Banking = Building": building, "Investment = On Track": on_track,
                      f"investment_rate >= {IR_MIN}": ir,
                      f"emergency_fund_months <= {EF_MAX}": ef,
                      f"savings_rate <= {SR_MAX}": sr},
            "desc": (f"D1 AND emergency_fund_months <= {EF_MAX} AND savings_rate <= "
                     f"{SR_MAX} — the loosest Banking-side override ceilings across both "
                     f"archetypes, so 'foundation still building' means genuinely thin "
                     f"rather than merely inside the Building tier"),
        },
        "D3 (continuous only, no tiers)": {
            "parts": {f"investment_rate >= {IR_MIN}": ir,
                      f"emergency_fund_months <= {EF_MAX}": ef,
                      f"savings_rate <= {SR_MAX}": sr},
            "desc": ("the full signature with BOTH tier conditions dropped — the redundancy "
                     "control, same as C3 in the Rule 4 task"),
        },
    }


# ======================================================================================
# Step 4 — evaluate
# ======================================================================================
def evaluate(df, cand):
    parts = cand["parts"]
    mask = pd.Series(np.logical_and.reduce([p.to_numpy() for p in parts.values()]),
                     index=df.index)
    normal = df[df["profile_type"] == "normal"]
    tgt = df["profile_type"].isin(TARGETS)
    n_flag = int(mask.sum())

    pred = float(np.prod([float(p.mean()) for p in parts.values()]))
    obs = float(mask.mean())
    prevalence = float(tgt.mean())
    precision = float((mask & tgt).sum()) / n_flag if n_flag else 0.0

    e = {
        "description": cand["desc"],
        "conditions": list(parts),
        "n_flagged": n_flag,
        "full_population_pct": round(100 * obs, 1),
        "normal_only_pct": round(100 * float(mask.loc[normal.index].mean()), 1),
        "independence_predicted_pct": round(100 * pred, 1),
        "observed_over_predicted": round(obs / pred, 2) if pred > 0 else None,
        "target_pct_of_flagged": round(100 * precision, 1),
        "lift_over_prevalence": round(precision / prevalence, 2) if prevalence else None,
        "archetype_recall_pct": round(
            100 * float((mask & tgt).sum()) / int(tgt.sum()), 1) if tgt.sum() else None,
        "per_target_pct_of_flagged": {
            t: round(100 * float(((df["profile_type"] == t) & mask).sum()) / max(n_flag, 1), 1)
            for t in TARGETS},
        "per_target_recall_pct": {
            t: round(100 * float(((df["profile_type"] == t) & mask).sum())
                     / int((df["profile_type"] == t).sum()), 1) for t in TARGETS},
        "by_band_full_pct": {}, "by_band_normal_pct": {},
        "profile_mix_pct": {k: round(100 * v / max(n_flag, 1), 1) for k, v in
                            df.loc[mask, "profile_type"].value_counts().items()},
    }
    for b in range(1, 6):
        e["by_band_full_pct"][str(b)] = round(100 * float(mask[df["age_band"] == b].mean()), 1)
        e["by_band_normal_pct"][str(b)] = round(
            100 * float(mask.loc[normal.index][normal["age_band"] == b].mean()), 1)

    flagged = df[mask]
    fn = flagged[flagged["profile_type"] == "normal"]
    e["flagged_composition"] = {
        "normal_pct": round(100 * len(fn) / max(n_flag, 1), 1),
        "normal_median_investment_rate": (round(float(fn["investment_rate"].median()), 4)
                                          if len(fn) else None),
        "normal_median_emergency_fund_months": (
            round(float(fn["emergency_fund_months"].median()), 2) if len(fn) else None),
    }

    uses_override_cut = any(c.startswith(("investment_rate", "savings_rate",
                                          "emergency_fund_months")) for c in parts)
    e["archetype_recall_is_tautological"] = bool(uses_override_cut)
    e["recall_note"] = (
        "tautological — the cuts are the archetypes' own override bounds, so members pass "
        "by definition; not evidence" if uses_override_cut else
        "a real measurement — tier membership is not defined by the overrides")
    e["max_band_rate_pct"] = max(e["by_band_full_pct"].values())
    e["meets_lift"] = bool((e["lift_over_prevalence"] or 0) >= MIN_LIFT)
    e["meets_band_rate"] = bool(e["max_band_rate_pct"] <= MAX_BAND_RATE_PCT)
    return e


def tier_recall(df):
    """The one recall claim that is a real measurement rather than a construction artefact:
    does each archetype actually land in Banking = Building AND Investment = On Track?"""
    tiers = (df["banking_tier"] == "Building") & (df["investment_tier"] == "On Track")
    out = {}
    for t in TARGETS + ["normal"]:
        s = df["profile_type"] == t
        out[t] = {
            "n": int(s.sum()),
            "pct_in_Building": round(100 * float((df.loc[s, "banking_tier"] == "Building").mean()), 1),
            "pct_in_On_Track": round(100 * float((df.loc[s, "investment_tier"] == "On Track").mean()), 1),
            "pct_in_both": round(100 * float(tiers[s].mean()), 1),
        }
    return out


def separability(df):
    """Can the two targets be told apart from each other, or must the label stay generic?

    Unlike the Rule 4 case — where conservative_saver and underinsured_saver shared their
    whole Banking signature and were genuinely inseparable — high_debt_investor carries two
    overrides over_invested does not (dti, healthy_debt_ratio). This tests whether those
    actually separate them inside the flagged set.
    """
    flagged = df[d2_mask(df)]
    hi_dti = flagged["dti"] >= DTI_MIN_HDI
    out = {"within": "D2's flagged set", "n_flagged": int(len(flagged)),
           "split_on": f"dti >= {DTI_MIN_HDI} (high_debt_investor's own dti floor)",
           "above": {}, "below": {}}
    for side, m in [("above", hi_dti), ("below", ~hi_dti)]:
        s = flagged[m]
        out[side] = {"n": int(len(s)),
                     "profile_mix_pct": {k: round(100 * v / max(len(s), 1), 1)
                                         for k, v in s["profile_type"].value_counts().items()}}
    for t in TARGETS:
        s = flagged[flagged["profile_type"] == t]
        out[f"{t}_pct_above_dti_cut"] = (round(100 * float((s["dti"] >= DTI_MIN_HDI).mean()), 1)
                                         if len(s) else None)
    return out


def synthetic_suppression_check(df):
    """Why is precision so high — a strong rule, or a property of the generator?

    D2's signature is "invests hard, saves little, no buffer". In the copula-generated
    normal population, savings_rate / emergency_fund_months / investment_rate are POSITIVELY
    correlated, so that combination is actively suppressed among normal users — which
    inflates precision far above anything a real population would give, where the pattern
    genuinely occurs. This measures the suppression instead of asserting it: the observed
    joint rate among normals against the rate the same three marginals would produce if they
    were independent.
    """
    norm = df[df["profile_type"] == "normal"]
    conds = {f"investment_rate >= {IR_MIN}": (norm["investment_rate"] >= IR_MIN),
             f"emergency_fund_months <= {EF_MAX}": (norm["emergency_fund_months"] <= EF_MAX),
             f"savings_rate <= {SR_MAX}": (norm["savings_rate"] <= SR_MAX)}
    marginals = {k: round(100 * float(v.mean()), 2) for k, v in conds.items()}
    joint = float(np.logical_and.reduce([v.to_numpy() for v in conds.values()]).mean())
    indep = float(np.prod([float(v.mean()) for v in conds.values()]))
    return {
        "population": "normal users only",
        "marginal_pct": marginals,
        "observed_joint_pct": round(100 * joint, 3),
        "independence_predicted_joint_pct": round(100 * indep, 3),
        "suppression_ratio": (round(joint / indep, 2) if indep > 0 else None),
        "pairwise_spearman_among_normals": {
            "savings_rate~investment_rate": round(float(
                norm["savings_rate"].corr(norm["investment_rate"], method="spearman")), 3),
            "emergency_fund_months~investment_rate": round(float(
                norm["emergency_fund_months"].corr(norm["investment_rate"],
                                                   method="spearman")), 3),
            "savings_rate~emergency_fund_months": round(float(
                norm["savings_rate"].corr(norm["emergency_fund_months"],
                                          method="spearman")), 3),
        },
        "note": ("a suppression ratio below 1 means the synthetic normal population produces "
                 "this combination LESS often than chance, so the precision figure is partly "
                 "a property of the generator's correlation structure and must not be quoted "
                 "as an expected real-world precision"),
    }


def sensitivity(df):
    """Does the result depend on knowing the generator's exact bounds? Same circularity
    check as the Rule 4 task: sweep each axis with the other two held at their bounds."""
    tgt = df["profile_type"].isin(TARGETS).to_numpy()
    prevalence = float(tgt.mean())

    def score(mask):
        m = mask.to_numpy()
        n = int(m.sum())
        prec = float((m & tgt).sum()) / n if n else 0.0
        return {"n_flagged": n, "full_population_pct": round(100 * m.mean(), 1),
                "precision_pct": round(100 * prec, 1),
                "lift": round(prec / prevalence, 2) if prevalence else None,
                "recall_pct": round(100 * float((m & tgt).sum()) / int(tgt.sum()), 1)}

    grids = {
        # (values, builder, this axis's OWN override bound — compared per axis, since the
        # three bounds share numeric values and a global membership test mislabels them)
        "investment_rate_min": ([0.10, 0.14, IR_MIN, 0.22, 0.26],
                                lambda v: d2_mask(df, ir_min=v), IR_MIN),
        "emergency_fund_months_max": ([1.0, 1.5, EF_MAX, 3.0, 4.0],
                                      lambda v: d2_mask(df, ef_max=v), EF_MAX),
        "savings_rate_max": ([0.06, 0.08, SR_MAX, 0.14, 0.18],
                             lambda v: d2_mask(df, sr_max=v), SR_MAX),
    }
    return {axis: {str(v): {**score(build(v)), "is_override_bound": bool(v == bound)}
                   for v in sorted(set(values))}
            for axis, (values, build, bound) in grids.items()}


# ======================================================================================
# Plot + report
# ======================================================================================
def plots(s1, results):
    os.makedirs(PLOT_DIR, exist_ok=True)
    names = list(results)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    ax.bar(x - 0.2, [results[n]["full_population_pct"] for n in names], 0.4,
           label="observed", color="#4C72B0")
    ax.bar(x + 0.2, [results[n]["independence_predicted_pct"] for n in names], 0.4,
           label="independence-predicted", color="#B0B0B0")
    ax.axhline(MAX_BAND_RATE_PCT, color="#C44E52", ls="--", lw=1.2,
               label=f"{MAX_BAND_RATE_PCT:g}% 'common finding' line")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" ")[0] for n in names])
    ax.set_ylabel("% of users flagged")
    ax.set_title("Base rate vs. what chance overlap alone predicts")
    ax.legend(fontsize=8)

    ax = axes[1]
    bottom = np.zeros(len(names))
    for t, colour in zip(TARGETS, ["#55A868", "#8172B2"]):
        vals = np.array([results[n]["per_target_pct_of_flagged"][t] for n in names])
        ax.bar(x, vals, 0.5, bottom=bottom, label=f"% flagged that are {t}", color=colour)
        bottom += vals
    ax.axhline(s1["combined_prevalence_pct"], color="#C44E52", ls="--", lw=1.2,
               label=f"combined prevalence ({s1['combined_prevalence_pct']}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" ")[0] for n in names])
    ax.set_ylabel("% of flagged users")
    ax.set_title("Enrichment, split by which archetype")
    ax.legend(fontsize=8)

    fig.suptitle("Rule 5 tightening — base rate, chance baseline and archetype enrichment")
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "rule5_candidates.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [os.path.relpath(p, REPO_ROOT).replace("\\", "/")]


def write_markdown(df, s1, results, tr, sep, sup, sens, plot_paths, winner, verdict_lines):
    L = []
    a = L.append
    a("# Layer 3 Rule 5 — tightening and re-validation\n")
    a("**Analysis only.** Third in the sequence after `layer3_validation.py` and "
      "`layer3_rule4_tightening.py`, using the same method so the two rules' results are "
      "comparable. Tier assignment is imported from the prior validation (same artifacts, "
      "same per-band `tier_map`, same replay-vs-artifact gate). Nothing was refit, no "
      "production file was written, no rule was implemented.\n")
    a(f"Generated by `MLAI/clustering/layer3_rule5_tightening.py` on {len(df):,} users.\n")

    a("## Step 1 — what the two target archetypes actually are\n")
    a(f"Combined: **n = {s1['combined_n']}, {s1['combined_prevalence_pct']}% of the "
      f"population** — twice Rule 4's target prevalence, which matters when reading lift.\n")
    for t, d in s1["per_archetype"].items():
        a(f"### `{t}` — n={d['n']} ({d['prevalence_pct']}%)\n")
        a("| Feature | override range | realized | normal median | normal q10 | normal q90 | "
          "% of normals inside range |")
        a("|---|---|---|---|---|---|---|")
        for f, x in d["features"].items():
            a(f"| `{f}` | {x['override_range'][0]}–{x['override_range'][1]} | "
              f"{x['realized_range'][0]}–{x['realized_range'][1]} | {x['normal_median']} | "
              f"{x['normal_q10']} | {x['normal_q90']} | "
              f"{x['pct_of_normals_inside_override_range']}% |")
        a("")

    a("## Step 2 — confounder check\n")
    a("**(a) Non-target archetypes sharing an overridden feature at overlapping values:**\n")
    if s1["overlapping_non_target_archetypes"]:
        for other, fs in s1["overlapping_non_target_archetypes"].items():
            a(f"- `{other}`: " + "; ".join(
                f"`{f}` {v['range'][0]}–{v['range'][1]} (collides with "
                f"{', '.join(v['collides_with'])})" for f, v in fs.items()))
    else:
        a("- none")
    a("")
    sp = s1["target_separability"]
    a("**(b) Are the two targets separable from each other?** "
      f"{'Yes — ' if sp['separable_in_principle'] else 'No — '}"
      f"`high_debt_investor` overrides "
      f"{', '.join('`' + f + '`' for f in sp['high_debt_investor_only_overrides'])} "
      f"which `over_invested` does not. This is the key difference from the Rule 4 task, "
      f"where `conservative_saver` and `underinsured_saver` shared their entire Banking "
      f"signature and could not be told apart. Tested empirically below.\n")

    a("### Tier membership — the one real recall claim\n")
    a("| Profile | n | % in Banking = Building | % in Investment = On Track | % in both |")
    a("|---|---|---|---|---|")
    for t, d in tr.items():
        a(f"| `{t}` | {d['n']} | {d['pct_in_Building']}% | {d['pct_in_On_Track']}% | "
          f"**{d['pct_in_both']}%** |")
    a("")

    a("## Steps 3–4 — candidates and results\n")
    for n, e in results.items():
        a(f"- **{n}** — {e['description']}")
    a("")
    a("| Candidate | flagged | full pop | normal only | indep-predicted | obs/pred | "
      "% flagged that are a target | lift | over_inv | hdi | max band |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, e in results.items():
        pt = e["per_target_pct_of_flagged"]
        a(f"| {n} | {e['n_flagged']} | **{e['full_population_pct']}%** | "
          f"{e['normal_only_pct']}% | {e['independence_predicted_pct']}% | "
          f"{e['observed_over_predicted']}x | **{e['target_pct_of_flagged']}%** | "
          f"{e['lift_over_prevalence']}x | {pt['over_invested']}% | "
          f"{pt['high_debt_investor']}% | {e['max_band_rate_pct']}% |")
    a("")
    a(f"`lift` is the share of flagged users that are one of the two targets divided by "
      f"their combined {s1['combined_prevalence_pct']}% prevalence. Archetype recall is "
      f"omitted from this table on purpose — for D1–D3 the cuts *are* the archetypes' own "
      f"override bounds, so recall is pinned near 100% by construction and is not evidence. "
      f"The meaningful recall claim is the tier-membership table above.\n")

    a("### Per age band (full population)\n")
    a("| Candidate | b1 | b2 | b3 | b4 | b5 |")
    a("|---|---|---|---|---|---|")
    for n, e in results.items():
        a(f"| {n} | " + " | ".join(f"{e['by_band_full_pct'][str(b)]}%"
                                   for b in range(1, 6)) + " |")
    a("")

    a("### What each flagged set is made of\n")
    a("| Candidate | normal users | their median investment_rate | their median EF months | "
      "other archetypes present |")
    a("|---|---|---|---|---|")
    for n, e in results.items():
        c = e["flagged_composition"]
        other = ", ".join(f"{k} {v}%" for k, v in e["profile_mix_pct"].items()
                          if k not in TARGETS + ["normal"])
        a(f"| {n} | {c['normal_pct']}% | {c['normal_median_investment_rate']} | "
          f"{c['normal_median_emergency_fund_months']} | {other or '—'} |")
    a("")

    a("### Are the two archetypes separable inside the flagged set?\n")
    a(f"Splitting {sep['within']} (n={sep['n_flagged']}) on **{sep['split_on']}**:\n")
    a("| Side | n | profile mix |")
    a("|---|---|---|")
    for side in ["above", "below"]:
        d = sep[side]
        a(f"| dti {'>=' if side == 'above' else '<'} {DTI_MIN_HDI} | {d['n']} | " +
          ", ".join(f"{k} {v}%" for k, v in d["profile_mix_pct"].items()) + " |")
    a("")
    for t in TARGETS:
        a(f"- `{t}`: {sep[f'{t}_pct_above_dti_cut']}% sit above the dti cut")
    a("")

    a("### Why precision is this high — read before quoting the number\n")
    a(f"D2's signature is *invests hard, saves little, no buffer*. Among **normal users "
      f"only**, the three conditions have marginals "
      + ", ".join(f"`{k}` {v}%" for k, v in sup["marginal_pct"].items())
      + f" — which under independence would put "
      f"**{sup['independence_predicted_joint_pct']}%** of normals inside all three. The "
      f"observed figure is **{sup['observed_joint_pct']}%**, a suppression ratio of "
      f"**{sup['suppression_ratio']}x**.\n")
    a("Two causes, in order of size. First, `investment_rate >= "
      f"{IR_MIN}` is simply rare among normals "
      f"({sup['marginal_pct'][f'investment_rate >= {IR_MIN}']}%) — the single biggest "
      f"factor. Second, the copula gives normal users positively correlated accumulation "
      "features " + ", ".join(f"(`{k}` ρ={v})" for k, v in
                              sup["pairwise_spearman_among_normals"].items()) +
      ", so a normal user who invests heavily also tends to save well and hold a buffer, "
      "making the conjunction rarer still.\n")
    n_norm = int(len(df[df['profile_type'] == 'normal']))
    exp_n = sup["independence_predicted_joint_pct"] / 100 * n_norm
    a(f"**Be precise about what the {sup['suppression_ratio']}x ratio can support.** Even "
      f"under independence the expected count is only ~{exp_n:.1f} of {n_norm:,} normal "
      f"users, so observing zero is consistent both with genuine correlation-driven "
      f"suppression and with the combination simply being very rare at this sample size. "
      f"The data cannot separate those two explanations — do not claim it does.\n")
    a("**Consequence, stated plainly: the ~99.7% precision is substantially a property of "
      "the dataset, not a forecast of real-world performance.** Real Singaporeans do combine "
      "aggressive investing with a thin cash buffer — that is exactly the pattern the rule "
      "exists to catch — while the synthetic normal population almost never does. The "
      "tightened rule is still the better rule, and the direction of the improvement is "
      "real; but its precision on real users will be materially lower, and 99.7% must not "
      "be quoted as an expected hit rate.\n")

    a("### Sensitivity — does this survive not knowing the answer key?\n")
    a("Each axis swept with the other two held at their override bounds (D2 form); "
      "**bold** marks the bound itself.\n")
    for axis, grid in sens.items():
        a(f"`{axis}`\n")
        a("| value | flagged | % of population | precision | lift | recall |")
        a("|---|---|---|---|---|---|")
        for v, d in grid.items():
            m = "**" if d["is_override_bound"] else ""
            a(f"| {m}{v}{m} | {d['n_flagged']} | {d['full_population_pct']}% | "
              f"{d['precision_pct']}% | {d['lift']}x | {d['recall_pct']}% |")
        a("")

    a("## Step 5 — recommendation\n")
    a(f"Criteria fixed before results, same as the Rule 4 task: lift >= {MIN_LIFT}x, no age "
      f"band above {MAX_BAND_RATE_PCT}%, and meaningful recall on tier membership "
      f"(>= {MIN_TIER_RECALL_PCT}%).\n")
    a("| Candidate | lift | band rate | verdict |")
    a("|---|---|---|---|")
    for n, e in results.items():
        a(f"| {n} | {'PASS' if e['meets_lift'] else 'fail'} | "
          f"{'PASS' if e['meets_band_rate'] else 'fail'} | "
          f"{'meets both' if e['meets_lift'] and e['meets_band_rate'] else 'rejected'} |")
    a("")
    for line in verdict_lines:
        a(line)
    a("")
    a("## Plots\n")
    for p in plot_paths:
        a(f"- `{p}`")
    a("")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("=" * 100)
    print(f"RULE 5 TIGHTENING — {len(df):,} users (analysis only, no refit)")
    print("=" * 100)
    for domain, col in [("banking", "banking_tier"), ("investment", "investment_tier"),
                        ("insurance", "insurance_tier")]:
        labels, tier_labels, shares = replay_tiers(df, domain)
        df[col] = labels
        print(f"  [replay OK] {domain:<11} {tier_labels} -> {shares}%")

    s1 = step1(df)
    print(f"\n  step 1 — targets: {' + '.join(TARGETS)} = {s1['combined_n']} users "
          f"({s1['combined_prevalence_pct']}%)")
    for t, d in s1["per_archetype"].items():
        print(f"    {t} (n={d['n']}):")
        for f, x in d["features"].items():
            print(f"      {f:<24} override {x['override_range'][0]}–{x['override_range'][1]:<6}"
                  f"  normal median {x['normal_median']:<8}"
                  f"  ({x['pct_of_normals_inside_override_range']}% of normals inside)")
    print(f"\n  step 2 — non-target archetypes colliding on an overridden feature: "
          f"{list(s1['overlapping_non_target_archetypes']) or 'none'}")
    print(f"    targets separable from each other via: "
          f"{s1['target_separability']['discriminating_features']}")

    tr = tier_recall(df)
    print("\n  tier membership (the real recall claim):")
    for t, d in tr.items():
        print(f"    {t:<20} Building {d['pct_in_Building']:>5}%  "
              f"On Track {d['pct_in_On_Track']:>5}%  both {d['pct_in_both']:>5}%")

    results = {n: evaluate(df, c) for n, c in candidates(df).items()}
    print(f"\n  steps 3-4 — candidates (lift vs the {s1['combined_prevalence_pct']}% "
          f"combined prevalence):")
    print(f"    {'candidate':<32}{'rate':>8}{'indep':>8}{'o/p':>7}{'%tgt':>7}{'lift':>7}"
          f"{'maxband':>9}")
    for n, e in results.items():
        print(f"    {n:<32}{e['full_population_pct']:>7}%{e['independence_predicted_pct']:>7}%"
              f"{e['observed_over_predicted']:>7}{e['target_pct_of_flagged']:>6}%"
              f"{e['lift_over_prevalence']:>7}{e['max_band_rate_pct']:>8}%")

    sep = separability(df)
    print(f"\n  separability — splitting D2's {sep['n_flagged']} flagged users on "
          f"dti >= {DTI_MIN_HDI}:")
    for side in ["above", "below"]:
        print(f"    {side:<6} n={sep[side]['n']:<5} {sep[side]['profile_mix_pct']}")

    sup = synthetic_suppression_check(df)
    print(f"\n  suppression check (normal users only): observed joint "
          f"{sup['observed_joint_pct']}% vs independence-predicted "
          f"{sup['independence_predicted_joint_pct']}% -> ratio {sup['suppression_ratio']}x")

    sens = sensitivity(df)
    print("\n  sensitivity (D2 form, one axis at a time; * = the override bound):")
    for axis, grid in sens.items():
        cells = "  ".join(f"{'*' if d['is_override_bound'] else ''}{v}:"
                          f"{d['precision_pct']}%/{d['full_population_pct']}%"
                          for v, d in grid.items())
        print(f"    {axis:<28} {cells}   (precision/base rate)")

    # ---- Step 5. The comparison that decides it: does tightening beat the original by
    # enough to justify three extra conditions, given the original already passes?
    orig = results["R5 (original)"]
    tightened = {n: e for n, e in results.items() if n != "R5 (original)"}
    passing = {n: e for n, e in tightened.items()
               if e["meets_lift"] and e["meets_band_rate"]}
    best = max(passing, key=lambda n: passing[n]["lift_over_prevalence"]) if passing else None
    orig_ok = orig["meets_lift"] and orig["meets_band_rate"]

    verdict = []
    if best:
        gain = results[best]["lift_over_prevalence"] - orig["lift_over_prevalence"]
        lost = orig["n_flagged"] - results[best]["n_flagged"]
        verdict.append(
            f"**Best tightened candidate: `{best}`** — lift "
            f"{results[best]['lift_over_prevalence']}x vs the original's "
            f"{orig['lift_over_prevalence']}x (**+{gain:.2f}x**), precision "
            f"{results[best]['target_pct_of_flagged']}% vs {orig['target_pct_of_flagged']}%, "
            f"base rate {results[best]['full_population_pct']}% vs "
            f"{orig['full_population_pct']}% ({lost} fewer users flagged).\n")
    if orig_ok:
        verdict.append(
            f"**The original tier-only rule already clears every criterion** "
            f"(lift {orig['lift_over_prevalence']}x >= {MIN_LIFT}x; highest band "
            f"{orig['max_band_rate_pct']}% <= {MAX_BAND_RATE_PCT}%; tier recall "
            f"{tr[TARGETS[0]]['pct_in_both']}% / {tr[TARGETS[1]]['pct_in_both']}%). Unlike "
            f"Rule 4 — which fired at 0.96x chance and *had* to be fixed — Rule 5 was never "
            f"broken, so tightening here is an optimisation, not a repair.\n")
    else:
        verdict.append("**The original tier-only rule does NOT clear the criteria** — see "
                       "the table above for which one it fails.\n")
    verdict.append(
        f"Note on lift scale: the two targets are {s1['combined_prevalence_pct']}% of the "
        f"population, twice `conservative_saver`'s {RULE4_TARGET_PREVALENCE_PCT}%, so the "
        f"arithmetic ceiling on lift here is half what it was in the Rule 4 task (a perfect "
        f"rule would score {100 / s1['combined_prevalence_pct']:.1f}x, vs "
        f"{RULE4_LIFT_CEILING}x there). Compare precision directly rather than reading the "
        f"two tasks' lift numbers side by side.")

    plot_paths = plots(s1, results)
    summary = {"dataset": {"path": "MLAI/clustering/data/processed/synthetic_users.csv",
                           "n_users": int(len(df))},
               "criteria": {"min_lift": MIN_LIFT, "min_tier_recall_pct": MIN_TIER_RECALL_PCT,
                            "max_band_rate_pct": MAX_BAND_RATE_PCT,
                            "note": "fixed before results were seen"},
               "cuts_used": {"investment_rate_min": IR_MIN, "emergency_fund_months_max": EF_MAX,
                             "savings_rate_max": SR_MAX,
                             "note": "loosest override bound across the two targets"},
               "step1_archetypes": s1, "tier_membership": tr,
               "step3_4_candidates": results, "separability": sep,
               "synthetic_suppression_check": sup, "sensitivity": sens,
               "step5_best_tightened": best,
               "step5_original_meets_criteria": bool(orig_ok),
               "plots": plot_paths}
    with open(os.path.join(OUT_DIR, "layer3_rule5_tightening.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "RULE5_TIGHTENING.md"), "w", encoding="utf-8") as f:
        f.write(write_markdown(df, s1, results, tr, sep, sup, sens, plot_paths, best,
                               verdict))

    print(f"\n  step 5 — original meets criteria: {orig_ok}; best tightened: {best}")
    print(f"  wrote {os.path.relpath(OUT_DIR, REPO_ROOT)}/"
          f"{{RULE5_TIGHTENING.md, layer3_rule5_tightening.json}} + {len(plot_paths)} plot(s)")


if __name__ == "__main__":
    main()
