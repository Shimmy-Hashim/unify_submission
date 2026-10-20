"""
layer3_rule4_tightening.py — can Layer 3's Rule 4 be tightened into a real signal?
(Unify, CCDS25-1170, July 2026)

ANALYSIS ONLY — extends layer3_validation.py, does not repeat it. Nothing is refit, no
production file is touched, and no Layer 3 rule is implemented.

THE PROBLEM THIS ADDRESSES
  Rule 4 as proposed is `Banking = Stable AND Investment = Getting Started`. The prior
  validation found it fires for 28.5% of users against an independence-predicted 29.7%
  (47.9% x 61.9%) — i.e. the conjunction of two common tiers, carrying essentially no
  information beyond the fact that both tiers are common. The archetype it is meant to
  describe, `conservative_saver`, is only 4.17% of the population.

WHAT IS TESTED
  Step 1  `conservative_saver`'s ACTUAL generator overrides, read straight out of
          MISMATCH_TYPES rather than guessed, and compared against the normal population's
          distribution on the same features. Every threshold used below is derived from
          those override bounds — no round numbers were invented.
  Step 2  Three tightened candidates, built to separate the contribution of the TIER
          conditions from the contribution of the CONTINUOUS signature.
  Step 3  For each: base rate (full / normal-only / per band), enrichment for
          `conservative_saver` (precision, lift, recall), and the independence-predicted
          rate, exactly as the prior validation framed it.

READ THE TWO METRICS DIFFERENTLY — they answer different questions, and the second one is
the decisive one:
  * observed / independence-predicted asks "is this conjunction more than chance overlap of
    its parts?". It is inflated whenever the conditions are near-redundant by construction
    (an investment_rate cut is not independent of the Investment tier, which investment_rate
    largely drives), so a high ratio here is necessary but nowhere near sufficient.
  * lift over the archetype's 4.17% prevalence asks "does this actually find the users it
    claims to describe?". This is the one a recommendation should rest on.

Outputs (new, additive):
    MLAI/clustering/results/layer3_rule4_tightening/RULE4_TIGHTENING.md
    MLAI/clustering/results/layer3_rule4_tightening/layer3_rule4_tightening.json
    MLAI/clustering/results/layer3_rule4_tightening/plots/*.png

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/layer3_rule4_tightening.py
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
# Tier assignment is imported from the prior validation, not reimplemented — same artifacts,
# same per-band tier_map path, same replay-vs-artifact gate.
from layer3_validation import replay_tiers, DATA_PATH, BANDS, HIGH_BASE_RATE_PP  # noqa: E402
from generate_synthetic_data import MISMATCH_TYPES, INVESTOR_PROB  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results/layer3_rule4_tightening")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

TARGET = "conservative_saver"
OVERRIDES = MISMATCH_TYPES[TARGET]          # the generator's own definition of the archetype

# Every cut below is an override BOUND, not a chosen number:
IR_MAX = OVERRIDES["investment_rate"][2]            # 0.03 — the archetype's investment_rate ceiling
SR_MIN = OVERRIDES["savings_rate"][1]               # 0.28 — its savings_rate floor
EF_MIN = OVERRIDES["emergency_fund_months"][1]      # 5.0  — its emergency_fund_months floor

# Recommendation criteria, fixed BEFORE seeing the numbers so the verdict is not fitted to
# whichever candidate happens to look best.
MIN_LIFT = 3.0                 # >= 3x the archetype's population prevalence
MIN_RECALL_PCT = 25.0          # a rule that fires for almost none of the archetype is useless
MAX_BAND_RATE_PCT = HIGH_BASE_RATE_PP   # 25% — the prior task's "is this personalised?" line


# ======================================================================================
# Step 1 — what the archetype actually is
# ======================================================================================
def step1(df):
    normal = df[df["profile_type"] == "normal"]
    target = df[df["profile_type"] == TARGET]
    out = {"archetype": TARGET,
           "n": int(len(target)),
           "population_prevalence_pct": round(100 * len(target) / len(df), 2),
           "generator_overrides": {f: {"kind": s[0], "low": s[1],
                                       "high": s[2] if len(s) > 2 else None}
                                   for f, s in OVERRIDES.items()},
           "features": {}}

    # spec[2] is safe because every one of conservative_saver's overrides is a
    # ('uniform', lo, hi) triple. Other profiles carry ('fixed', value) pairs, so pointing
    # TARGET at one of those would need this unpacking reworked, not just retargeted.
    for f, spec in OVERRIDES.items():
        lo, hi = spec[1], spec[2]
        nv, tv = normal[f], target[f]
        out["features"][f] = {
            "override_range": [lo, hi],
            "normal_median": round(float(nv.median()), 4),
            "normal_q25": round(float(nv.quantile(0.25)), 4),
            "normal_q75": round(float(nv.quantile(0.75)), 4),
            "normal_q90": round(float(nv.quantile(0.90)), 4),
            "pct_of_normals_inside_override_range":
                round(100 * float(((nv >= lo) & (nv <= hi)).mean()), 1),
            "override_low_as_normal_percentile":
                round(float((nv < lo).mean() * 100), 1),
            "archetype_median": round(float(tv.median()), 4),
        }

    # Which OTHER archetypes overlap this signature? Decides whether a Banking-side condition
    # can discriminate at all, or whether the Investment side is doing all the work.
    out["overlapping_archetypes"] = {
        t: {f: list(s[1:]) for f, s in MISMATCH_TYPES[t].items() if f in OVERRIDES}
        for t in MISMATCH_TYPES
        if t != TARGET and set(MISMATCH_TYPES[t]) & set(OVERRIDES)}

    # The confounder any investment_rate cut inherits: the generator's non-investor hurdle.
    out["non_investor_confounder"] = {
        "investor_prob": INVESTOR_PROB,
        "pct_of_normals_with_zero_investment_rate":
            round(100 * float((normal["investment_rate"] == 0).mean()), 1),
        "pct_of_normals_at_or_below_ir_max":
            round(100 * float((normal["investment_rate"] <= IR_MAX).mean()), 1),
        "note": ("investment_rate <= {} catches every non-investor in the normal population, "
                 "not only conservative savers — so a share of any flagged set is ordinary "
                 "users who simply do not invest".format(IR_MAX)),
    }
    return out


# ======================================================================================
# Step 2 — the candidates
# ======================================================================================
def candidates(df):
    """Deliberately nested so each row isolates one added condition:
       original -> + investment_rate cut -> + Banking headroom -> tiers removed entirely."""
    stable = df["banking_tier"] == "Stable"
    getting_started = df["investment_tier"] == "Getting Started"
    ir = df["investment_rate"] <= IR_MAX
    sr = df["savings_rate"] >= SR_MIN
    ef = df["emergency_fund_months"] >= EF_MIN

    return {
        "R4 (original)": {
            "parts": {"Banking = Stable": stable,
                      "Investment = Getting Started": getting_started},
            "desc": "Banking = Stable AND Investment = Getting Started",
        },
        "C1 (+ investment_rate cut)": {
            "parts": {"Banking = Stable": stable,
                      "Investment = Getting Started": getting_started,
                      f"investment_rate <= {IR_MAX}": ir},
            "desc": (f"R4 AND investment_rate <= {IR_MAX} — the archetype's own "
                     f"investment_rate ceiling"),
        },
        "C2 (+ Banking headroom)": {
            "parts": {"Banking = Stable": stable,
                      "Investment = Getting Started": getting_started,
                      f"investment_rate <= {IR_MAX}": ir,
                      f"savings_rate >= {SR_MIN}": sr,
                      f"emergency_fund_months >= {EF_MIN}": ef},
            "desc": (f"C1 AND savings_rate >= {SR_MIN} AND emergency_fund_months >= {EF_MIN} "
                     f"— the archetype's Banking-side override floors, so 'strong foundation' "
                     f"means clearly inside the signature rather than barely inside a tier"),
        },
        "C3 (continuous only, no tiers)": {
            "parts": {f"investment_rate <= {IR_MAX}": ir,
                      f"savings_rate >= {SR_MIN}": sr,
                      f"emergency_fund_months >= {EF_MIN}": ef},
            "desc": ("the full override signature with BOTH tier conditions dropped — the "
                     "control that shows whether the tiers contribute anything at all"),
        },
    }


# ======================================================================================
# Step 3 — evaluate
# ======================================================================================
def tier_membership_recall(df):
    """Share of TARGET landing in BOTH of Rule 4's tier conditions — the recall figure the
    acceptance criterion is gated on.

    Archetype recall against a continuous cut is TAUTOLOGICAL: every cut here is the
    archetype's own override bound, so its members pass by definition. The output marks
    that with `recall_is_tautological` and a note saying the number "is not evidence" — so
    gating the criterion on it would mean gating on something this file itself disclaims.
    Tier membership is not defined by the overrides, so it is a real measurement. This is
    the same treatment the sibling Rule 5 task applies (its `tier_membership` block); it is
    computed but deliberately NOT serialised here, so the committed
    layer3_rule4_tightening.json stays reproducible from this script unchanged.
    """
    tgt = df["profile_type"] == TARGET
    in_both = (df["banking_tier"] == "Stable") & (df["investment_tier"] == "Getting Started")
    return round(100 * float(in_both[tgt].mean()), 1) if tgt.any() else 0.0


def evaluate(df, cand, tier_recall_pct):
    parts = cand["parts"]
    mask = np.logical_and.reduce([p.to_numpy() for p in parts.values()])
    mask = pd.Series(mask, index=df.index)
    normal = df[df["profile_type"] == "normal"]
    tgt = df["profile_type"] == TARGET
    n_flag = int(mask.sum())

    # Independence baseline: what this conjunction would fire at if its parts were unrelated.
    pred = float(np.prod([float(p.mean()) for p in parts.values()]))
    obs = float(mask.mean())

    prevalence = float(tgt.mean())
    precision = float((mask & tgt).sum()) / n_flag if n_flag else 0.0
    recall = float((mask & tgt).sum()) / int(tgt.sum()) if tgt.sum() else 0.0

    e = {
        "description": cand["desc"],
        "conditions": list(parts),
        "n_flagged": n_flag,
        "full_population_pct": round(100 * obs, 1),
        "normal_only_pct": round(100 * float(mask.loc[normal.index].mean()), 1),
        "independence_predicted_pct": round(100 * pred, 1),
        "observed_over_predicted": round(obs / pred, 2) if pred > 0 else None,
        "conservative_saver_pct_of_flagged": round(100 * precision, 1),
        "lift_over_prevalence": round(precision / prevalence, 2) if prevalence else None,
        "recall_of_archetype_pct": round(100 * recall, 1),
        "by_band_full_pct": {}, "by_band_normal_pct": {},
        "profile_mix_pct": {k: round(100 * v / max(n_flag, 1), 1) for k, v in
                            df.loc[mask, "profile_type"].value_counts().items()},
    }
    for b in range(1, 6):
        e["by_band_full_pct"][str(b)] = round(
            100 * float(mask[df["age_band"] == b].mean()), 1)
        nb = normal["age_band"] == b
        e["by_band_normal_pct"][str(b)] = round(
            100 * float(mask.loc[normal.index][nb].mean()), 1)

    # What the non-archetype majority of the flagged set actually is. A flag can be
    # "enriched" and still be mostly ordinary non-investors, which is a different product
    # decision from finding a conservative saver.
    flagged = df[mask]
    fn = flagged[flagged["profile_type"] == "normal"]
    e["flagged_composition"] = {
        "normal_pct": round(100 * len(fn) / max(n_flag, 1), 1),
        "normal_non_investors_pct": round(
            100 * int((fn["has_investments"] == 0).sum()) / max(n_flag, 1), 1),
        "normal_investors_pct": round(
            100 * int((fn["has_investments"] == 1).sum()) / max(n_flag, 1), 1),
    }
    # Recall needs a health warning. Every continuous cut here IS an override bound, so an
    # archetype member satisfies it by definition and recall is pinned at ~100% carrying no
    # information. The tier conditions are NOT constructed that way, so R4's recall is a real
    # measurement — it says every conservative_saver genuinely lands in both tiers.
    # Detected from the condition LABELS, which is why they are written as
    # "<feature> <op> <value>" — a label reworded to read more nicely would silently turn
    # this off and let a tautological recall be reported as evidence.
    uses_override_cut = any(c.startswith(("investment_rate", "savings_rate",
                                          "emergency_fund_months")) for c in parts)
    e["recall_is_tautological"] = bool(uses_override_cut)
    e["recall_note"] = (
        "tautological — the cut is the archetype's own override bound, so every member "
        "passes it by definition; this number is not evidence"
        if uses_override_cut else
        "a real measurement — tier membership is not defined by the overrides")
    e["max_band_rate_pct"] = max(e["by_band_full_pct"].values())
    e["meets_lift"] = bool((e["lift_over_prevalence"] or 0) >= MIN_LIFT)
    # Gated on tier membership, NOT on recall_of_archetype_pct above — see
    # tier_membership_recall() for why that figure cannot carry a criterion.
    e["meets_recall"] = bool(tier_recall_pct >= MIN_RECALL_PCT)
    e["meets_band_rate"] = bool(e["max_band_rate_pct"] <= MAX_BAND_RATE_PCT)
    e["meets_all_criteria"] = bool(e["meets_lift"] and e["meets_recall"]
                                   and e["meets_band_rate"])
    return e


def sensitivity(df):
    """How much of the result depends on knowing the generator's exact override bounds?

    This is the circularity check. The thresholds above were reverse-engineered from
    MISMATCH_TYPES, which no real user carries — so if precision only holds at exactly
    (0.03, 0.28, 5.0) and collapses one notch either side, the finding is an artefact of
    reading the answer key rather than a usable rule. Each axis is swept with the other two
    held at their override values, in the C2 form (tiers included).
    """
    stable = df["banking_tier"] == "Stable"
    getting_started = df["investment_tier"] == "Getting Started"
    tgt = (df["profile_type"] == TARGET).to_numpy()
    prevalence = float(tgt.mean())

    def c2_mask(ir_max=IR_MAX, sr_min=SR_MIN, ef_min=EF_MIN):
        """C2's mask, with any single axis overridable and the rest at their bounds.

        Written once rather than spelled out per axis: "sweep one, hold the other two at
        their override values" is then structural, instead of a property you have to
        confirm by reading three near-identical condition chains and checking that each
        one held the correct two constants.
        """
        return (stable & getting_started
                & (df["investment_rate"] <= ir_max)
                & (df["savings_rate"] >= sr_min)
                & (df["emergency_fund_months"] >= ef_min))

    def score(mask):
        m = mask.to_numpy()
        n = int(m.sum())
        prec = float((m & tgt).sum()) / n if n else 0.0
        return {"n_flagged": n, "full_population_pct": round(100 * m.mean(), 1),
                "precision_pct": round(100 * prec, 1),
                "lift": round(prec / prevalence, 2) if prevalence else None,
                "recall_pct": round(100 * float((m & tgt).sum()) / int(tgt.sum()), 1)}

    grids = {
        # (values, builder, this axis's OWN override bound — compared per axis rather than
        # against all three, because the bounds are plain numbers that can coincide with a
        # different axis's sweep point and a global membership test then marks the wrong
        # row as the bound. No collision exists at the values below, but the sibling Rule 5
        # task hit exactly that (its 0.10 sweep point equals another axis's bound), so this
        # is written the safe way in both.)
        "investment_rate_max": ([0.01, 0.02, IR_MAX, 0.05, 0.08, 0.12],
                                lambda v: c2_mask(ir_max=v), IR_MAX),
        "savings_rate_min": ([0.20, 0.24, SR_MIN, 0.32, 0.36],
                             lambda v: c2_mask(sr_min=v), SR_MIN),
        "emergency_fund_months_min": ([3.0, 4.0, EF_MIN, 6.0, 8.0],
                                      lambda v: c2_mask(ef_min=v), EF_MIN),
    }
    out = {}
    for axis, (values, build, bound) in grids.items():
        out[axis] = {str(v): {**score(build(v)), "is_override_bound": bool(v == bound)}
                     for v in sorted(set(values))}
    return out


# ======================================================================================
# Plots + report
# ======================================================================================
def plots(s1, results):
    os.makedirs(PLOT_DIR, exist_ok=True)
    written = []
    names = list(results)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(names))
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
    ax.bar(x - 0.2, [results[n]["conservative_saver_pct_of_flagged"] for n in names], 0.4,
           label=f"% of flagged that are {TARGET}", color="#55A868")
    ax.bar(x + 0.2, [results[n]["recall_of_archetype_pct"] for n in names], 0.4,
           label="% of the archetype caught (recall)", color="#8172B2")
    ax.axhline(s1["population_prevalence_pct"], color="#C44E52", ls="--", lw=1.2,
               label=f"{TARGET} prevalence ({s1['population_prevalence_pct']}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" ")[0] for n in names])
    ax.set_ylabel("%")
    ax.set_title("Does it actually find the archetype?")
    ax.legend(fontsize=8)

    fig.suptitle("Rule 4 tightening — base rate, chance baseline and archetype enrichment")
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "rule4_candidates.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)
    return [os.path.relpath(w, REPO_ROOT).replace("\\", "/") for w in written]


def write_markdown(df, s1, results, sens, plot_paths, winner):
    L = []
    a = L.append
    a("# Layer 3 Rule 4 — tightening and re-validation\n")
    a("**Analysis only.** Extends `layer3_validation.py`; tier assignment is imported from "
      "it (same artifacts, same per-band `tier_map`, same replay-vs-artifact gate). Nothing "
      "was refit, no production file was written, and no Layer 3 rule was implemented.\n")
    a(f"Generated by `MLAI/clustering/layer3_rule4_tightening.py` on {len(df):,} users.\n")

    a(f"## Step 1 — what `{TARGET}` actually is\n")
    a(f"Read directly from `MISMATCH_TYPES` in `generate_synthetic_data.py`. "
      f"n = {s1['n']}, **{s1['population_prevalence_pct']}% of the population**.\n")
    a("| Feature | override range | normal median | normal q75 | normal q90 | "
      "% of normals inside the range | override floor = normal percentile |")
    a("|---|---|---|---|---|---|---|")
    for f, d in s1["features"].items():
        a(f"| `{f}` | {d['override_range'][0]}–{d['override_range'][1]} | "
          f"{d['normal_median']} | {d['normal_q75']} | {d['normal_q90']} | "
          f"{d['pct_of_normals_inside_override_range']}% | "
          f"{d['override_low_as_normal_percentile']}th |")
    a("")
    a("So the archetype's signature is: **saves in the top quartile of normal users, holds a "
      "comfortable emergency fund, and invests essentially nothing.** Every threshold below "
      "is one of these override bounds — no round numbers were invented.\n")

    a("**Which other archetypes share part of this signature** (this decides whether a "
      "Banking-side condition can discriminate at all):\n")
    for t, fs in s1["overlapping_archetypes"].items():
        a(f"- `{t}`: " + ", ".join(f"`{f}` {v[0]}–{v[1]}" for f, v in fs.items()))
    a("")
    nc = s1["non_investor_confounder"]
    a(f"**Confounder inherited by any `investment_rate` cut:** the generator applies a "
      f"non-investor hurdle (`INVESTOR_PROB = {nc['investor_prob']}`), so "
      f"**{nc['pct_of_normals_with_zero_investment_rate']}% of normal users have an "
      f"investment_rate of exactly 0**, and "
      f"{nc['pct_of_normals_at_or_below_ir_max']}% sit at or below {IR_MAX}. Those users are "
      f"ordinary non-investors, not conservative savers — they will show up in any flagged "
      f"set built on this feature.\n")

    a("## Steps 2–3 — candidates and results\n")
    for n, e in results.items():
        a(f"- **{n}** — {e['description']}")
    a("")
    a("| Candidate | flagged | full pop | normal only | independence-predicted | obs/pred | "
      f"% flagged that are `{TARGET}` | lift | recall | max band rate |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for n, e in results.items():
        a(f"| {n} | {e['n_flagged']} | **{e['full_population_pct']}%** | "
          f"{e['normal_only_pct']}% | {e['independence_predicted_pct']}% | "
          f"{e['observed_over_predicted']}x | "
          f"**{e['conservative_saver_pct_of_flagged']}%** | "
          f"{e['lift_over_prevalence']}x | {e['recall_of_archetype_pct']}% | "
          f"{e['max_band_rate_pct']}% |")
    a("")
    a(f"`lift` is the share of flagged users that are `{TARGET}` divided by its "
      f"{s1['population_prevalence_pct']}% population prevalence. `obs/pred` compares the "
      f"observed rate to the product of its parts' marginals — useful, but inflated whenever "
      f"the conditions are near-redundant by construction, so **lift is the decisive "
      f"number, not obs/pred**.\n")
    a("**Ignore the recall column for C1–C3.** Every continuous cut used here *is* one of "
      "the archetype's override bounds, so an archetype member clears it by definition and "
      "recall is pinned at 100% carrying no information. R4's 100% recall is different and "
      "is a real measurement: tier membership is not defined by the overrides, so it says "
      f"**every single `{TARGET}` genuinely lands in Banking = Stable AND Investment = "
      "Getting Started**. The original rule's problem was never that it missed them — it is "
      "that it swept up 1,700 other users alongside.\n")

    a("### Per age band (full population)\n")
    a("| Candidate | b1 | b2 | b3 | b4 | b5 |")
    a("|---|---|---|---|---|---|")
    for n, e in results.items():
        a(f"| {n} | " + " | ".join(f"{e['by_band_full_pct'][str(b)]}%"
                                   for b in range(1, 6)) + " |")
    a("")

    a("### What each flagged set is actually made of\n")
    a("| Candidate | normal users | ...of which non-investors | ...of which investors | "
      "top archetypes |")
    a("|---|---|---|---|---|")
    for n, e in results.items():
        c = e["flagged_composition"]
        arch = ", ".join(f"{k} {v}%" for k, v in list(e["profile_mix_pct"].items())[:3]
                         if k != "normal")
        a(f"| {n} | {c['normal_pct']}% | {c['normal_non_investors_pct']}% | "
          f"{c['normal_investors_pct']}% | {arch} |")
    a("")

    a("### Do the tier conditions still earn their place?\n")
    c2, c3 = results["C2 (+ Banking headroom)"], results["C3 (continuous only, no tiers)"]
    a(f"C2 and C3 differ only in whether the two tier conditions are applied on top of the "
      f"continuous signature. C3 flags **{c3['n_flagged']}** users, C2 flags "
      f"**{c2['n_flagged']}** — the tier conditions exclude "
      f"**{c3['n_flagged'] - c2['n_flagged']} user(s)** in the whole dataset. Once the "
      f"continuous signature is applied, the tiers are very nearly a no-op: they are not "
      f"wrong, they are redundant. Keeping them costs nothing and keeps Layer 3 phrased in "
      f"tier language; dropping them changes almost no one's result.\n")

    a("### Sensitivity — does this survive not knowing the answer key?\n")
    a("The thresholds above were reverse-engineered from `MISMATCH_TYPES`, which no real "
      "user carries. If precision only holds at exactly those values, the result is an "
      "artefact of reading the generator's answer key. Each axis is swept with the other two "
      "held at their override bounds (C2 form, tiers included); **bold** marks the override "
      "bound itself.\n")
    for axis, grid in sens.items():
        a(f"`{axis}`\n")
        a("| value | flagged | % of population | precision | lift | recall |")
        a("|---|---|---|---|---|---|")
        for v, d in grid.items():
            mark = "**" if d["is_override_bound"] else ""
            a(f"| {mark}{v}{mark} | {d['n_flagged']} | {d['full_population_pct']}% | "
              f"{d['precision_pct']}% | {d['lift']}x | {d['recall_pct']}% |")
        a("")

    a("## Step 4 — recommendation\n")
    a(f"Criteria were fixed before the numbers were seen: lift >= {MIN_LIFT}x, recall >= "
      f"{MIN_RECALL_PCT}%, and no age band above {MAX_BAND_RATE_PCT}%.\n")
    a("| Candidate | lift | recall | band rate | verdict |")
    a("|---|---|---|---|---|")
    for n, e in results.items():
        a(f"| {n} | {'PASS' if e['meets_lift'] else 'fail'} | "
          f"{'PASS' if e['meets_recall'] else 'fail'} | "
          f"{'PASS' if e['meets_band_rate'] else 'fail'} | "
          f"{'**MEETS ALL**' if e['meets_all_criteria'] else 'rejected'} |")
    a("")
    if winner:
        e = results[winner]
        a(f"**Recommended: `{winner}`** — {e['description']}.\n")
        a(f"It flags {e['full_population_pct']}% of users against an "
          f"independence-predicted {e['independence_predicted_pct']}% "
          f"({e['observed_over_predicted']}x), of whom "
          f"{e['conservative_saver_pct_of_flagged']}% are `{TARGET}` "
          f"({e['lift_over_prevalence']}x its prevalence), catching "
          f"{e['recall_of_archetype_pct']}% of all `{TARGET}` users. Highest single-band "
          f"rate is {e['max_band_rate_pct']}%.\n")
        a("**Three caveats that must travel with it:**\n")
        a(f"1. {e['flagged_composition']['normal_non_investors_pct']}% of the flagged set are "
          f"ordinary `normal` users who simply hold no investments, and "
          f"{e['profile_mix_pct'].get('underinsured_saver', 0)}% are `underinsured_saver` "
          f"(which shares this archetype's savings and emergency-fund overrides but has no "
          f"`investment_rate` override). The flag means **'saves well, doesn't invest'** — "
          f"which is the right thing to surface, but it is not the same claim as "
          f"'is a conservative saver'.")
        a("2. Its thresholds are the generator's own override bounds. On synthetic data that "
          "is close to circular — see the sensitivity table above for how far the result "
          "survives moving off those exact values, which is the only evidence here that "
          "bears on whether it would hold against real users.")
        a("3. Recall cannot be used to defend it, for the reason given above.\n")
    else:
        a("**No candidate meets all three criteria.** Reporting that rather than promoting "
          "the least-bad option — see the per-criterion table above for where each fails.\n")

    a("## Plots\n")
    for p in plot_paths:
        a(f"- `{p}`")
    a("")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("=" * 100)
    print(f"RULE 4 TIGHTENING — {len(df):,} users (analysis only, no refit)")
    print("=" * 100)
    for domain, col in [("banking", "banking_tier"), ("investment", "investment_tier"),
                        ("insurance", "insurance_tier")]:
        labels, tier_labels, shares = replay_tiers(df, domain)
        df[col] = labels
        print(f"  [replay OK] {domain:<11} {tier_labels} -> {shares}%")

    s1 = step1(df)
    print(f"\n  step 1 — {TARGET}: n={s1['n']} "
          f"({s1['population_prevalence_pct']}% of population)")
    for f, d in s1["features"].items():
        print(f"    {f:<24} override {d['override_range'][0]}–{d['override_range'][1]:<6} "
              f"normal median {d['normal_median']:<8} "
              f"({d['pct_of_normals_inside_override_range']}% of normals inside range)")
    print(f"    non-investor confounder: "
          f"{s1['non_investor_confounder']['pct_of_normals_with_zero_investment_rate']}% "
          f"of normals have investment_rate == 0")

    cands = candidates(df)
    tier_recall_pct = tier_membership_recall(df)
    print(f"    tier-membership recall (the criterion's basis): "
          f"{tier_recall_pct}% of {TARGET} land in both tiers")
    results = {n: evaluate(df, c, tier_recall_pct) for n, c in cands.items()}

    print(f"\n  steps 2-3 — candidates (lift is vs the {s1['population_prevalence_pct']}% "
          f"prevalence of {TARGET}):")
    print(f"    {'candidate':<32}{'rate':>8}{'indep':>8}{'o/p':>7}"
          f"{'%cons':>8}{'lift':>7}{'recall':>8}")
    for n, e in results.items():
        print(f"    {n:<32}{e['full_population_pct']:>7}%{e['independence_predicted_pct']:>7}%"
              f"{e['observed_over_predicted']:>7}{e['conservative_saver_pct_of_flagged']:>7}%"
              f"{e['lift_over_prevalence']:>7}{e['recall_of_archetype_pct']:>7}%")

    passing = [n for n, e in results.items()
               if e["meets_all_criteria"] and n != "R4 (original)"]
    # Among passers, prefer the highest lift; recall breaks ties.
    winner = (max(passing, key=lambda n: (results[n]["lift_over_prevalence"],
                                          results[n]["recall_of_archetype_pct"]))
              if passing else None)

    sens = sensitivity(df)
    print("\n  sensitivity (C2 form, one axis at a time; * = the override bound):")
    for axis, grid in sens.items():
        cells = "  ".join(f"{'*' if d['is_override_bound'] else ''}{v}:"
                          f"{d['precision_pct']}%/{d['full_population_pct']}%"
                          for v, d in grid.items())
        print(f"    {axis:<28} {cells}   (precision/base rate)")

    print(f"\n  step 4 — criteria: lift >= {MIN_LIFT}x, recall >= {MIN_RECALL_PCT}%, "
          f"every band <= {MAX_BAND_RATE_PCT}%")
    for n, e in results.items():
        print(f"    {n:<32} lift {'PASS' if e['meets_lift'] else 'fail'}  "
              f"recall {'PASS' if e['meets_recall'] else 'fail'}  "
              f"band {'PASS' if e['meets_band_rate'] else 'fail'}  -> "
              f"{'MEETS ALL' if e['meets_all_criteria'] else 'rejected'}")
    print(f"\n  RECOMMENDATION: {winner if winner else 'none — no candidate meets all three'}")

    plot_paths = plots(s1, results)
    summary = {"dataset": {"path": "MLAI/clustering/data/processed/synthetic_users.csv",
                           "n_users": int(len(df))},
               "criteria": {"min_lift": MIN_LIFT, "min_recall_pct": MIN_RECALL_PCT,
                            "max_band_rate_pct": MAX_BAND_RATE_PCT,
                            "note": "fixed before results were seen"},
               "step1_archetype": s1, "step2_3_candidates": results,
               "sensitivity": sens, "step4_recommendation": winner, "plots": plot_paths}
    with open(os.path.join(OUT_DIR, "layer3_rule4_tightening.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "RULE4_TIGHTENING.md"), "w", encoding="utf-8") as f:
        f.write(write_markdown(df, s1, results, sens, plot_paths, winner))
    print(f"\n  wrote {os.path.relpath(OUT_DIR, REPO_ROOT)}/"
          f"{{RULE4_TIGHTENING.md, layer3_rule4_tightening.json}} + {len(plot_paths)} plot(s)")


if __name__ == "__main__":
    main()
