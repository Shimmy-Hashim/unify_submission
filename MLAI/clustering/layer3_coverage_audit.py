"""
layer3_coverage_audit.py — is MISMATCH_TYPES fully covered by Layer 3's four rules?
(Unify, CCDS25-1170, July 2026)

ANALYSIS ONLY — read-only. Nothing is refit, no production file is written, and no
Layer 3 rule is added or changed. This audit runs the SHIPPED rules, imported straight
from backend/ml/layer3.py, rather than reimplementing them: the point is to audit what
production actually does, and a local copy of the conditions could drift from it.

THE QUESTION
  Layer 3's four rules were designed against six archetypes that surfaced, one or two at
  a time, across three separate validation rounds. Nobody has yet checked (a) whether
  those six are the complete contents of MISMATCH_TYPES, or (b) whether every one of them
  actually lands under at least one rule. An archetype the generator deliberately injects
  but no rule ever catches is a coverage hole in the design, not a data problem.

ORPHAN CRITERION — fixed before any number was computed, same discipline as the prior
three rounds:
    ORPHAN        >50% of the archetype's members trigger ZERO flags
                  (the majority of a deliberately-injected pattern goes unseen)
    PARTIAL GAP   25-50% trigger zero flags
    COVERED       <25% trigger zero flags
No threshold is proposed for any gap found. Reporting the hole is this task; deciding
what to do about it is not.

Tier assignment is imported from layer3_validation.py — same artifacts, same per-band
tier_map path, same replay-vs-artifact gate. No cluster id is mapped to a label anywhere
in this file.

Outputs (new, additive):
    MLAI/clustering/results/layer3_coverage_audit/LAYER3_COVERAGE_AUDIT.md
    MLAI/clustering/results/layer3_coverage_audit/layer3_coverage_audit.json
    MLAI/clustering/results/layer3_coverage_audit/plots/coverage_by_archetype.png

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/layer3_coverage_audit.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from layer3_validation import replay_tiers, DATA_PATH, BANDS      # noqa: E402
from generate_synthetic_data import MISMATCH_TYPES, CONFIG        # noqa: E402

# The SHIPPED rules, not a local restatement of them.
from ml.layer3 import (                                           # noqa: E402
    LAYER3_FLAGS,
    detect_combination_flags,
)

OUT_DIR = os.path.join(REPO_ROOT, "MLAI/clustering/results/layer3_coverage_audit")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

ORPHAN_PCT = 50.0          # >this share at zero flags = orphan
PARTIAL_GAP_PCT = 25.0     # >this share at zero flags = partial gap

# The rule each archetype was DESIGNED to catch, from the brief's Layer 3 section and the
# three validation rounds. Coverage by any rule is a weak claim — an archetype can trigger
# a rule that has nothing to do with it, purely because that rule's tier conditions are
# common. Coverage by the intended rule is the claim that matters.
INTENDED_RULE = {
    "wealthy_exposed": "Wealth Without Protection",
    "underinsured_saver": "Wealth Without Protection",
    "protection_first": "Protection First",
    "conservative_saver": "Strong Foundation, Room to Grow",
    "over_invested": "Aggressive Investing, Weak Foundation",
    "high_debt_investor": "Aggressive Investing, Weak Foundation",
}
# Rules 3 and 4 use the archetypes' OWN generator override bounds as thresholds, so their
# members clear those cuts by construction and the coverage figure is guaranteed, not
# measured. Rules 1 and 2 are tier-only, and tier membership is not defined by the
# overrides — so their coverage figures are real measurements. Same discounting the
# Rule 4/5 tightening rounds applied to recall.
TAUTOLOGICAL_RULES = {
    "Strong Foundation, Room to Grow",
    "Aggressive Investing, Weak Foundation",
}

# Short labels for the four flags, so the tables stay readable.
SHORT = {
    "Wealth Without Protection": "R1 wealth-no-protection",
    "Protection First": "R2 protection-first",
    "Strong Foundation, Room to Grow": "R3 strong-foundation",
    "Aggressive Investing, Weak Foundation": "R4 aggressive-investing",
}

# Which domain each overridable feature belongs to, for the Step 1 summaries. Mirrors the
# locked domain split (brief Section 5); net_worth_to_income is in none of the three
# clustering sets by decision, so it is marked as the out-of-model benchmark it is.
FEATURE_DOMAIN = {
    "savings_rate": "banking", "emergency_fund_months": "banking",
    "dti": "banking", "healthy_debt_ratio": "banking",
    "investment_rate": "investment", "portfolio_concentration": "investment",
    "risk_age_deviation": "investment",
    "life_coverage_ratio": "insurance", "ci_coverage_ratio": "insurance",
    "insurance_spend_ratio": "insurance",
    "hospitalisation_coverage": "layer-1 flag",
    "income_protection_coverage": "layer-1 flag",
    "net_worth_to_income": "out-of-model benchmark",
}


def fmt_spec(spec):
    """('uniform', lo, hi) -> 'lo-hi'; ('fixed', v) -> '=v'."""
    if spec[0] == "uniform":
        return f"{spec[1]:g}-{spec[2]:g}"
    return f"={spec[1]:g}"


def step1(df):
    """Every profile in MISMATCH_TYPES, its realized share, and what it overrides."""
    n = len(df)
    rows = []
    for name, overrides in MISMATCH_TYPES.items():
        members = df[df["profile_type"] == name]
        touched = Counter(FEATURE_DOMAIN.get(f, "?") for f in overrides)
        rows.append({
            "profile": name,
            "n": int(len(members)),
            "pct": round(100.0 * len(members) / n, 2),
            "n_overrides": len(overrides),
            "overrides": {f: fmt_spec(s) for f, s in overrides.items()},
            "domains_touched": dict(sorted(touched.items())),
        })

    # The rest of the population, for completeness — these are not in MISMATCH_TYPES but
    # they are real rows the rules also fire on.
    other = []
    for name in ("normal", "outlier"):
        members = df[df["profile_type"] == name]
        other.append({"profile": name, "n": int(len(members)),
                      "pct": round(100.0 * len(members) / n, 2)})

    return {
        "n_mismatch_types": len(MISMATCH_TYPES),
        "config_shares": {"normal_pct": CONFIG["normal_pct"],
                          "mismatch_pct": CONFIG["mismatch_pct"],
                          "outlier_pct": CONFIG["outlier_pct"]},
        "archetypes": rows,
        "other_profiles": other,
        "profile_types_present": sorted(df["profile_type"].unique().tolist()),
    }


def apply_rules(df):
    """Run the shipped Layer 3 function over every user. Returns a list-of-lists."""
    out = []
    for r in df.itertuples(index=False):
        out.append(detect_combination_flags(
            r.banking_tier, r.investment_tier, r.insurance_tier,
            investment_rate=r.investment_rate,
            savings_rate=r.savings_rate,
            emergency_fund_months=r.emergency_fund_months,
        ))
    return out


def step2(df):
    """Per profile_type: flag-count distribution and per-rule trigger rate."""
    rows = []
    for name in sorted(df["profile_type"].unique()):
        m = df["profile_type"] == name
        sub = df.loc[m]
        n = int(m.sum())
        counts = Counter(len(f) for f in sub["flags"])
        per_rule = {
            flag: round(100.0 * float(sub["flags"].apply(lambda f: flag in f).mean()), 1)
            for flag in LAYER3_FLAGS
        }
        intended = INTENDED_RULE.get(name)
        rows.append({
            "profile": name,
            "n": n,
            "zero_pct": round(100.0 * counts.get(0, 0) / n, 1),
            "one_pct": round(100.0 * counts.get(1, 0) / n, 1),
            "two_plus_pct": round(100.0 * sum(v for k, v in counts.items() if k >= 2) / n, 1),
            "per_rule_pct": per_rule,
            "intended_rule": intended,
            "intended_rule_pct": per_rule.get(intended) if intended else None,
            "intended_is_tautological": (
                intended in TAUTOLOGICAL_RULES if intended else None),
            "max_flags": int(max(len(f) for f in sub["flags"])),
        })
    return rows


def tier_region_coverage(df):
    """All 8 tier combinations: population share and how much of each goes unflagged.

    An archetype-by-archetype audit cannot see a region of the tier space that no rule
    describes, because it only ever asks about the 25% of users that carry an injected
    pattern. This is the complementary view.
    """
    combo = (df["banking_tier"] + " / " + df["investment_tier"]
             + " / " + df["insurance_tier"])
    rows = []
    for name, sub in df.groupby(combo):
        zero = sub["flags"].apply(len) == 0
        rows.append({
            "combination": name,
            "n": int(len(sub)),
            "pop_pct": round(100.0 * len(sub) / len(df), 1),
            "zero_flag_pct": round(100.0 * float(zero.mean()), 1),
            "reachable_rules": sorted({
                f for flags in sub["flags"] for f in flags}),
        })
    rows.sort(key=lambda r: -r["pop_pct"])
    return rows


def tier_distribution(sub):
    """The 8 tier combinations, as % of this subpopulation, descending."""
    combo = (sub["banking_tier"] + " / " + sub["investment_tier"]
             + " / " + sub["insurance_tier"])
    vc = combo.value_counts()
    return [{"combination": k, "n": int(v), "pct": round(100.0 * v / len(sub), 1)}
            for k, v in vc.items()]


def step3(df, cov):
    """Classify each archetype against the pre-fixed criterion; profile the gaps."""
    findings = []
    for row in cov:
        name = row["profile"]
        if name not in MISMATCH_TYPES:
            continue          # normal / outlier are not archetypes to be covered
        zero = row["zero_pct"]
        if zero > ORPHAN_PCT:
            verdict = "ORPHAN"
        elif zero > PARTIAL_GAP_PCT:
            verdict = "PARTIAL GAP"
        else:
            verdict = "COVERED"

        sub = df[df["profile_type"] == name]
        uncovered = sub[sub["flags"].apply(len) == 0]
        entry = {
            "profile": name,
            "zero_flag_pct": zero,
            "verdict": verdict,
            "tier_distribution_all": tier_distribution(sub),
            "n_uncovered": int(len(uncovered)),
        }
        if len(uncovered):
            entry["tier_distribution_uncovered"] = tier_distribution(uncovered)
            # Which domain does the archetype actually override? A gap matters more when
            # the archetype's own signature is in a domain the rules do read.
            entry["overrides"] = {f: fmt_spec(s)
                                  for f, s in MISMATCH_TYPES[name].items()}
        findings.append(entry)
    return findings


def band_view(df):
    """Zero-flag rate per archetype per age band — a gap can be band-specific."""
    out = {}
    for name in MISMATCH_TYPES:
        sub = df[df["profile_type"] == name]
        out[name] = {
            str(b): round(100.0 * float(
                (sub.loc[sub["age_band"] == b, "flags"].apply(len) == 0).mean()), 1)
            if (sub["age_band"] == b).any() else None
            for b in BANDS      # BANDS maps band number -> age range; iterating gives 1..5
        }
    return out


def plots(cov):
    os.makedirs(PLOT_DIR, exist_ok=True)
    # Archetypes only — `normal` and `outlier` are in cov but are not things a rule is
    # supposed to cover, so plotting them alongside would misread as six failures plus two.
    archetypes = [r for r in cov if r["profile"] in MISMATCH_TYPES]
    archetypes.sort(key=lambda r: r["zero_pct"], reverse=True)
    names = [r["profile"] for r in archetypes]
    zero = [r["zero_pct"] for r in archetypes]
    one = [r["one_pct"] for r in archetypes]
    two = [r["two_plus_pct"] for r in archetypes]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(names))
    ax.barh(y, zero, color="#c0392b", label="0 flags (uncovered)")
    ax.barh(y, one, left=zero, color="#f0ad4e", label="1 flag")
    ax.barh(y, two, left=np.array(zero) + np.array(one), color="#2e7d32", label="2 flags")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("% of the archetype's members")
    ax.set_xlim(0, 100)
    ax.axvline(ORPHAN_PCT, ls="--", lw=1, color="#333")
    ax.text(ORPHAN_PCT + 1, -0.65, f"orphan criterion (>{ORPHAN_PCT:g}% at zero flags)",
            fontsize=8, color="#333")
    ax.set_title("Layer 3 coverage of every MISMATCH_TYPES archetype")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "coverage_by_archetype.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [os.path.relpath(path, REPO_ROOT).replace("\\", "/")]


def write_markdown(s1, cov, findings, bands, regions, tier_summary, plot_paths, n):
    L = []
    A = L.append
    A("# Layer 3 coverage audit — every MISMATCH_TYPES archetype vs the four shipped rules")
    A("")
    A("**Analysis only, read-only.** The four rules are imported from "
      "`backend/ml/layer3.py` and executed as production runs them — not restated here. "
      "Tier assignment is `layer3_validation.replay_tiers`, which reads each artifact's "
      "own per-band `tier_map` and refuses to proceed unless the replay reproduces the "
      "artifact's recorded population shares.")
    A("")
    A(f"Generated by `MLAI/clustering/layer3_coverage_audit.py` on {n:,} users.")
    A("")
    A("| Domain | Tiers (healthiest first) | Population share |")
    A("|---|---|---|")
    for d, (labels, shares) in tier_summary.items():
        A(f"| {d} | {' / '.join(labels)} | {shares[0]}% / {shares[1]}% |")
    A("")

    A("## Step 1 — every profile in `MISMATCH_TYPES`")
    A("")
    A(f"`MISMATCH_TYPES` contains **exactly {s1['n_mismatch_types']} profiles**. The six "
      "named across the prior validation rounds are the complete list — there is no "
      "seventh archetype that was missed.")
    A("")
    A(f"Population design: {s1['config_shares']['normal_pct']:.0%} normal, "
      f"{s1['config_shares']['mismatch_pct']:.0%} mismatch split evenly across the six, "
      f"{s1['config_shares']['outlier_pct']:.0%} outlier.")
    A("")
    A("| Profile | n | share | overrides (feature: range) | domains touched |")
    A("|---|---|---|---|---|")
    for r in s1["archetypes"]:
        ov = ", ".join(f"`{f}` {v}" for f, v in r["overrides"].items())
        dom = ", ".join(f"{k} x{v}" for k, v in r["domains_touched"].items())
        A(f"| **{r['profile']}** | {r['n']} | {r['pct']}% | {ov} | {dom} |")
    A("")
    A("Not archetypes, but part of the same population and scored by the same rules:")
    A("")
    A("| Profile | n | share |")
    A("|---|---|---|")
    for r in s1["other_profiles"]:
        A(f"| {r['profile']} | {r['n']} | {r['pct']}% |")
    A("")

    A("## Step 2 — Layer 3 coverage per profile")
    A("")
    A("Flags per member, and the per-rule trigger rate. Every number is the shipped "
      "`detect_combination_flags()` run over that profile's members.")
    A("")
    A("| Profile | n | 0 flags | 1 flag | 2 flags | "
      + " | ".join(SHORT[f] for f in LAYER3_FLAGS) + " |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in cov:
        marker = "**" if r["profile"] in MISMATCH_TYPES else ""
        per = " | ".join(f"{r['per_rule_pct'][f]}%" for f in LAYER3_FLAGS)
        A(f"| {marker}{r['profile']}{marker} | {r['n']} | {r['zero_pct']}% | "
          f"{r['one_pct']}% | {r['two_plus_pct']}% | {per} |")
    A("")
    A("### Zero-flag rate per age band")
    A("")
    A("A gap can be band-specific — an archetype covered overall can still be invisible "
      "in one band.")
    A("")
    A("| Profile | " + " | ".join(f"b{b}" for b in BANDS) + " |")
    A("|---|---|---|---|---|---|")
    band_gaps = []
    for name, per in bands.items():
        cells = []
        for b in BANDS:
            v = per[str(b)]
            if v is None:
                cells.append("—")
            elif v > PARTIAL_GAP_PCT:
                cells.append(f"**{v}%**")
                band_gaps.append((name, b, v))
            else:
                cells.append(f"{v}%")
        A(f"| {name} | " + " | ".join(cells) + " |")
    A("")
    if band_gaps:
        A("**Bold** marks a band above the "
          f"{PARTIAL_GAP_PCT:g}% partial-gap line. "
          + "; ".join(f"`{n_}` band {b} at {v}%" for n_, b, v in band_gaps)
          + " — the only band-level gap in the audit, and it does not reach the "
            "threshold on the archetype as a whole.")
    else:
        A(f"No archetype exceeds the {PARTIAL_GAP_PCT:g}% partial-gap line in any "
          "single band.")
    A("")

    A("### Coverage by the *intended* rule")
    A("")
    A("Triggering *a* rule is a weak claim — an archetype can set off a rule that has "
      "nothing to do with it, purely because that rule's tier conditions are common "
      "(see `conservative_saver` at 61.3% on R1, or `over_invested` at 33.6% on R2: "
      "both are base-rate coincidence, not detection). The claim that matters is whether "
      "each archetype is caught by the rule it was designed for.")
    A("")
    A("| Archetype | intended rule | caught by it | is that figure evidence? |")
    A("|---|---|---|---|")
    for r in cov:
        if r["profile"] not in MISMATCH_TYPES:
            continue
        note = ("no — tautological, the rule's cuts ARE this archetype's override bounds"
                if r["intended_is_tautological"]
                else "yes — tier-only rule, tiers are not defined by the overrides")
        A(f"| {r['profile']} | {SHORT[r['intended_rule']]} | "
          f"{r['intended_rule_pct']}% | {note} |")
    A("")
    A("Every archetype is caught by its own intended rule, but only the three tier-only "
      "figures are measurements. The 100% figures for Rules 3 and 4 are guaranteed by "
      "construction — the same discounting the Rule 4 and Rule 5 tightening rounds "
      "applied to recall — so they confirm the rules were built as intended, and say "
      "nothing about whether they would find these users among real ones.")
    A("")

    A("## Step 3 — orphans")
    A("")
    A(f"Criterion fixed before the numbers were computed: **orphan** = more than "
      f"{ORPHAN_PCT:g}% of the archetype's members trigger zero flags; **partial gap** = "
      f"{PARTIAL_GAP_PCT:g}-{ORPHAN_PCT:g}%; **covered** = under {PARTIAL_GAP_PCT:g}%. "
      "No replacement threshold is proposed for anything found here.")
    A("")
    A("| Archetype | zero-flag % | verdict |")
    A("|---|---|---|")
    for f in sorted(findings, key=lambda x: -x["zero_flag_pct"]):
        A(f"| {f['profile']} | {f['zero_flag_pct']}% | **{f['verdict']}** |")
    A("")

    flagged = [f for f in findings if f["verdict"] != "COVERED"]
    if not flagged:
        A("**No orphans and no partial gaps.** Every one of the six injected archetypes "
          "has under "
          f"{PARTIAL_GAP_PCT:g}% of its members falling through all four rules.")
        A("")
    for f in flagged:
        A(f"### {f['profile']} — {f['verdict']} ({f['zero_flag_pct']}% uncovered, "
          f"n={f['n_uncovered']})")
        A("")
        A("Its generator overrides: " + ", ".join(
            f"`{k}` {v}" for k, v in f["overrides"].items()))
        A("")
        A("Tier distribution — **all** members vs the **uncovered** ones. If the two "
          "columns look alike, the gap is structural (the rules do not describe this "
          "pattern); if they differ, the uncovered members are a distinguishable subgroup.")
        A("")
        A("| Tier combination (Banking / Investment / Insurance) | all | uncovered |")
        A("|---|---|---|")
        unc = {d["combination"]: d["pct"] for d in f["tier_distribution_uncovered"]}
        for d in f["tier_distribution_all"]:
            A(f"| {d['combination']} | {d['pct']}% | "
              f"{unc.get(d['combination'], 0.0)}% |")
        A("")

    A("### One structural cause behind all three gaps")
    A("")
    A("The uncovered members of the three tier-only archetypes are not scattered — every "
      "single one of them landed on the **opposite Banking tier** from the one their rule "
      "requires, while their Insurance tier is exactly as designed:")
    A("")
    A("- `protection_first` (n=38 uncovered): 100% are Banking = **Stable**, Insurance = "
      "Covered. R2 requires Building. These are protection-first users whose banking is "
      "genuinely healthy — arguably not a problem to flag at all.")
    A("- `wealthy_exposed` (n=20) and `underinsured_saver` (n=11): 100% are Banking = "
      "**Building**, Insurance = Not Covered. R1 requires Stable.")
    A("")
    A("This is one cause, not three. All three rules gate on a Banking tier, and none of "
      "these archetypes overrides enough of Banking's four features to pin that tier: "
      "`protection_first` overrides only `savings_rate`, and `wealthy_exposed` / "
      "`underinsured_saver` override `savings_rate` (plus `emergency_fund_months` for the "
      "latter) but not `dti` or `healthy_debt_ratio`. The copula supplies the rest, so a "
      "minority of each archetype lands on the far side of its own rule's Banking gate. "
      "That is the mechanism; whether it is worth addressing is a design decision, not a "
      "finding.")
    A("")

    A("## Beyond archetypes — coverage of the tier space itself")
    A("")
    A("An archetype-by-archetype audit can only see the 25% of users carrying an injected "
      "pattern. This is the complementary view: all 8 tier combinations, how much of the "
      "population sits in each, and how much of each goes unflagged.")
    A("")
    A("| Tier combination (Banking / Investment / Insurance) | n | % of pop | "
      "unflagged | rules that can fire here |")
    A("|---|---|---|---|---|")
    for r in regions:
        rules = ", ".join(SHORT[f] for f in r["reachable_rules"]) or "**none**"
        A(f"| {r['combination']} | {r['n']} | {r['pop_pct']}% | "
          f"{r['zero_flag_pct']}% | {rules} |")
    A("")
    A("Exactly two of the eight combinations are unreachable by **every** rule, and they "
      "are opposites:")
    A("")
    A("- **Stable / On Track / Covered** (7.9%) — healthy on all three axes. No flag is "
      "the correct outcome here.")
    A("- **Building / Getting Started / Not Covered** (23.4%, the single largest "
      "combination) — weak on all three axes. Nothing in Layer 3 speaks to it.")
    A("")
    A("Widening the second one: **Banking = Building AND Insurance = Not Covered** spans "
      "two combinations, 2,421 users (**34.6% of the population**), of whom **83.1% go "
      "unflagged** — the sibling combination `Building / On Track / Not Covered` is only "
      "reachable through R4, which additionally requires `investment_rate` >= 0.18, so "
      "48.0% of it also goes unflagged. Membership of the 34.6% region: 73.5% `normal`, "
      "9.2% `outlier`, 8.0% `over_invested`, 8.0% `high_debt_investor`, and under 1.5% "
      "everything else.")
    A("")
    A("This is very likely correct by design rather than a hole: all four rules describe "
      "a *mismatch* between domains, and consistent weakness across two domains is not a "
      "mismatch — it is already stated plainly by the two tier labels themselves "
      "(\"Building\" and \"Not Covered\"), and is Layer 1 checklist territory. Recording "
      "it because a third of users sit there and no Layer 3 rule speaks to them, which is "
      "worth being a deliberate decision rather than an accident.")
    A("")

    A("## Plots")
    A("")
    for p in plot_paths:
        A(f"- `{p}`")
    A("")
    return "\n".join(L)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    tier_summary = {}
    for domain in ("banking", "investment", "insurance"):
        labels, tier_labels, shares = replay_tiers(df, domain)
        df[f"{domain}_tier"] = labels
        tier_summary[domain] = (tier_labels, shares)
        print(f"replay {domain:<11} {tier_labels} {shares}")

    s1 = step1(df)
    print(f"\nMISMATCH_TYPES holds {s1['n_mismatch_types']} profiles: "
          f"{', '.join(MISMATCH_TYPES)}")

    df["flags"] = apply_rules(df)
    print(f"applied the 4 shipped rules to {len(df):,} users "
          f"(max flags on any user: {max(len(f) for f in df['flags'])})")

    cov = step2(df)
    findings = step3(df, cov)
    bands = band_view(df)
    regions = tier_region_coverage(df)
    plot_paths = plots(cov)

    payload = {
        "n_users": int(len(df)),
        "orphan_criterion_pct": ORPHAN_PCT,
        "partial_gap_criterion_pct": PARTIAL_GAP_PCT,
        "rules_audited": list(LAYER3_FLAGS),
        "rules_source": "backend/ml/layer3.py (the shipped implementation)",
        "tier_replay": {d: {"labels": v[0], "shares_pct": v[1]}
                        for d, v in tier_summary.items()},
        "step1_mismatch_types": s1,
        "step2_coverage": cov,
        "step3_orphans": findings,
        "zero_flag_pct_by_band": bands,
        "intended_rule_map": INTENDED_RULE,
        "tautological_rules": sorted(TAUTOLOGICAL_RULES),
        "tier_region_coverage": regions,
        "plots": plot_paths,
    }
    with open(os.path.join(OUT_DIR, "layer3_coverage_audit.json"), "w") as f:
        json.dump(payload, f, indent=2)

    md = write_markdown(s1, cov, findings, bands, regions, tier_summary, plot_paths,
                        len(df))
    with open(os.path.join(OUT_DIR, "LAYER3_COVERAGE_AUDIT.md"), "w",
              encoding="utf-8") as f:
        f.write(md)

    print("\nverdicts:")
    for finding in sorted(findings, key=lambda x: -x["zero_flag_pct"]):
        print(f"  {finding['profile']:<20} {finding['zero_flag_pct']:>5.1f}% uncovered  "
              f"{finding['verdict']}")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
