"""
layer3.py — Layer 3 combination-detection rules.

Layer 3 sits on top of the three already-assigned domain tiers (Banking,
Investment, Insurance) and detects cross-domain patterns that no single
domain score can express: a user can look fine on two axes and still be
carrying a real mismatch between them. It is the layer the domain split
was built to make possible (brief Section 5, Step 5 finding: the near-zero
population-wide Banking↔Investment correlation collapses specifically
because of injected contradiction archetypes — "exactly the signal a
future Layer 3 should catch").

This module is deliberately dumb about how tiers are produced. It does NOT
load artifacts, does NOT compute distances, and does NOT map cluster ids to
labels — it consumes tier labels that Layer 2 has already resolved through
each artifact's own per-band ``tier_map``. That separation is load-bearing:
cluster id does not mean the same tier across age bands (Banking's cluster 0
is "Building" in band 1 but "Stable" in bands 2–5), and the only safe place
for that knowledge is the artifact replay, not here.

Validation lineage — every threshold below traces to a specific result:
  - MLAI/clustering/results/layer3_validation/LAYER3_VALIDATION.md
        base rates for all five originally-proposed rules, and the
        net-worth sub-split test that got Rule 1 shipped as one flag.
  - MLAI/clustering/results/layer3_rule4_tightening/RULE4_TIGHTENING.md
        Rule 3's "C2" candidate (the tier-only form fired at 0.96x chance).
  - MLAI/clustering/results/layer3_rule5_tightening/RULE5_TIGHTENING.md
        Rule 4's "D1" candidate (the tier-only form was already sound;
        D1 is an optimisation, not a repair).

Call site: score_pipeline.py, which resolves the three tier labels through
tier_assignment.py and passes them here alongside investment_rate,
savings_rate and emergency_fund_months from scoring.py's compute_* functions.
It only calls this when all three tiers resolved — a rule skipped for a None
tier is indistinguishable from a rule that ran and did not fire.

Acceptance criteria were fixed before the numbers were seen in all three:
lift >= 3.0x, no age band above 25%, and recall discounted wherever the
thresholds are the generator's own override bounds (circular by
construction). The precision figures quoted below are measured on
synthetic data and are dataset-inflated — see RULE5_TIGHTENING.md's
suppression check. They justify the rule shapes; they are not forecasts of
a real-world hit rate.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Tier labels (locked, brief Section 5)
# ---------------------------------------------------------------------------
# The single source of truth for these strings is each domain artifact's
# tier_map in backend/ml/artifacts/domain_split/. They are restated here as
# constants so the rule logic reads in tier language rather than in string
# literals — and verify_tier_labels_against_artifacts() below re-reads the
# artifacts to prove the two have not drifted apart. Never compare a tier
# against a bare string literal anywhere in this module.
BANKING_STABLE = "Stable"
BANKING_BUILDING = "Building"
INVESTMENT_ON_TRACK = "On Track"
INVESTMENT_GETTING_STARTED = "Getting Started"
INSURANCE_COVERED = "Covered"
INSURANCE_NOT_COVERED = "Not Covered"

# Healthiest-first per domain, matching each artifact's tier_labels ordering.
BANKING_TIERS = (BANKING_STABLE, BANKING_BUILDING)
INVESTMENT_TIERS = (INVESTMENT_ON_TRACK, INVESTMENT_GETTING_STARTED)
INSURANCE_TIERS = (INSURANCE_COVERED, INSURANCE_NOT_COVERED)

# ---------------------------------------------------------------------------
# Flag identifiers (the four locked rule names)
# ---------------------------------------------------------------------------
FLAG_WEALTH_WITHOUT_PROTECTION = "Wealth Without Protection"
FLAG_PROTECTION_FIRST = "Protection First"
FLAG_STRONG_FOUNDATION_ROOM_TO_GROW = "Strong Foundation, Room to Grow"
FLAG_AGGRESSIVE_INVESTING_WEAK_FOUNDATION = "Aggressive Investing, Weak Foundation"

# Canonical evaluation order. detect_combination_flags() returns triggered
# flags in this order so the result is deterministic and directly comparable
# between users (e.g. for a cached checklist_flags field).
LAYER3_FLAGS = (
    FLAG_WEALTH_WITHOUT_PROTECTION,
    FLAG_PROTECTION_FIRST,
    FLAG_STRONG_FOUNDATION_ROOM_TO_GROW,
    FLAG_AGGRESSIVE_INVESTING_WEAK_FOUNDATION,
)

# ---------------------------------------------------------------------------
# Thresholds — provenance, not magic numbers
# ---------------------------------------------------------------------------
# Rule 3 ("Strong Foundation, Room to Grow"), all three from the C2 candidate
# in MLAI/clustering/layer3_rule4_tightening.py — see
# results/layer3_rule4_tightening/RULE4_TIGHTENING.md. Each value is one of
# conservative_saver's own override bounds in MISMATCH_TYPES, not a round
# number chosen for readability:
#   investment_rate      override range 0.0–0.03  -> 0.03 is its CEILING
#   savings_rate         override range 0.28–0.55 -> 0.28 is its FLOOR
#   emergency_fund_months override range 5.0–18.0 -> 5.0  is its FLOOR
# C2 result: 5.5% base rate, 76.2% of flagged users are conservative_saver
# (18.28x its 4.17% prevalence), 7.02x the independence-predicted rate,
# worst single age band 6.5%. The tier-only version of this rule fired at
# 28.5% — 0.96x chance, i.e. no signal at all — which is why the three
# continuous conditions exist.
RULE3_INVESTMENT_RATE_MAX = 0.03
RULE3_SAVINGS_RATE_MIN = 0.28
RULE3_EMERGENCY_FUND_MONTHS_MIN = 5.0

# Rule 4 ("Aggressive Investing, Weak Foundation"), from the D1 candidate in
# MLAI/clustering/layer3_rule5_tightening.py — see
# results/layer3_rule5_tightening/RULE5_TIGHTENING.md. This rule targets TWO
# archetypes at once (over_invested, high_debt_investor), so the cut is the
# LOOSER of their two investment_rate floors (0.22 and 0.18 respectively) —
# taking 0.22 would have excluded high_debt_investor by construction.
# D1 result: 8.9% base rate, 93.6% of flagged users are one of the two
# targets (11.24x their combined 8.33% prevalence), 4.35x the
# independence-predicted rate, worst single age band 9.7%.
# Note the arithmetic ceiling on lift differs between the two rules (12.0x
# here at 8.33% prevalence vs 24.0x for Rule 3 at 4.17%) — compare the
# precision figures directly, not the two lift numbers.
RULE4_INVESTMENT_RATE_MIN = 0.18

# Deliberately NOT here: net_worth_to_income_ratio. A net-worth sub-split of
# Rule 1 (to separate wealthy_exposed from underinsured_saver) was tested and
# rejected — the band threshold sits ON the density peak, not in a valley, in
# all 5 age bands, and underinsured_saver is not a distinct mode at all.
# See LAYER3_VALIDATION.md "Verdict on the split assumption". Do not add it.

_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "domain_split"


def detect_combination_flags(
    banking_tier: str | None,
    investment_tier: str | None,
    insurance_tier: str | None,
    investment_rate: float | None = None,
    savings_rate: float | None = None,
    emergency_fund_months: float | None = None,
) -> list[str]:
    """
    Evaluates the four locked Layer 3 combination rules for one user.

    Pure function: no Firestore access, no artifact loading, no I/O. Same
    contract as the compute_* functions in scoring.py — data is passed in,
    and missing data is None rather than a fabricated zero.

    The three continuous argument names match the compute_* return keys
    exactly, so no translation through CLUSTERING_FEATURE_KEY_MAP is needed
    here (that map exists for dti / net_worth_to_income, neither of which any
    Layer 3 rule uses).

    Args:
        banking_tier:    "Stable" or "Building", already resolved by Layer 2
                         through banking_model.json's per-band tier_map.
        investment_tier: "On Track" or "Getting Started", likewise.
        insurance_tier:  "Covered" or "Not Covered", likewise. (Insurance is
                         Approach B — one pooled model, no per-band map.)
        investment_rate: From compute_investment_features()["investment_rate"].
        savings_rate:    From compute_banking_features()["savings_rate"].
        emergency_fund_months:
                         From compute_banking_features()["emergency_fund_months"].

    Returns:
        List of triggered flag identifiers, ordered per LAYER3_FLAGS. Empty
        list when nothing fires — the common case, and not an error state.

    Missing data (None):
        A tier of None means "not computable for this user" (e.g. no
        insurance policies parsed yet), and every rule that reads it is
        simply skipped. Same for the continuous features: Rule 3 needs all
        three, Rule 4 needs investment_rate, and a rule with a missing input
        does not fire. None is never coerced to 0.0 — treating an unknown
        savings rate as zero would silently invent a finding about someone
        whose banking data has not synced.

    Raises:
        ValueError: on a tier string that is not one of that domain's two
        locked labels. This is loud on purpose — a typo or a stale label
        ("Strong", "Weak", lowercased text) would otherwise mean every rule
        involving that domain silently never fires, which looks identical to
        a healthy user.
    """
    _validate_tier("banking_tier", banking_tier, BANKING_TIERS)
    _validate_tier("investment_tier", investment_tier, INVESTMENT_TIERS)
    _validate_tier("insurance_tier", insurance_tier, INSURANCE_TIERS)

    flags: list[str] = []

    # 1. Wealth Without Protection — tier-only by decision (the net-worth
    #    sub-split was tested and rejected). ~31.7% of the synthetic
    #    population: framed to the user as common, not rare.
    if banking_tier == BANKING_STABLE and insurance_tier == INSURANCE_NOT_COVERED:
        flags.append(FLAG_WEALTH_WITHOUT_PROTECTION)

    # 2. Protection First — tier-only, validated as-is. Strongly
    #    band-conditional (33.8% in band 3 vs ~5% in bands 1 and 5), which is
    #    a property of the data, not a defect in the rule.
    if insurance_tier == INSURANCE_COVERED and banking_tier == BANKING_BUILDING:
        flags.append(FLAG_PROTECTION_FIRST)

    # 3. Strong Foundation, Room to Grow — tiers PLUS conservative_saver's
    #    three override bounds. Without the continuous conditions this rule
    #    carries no information (0.96x chance).
    if (
        banking_tier == BANKING_STABLE
        and investment_tier == INVESTMENT_GETTING_STARTED
        and investment_rate is not None
        and savings_rate is not None
        and emergency_fund_months is not None
        and investment_rate <= RULE3_INVESTMENT_RATE_MAX
        and savings_rate >= RULE3_SAVINGS_RATE_MIN
        and emergency_fund_months >= RULE3_EMERGENCY_FUND_MONTHS_MIN
    ):
        flags.append(FLAG_STRONG_FOUNDATION_ROOM_TO_GROW)

    # 4. Aggressive Investing, Weak Foundation — one combined flag covering
    #    both over_invested and high_debt_investor. The two ARE separable
    #    (dti >= 0.40 splits them at 96.0% / 99.6% purity), but splitting was
    #    deliberately deprioritised — future v2 candidate, see the brief.
    if (
        banking_tier == BANKING_BUILDING
        and investment_tier == INVESTMENT_ON_TRACK
        and investment_rate is not None
        and investment_rate >= RULE4_INVESTMENT_RATE_MIN
    ):
        flags.append(FLAG_AGGRESSIVE_INVESTING_WEAK_FOUNDATION)

    # Structural invariant, same spirit as scoring.py's module-level asserts.
    # Rules 1/2 are mutually exclusive through banking_tier, as are rules 3/4,
    # so at most one from each pair can fire. If this ever trips, a rule's
    # tier condition has been edited into an overlap.
    assert len(flags) <= 2, f"rules are not pairwise exclusive any more: {flags}"
    return flags


def _validate_tier(arg_name: str, value: str | None, allowed: tuple[str, ...]) -> None:
    """Rejects unrecognised tier labels; None (not computable) is allowed."""
    if value is not None and value not in allowed:
        raise ValueError(
            f"{arg_name}={value!r} is not one of the locked labels {allowed}. "
            "Tier labels must come from the artifact's tier_map "
            "(backend/ml/artifacts/domain_split/), never from a literal."
        )


def verify_tier_labels_against_artifacts(artifact_dir: Path | None = None) -> dict:
    """
    Re-reads the production artifacts and asserts this module's tier-label
    constants still match them exactly, per domain and — for the two
    Approach A domains — in every age band's tier_map.

    Not called at import (this module is meant to stay I/O-free); it is run by
    backend/scripts/verify_layer3_flags.py. The point is to catch label drift
    between an artifact rebuild and these constants, which would otherwise
    turn every rule into a silent no-op rather than an error.

    Returns:
        Dict mapping domain -> the artifact's declared tier_labels list.
    """
    artifact_dir = artifact_dir or _ARTIFACT_DIR
    expected = {
        "banking": list(BANKING_TIERS),
        "investment": list(INVESTMENT_TIERS),
        "insurance": list(INSURANCE_TIERS),
    }
    declared = {}
    for domain, labels in expected.items():
        model = json.loads(
            (artifact_dir / f"{domain}_model.json").read_text(encoding="utf-8")
        )
        got = model["shipped_tiers_summary"]["tier_labels"]
        if got != labels:
            raise AssertionError(
                f"{domain}: artifact tier_labels {got} != layer3.py constants "
                f"{labels} — the artifact was rebuilt with different wording, "
                "or these constants are stale. Fix before shipping."
            )
        declared[domain] = got

        # Approach A domains carry a per-band tier_map whose labels must all
        # be drawn from the same locked pair, attached by health_rank. This is
        # the check that would catch a rebuild that pinned labels to cluster
        # ids instead (Banking's cluster 0 flips meaning across bands).
        for band, band_value in (model.get("bands") or {}).items():
            for cluster_id, entry in band_value["tier_map"].items():
                if entry["tier_label"] != labels[entry["health_rank"]]:
                    raise AssertionError(
                        f"{domain} band {band} cluster {cluster_id}: label "
                        f"{entry['tier_label']!r} does not match health_rank "
                        f"{entry['health_rank']} -> {labels[entry['health_rank']]!r}"
                    )
    return declared


if __name__ == "__main__":
    print("Layer 3 combination rules — locked (brief Section 5)\n")
    for i, flag in enumerate(LAYER3_FLAGS, start=1):
        print(f"  {i}. {flag}")
    print("\nThresholds (all traced to a validation result):")
    print(f"  rule 3  investment_rate       <= {RULE3_INVESTMENT_RATE_MAX}")
    print(f"  rule 3  savings_rate          >= {RULE3_SAVINGS_RATE_MIN}")
    print(f"  rule 3  emergency_fund_months >= {RULE3_EMERGENCY_FUND_MONTHS_MIN}")
    print(f"  rule 4  investment_rate       >= {RULE4_INVESTMENT_RATE_MIN}")
    print("\nArtifact label check:")
    for domain, labels in verify_tier_labels_against_artifacts().items():
        print(f"  {domain:<11} {labels}")
    print("\nFull test suite: python scripts/verify_layer3_flags.py")
