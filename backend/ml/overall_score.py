"""
overall_score.py — the locked overall-score formula and the Layer 1 checklist.

Two things live here, both pure functions over already-resolved inputs:

  compute_overall_score(...)     the locked non-compensatory 0-100 formula
  evaluate_layer1_checklist(...) the two scored Layer 1 flags + their copy

They share a module because they share inputs: income_protection_coverage and
hospitalisation_coverage are simultaneously two thirds of the score's Insurance
component and the two user-facing checklist checks. Splitting them apart would
mean two files reading the same two booleans and disagreeing over time.

This is a sibling of layer3.py rather than part of scoring.py for the same
reason layer3.py is: scoring.py computes FEATURES from raw Firestore-shaped
data. This module consumes tiers and flags that something else has already
resolved, and never touches an account, a policy or an artifact.

WHY THE FORMULA IS SHAPED THIS WAY (brief Section 5, "Overall score formula —
locked", July 2026, 3 validation rounds — do not re-derive it here):
  Round 1 rejected continuous per-domain scores: Insurance INVERTS (every
  candidate ordered better-covered users below worse-covered ones inside the
  same tier), so averaging three continuous scores fails in exactly the domain
  the app exists to surface.
  Round 2 rejected plain Option B (equal domain thirds). Averaging three
  binaries into one slot shrinks that slot's variance — realised influence came
  out 41.3/45.3/13.5, not 33/33/33 — and Banking-Stable users with zero
  insurance scored 3.13 points ABOVE the normal-population mean. That is the
  same compensatory failure that disqualified the original cluster-derived
  centroid, resurfacing one layer up in how components combine.
  Round 3 rejected gating the BASE (mean of banking + investment only): it
  makes insurance purely punitive, and the protection_first archetype fell to
  last place, below the normal population.
  What shipped caps Option B's own score — a score that already counts
  insurance POSITIVELY — rather than gating a base that ignores it. Insurance
  can lift you and can cap you; it can never be bought with banking strength.

Validation lineage — every number below traces to a specific result:
  - MLAI/clustering/results/gated_score_validation/GATED_SCORE_VALIDATION.md
        the locked ceiling ladder, the three harshness settings it was chosen
        against, and the key check (zero-insurance/strong-banking group lands
        5.55 points BELOW the normal mean; cap fires on 11.26% of users).
  - MLAI/clustering/results/overall_score_validation/OVERALL_SCORE_VALIDATION.md
        Option B's compensatory failure, and the has_investments drop that
        scoring.py's LAYER1_CHECKLIST_FEATURES now reflects.

Call site: score_pipeline.py, same as layer3.py. It resolves the three tier
labels through tier_assignment.py, reads dependants from users/{uid} and the
two coverage flags from scoring.compute_insurance_features(), then calls
compute_score_breakdown() and evaluate_layer1_checklist() and caches the
results onto users/{uid}/financialProfile/profile per brief Section 8.
"""

# layer3.py is the single place tier-label strings are written down in backend/;
# importing them keeps one source of truth and lets its
# verify_tier_labels_against_artifacts() cover this module too. The fallback is
# only for running this file directly as a script (__main__ block below).
try:
    from .layer3 import (
        BANKING_STABLE, BANKING_TIERS,
        INSURANCE_COVERED, INSURANCE_TIERS,
        INVESTMENT_ON_TRACK, INVESTMENT_TIERS,
        _validate_tier,
    )
except ImportError:  # pragma: no cover — direct `python ml/overall_score.py`
    from layer3 import (  # type: ignore[no-redef]
        BANKING_STABLE, BANKING_TIERS,
        INSURANCE_COVERED, INSURANCE_TIERS,
        INVESTMENT_ON_TRACK, INVESTMENT_TIERS,
        _validate_tier,
    )

# ---------------------------------------------------------------------------
# The two 0/100 domain components
# ---------------------------------------------------------------------------
# Binary by design, not by simplification: both Banking and Investment ship as
# two tiers (brief Section 5, "FINAL ALGORITHM/TIER DECISIONS — LOCKED"), and
# Round 1 established there is no defensible continuous score inside a tier.
DOMAIN_SCORE_MAX = 100.0
DOMAIN_SCORE_MIN = 0.0

# ---------------------------------------------------------------------------
# The Insurance third — three equally-weighted 0/100 indicators
# ---------------------------------------------------------------------------
# insurance_tier == "Covered" is the Layer 2 clustering verdict (life/CI
# coverage ratios + insurance spend ratio); the other two are the Layer 1
# checklist's scored binary flags. Three indicators, mean, so the third takes
# exactly four values: 0, 33.33, 66.67, 100.
INSURANCE_INDICATOR_COUNT = 3

# ---------------------------------------------------------------------------
# The ceiling ladder — NOT invented here
# ---------------------------------------------------------------------------
# The VALUES are the project's own historical TIER_THRESHOLDS from
# SYNTHETIC_IDEALS (0-39 At Risk / 40-69 Building / 70-100 Stable). The
# distance-to-centroid mechanism those bands belonged to is superseded, but the
# boundaries are the only score banding this project has ever written down, so
# the ceilings are taken from them rather than made up:
#     39.0  = the top of At Risk  -> a user capped here can never read "Building"
#     69.0  = the top of Building -> a user capped here can never read "Stable"
#     100.0 = no cap
#
# Honesty, at the same standard applied to SYNTHETIC_IDEALS itself:
#   GROUNDED: the three values, per above.
#   GROUNDED: "zero insurance coverage must never read as Stable" — follows
#             directly from the brief's stated principle that critical gaps
#             surface regardless of overall score.
#   JUDGEMENT CALL, no independent grounding: the MAPPING below, i.e. that
#             clearing 0 of 3 costs two bands, 1 of 3 costs one band, and 2 of 3
#             costs nothing. gated_score_validation.py ran this at three
#             harshness settings (lenient / band_ladder / strict) and reported
#             the sensitivity; "band_ladder" is the locked primary. It is a
#             defensible reading of proportionality, not a benchmark.
#
# Keyed on the integer count of indicators cleared rather than on the
# insurance_third float, deliberately: 2/3 is not exactly representable, and a
# ladder keyed on 66.666... would be one rounding change away from silently
# handing a 69 cap to a user who cleared two indicators.
CEILING_BY_INDICATORS_CLEARED = {0: 39.0, 1: 69.0, 2: 100.0, 3: 100.0}

# ---------------------------------------------------------------------------
# Layer 1 checklist — 2 scored checks, dependants as a tone selector only
# ---------------------------------------------------------------------------
# The full Layer 1 set is 3 (scoring.py's LAYER1_CHECKLIST_FEATURES), but only
# these two are ever scored. dependants is the third, and it is NOT a check: it
# never passes, never fails, and contributes nothing to any score. Three
# validation rounds confirmed it has no home in any tested formula, which is the
# expected result, not a gap. Its entire job is choosing which failure message a
# failed check shows. There is deliberately NO numeric or threshold logic on it
# anywhere in this module — no "dependants >= 2" tier, no coverage multiple.
CHECK_INCOME_PROTECTION = "income_protection_coverage"
CHECK_HOSPITALISATION = "hospitalisation_coverage"

# Canonical evaluation order, so a cached checklist_flags array is deterministic
# and directly comparable between users and between runs.
LAYER1_CHECKS = (CHECK_INCOME_PROTECTION, CHECK_HOSPITALISATION)

# Message copy. Two failure variants per check, selected on whether the user has
# anyone depending on them — the consequence of the gap is genuinely different,
# not just worded differently. Pass copy does not branch: "you have it" is the
# same fact either way. Tone follows the tier-label design principle from the
# brief — describe the situation, never grade the person.
_PASS_MESSAGES = {
    CHECK_INCOME_PROTECTION:
        "You have income protection or disability cover in place.",
    CHECK_HOSPITALISATION:
        "You have hospitalisation cover beyond MediShield Life.",
}
_FAIL_MESSAGES_NO_DEPENDANTS = {
    CHECK_INCOME_PROTECTION:
        "You have no income protection cover. If illness or injury stopped your "
        "pay, there is nothing in place to replace it.",
    CHECK_HOSPITALISATION:
        "You have no hospitalisation cover beyond MediShield Life. A large "
        "hospital bill would come out of your own savings.",
}
_FAIL_MESSAGES_WITH_DEPENDANTS = {
    CHECK_INCOME_PROTECTION:
        "You have no income protection cover. If illness or injury stopped your "
        "pay, the people who depend on you would feel it first.",
    CHECK_HOSPITALISATION:
        "You have no hospitalisation cover beyond MediShield Life. A large "
        "hospital bill would come out of the savings your household relies on.",
}
# Not a third failure mode — "we don't know yet". Kept distinct so an
# incomplete profile never reads as a confirmed protection gap.
#
# ⚠️ This copy was corrected in July 2026 (Phase 3b). It previously said "No
# insurance policies have been added yet", which described a state that CANNOT
# reach this branch: compute_insurance_features() returns real zeros for a user
# with zero policies — a genuine "Not Covered" tier and two genuinely FAILED
# checks, not unknown ones. Both coverage flags only arrive here as None when
# the insurance features could not be computed at all, and the thing that stops
# them being computed is income: every insurance ratio divides by it, so income
# = 0 or unset makes the whole set unavailable (ml/score_pipeline.py's
# income_not_set path). Income is therefore what the user needs to act on, and
# what this copy now names.
#
# The same branch is also reached on score_pipeline's other whole-pipeline
# unavailable states (unsupported_age, date_of_birth_not_set,
# no_financial_profile), where "income" is not literally the missing piece —
# all four are "onboarding is not finished yet", and income is both the
# dominant case and the only one the user can fix from the checklist itself.
_UNKNOWN_MESSAGES = {
    CHECK_INCOME_PROTECTION:
        "Your monthly income has not been set yet, so income protection cover "
        "cannot be checked.",
    CHECK_HOSPITALISATION:
        "Your monthly income has not been set yet, so hospitalisation cover "
        "cannot be checked.",
}


def compute_overall_score(
    banking_tier: str | None,
    investment_tier: str | None,
    insurance_tier: str | None,
    income_protection_coverage: bool | int | float | None,
    hospitalisation_coverage: bool | int | float | None,
) -> float | None:
    """
    The locked overall Financial Health Score, 0-100.

        banking_score    = 100 if banking_tier    == "Stable"   else 0
        investment_score = 100 if investment_tier == "On Track" else 0
        insurance_third  = 100 * mean(insurance_tier == "Covered",
                                      income_protection_coverage,
                                      hospitalisation_coverage)
        option_b_score   = (banking + investment + insurance_third) / 3
        overall_score    = min(option_b_score, ceiling(insurance_third))

    Pure function: no Firestore access, no artifact loading, no I/O. Same
    contract as scoring.py's compute_* functions and layer3.py — data is passed
    in, and missing data is None rather than a fabricated zero.

    Args:
        banking_tier:    "Stable" or "Building", already resolved by Layer 2
                         through banking_model.json's per-band tier_map.
        investment_tier: "On Track" or "Getting Started", likewise.
        insurance_tier:  "Covered" or "Not Covered", likewise.
        income_protection_coverage:
                         From compute_insurance_features(). 0/1, or the
                         equivalent bool.
        hospitalisation_coverage:
                         From compute_insurance_features(). 0/1, or bool.

    Returns:
        A float in [0, 100], or None when the score is not computable — see
        below. Not rounded: rounding is a presentation decision, and the raw
        value is what the validation figures are stated against.

    Missing data (None):
        ALL FIVE inputs are required. If any is None the function returns None
        — "no score yet" — and never substitutes a default. This is stricter
        than layer3.py, and deliberately so: a Layer 3 rule with a missing
        input simply does not fire, which is a safe silence, whereas a score
        computed from four of five inputs is a number a user would read and act
        on. Treating a missing hospitalisation flag as 0 would invent a
        protection gap; treating it as 1 would hide a real one. Both are worse
        than showing nothing. The caller decides what "no score yet" looks like.

    Raises:
        ValueError: on a tier string that is not one of that domain's two
        locked labels, or on a coverage flag that is not 0/1/True/False. Loud
        on purpose, for the same reason as layer3.py: a stale label would
        otherwise silently score as "not the good tier", which is
        indistinguishable from a genuinely weak user.
    """
    breakdown = compute_score_breakdown(
        banking_tier, investment_tier, insurance_tier,
        income_protection_coverage, hospitalisation_coverage,
    )
    return None if breakdown is None else breakdown["overall_score"]


def compute_score_breakdown(
    banking_tier: str | None,
    investment_tier: str | None,
    insurance_tier: str | None,
    income_protection_coverage: bool | int | float | None,
    hospitalisation_coverage: bool | int | float | None,
) -> dict | None:
    """
    compute_overall_score()'s working, exposed.

    Same inputs, same validation, same None contract — this is where the
    arithmetic actually happens; compute_overall_score() is a thin reader of
    ``overall_score``. Separate because the intermediate values are worth
    caching and displaying: which of the three insurance indicators a user
    cleared is exactly the "critical gaps surface regardless of overall score"
    information the brief requires, and ``capped`` answers "would this user
    score higher if they closed their protection gap?" without recomputing.

    Returns:
        None when any input is None, else a dict of:
            banking_score, investment_score  0.0 or 100.0
            insurance_indicators             the three 0/1 indicators, in the
                                             order tier / income protection /
                                             hospitalisation
            insurance_indicators_cleared     0-3
            insurance_third                  0-100
            option_b_score                   the uncapped score
            ceiling                          39.0 / 69.0 / 100.0
            capped                           True when the ceiling bound
            overall_score                    min(option_b_score, ceiling)
    """
    _validate_tier("banking_tier", banking_tier, BANKING_TIERS)
    _validate_tier("investment_tier", investment_tier, INVESTMENT_TIERS)
    _validate_tier("insurance_tier", insurance_tier, INSURANCE_TIERS)
    income_protection = _as_indicator(
        "income_protection_coverage", income_protection_coverage)
    hospitalisation = _as_indicator(
        "hospitalisation_coverage", hospitalisation_coverage)

    if None in (banking_tier, investment_tier, insurance_tier,
                income_protection, hospitalisation):
        return None

    banking_score = (
        DOMAIN_SCORE_MAX if banking_tier == BANKING_STABLE else DOMAIN_SCORE_MIN)
    investment_score = (
        DOMAIN_SCORE_MAX if investment_tier == INVESTMENT_ON_TRACK
        else DOMAIN_SCORE_MIN)

    indicators = (
        1.0 if insurance_tier == INSURANCE_COVERED else 0.0,
        income_protection,
        hospitalisation,
    )
    assert len(indicators) == INSURANCE_INDICATOR_COUNT
    cleared = int(sum(indicators))
    insurance_third = DOMAIN_SCORE_MAX * sum(indicators) / INSURANCE_INDICATOR_COUNT

    option_b_score = (banking_score + investment_score + insurance_third) / 3.0
    ceiling = CEILING_BY_INDICATORS_CLEARED[cleared]

    return {
        "banking_score": banking_score,
        "investment_score": investment_score,
        "insurance_indicators": indicators,
        "insurance_indicators_cleared": cleared,
        "insurance_third": insurance_third,
        "option_b_score": option_b_score,
        "ceiling": ceiling,
        "capped": option_b_score > ceiling,
        "overall_score": min(option_b_score, ceiling),
    }


def evaluate_layer1_checklist(
    income_protection_coverage: bool | int | float | None,
    hospitalisation_coverage: bool | int | float | None,
    dependants: int | None,
) -> list[dict]:
    """
    The Layer 1 rule-based checklist: two scored binary checks, plus dependants
    as a tone selector that is never itself scored.

    Pure function, no I/O. Always returns one entry per check in LAYER1_CHECKS
    order, including checks that could not be evaluated — a checklist that
    silently drops a row reads as "you're fine on that one".

    Args:
        income_protection_coverage: From compute_insurance_features(). 0/1/bool.
        hospitalisation_coverage:   From compute_insurance_features(). 0/1/bool.
        dependants: From the users/{uid} document, collected at onboarding.
            TONE ONLY. It selects which failure message a failed check shows
            (0 vs 1+) and does nothing else — it cannot make a check pass or
            fail, and no threshold is applied to its value beyond "> 0".

    Returns:
        A list of dicts, one per check:
            check    the check identifier (a LAYER1_CHECKS member)
            passed   True / False, or None when the input was not available
            message  user-facing copy for that outcome
            scored   always True here — every entry in this list feeds the
                     overall score's Insurance third. Present so a future
                     unscored/advisory check cannot be mistaken for a scored
                     one by whatever renders the list.

    Raises:
        ValueError: on a coverage flag that is not 0/1/True/False, or on a
        negative dependants count. Same principle as the tier strings: a value
        that cannot be interpreted is not quietly interpreted anyway.
    """
    if dependants is not None:
        if isinstance(dependants, bool) or not isinstance(dependants, int):
            raise ValueError(
                f"dependants={dependants!r} must be an int or None; it is read "
                "verbatim from the users/{uid} document's dependants field.")
        if dependants < 0:
            raise ValueError(f"dependants={dependants!r} cannot be negative.")

    # None means the field was never collected. It falls back to the neutral
    # copy because a message must be shown either way, and it CANNOT change any
    # pass/fail outcome — this is a copy default, not a fabricated score input.
    has_dependants = bool(dependants) if dependants is not None else False

    values = {
        CHECK_INCOME_PROTECTION: _as_indicator(
            CHECK_INCOME_PROTECTION, income_protection_coverage),
        CHECK_HOSPITALISATION: _as_indicator(
            CHECK_HOSPITALISATION, hospitalisation_coverage),
    }

    flags = []
    for check in LAYER1_CHECKS:
        value = values[check]
        if value is None:
            passed, message = None, _UNKNOWN_MESSAGES[check]
        elif value == 1.0:
            passed, message = True, _PASS_MESSAGES[check]
        else:
            passed = False
            message = (_FAIL_MESSAGES_WITH_DEPENDANTS if has_dependants
                       else _FAIL_MESSAGES_NO_DEPENDANTS)[check]
        flags.append({
            "check": check,
            "passed": passed,
            "message": message,
            "scored": True,
        })

    assert len(flags) == len(LAYER1_CHECKS)
    return flags


def _as_indicator(arg_name: str, value) -> float | None:
    """
    Normalises a binary coverage flag to 1.0 / 0.0, preserving None.

    compute_insurance_features() returns these as ints, but a Firestore round
    trip can hand back a bool or a float, so all three are accepted. Anything
    else raises rather than being coerced: a 2, a 0.5 or a "yes" reaching here
    means an upstream bug, and truthiness would turn every one of them into a
    silent pass.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and value in (0, 1):
        return float(value)
    raise ValueError(
        f"{arg_name}={value!r} must be 0, 1, True, False or None — a binary "
        "coverage flag as returned by compute_insurance_features()."
    )


if __name__ == "__main__":
    print("Overall score formula — locked (brief Section 5)\n")
    print("  overall_score = min(option_b_score, ceiling(insurance_third))\n")
    print("Ceiling ladder (values from the project's own TIER_THRESHOLDS):")
    for cleared, ceiling in CEILING_BY_INDICATORS_CLEARED.items():
        third = 100.0 * cleared / INSURANCE_INDICATOR_COUNT
        print(f"  {cleared} of 3 indicators cleared  (insurance_third "
              f"{third:6.2f})  ->  ceiling {ceiling:5.1f}")

    print("\nThe pattern the formula exists to fix "
          "(Banking=Stable, Investment=On Track, zero insurance):")
    b = compute_score_breakdown(
        BANKING_STABLE, INVESTMENT_ON_TRACK, "Not Covered", 0, 0)
    print(f"  option_b_score {b['option_b_score']:.2f}  ceiling "
          f"{b['ceiling']:.1f}  ->  overall_score {b['overall_score']:.2f} "
          f"(capped: {b['capped']})")

    print("\nLayer 1 checklist — both checks failing, by dependants:")
    for dependants in (0, 2):
        print(f"  dependants={dependants}:")
        for flag in evaluate_layer1_checklist(0, 0, dependants):
            print(f"    [{'PASS' if flag['passed'] else 'FAIL'}] {flag['message']}")

    print("\nFull test suite: python scripts/verify_overall_score.py")
