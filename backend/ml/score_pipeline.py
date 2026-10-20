"""
score_pipeline.py — Phase 3b: the orchestrator that finally makes the score live.

This is the wiring layer, and deliberately nothing more. Every decision it
reports was made and validated somewhere else:

    scoring.compute_*_features()          raw Firestore data -> features
    tier_assignment.assign_domain_tier_*  features + age band -> tier
    layer3.detect_combination_flags()     three tiers -> cross-domain flags
    overall_score.compute_score_breakdown three tiers + 2 flags -> 0-100
    overall_score.evaluate_layer1_checklist            -> the two scored checks

Nothing here re-derives a threshold, a label, a formula or a None-rule. Where a
question like "is this score computable?" already has an answer inside one of
those modules, this module asks it rather than re-implementing the test — most
importantly, ``overall_score`` is None because compute_score_breakdown() said so
on partial input, not because this file checked for missing tiers first.

PURE, BY DESIGN — I/O LIVES IN data/score_profile.py
Every module in backend/ml/ is I/O-free (scoring.py, layer3.py,
overall_score.py; tier_assignment.py reads only its own committed artifacts),
and every Firestore reader/writer lives in backend/data/. This module keeps that
split: it takes already-fetched documents and returns a plain dict.
data/score_profile.py does the fetching, the caching and the reading back. That
is also what makes the whole pipeline testable without Firebase, Plaid or a
network — verify_score_pipeline.py exercises every branch below with constructed
dicts.

THE THREE "UNAVAILABLE" SHAPES, AND WHY THEY ARE NOT ERRORS
An honest "we can't tell you yet, here's why" is the product requirement (brief
Section 8: income = 0 must surface as "score unavailable", never a fabricated
number). So this module never raises for a user-data problem. It returns a
result whose reason code says which of these happened:

  income_not_set        monthly_income_sgd is 0 or absent. Phase 2's compute_*
                        functions raise ValueError on income <= 0; that contract
                        is caught HERE, at the orchestrator boundary, and turned
                        into a reason code. No domain is assessable without
                        income (Banking needs it for savings_rate and dti,
                        Investment for investment_rate, Insurance for all three
                        of its ratios), so this is a whole-pipeline state.
  unsupported_age       date_of_birth resolves to an age outside 20-70.
                        tier_assignment.age_band_for_date_of_birth() raises by
                        default and this module does NOT pass clamp=True: a tier
                        from a model that never saw anyone this user's age is
                        exactly the fabricated number the brief forbids. Also a
                        whole-pipeline state — all three models are age-indexed.
  incomplete_domains    the ordinary partial case. Some domains have tiers, some
                        do not; the ones that do are returned as real results
                        and only the overall score is withheld.

Anything else — an unreadable artifact, a stale tier label, a bug — still
raises. Those are not user-data states and must not be dressed up as one.
"""

import logging
from datetime import date, datetime, timezone

from . import coverage_gaps, layer3, overall_score, scoring, tier_assignment

logger = logging.getLogger(__name__)

# 1.1.0 added ``insurance_coverage`` — a derived, read-only block. No scoring
# behaviour changed with it; a 1.0.0 cache differs from a 1.1.0 one only by
# that key's absence.
# 1.2.0 added ``healthy_debt_ratio`` to ml_profile and ``net_worth_sgd`` to the
# scoreHistory snapshot — both read-only pass-throughs of values already
# computed elsewhere. Again no scoring behaviour changed.
SCHEMA_VERSION = "1.2.0"

# The three domains, in the order every user-facing surface shows them. Same
# spelling as the artifact filename stems and scoring.py's dict keys.
DOMAIN_ORDER = ("banking", "investment", "insurance")

# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------
# Machine-readable, stable, and the ONLY thing the client should branch on —
# copy can be re-worded without breaking routing. Nothing here is a synonym for
# another: each one implies a different next action for the user.
REASON_INCOME_NOT_SET = "income_not_set"            # -> ask for income
REASON_UNSUPPORTED_AGE = "unsupported_age"          # -> nothing the user can do
REASON_DATE_OF_BIRTH_NOT_SET = "date_of_birth_not_set"   # -> finish onboarding
REASON_NO_FINANCIAL_PROFILE = "no_financial_profile"     # -> finish onboarding
REASON_INCOMPLETE_DOMAINS = "incomplete_domains"    # -> connect the missing side
REASON_MISSING_FEATURES = "missing_features"        # per domain, with the list

# Per-domain assessment states.
STATUS_ASSESSED = "assessed"
STATUS_NOT_ASSESSED = "not_assessed"


def compute_score_profile(
    user_doc: dict | None,
    profile: dict | None,
    bank_accounts: list | None,
    investment_accounts: list | None,
    investment_transactions: list | None,
    insurance_policies: list | None,
    now: datetime | None = None,
) -> dict:
    """
    Runs the whole Layer 1 / Layer 2 / Layer 3 / overall-score pipeline for one
    user, from already-fetched Firestore documents.

    Pure function: no Firestore access, no network, no writes. The only file it
    touches is the committed artifact set, via tier_assignment.

    Args:
        user_doc:      users/{uid} dict — date_of_birth and dependants.
        profile:       users/{uid}/financialProfile/profile dict. None means the
                       document does not exist (onboarding incomplete).
        bank_accounts: users/{uid}/plaidAccounts docs.
        investment_accounts:     users/{uid}/investmentAccounts docs.
        investment_transactions: their transactions, flattened.
        insurance_policies:      users/{uid}/insurancePolicies docs.
        now:           Injectable clock for tests; defaults to UTC now.

    Returns:
        A dict — see RESULT SHAPE below. Never raises for a user-data problem.

    RESULT SHAPE
        schema_version    str
        computed_at       tz-aware datetime (UTC)
        age_band          1-5, or None when it could not be determined
        age_band_range    "40-49", or None
        age               the user's age in years, or None
        domains           {domain: {status, tier_index, tier_label, reason,
                                    missing_features}}
                          status is "assessed" or "not_assessed"; a
                          not_assessed domain carries a reason and, for
                          missing_features, exactly which ones were missing.
        layer3_flags      list of active flag names (empty list, never None)
        layer3_evaluated  bool — False means "not all three tiers were
                          available", NOT "evaluated and nothing fired". A
                          client that shows "no issues found" must check this.
        layer3_reason     None, or why it was not evaluated
        overall_score     float 0-100, or None
        overall_score_reason   None when a score is present, else the reason
        score_breakdown   compute_score_breakdown()'s dict, or None
        checklist_flags   evaluate_layer1_checklist()'s list — ALWAYS present,
                          one entry per check, with passed=None where unknown
        insurance_coverage
                          coverage_gaps.compute_coverage_gaps()'s dict, or None
                          when the insurance features could not be computed.
                          DERIVED, READ-ONLY: the two continuous coverage
                          ratios beside the age-band benchmark each is measured
                          against. Nothing in this pipeline reads it back — it
                          exists so the Insurance screen's Coverage Gaps
                          section can say how far short a user is without
                          re-deriving a ratio the backend already owns.
        healthy_debt_ratio
                          compute_banking_features()'s own value, or None when
                          it was not computable (no classifiable debt, or no
                          bank accounts). None does not by itself withhold the
                          Banking tier: with savings_rate, dti and
                          emergency_fund_months present, tier_assignment scores
                          those three, so a debt-free user can be "assessed"
                          with this None beside the tier.
                          DERIVED, READ-ONLY, same standing as
                          insurance_coverage: read by nothing below, published
                          so Home can show the debt split the Banking tier was
                          assigned from without a second Dart classification.
        net_worth_sgd     financialProfile's net_worth_sgd, passed through
                          unchanged for the scoreHistory snapshot. Owned and
                          computed solely by data/financial_profile.py; None
                          only when there is no profile document at all.
        unavailable_reason
                          None, or the whole-pipeline reason (income_not_set /
                          unsupported_age / date_of_birth_not_set /
                          no_financial_profile). When set, every domain is
                          not_assessed for the same reason.
    """
    now = now or datetime.now(timezone.utc)
    result = _empty_result(now)

    # --- profile document: existence is the "user has onboarded" signal, the
    # same rule refresh_financial_profile() uses to decide whether to write.
    if profile is None:
        return _unavailable(result, REASON_NO_FINANCIAL_PROFILE)

    # Net worth does not depend on income or age, so it is recorded before
    # either guard: a day whose score is unavailable still has a real net worth.
    # Copied, never recomputed — financial_profile.compute_net_worth() is its
    # one source, and a second calculation here could disagree with the value
    # Home displays from the same document.
    result["net_worth_sgd"] = profile.get("net_worth_sgd")

    # --- age band. Required by ALL THREE domains: it selects the fitted model
    # for Banking/Investment and supplies Insurance's fourth input dimension
    # (tier_assignment's Phase 3a finding — Insurance is pooled but NOT
    # age-blind). So a bad age is fatal to the whole pipeline, not to one third.
    date_of_birth = (user_doc or {}).get("date_of_birth")
    if date_of_birth is None:
        return _unavailable(result, REASON_DATE_OF_BIRTH_NOT_SET)
    try:
        # clamp=False deliberately — see the module docstring.
        age_band = tier_assignment.age_band_for_date_of_birth(
            date_of_birth, today=now.date())
    except ValueError as exc:
        logger.info("score unavailable: %s", exc)
        return _unavailable(result, REASON_UNSUPPORTED_AGE)
    except TypeError as exc:
        # A date_of_birth that is not a date at all is a data-shape bug, not a
        # user state — but it is still not worth a 500 on a score request.
        logger.warning("score unavailable: unusable date_of_birth %r (%s)",
                       date_of_birth, exc)
        return _unavailable(result, REASON_DATE_OF_BIRTH_NOT_SET)

    age = _age_in_years(date_of_birth, now.date())
    low, high = tier_assignment.AGE_BAND_RANGES[age_band]
    # tier_assignment owns the band definition; this is the local age
    # calculation (needed for risk_age_deviation, which wants exact years, not a
    # band) proving it agrees with that owner rather than drifting from it.
    assert low <= age <= high, (
        f"local age {age} is outside band {age_band} ({low}-{high}) — "
        "score_pipeline's age arithmetic has drifted from "
        "tier_assignment.age_band_for_date_of_birth()")
    result["age"] = age
    result["age_band"] = age_band
    result["age_band_range"] = f"{low}-{high}"

    # --- income. annual_income_from_financial_profile() raises when the field
    # is absent; the compute_* functions raise on <= 0. Both are the same user
    # state, and both are caught here rather than allowed to 500.
    try:
        annual_income = scoring.annual_income_from_financial_profile(profile)
        if annual_income <= 0:
            raise ValueError("monthly_income_sgd is 0")
    except ValueError as exc:
        logger.info("score unavailable for this user: %s", exc)
        return _unavailable(result, REASON_INCOME_NOT_SET)

    # --- features, per domain, through the Phase 2 functions unchanged.
    features_by_domain = _compute_features(
        profile, bank_accounts or [], investment_accounts or [],
        investment_transactions or [], insurance_policies or [],
        annual_income, age, now)

    # --- Layer 2. A domain is assessed only when tier_assignment returns a
    # tier; it returns None whenever any required feature is missing, which is
    # the check this module reuses rather than duplicating.
    for domain in DOMAIN_ORDER:
        result["domains"][domain] = _assign_domain(
            domain, features_by_domain[domain], age_band)

    tiers = {d: result["domains"][d]["tier_label"] for d in DOMAIN_ORDER}
    all_assessed = all(tiers[d] is not None for d in DOMAIN_ORDER)

    # --- Layer 3. Needs all three tiers by design: every rule reads at least
    # two domains, and detect_combination_flags() silently skips rules with a
    # None tier — which would look identical to "evaluated, nothing fired".
    # Rather than publish that ambiguity, it is only run when complete, and
    # layer3_evaluated records which of the two happened.
    banking_features = features_by_domain["banking"]
    investment_features = features_by_domain["investment"]

    # The exact number the Banking tier was just assigned from — not recomputed.
    result["healthy_debt_ratio"] = banking_features.get("healthy_debt_ratio")

    if all_assessed:
        result["layer3_flags"] = layer3.detect_combination_flags(
            tiers["banking"], tiers["investment"], tiers["insurance"],
            investment_rate=investment_features.get("investment_rate"),
            savings_rate=banking_features.get("savings_rate"),
            emergency_fund_months=banking_features.get("emergency_fund_months"),
        )
        result["layer3_evaluated"] = True
    else:
        result["layer3_reason"] = REASON_INCOMPLETE_DOMAINS

    # --- Layer 1 checklist. Always returned, including when its inputs are
    # unknown: evaluate_layer1_checklist() emits passed=None entries rather than
    # dropping rows, because a missing row reads as a passing one.
    insurance_features = features_by_domain["insurance"]
    result["checklist_flags"] = overall_score.evaluate_layer1_checklist(
        insurance_features.get("income_protection_coverage"),
        insurance_features.get("hospitalisation_coverage"),
        (user_doc or {}).get("dependants"),
    )

    # --- Coverage Gaps. Strictly derived from features that are already
    # computed, and read by nothing below: the two continuous coverage ratios
    # published next to the age-band benchmark each is measured against, so the
    # Insurance screen can show HOW SHORT rather than only the binary tier.
    # Reached only past the income guard above, so the ratios are real numbers.
    result["insurance_coverage"] = coverage_gaps.compute_coverage_gaps(
        insurance_features, annual_income, age_band)

    # --- the overall score. compute_score_breakdown() returns None if ANY of
    # its five inputs is None; that is the availability test, reused verbatim.
    # This module never pre-checks "are all three tiers present?" and skips the
    # call — the formula module owns that rule.
    breakdown = overall_score.compute_score_breakdown(
        tiers["banking"], tiers["investment"], tiers["insurance"],
        insurance_features.get("income_protection_coverage"),
        insurance_features.get("hospitalisation_coverage"),
    )
    if breakdown is None:
        result["overall_score_reason"] = REASON_INCOMPLETE_DOMAINS
    else:
        result["score_breakdown"] = breakdown
        result["overall_score"] = breakdown["overall_score"]

    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _empty_result(now: datetime) -> dict:
    """The full shape, with every field present. A result never gains or loses
    keys between branches — the client renders one schema, not four."""
    return {
        "schema_version": SCHEMA_VERSION,
        "computed_at": now,
        "age": None,
        "age_band": None,
        "age_band_range": None,
        "domains": {
            domain: _domain_result(STATUS_NOT_ASSESSED, reason=None)
            for domain in DOMAIN_ORDER
        },
        "layer3_flags": [],
        "layer3_evaluated": False,
        "layer3_reason": None,
        "overall_score": None,
        "overall_score_reason": None,
        "score_breakdown": None,
        "checklist_flags": overall_score.evaluate_layer1_checklist(
            None, None, None),
        # None, never an empty dict: "the ratios could not be computed" and
        # "computed, and both are zero" are different things to show, and a
        # zero-policy user genuinely produces the second.
        "insurance_coverage": None,
        # None means "not computable", never 0.0 — 0.0 is a real ratio (all
        # classified debt is unhealthy).
        "healthy_debt_ratio": None,
        "net_worth_sgd": None,
        "unavailable_reason": None,
    }


def _domain_result(status: str, reason: str | None, tier_index: int | None = None,
                   tier_label: str | None = None,
                   missing_features: list | None = None) -> dict:
    return {
        "status": status,
        "tier_index": tier_index,
        "tier_label": tier_label,
        "reason": reason,
        "missing_features": missing_features or [],
    }


def _unavailable(result: dict, reason: str) -> dict:
    """A whole-pipeline unavailable state: no domain is assessable, so every
    domain carries the SAME reason rather than a vaguer per-domain one."""
    result["unavailable_reason"] = reason
    result["overall_score_reason"] = reason
    result["layer3_reason"] = reason
    for domain in DOMAIN_ORDER:
        result["domains"][domain] = _domain_result(STATUS_NOT_ASSESSED, reason)
    return result


def _compute_features(
    profile: dict,
    bank_accounts: list,
    investment_accounts: list,
    investment_transactions: list,
    insurance_policies: list,
    annual_income: float,
    age: int,
    now: datetime,
) -> dict:
    """Phase 2's compute_* functions, called unchanged. Returns their raw output
    per domain (compute-function key spelling, not artifact spelling).

    ``now`` is the run's own clock. Investment is the only time-windowed
    feature (investment_rate's 30-day deposit window), so it is the only call
    that takes it — without it, an injected test clock never reached the
    window and a fixture deposit aged out against the real date."""
    return {
        "banking": scoring.compute_banking_features(
            profile, bank_accounts, annual_income),
        "investment": scoring.compute_investment_features(
            investment_accounts, investment_transactions, annual_income, age,
            now=now),
        # Zero policies is a REAL state, not an unknown one: with no policies,
        # every ratio is a true 0.0 and both coverage flags a true 0, so the
        # user gets a genuine "Not Covered" tier. That is the locked semantic —
        # see verify_score_pipeline.py's zero-policy section, which pins it.
        "insurance": scoring.compute_insurance_features(
            insurance_policies, annual_income),
    }


def _assign_domain(domain: str, computed_features: dict, age_band: int) -> dict:
    """
    One domain's Layer 2 result.

    The feature dict handed to tier_assignment is keyed by the ARTIFACT's names,
    built from scoring.py's own per-domain list and CLUSTERING_FEATURE_KEY_MAP —
    Banking's ``dti`` is ``debt_to_income_ratio`` on the compute side, and
    renaming it by hand here is exactly the silent mismatch the map exists to
    prevent.
    """
    wanted = scoring.LAYER2_CLUSTERING_FEATURES_BY_DOMAIN[domain]
    features = {
        name: computed_features.get(
            scoring.CLUSTERING_FEATURE_KEY_MAP.get(name, name))
        for name in wanted
    }
    tier_index = tier_assignment.assign_domain_tier_index(
        domain, features, age_band)
    if tier_index is None:
        return _domain_result(
            STATUS_NOT_ASSESSED, REASON_MISSING_FEATURES,
            missing_features=[n for n in wanted if features[n] is None])
    # The label comes from the artifact via tier_assignment, never from a
    # literal and never by indexing a hardcoded list.
    return _domain_result(
        STATUS_ASSESSED, None, tier_index=tier_index,
        tier_label=tier_assignment.tier_labels(domain)[tier_index])


def _age_in_years(date_of_birth: date | datetime, today: date) -> int:
    """
    Age in whole years — the same rule as AppUser.age and
    tier_assignment.age_band_for_date_of_birth().

    Needed separately from the age BAND because compute_investment_features()
    takes exact years (risk_age_deviation's target is (100 - age) / 100), while
    the models are indexed by band. tier_assignment remains the authority on the
    band; compute_score_profile() asserts the two agree.
    """
    if isinstance(date_of_birth, datetime):
        date_of_birth = date_of_birth.date()
    return (today.year - date_of_birth.year
            - ((today.month, today.day)
               < (date_of_birth.month, date_of_birth.day)))


if __name__ == "__main__":
    from datetime import timedelta

    NOW = datetime.now(timezone.utc)
    DOB = NOW.date() - timedelta(days=365 * 42)
    USER = {"date_of_birth": DOB, "dependants": 2}
    PROFILE = {"monthly_income_sgd": 5000.0, "monthly_expenses_sgd": 2500.0,
               "net_worth_sgd": 120000.0}
    ACCOUNTS = [
        {"account_type": "depository", "account_subtype": "checking",
         "balance_current": 30000.0},
        {"account_type": "credit", "account_subtype": "credit card",
         "balance_current": 4000.0,
         "liability": {"monthly_payment_amount": 300.0}},
    ]
    INVESTMENTS = [{"total_value_sgd": 50000.0, "concentration_risk": 0.25,
                    "asset_allocation": {"equities": 0.6}}]
    TXNS = [{"type": "deposit", "amount": -600.0, "date": NOW}]
    POLICIES = [{"policy_type": "life", "sum_assured_sgd": 500000.0,
                 "annual_premium_sgd": 1800.0},
                {"policy_type": "hospitalisation", "annual_premium_sgd": 600.0}]

    print("score_pipeline — one worked example\n")
    out = compute_score_profile(USER, PROFILE, ACCOUNTS, INVESTMENTS, TXNS,
                                POLICIES, now=NOW)
    print(f"  age {out['age']}  band {out['age_band']} "
          f"({out['age_band_range']})")
    for domain in DOMAIN_ORDER:
        d = out["domains"][domain]
        print(f"  {domain:<11} {d['status']:<13} {d['tier_label'] or '—'}"
              + (f"  missing: {d['missing_features']}"
                 if d["missing_features"] else ""))
    print(f"  layer3       {out['layer3_flags'] or 'no flags'} "
          f"(evaluated: {out['layer3_evaluated']})")
    print(f"  overall      {out['overall_score']} "
          f"({out['overall_score_reason'] or 'available'})")
    for flag in out["checklist_flags"]:
        print(f"  layer1       [{flag['passed']}] {flag['check']}")

    print("\n  the three unavailable states:")
    for label, kwargs in [
        ("income_not_set", {"profile": {"monthly_income_sgd": 0.0}}),
        ("unsupported_age", {"user_doc": {"date_of_birth":
                                          NOW.date() - timedelta(days=365 * 80),
                                          "dependants": 0}}),
        ("no_financial_profile", {"profile": None}),
    ]:
        args = {"user_doc": USER, "profile": PROFILE, "bank_accounts": ACCOUNTS,
                "investment_accounts": INVESTMENTS,
                "investment_transactions": TXNS,
                "insurance_policies": POLICIES, "now": NOW, **kwargs}
        print(f"    {label:<22} -> "
              f"{compute_score_profile(**args)['unavailable_reason']}")

    print("\nFull test suite: python scripts/verify_score_pipeline.py")
