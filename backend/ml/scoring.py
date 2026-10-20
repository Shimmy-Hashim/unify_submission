"""
scoring.py — computes the Layer 1 and Layer 2 features from raw Firestore data.

Turns a user's stored Plaid accounts, investment holdings and parsed insurance
policies into the named features the rest of the pipeline consumes:

    compute_banking_features()     savings_rate, emergency_fund_months,
                                   debt_to_income_ratio, healthy_debt_ratio,
                                   net_worth_to_income_ratio
    compute_investment_features()  has_investments, investment_rate,
                                   portfolio_concentration, risk_age_deviation
    compute_insurance_features()   life / CI coverage ratios, the two binary
                                   coverage flags, insurance_spend_ratio

All three are pure — data is passed in, nothing is read from Firestore here,
and a feature whose inputs are missing comes back None rather than a fabricated
zero. Turning features into tiers is tier_assignment.py's job; combining tiers
into a score is overall_score.py's.

This module also owns the feature accounting itself — which of the 15 features
in the brief's Section 5 table belongs to which layer — enforced by the
module-level invariant below and checked against the live production artifacts
by verify_clustering_features_against_artifacts().

Reference: Metzler, Zhou & Grace (2021), Quantitative Finance and Economics,
5(3). Cited for the three-tier finding, and for its rejection of a
silhouette-competitive k that split on a single trivial axis. NOT cited as
support for any particular algorithm or distance measure over this project's
continuous features — the paper uses PAM/Gower on mixed-type survey data and
explicitly finds K-Means unsuitable there.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature accounting (brief Section 5 — domain split locked July 2026)
# ---------------------------------------------------------------------------
# Section 5's full feature table lists 15 features. Every one of them has
# exactly one home, and the module-level guard below enforces that so a feature
# can never quietly change status or vanish:
#
#   Layer 2 — ML clustering (Phase 3 inference): 10 CONTINUOUS features, split
#   4 / 3 / 3 across THREE INDEPENDENT domain models. Not one flat vector — see
#   the note on the restructure below.
#
#   Layer 1 — rule-based checklist (Phase 3): 3. Two scored binary checks plus
#   dependants as a tone-only modifier. Threshold/flag inputs, never clustering
#   dimensions.
#
#   Validation-only: 1. net_worth_to_income, excluded from clustering entirely.
#
#   Unlayered: 1. has_investments, computed and displayed but never scored.
#
# ⚠️ WHY THIS IS THREE LISTS AND NOT ONE (restructured July 2026).
# The previous flat LAYER2_CLUSTERING_FEATURES described a superseded design:
# one 11-feature vector -> one scaler -> distance to one ideal centroid. The
# locked architecture is three separate clusterings, each with its own fitted
# model AND its own fitted RobustScaler, mirrored exactly by the production
# artifacts (backend/ml/artifacts/domain_split/{domain}_model.json +
# {domain}_scaler.pkl, one per domain — see its README). A flat list would have
# forced Phase 3 to re-split it against knowledge the artifacts already hold,
# and would have silently lost the per-domain ORDER, which is load-bearing:
# each scaler's center_/scale_ arrays are positional against that domain's own
# feature list. The dict keys below are the artifact filename stems, so
# Phase 3 iterates domains rather than hardcoding three paths.
#
# The flat name LAYER2_CLUSTERING_FEATURES is deliberately RETIRED rather than
# rebound to the dict: a stale `for f in LAYER2_CLUSTERING_FEATURES` would then
# silently iterate domain names and look plausible, whereas a NameError is
# loud. Anything still reaching for the old name wants
# LAYER2_CLUSTERING_FEATURES_BY_DOMAIN or one of the three constants.
#
# net_worth_to_income is NOT in any clustering list, and its absence is asserted
# below rather than merely intended. Including it was directly tested and
# degraded both domains it could join — Banking's gate-passers dropped 9->4 and
# the winning partition became a wealth-level split rather than a banking-health
# one; Investment collapsed to a near-pure single-axis split with ZERO
# gate-passers. Concretely, over_invested users (catastrophic banking health,
# ordinary net worth) became statistically indistinguishable from normal: net
# worth was masking genuine banking distress. It survives as Banking's
# out-of-model validation benchmark, where it is the most binding one
# (lowest-headroom variable in 18 of 28 survivors).
#
# Feature ORDER within each list matches that domain's artifact exactly
# (clustering_features and scaler.features, which the artifacts declare
# separately and which must agree). verify_clustering_features_against_artifacts()
# below re-reads the artifacts and proves it, the same way layer3.py proves its
# tier-label constants — do not reorder these by hand.
BANKING_CLUSTERING_FEATURES = [
    "savings_rate", "emergency_fund_months", "dti", "healthy_debt_ratio",
]
INVESTMENT_CLUSTERING_FEATURES = [
    "investment_rate", "portfolio_concentration", "risk_age_deviation",
]
INSURANCE_CLUSTERING_FEATURES = [
    "life_coverage_ratio", "ci_coverage_ratio", "insurance_spend_ratio",
]

# Keyed by artifact filename stem: artifacts/domain_split/{key}_model.json.
# Same three keys, same spelling, as layer3.py's artifact verification.
LAYER2_CLUSTERING_FEATURES_BY_DOMAIN = {
    "banking": BANKING_CLUSTERING_FEATURES,
    "investment": INVESTMENT_CLUSTERING_FEATURES,
    "insurance": INSURANCE_CLUSTERING_FEATURES,
}

# Excluded from all three clustering models; used only to check Banking's
# results afterwards, never fed to any model. Carried in its artifact-side
# spelling, which is how banking_model.json's out_of_model_benchmarks names it.
VALIDATION_ONLY_FEATURES = [
    "net_worth_to_income",         # compute_banking_features() via the key map
]

# Artifact/benchmark names -> the keys the compute_* functions in this module
# return. Two names differ; Phase 3's feature-vector assembler must map through
# this rather than assuming the names line up.
#
# Post-restructure status of each entry, since only one is still on the
# clustering path:
#   dti                 STILL REQUIRED — Banking's third clustering feature. A
#                       Phase 3 assembler that skips the map builds Banking's
#                       vector from a missing key.
#   net_worth_to_income NO LONGER on any clustering path (validation-only), but
#                       still required by whatever consumes Banking's
#                       out-of-model benchmark, which uses the same spelling.
# Kept as one map because both are the same artifact-name -> compute-name
# translation; the guard below asserts every entry still corresponds to a
# feature that exists somewhere in the accounting.
CLUSTERING_FEATURE_KEY_MAP = {
    "dti": "debt_to_income_ratio",
    "net_worth_to_income": "net_worth_to_income_ratio",
}

LAYER1_CHECKLIST_FEATURES = [
    "income_protection_coverage",  # compute_insurance_features() — SCORED
    "hospitalisation_coverage",    # compute_insurance_features() — SCORED (moved from L2)
    "dependants",                  # users/{uid} document — TONE ONLY, never scored
]

# has_investments belongs to NO layer any more (July 2026, overall-score
# validation). It was dropped from the Layer 1 checklist as redundant with the
# Investment tier — a user in "On Track" holds investments by construction —
# which is the same redundancy argument that kept it out of Layer 2 clustering
# in the first place (redundant with investment_rate; the only one of the five
# original exclusions that ever had a recorded rationale).
#
# It is deliberately given its own named bucket rather than deleted from the
# accounting below. The invariant's job is "every feature in the brief's
# Section 5 table has exactly one home, and none of them vanishes quietly" —
# dropping the count to 14 would make a genuinely lost feature and a
# deliberately demoted one look identical. The feature itself is NOT dead: it is
# still computed and returned by compute_investment_features() and still drives
# the Investments screen; it is simply not an input to any scoring layer.
# Same treatment in spirit as net_worth_to_income, which was likewise demoted
# out of a scoring role (to validation-only) without leaving the accounting.
UNLAYERED_COMPUTED_FEATURES = [
    "has_investments",             # compute_investment_features() — display only
]

# Structural invariant (module-level, same pattern as the EQUITY_TARGETS check
# below): enforce the confirmed split at import so a regression to the old
# arrangement fails loudly rather than silently. The load-bearing rules are that
# Layer 2 stays CONTINUOUS-ONLY (the ablation's whole finding), that it stays
# 4/3/3 across three domains rather than drifting back to one flat vector, and
# that net_worth_to_income never re-enters a clustering set.
_KNOWN_BINARY_FLAGS = {
    "hospitalisation_coverage", "has_investments", "income_protection_coverage",
}
_DOMAIN_FEATURE_COUNTS = {"banking": 4, "investment": 3, "insurance": 3}
_ALL_CLUSTERING_FEATURES = [
    f for feats in LAYER2_CLUSTERING_FEATURES_BY_DOMAIN.values() for f in feats
]

assert set(LAYER2_CLUSTERING_FEATURES_BY_DOMAIN) == set(_DOMAIN_FEATURE_COUNTS), (
    "Layer 2 is exactly three domains, keyed by artifact filename stem "
    f"{sorted(_DOMAIN_FEATURE_COUNTS)} — got "
    f"{sorted(LAYER2_CLUSTERING_FEATURES_BY_DOMAIN)}")
for _domain, _expected_n in _DOMAIN_FEATURE_COUNTS.items():
    assert len(LAYER2_CLUSTERING_FEATURES_BY_DOMAIN[_domain]) == _expected_n, (
        f"{_domain} clusters on exactly {_expected_n} features (brief Section 5, "
        f"'Final domain split') — got "
        f"{LAYER2_CLUSTERING_FEATURES_BY_DOMAIN[_domain]}")
assert len(_ALL_CLUSTERING_FEATURES) == len(set(_ALL_CLUSTERING_FEATURES)) == 10, (
    "the three domain sets must be disjoint and total exactly 10 clustering "
    f"features (4+3+3) — got {_ALL_CLUSTERING_FEATURES}")
assert not (set(_ALL_CLUSTERING_FEATURES) & _KNOWN_BINARY_FLAGS), (
    "Layer 2 must stay continuous-only; binary flags belong in Layer 1 — "
    f"offenders: {set(_ALL_CLUSTERING_FEATURES) & _KNOWN_BINARY_FLAGS}")
# The single most important line here: net worth masked genuine banking distress
# when clustered on, so it must never reappear in a clustering set under either
# spelling.
assert not ({"net_worth_to_income", "net_worth_to_income_ratio"}
            & set(_ALL_CLUSTERING_FEATURES)), (
    "net_worth_to_income is excluded from ALL clustering (it masked genuine "
    "banking distress — brief Section 5, Option 4). It is validation-only")
assert VALIDATION_ONLY_FEATURES == ["net_worth_to_income"], (
    "net_worth_to_income is the only validation-only feature")

assert len(LAYER1_CHECKLIST_FEATURES) == 3, (
    "Layer 1 is exactly income_protection_coverage + hospitalisation_coverage "
    "(the two scored binary checks feeding the overall score's Insurance third) "
    "plus dependants (tone-only). has_investments was dropped July 2026")

# All four buckets pairwise disjoint, and together exactly the brief's 15.
_BUCKETS = {
    "clustering": set(_ALL_CLUSTERING_FEATURES),
    "layer1": set(LAYER1_CHECKLIST_FEATURES),
    "validation_only": set(VALIDATION_ONLY_FEATURES),
    "unlayered": set(UNLAYERED_COMPUTED_FEATURES),
}
for _a, _b in [("clustering", "layer1"), ("clustering", "validation_only"),
               ("clustering", "unlayered"), ("layer1", "validation_only"),
               ("layer1", "unlayered"), ("validation_only", "unlayered")]:
    assert not (_BUCKETS[_a] & _BUCKETS[_b]), (
        f"a feature cannot be both {_a} and {_b}: {_BUCKETS[_a] & _BUCKETS[_b]}")
assert len(set().union(*_BUCKETS.values())) == 15, (
    "the four buckets must together cover exactly the 15 features in the brief's "
    "Section 5 table — 10 clustering (4 banking + 3 investment + 3 insurance) + "
    "3 Layer 1 + 1 validation-only + 1 unlayered")

# Every key-map entry must still name a feature that exists somewhere in the
# accounting, so a renamed or dropped feature can't leave a dangling translation.
assert set(CLUSTERING_FEATURE_KEY_MAP) <= set().union(*_BUCKETS.values()), (
    "CLUSTERING_FEATURE_KEY_MAP has entries for features that no longer exist: "
    f"{set(CLUSTERING_FEATURE_KEY_MAP) - set().union(*_BUCKETS.values())}")

_DOMAIN_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "domain_split"


def verify_clustering_features_against_artifacts(artifact_dir: Path | None = None) -> dict:
    """
    Re-reads the three production artifacts and asserts this module's per-domain
    feature lists still match them exactly — same name, same ORDER, in both
    places each artifact declares them (``clustering_features`` and
    ``scaler.features``).

    Not called at import (this module stays I/O-free until asked); it is run by
    the __main__ block below. Direct analogue of layer3.py's
    verify_tier_labels_against_artifacts(), and it exists for the same class of
    bug: order is silent. A reordered list still has the right length and the
    right members, so every assert above passes, while Phase 3 would apply each
    scaler's center_/scale_ to the wrong column and produce a plausible,
    entirely wrong tier.

    Returns:
        Dict mapping domain -> the artifact's declared clustering_features.
    """
    artifact_dir = artifact_dir or _DOMAIN_ARTIFACT_DIR
    declared = {}
    for domain, expected in LAYER2_CLUSTERING_FEATURES_BY_DOMAIN.items():
        model = json.loads(
            (artifact_dir / f"{domain}_model.json").read_text(encoding="utf-8")
        )
        got = model["clustering_features"]
        if got != expected:
            raise AssertionError(
                f"{domain}: artifact clustering_features {got} != scoring.py "
                f"{expected} — the artifact was rebuilt on a different feature "
                "set, or these constants are stale. Fix before shipping.")
        # The scaler declares its own ordered feature list; if the two ever
        # disagree the artifact itself is internally inconsistent.
        scaler_feats = model["scaler"]["features"]
        if scaler_feats != expected:
            raise AssertionError(
                f"{domain}: artifact scaler.features {scaler_feats} != "
                f"clustering_features {expected} — artifact is internally "
                "inconsistent; the scaler would be applied to the wrong columns.")
        declared[domain] = got
    return declared


def annual_income_from_financial_profile(profile: dict) -> float:
    """Bridges the users/{uid}/financialProfile document (Section 8, which
    stores MONTHLY income as ``monthly_income_sgd``) to the ANNUAL income
    denominator the insurance features use: annual = monthly × 12.

    Raises ValueError when the field is missing — a profile written by the
    Home Phase 1 client always carries it, so absence means the document
    predates the schema or was written by something else.

    Zero is passed through as 0.0 (the profile allows zero income —
    students / NS / between jobs); compute_insurance_features() then raises
    its own ValueError, which a Phase 3 caller should surface as "score
    unavailable" rather than fabricating ratios against a zero denominator.
    """
    monthly = profile.get("monthly_income_sgd")
    if monthly is None:
        raise ValueError(
            "financialProfile document has no monthly_income_sgd field"
        )
    return float(monthly) * 12.0


def compute_insurance_features(policies: list, annual_income_sgd: float) -> dict:
    """
    Computes the five Insurance Score component features from a user's parsed
    insurance policies (dicts shaped like InsurancePolicyExtraction, with the
    strict ``policy_type`` enum).

    Type guards: the coverage ratios and flags are computed only from policies
    whose ``policy_type`` matches the feature. personal_accident and "other"
    policies are explicitly excluded from all four coverage calculations —
    their benefit figures are not life/CI sums assured, and including them
    corrupts the ratios (an accidental-death benefit figure inflating the life
    coverage ratio was a confirmed real-world case during gold-set testing).
    They still count toward insurance_spend_ratio, which is a total premium
    across ALL policies regardless of type.

    ``policy_type_raw`` is deliberately not read here — it is a free-text
    audit/report field and must never drive scoring logic.

    Args:
        policies:          List of policy dicts. Relevant keys: ``policy_type``,
                           ``sum_assured_sgd``, ``annual_premium_sgd``.
        annual_income_sgd: User's annual income; must be > 0 for the ratios.

    Returns:
        Dict with keys ``life_coverage_ratio``, ``ci_coverage_ratio``,
        ``hospitalisation_coverage`` (0/1), ``income_protection_coverage``
        (0/1), and ``insurance_spend_ratio``.
    """
    if annual_income_sgd <= 0:
        raise ValueError("annual_income_sgd must be positive")

    life_sum = sum(
        p.get("sum_assured_sgd") or 0.0
        for p in policies
        if p.get("policy_type") == "life"
    )
    ci_sum = sum(
        p.get("sum_assured_sgd") or 0.0
        for p in policies
        if p.get("policy_type") == "critical_illness"
    )
    has_hospitalisation = any(
        p.get("policy_type") == "hospitalisation" for p in policies
    )
    has_income_protection = any(
        p.get("policy_type") == "income_protection" for p in policies
    )
    # Spend ratio intentionally spans every policy type, including
    # personal_accident and other — premiums are real spend regardless of
    # which coverage bucket the policy falls into.
    total_premiums = sum(p.get("annual_premium_sgd") or 0.0 for p in policies)

    return {
        "life_coverage_ratio": life_sum / annual_income_sgd,
        "ci_coverage_ratio": ci_sum / annual_income_sgd,
        "hospitalisation_coverage": 1 if has_hospitalisation else 0,
        "income_protection_coverage": 1 if has_income_protection else 0,
        "insurance_spend_ratio": total_premiums / annual_income_sgd,
    }


# ---------------------------------------------------------------------------
# Home Phase 2 — Banking & Investment component features
# ---------------------------------------------------------------------------
#
# Shared contract (mirrors compute_insurance_features, extended for the
# missing-data realities of banking/investment data):
#   - annual_income_sgd <= 0            -> ValueError, same as insurance.
#     Phase 3 surfaces this as "score unavailable", never a fabricated ratio.
#   - annual_income_sgd is None         -> the caller has already handled the
#     income error (e.g. caught annual_income_from_financial_profile's
#     ValueError) and explicitly asks for the income-independent features
#     only; income-dependent features come back None.
#   - missing underlying data           -> that feature alone is None
#     ("cannot compute"), never a misleading zero — the same null-vs-zero
#     distinction as null sum_assured_sgd and the Flutter "—" states.

# Healthy vs unhealthy debt classification by plaidAccounts.account_subtype
# (Section 5: structured debt — mortgage / student loan / car / home equity —
# is healthy; revolving and unsecured consumer debt is not). Plaid subtype
# strings:
_HEALTHY_DEBT_SUBTYPES = {
    "mortgage",
    "student",
    "auto",
    # Home equity is revolving, but asset-secured and typically lower-rate, and
    # more often used for structured purposes than discretionary spending —
    # closer in character to a mortgage than to a credit card. A defensible but
    # debatable call, locked in brief Section 5's debt subtype resolution.
    "home equity",
}
_UNHEALTHY_DEBT_SUBTYPES = {
    "credit card",     # revolving, carried balance
    "paypal",          # PayPal credit — revolving, credit-card-like
    "consumer",        # Plaid's personal/consumer loan subtypes
    "loan",            # Plaid's generic personal loan subtype
    "line of credit",
    "overdraft",
}
# Anything else is deliberately unclassified: excluded from the ratio and
# logged, never silently guessed into a bucket. "business", "commercial" and
# "construction" are a permanent scope boundary (business debt is not a signal
# about personal financial health), not a pending decision; "other" and any
# subtype Plaid adds later stay out until a classification is actually decided.

# Risk-age benchmark: target equity fraction = (100 - age) / 100, the
# "100 minus age" rule. Honest citation state: a widely-used personal-
# finance heuristic with no single traceable origin — it emerged through
# financial-planning practice and media in the 1980s-90s, is not credited
# to any specific study or institution, and has NO Singapore-specific
# basis (MAS/LIA/SingStat publish no age-equity glidepath). Same
# disclosure category as the ~0.85 USD->SGD factor. Clamped to [0, 1].
# Standardized July 2026: this previously used a "110 minus age" variant
# while MLAI/clustering/generate_synthetic_data.py's EQUITY_TARGETS (which
# built the training data) used 100 — training and inference must share
# one convention, so both now use 100 (the generator asserts its targets
# equal (100 - band midpoint)/100; the assert below pins this side).
RISK_AGE_EQUITY_BASE = 100
assert RISK_AGE_EQUITY_BASE == 100, (
    "RISK_AGE_EQUITY_BASE must stay in lockstep with EQUITY_TARGETS in "
    "MLAI/clustering/generate_synthetic_data.py (both '100 minus age') — "
    "change both together and regenerate/re-benchmark, never one side alone")

# Same 30-day window the financialProfile expense derivation and the
# Savings screen use; investment deposits are summed over it.
_INVESTMENT_RATE_WINDOW_DAYS = 30


def compute_banking_features(
    profile: dict | None,
    accounts: list,
    annual_income_sgd: float | None,
) -> dict:
    """
    Computes the five Banking Score component features (Section 5 feature
    table) from the user's financialProfile document and plaidAccounts docs.

    Args:
        profile:           users/{uid}/financialProfile/profile dict (for
                           monthly_expenses_sgd and net_worth_sgd). May be
                           None only alongside annual_income_sgd=None.
        accounts:          List of users/{uid}/plaidAccounts docs, including
                           their ``liability`` maps.
        annual_income_sgd: From annual_income_from_financial_profile().
                           Must be > 0; None means the caller already
                           handled the invalid-income case and wants the
                           income-independent features only.

    Returns:
        Dict with keys ``savings_rate``, ``emergency_fund_months``,
        ``debt_to_income_ratio``, ``healthy_debt_ratio``,
        ``net_worth_to_income_ratio``. Any feature whose inputs are missing
        is None. With no plaidAccounts at all, every feature is None —
        nothing about a user's banking can honestly be computed from an
        empty picture (monthly_expenses_sgd/net_worth_sgd are 0.0-seeded in
        that state, not measured).

    Known, deliberate biases consumed here (Section 5, for the report):
    debt_to_income_ratio is a lower bound — Plaid's Liabilities product
    does not cover personal/auto loans or lines of credit (those accounts
    carry liability=None and contribute zero payment), and covered credit
    cards report minimum_payment_amount, not balance-driven obligation.
    """
    if annual_income_sgd is not None and annual_income_sgd <= 0:
        raise ValueError("annual_income_sgd must be positive")

    features = {
        "savings_rate": None,
        "emergency_fund_months": None,
        "debt_to_income_ratio": None,
        "healthy_debt_ratio": None,
        "net_worth_to_income_ratio": None,
    }
    if not accounts:
        return features

    monthly_income = (
        annual_income_sgd / 12.0 if annual_income_sgd is not None else None
    )
    monthly_expenses = (
        profile.get("monthly_expenses_sgd") if profile else None
    )
    net_worth = profile.get("net_worth_sgd") if profile else None

    # monthly_expenses_sgd is derived from depository transactions, so it
    # only means something when at least one depository account exists.
    depository = [
        a for a in accounts if a.get("account_type") == "depository"
    ]
    if depository and monthly_expenses is not None:
        if monthly_income is not None:
            features["savings_rate"] = (
                monthly_income - monthly_expenses
            ) / monthly_income
        if monthly_expenses > 0:
            liquid = sum(
                a.get("balance_current") or 0.0 for a in depository
            )
            # Zero expenses would mean "infinite months" — left None
            # instead (cannot compute a meaningful ratio). Unlike
            # healthy_debt_ratio, a None here still blocks the Banking tier —
            # see tier_assignment.OPTIONAL_FEATURES_BY_DOMAIN for why.
            features["emergency_fund_months"] = liquid / monthly_expenses

    if monthly_income is not None:
        # Monthly-basis on both sides: monthly liability payments over
        # monthly income. Accounts without liability data contribute zero
        # (locked Section 5/6 scoping decision — no estimation).
        monthly_debt_payments = sum(
            (a.get("liability") or {}).get("monthly_payment_amount") or 0.0
            for a in accounts
        )
        features["debt_to_income_ratio"] = (
            monthly_debt_payments / monthly_income
        )

    healthy = unhealthy = 0.0
    unclassified: dict[str, float] = {}
    for account in accounts:
        if account.get("account_type") not in ("credit", "loan"):
            continue
        balance = account.get("balance_current") or 0.0
        subtype = (account.get("account_subtype") or "").lower()
        if subtype in _HEALTHY_DEBT_SUBTYPES:
            healthy += balance
        elif subtype in _UNHEALTHY_DEBT_SUBTYPES:
            unhealthy += balance
        else:
            unclassified[subtype] = unclassified.get(subtype, 0.0) + balance
    if unclassified:
        logger.warning(
            "healthy_debt_ratio: unclassified debt subtypes excluded "
            "from the ratio: %s", unclassified,
        )
    classified_debt = healthy + unhealthy
    if classified_debt > 0:
        features["healthy_debt_ratio"] = healthy / classified_debt
    # No classifiable debt at all -> None (a proportion of nothing is
    # undefined; debt_to_income_ratio == 0.0 already signals "debt-free").
    # None does not block the Banking tier: tier_assignment scores Banking on
    # its remaining dimensions when this is None (OPTIONAL_FEATURES_BY_DOMAIN),
    # as long as savings_rate, dti and emergency_fund_months are real numbers.

    if annual_income_sgd is not None and net_worth is not None:
        features["net_worth_to_income_ratio"] = (
            net_worth / annual_income_sgd
        )

    return features


def compute_investment_features(
    accounts: list,
    transactions: list,
    annual_income_sgd: float | None,
    age: int | None,
    now: datetime | None = None,
) -> dict:
    """
    Computes the four Investment Score component features (Section 5
    feature table) from the user's investmentAccounts docs and their
    transactions subcollection.

    Args:
        accounts:          List of users/{uid}/investmentAccounts docs
                           (with cached asset_allocation and
                           concentration_risk).
        transactions:      Flat list of investment transaction docs across
                           all accounts (keys: ``type``, ``amount``,
                           ``date``).
        annual_income_sgd: Same contract as compute_banking_features.
        age:               Computed from users/{uid}.date_of_birth (never
                           stored). None -> risk_age_deviation is None.
        now:               Injectable clock for tests; defaults to UTC now.

    Returns:
        Dict with keys ``has_investments`` (bool), ``investment_rate``,
        ``portfolio_concentration``, ``risk_age_deviation``. With no
        investmentAccounts, has_investments is False (genuinely known) and
        the other three are None, not zero.
    """
    if annual_income_sgd is not None and annual_income_sgd <= 0:
        raise ValueError("annual_income_sgd must be positive")

    features = {
        "has_investments": bool(accounts),
        "investment_rate": None,
        "portfolio_concentration": None,
        "risk_age_deviation": None,
    }
    if not accounts:
        return features

    if annual_income_sgd is not None:
        # Same deposit logic verified in Plaid Phase 3: deposits are cash
        # credited into the account (Plaid sign convention: negative), so
        # the contribution amount is abs(amount) of type == "deposit" rows.
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=_INVESTMENT_RATE_WINDOW_DAYS)
        deposits = sum(
            abs(t.get("amount") or 0.0)
            for t in transactions
            if t.get("type") == "deposit"
            and t.get("date") is not None
            and t["date"] >= cutoff
        )
        features["investment_rate"] = deposits / (annual_income_sgd / 12.0)

    # concentration_risk is cached per account by the Phase 3 holdings sync
    # — reused, not recomputed. Aggregation across accounts: max, i.e. the
    # worst single account. Caveat (flagged, not hidden): for multi-account
    # users this overstates portfolio-wide concentration — a one-holding
    # small account reads as 1.0 even if the overall portfolio is
    # diversified. True portfolio-wide concentration needs cross-account
    # holding merges (recomputing from raw holdings), out of scope here.
    cached = [
        a.get("concentration_risk")
        for a in accounts
        if a.get("concentration_risk") is not None
    ]
    if cached:
        features["portfolio_concentration"] = max(cached)

    if age is not None:
        total_value = sum(a.get("total_value_sgd") or 0.0 for a in accounts)
        if total_value > 0:
            equities_value = sum(
                (a.get("asset_allocation") or {}).get("equities", 0.0)
                * (a.get("total_value_sgd") or 0.0)
                for a in accounts
            )
            actual_equity = equities_value / total_value
            target_equity = min(
                1.0, max(0.0, (RISK_AGE_EQUITY_BASE - age) / 100.0)
            )
            # Positive = more equity-heavy than the age target (aggressive),
            # negative = more conservative. Both fractions in [0, 1].
            features["risk_age_deviation"] = actual_equity - target_equity

    return features


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("compute_insurance_features sanity check:")
    # The personal_accident policy carries a benefit figure that must NOT
    # leak into life_coverage_ratio, but its premium counts toward spend.
    policies = [
        {"policy_type": "life", "sum_assured_sgd": 500000.0,
         "annual_premium_sgd": 1200.0},
        {"policy_type": "critical_illness", "sum_assured_sgd": 200000.0,
         "annual_premium_sgd": 800.0},
        {"policy_type": "hospitalisation", "sum_assured_sgd": None,
         "annual_premium_sgd": 600.0},
        {"policy_type": "personal_accident", "sum_assured_sgd": 100000.0,
         "annual_premium_sgd": 213.24},
        {"policy_type": "other", "sum_assured_sgd": 50000.0,
         "annual_premium_sgd": 400.0},
    ]
    features = compute_insurance_features(policies, annual_income_sgd=100000.0)
    for name, value in features.items():
        print(f"  {name} = {value}")
    assert features["life_coverage_ratio"] == 5.0, "PA/other leaked into life ratio"
    assert features["ci_coverage_ratio"] == 2.0, "PA/other leaked into CI ratio"
    assert features["hospitalisation_coverage"] == 1
    assert features["income_protection_coverage"] == 0
    assert abs(features["insurance_spend_ratio"] - 0.0321324) < 1e-9
    print("  all assertions passed — personal_accident/other excluded correctly")

    print("\ncompute_banking_features sanity check:")
    # Monthly income 5000 (annual 60000), expenses 3000; liquid 12000;
    # mortgage 200k + student 30k healthy, credit card 5k + personal loan
    # ("loan") 10k unhealthy, one unclassified "business" 7k excluded.
    profile = {"monthly_expenses_sgd": 3000.0, "net_worth_sgd": 90000.0}
    accounts = [
        {"account_type": "depository", "account_subtype": "checking",
         "balance_current": 4000.0, "liability": None},
        {"account_type": "depository", "account_subtype": "savings",
         "balance_current": 8000.0, "liability": None},
        {"account_type": "loan", "account_subtype": "mortgage",
         "balance_current": 200000.0,
         "liability": {"monthly_payment_amount": 1500.0}},
        {"account_type": "loan", "account_subtype": "student",
         "balance_current": 30000.0,
         "liability": {"monthly_payment_amount": 100.0}},
        {"account_type": "credit", "account_subtype": "credit card",
         "balance_current": 5000.0,
         "liability": {"monthly_payment_amount": 150.0}},
        {"account_type": "loan", "account_subtype": "loan",
         "balance_current": 10000.0, "liability": None},  # uncovered: 0 debt
        {"account_type": "loan", "account_subtype": "business",
         "balance_current": 7000.0, "liability": None},  # unclassified
    ]
    b = compute_banking_features(profile, accounts, annual_income_sgd=60000.0)
    for name, value in b.items():
        print(f"  {name} = {value}")
    assert abs(b["savings_rate"] - 0.4) < 1e-9            # (5000-3000)/5000
    assert abs(b["emergency_fund_months"] - 4.0) < 1e-9   # 12000/3000
    assert abs(b["debt_to_income_ratio"] - 0.35) < 1e-9   # 1750/5000
    assert abs(b["healthy_debt_ratio"] - 230000 / 245000) < 1e-9
    assert abs(b["net_worth_to_income_ratio"] - 1.5) < 1e-9  # 90000/60000
    empty = compute_banking_features(profile, [], annual_income_sgd=60000.0)
    assert all(v is None for v in empty.values()), "no accounts must be all-None"
    no_income = compute_banking_features(profile, accounts, None)
    assert no_income["savings_rate"] is None
    assert no_income["debt_to_income_ratio"] is None
    assert no_income["net_worth_to_income_ratio"] is None
    assert abs(no_income["emergency_fund_months"] - 4.0) < 1e-9
    assert abs(no_income["healthy_debt_ratio"] - 230000 / 245000) < 1e-9
    try:
        compute_banking_features(profile, accounts, 0.0)
        raise AssertionError("income=0 must raise ValueError")
    except ValueError:
        pass
    print("  all assertions passed — incl. all-None (no accounts), "
          "income=None and income=0 contracts")

    print("\ncompute_investment_features sanity check:")
    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    inv_accounts = [
        {"total_value_sgd": 30000.0, "concentration_risk": 0.4,
         "asset_allocation": {"equities": 0.9, "bonds": 0.1}},
        {"total_value_sgd": 10000.0, "concentration_risk": 0.7,
         "asset_allocation": {"equities": 0.2, "cash": 0.8}},
    ]
    inv_txns = [
        {"type": "deposit", "amount": -1000.0,
         "date": now - timedelta(days=10)},
        {"type": "deposit", "amount": -500.0,
         "date": now - timedelta(days=25)},
        {"type": "deposit", "amount": -999.0,
         "date": now - timedelta(days=45)},   # outside 30-day window
        {"type": "buy", "amount": 800.0,
         "date": now - timedelta(days=5)},    # not a deposit
    ]
    i = compute_investment_features(
        inv_accounts, inv_txns, annual_income_sgd=60000.0, age=30, now=now)
    for name, value in i.items():
        print(f"  {name} = {value}")
    assert i["has_investments"] is True
    assert abs(i["investment_rate"] - 0.3) < 1e-9         # 1500/5000
    assert i["portfolio_concentration"] == 0.7            # max across accounts
    # actual equity = (0.9*30000 + 0.2*10000)/40000 = 0.725; target (100-30)/100
    assert abs(i["risk_age_deviation"] - (0.725 - 0.7)) < 1e-9
    none_inv = compute_investment_features([], [], 60000.0, 30, now=now)
    assert none_inv["has_investments"] is False
    assert none_inv["investment_rate"] is None
    assert none_inv["portfolio_concentration"] is None
    assert none_inv["risk_age_deviation"] is None
    no_income_inv = compute_investment_features(
        inv_accounts, inv_txns, None, None, now=now)
    assert no_income_inv["has_investments"] is True
    assert no_income_inv["portfolio_concentration"] == 0.7
    assert no_income_inv["investment_rate"] is None
    assert no_income_inv["risk_age_deviation"] is None    # age unknown
    try:
        compute_investment_features(inv_accounts, inv_txns, 0.0, 30, now=now)
        raise AssertionError("income=0 must raise ValueError")
    except ValueError:
        pass
    print("  all assertions passed — incl. no-accounts, income=None/0 and "
          "age=None contracts")

    print("\nfeature accounting consistency check:")
    artifact = json.loads(
        (Path(__file__).parent / "artifacts"
         / "synthetic_ideal_centroids.json").read_text()
    )
    # Check against the DEPRECATED single-model artifact (July 2026): it holds the
    # pre-ablation 10-feature set, and is now expected to stay that way permanently
    # rather than transitionally — the 11-feature re-export it was waiting for was
    # overtaken by the domain-split decision, and the live artifacts are the three
    # models in artifacts/domain_split/. The check is kept because this file is still
    # committed and a silent edit to it should still fail loudly.
    _ARTIFACT_PRE_ABLATION_10 = [
        "savings_rate", "emergency_fund_months", "dti", "net_worth_to_income",
        "investment_rate", "portfolio_concentration", "risk_age_deviation",
        "life_coverage_ratio", "ci_coverage_ratio", "hospitalisation_coverage",
    ]
    _artifact_feats = artifact["clustering_features"]
    if _artifact_feats == _ARTIFACT_PRE_ABLATION_10:
        print("  NOTE: the single-model artifact holds the pre-ablation 10-feature "
              "set — expected and permanent; that artifact is DEPRECATED (see "
              "artifacts/DEPRECATED_single_model_artifacts.md). Live artifacts: "
              "artifacts/domain_split/")
    else:
        raise AssertionError(
            "the deprecated single-model artifact's clustering_features no longer "
            "match the known pre-ablation 10-feature set — genuine drift, or the "
            "file was edited by hand")
    # The artifact must at least be internally consistent (every band's centroid
    # keyed by its own declared feature list).
    for band, ideals in artifact["synthetic_ideal_centroids"].items():
        assert list(ideals) == _artifact_feats, (
            f"band {band} centroid keys inconsistent within the artifact")

    # The LIVE artifacts: per-domain feature lists, name AND order.
    for domain, declared in verify_clustering_features_against_artifacts().items():
        print(f"  {domain:<11}{len(declared)} features, order matches "
              f"{domain}_model.json + its scaler: {declared}")

    # Every feature in the accounting is actually producible: clustering and
    # validation-only via the compute_* functions (through the key map), Layer 1
    # via the functions plus the users-doc dependants field.
    computed_keys = set(b) | set(i) | set(features) | {"dependants"}
    clustering_keys = {
        CLUSTERING_FEATURE_KEY_MAP.get(f, f) for f in _ALL_CLUSTERING_FEATURES
    }
    validation_keys = {
        CLUSTERING_FEATURE_KEY_MAP.get(f, f) for f in VALIDATION_ONLY_FEATURES
    }
    assert len(clustering_keys) == 10, clustering_keys
    assert clustering_keys <= computed_keys, clustering_keys - computed_keys
    assert validation_keys <= computed_keys, validation_keys - computed_keys
    assert set(LAYER1_CHECKLIST_FEATURES) <= computed_keys, (
        set(LAYER1_CHECKLIST_FEATURES) - computed_keys)
    # has_investments is unlayered but must still be genuinely produced — that is
    # the whole point of keeping it in the accounting rather than deleting it.
    assert set(UNLAYERED_COMPUTED_FEATURES) <= computed_keys, (
        set(UNLAYERED_COMPUTED_FEATURES) - computed_keys)

    # The key map earns its place: every entry must actually translate (i.e. the
    # artifact-side name must NOT be a compute_* key, and the compute-side name
    # must be), otherwise it is a no-op hiding a real mismatch.
    for artifact_name, computed_name in CLUSTERING_FEATURE_KEY_MAP.items():
        assert artifact_name != computed_name
        assert artifact_name not in computed_keys, (
            f"{artifact_name!r} is now a real compute_* key — the map entry is "
            "stale and should be removed")
        assert computed_name in computed_keys, (
            f"{artifact_name!r} maps to {computed_name!r}, which no compute_* "
            "function returns")
    # dti is the only clustering feature still needing translation; every other
    # clustering name must already match its compute_* key one-for-one.
    needs_map = [f for f in _ALL_CLUSTERING_FEATURES if f not in computed_keys]
    assert needs_map == ["dti"], (
        f"expected dti to be the only clustering name needing the key map, got "
        f"{needs_map}")

    # Post-translation, the four buckets are still disjoint and still 15.
    assert not clustering_keys & set(LAYER1_CHECKLIST_FEATURES)
    assert not clustering_keys & validation_keys
    assert len(clustering_keys
               | validation_keys
               | set(LAYER1_CHECKLIST_FEATURES)
               | set(UNLAYERED_COMPUTED_FEATURES)) == 15
    print("  all assertions passed — 10 clustering features (4 banking / "
          "3 investment / 3 insurance) all producible, order-checked against the "
          "live artifacts; net_worth_to_income absent from all three; "
          "3 Layer 1 + 1 validation-only + 1 unlayered; 15 total, disjoint")
