"""Phase 3b verification — the score pipeline, its cache, and the endpoints.

    cd backend && venv\\Scripts\\activate && python scripts/verify_score_pipeline.py
    python scripts/verify_score_pipeline.py --pure-only     # skip section B

Two sections, deliberately separable:

  A. THE PURE PIPELINE — no Firebase, no Plaid, no network. Every branch of
     ml/score_pipeline.py driven with constructed Firestore-shaped dicts, in the
     style of verify_tier_assignment.py, plus the cache payload mapping. This is
     the section that can run anywhere, on any machine, with no credentials.

  B. END TO END — one throwaway Firebase user and a real Plaid Sandbox Item, in
     the style of verify_financial_profile.py / verify_home_phase2.py: the real
     endpoints over TestClient, the real Firestore writes, every sync hook, and
     the non-fatal contract proved by making the pipeline throw on purpose.
     Everything is deleted afterwards.

What section A proves that matters most: the pipeline's "unavailable" states are
returned, never raised, and never fabricated. income = 0, an age outside 20-70,
a missing date of birth and a missing profile document all come back as 200-able
results carrying a reason code — the brief's locked requirement (Section 8:
"Phase 3's ML inference endpoint must catch this and surface 'score
unavailable'… never silently substitute a fabricated score").
"""
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from data import score_profile  # noqa: E402
from ml import coverage_gaps, layer3, overall_score, scoring, tier_assignment  # noqa: E402
from ml import score_pipeline  # noqa: E402
from ml.score_pipeline import (  # noqa: E402
    REASON_DATE_OF_BIRTH_NOT_SET,
    REASON_INCOME_NOT_SET,
    REASON_INCOMPLETE_DOMAINS,
    REASON_MISSING_FEATURES,
    REASON_NO_FINANCIAL_PROFILE,
    REASON_UNSUPPORTED_AGE,
    STATUS_ASSESSED,
    STATUS_NOT_ASSESSED,
    compute_score_profile,
)

PURE_ONLY = "--pure-only" in sys.argv

passed, failed = [], []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def expect(name, got, want):
    check(name, got == want, f"got {got!r}, want {want!r}")


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


# ===========================================================================
# A. THE PURE PIPELINE
# ===========================================================================
print("=" * 78)
print("A1. A complete user — every domain assessable")
print("=" * 78)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
# Age 42 on the frozen clock -> band 3 (40-49). Stated as a literal date rather
# than "now - 42 years" so the expected band below is a constant, not arithmetic
# that could drift with the test's own bug.
DOB = date(1984, 3, 10)
MONTHLY_INCOME = 5000.0
MONTHLY_EXPENSES = 2500.0
ANNUAL_INCOME = MONTHLY_INCOME * 12

USER = {"date_of_birth": datetime.combine(DOB, datetime.min.time(),
                                          tzinfo=timezone.utc),
        "dependants": 2}
PROFILE = {"monthly_income_sgd": MONTHLY_INCOME,
           "monthly_expenses_sgd": MONTHLY_EXPENSES,
           "net_worth_sgd": 150000.0}
BANK_ACCOUNTS = [
    {"account_type": "depository", "account_subtype": "checking",
     "balance_current": 25000.0},
    {"account_type": "loan", "account_subtype": "mortgage",
     "balance_current": 300000.0,
     "liability": {"monthly_payment_amount": 900.0}},
    {"account_type": "credit", "account_subtype": "credit card",
     "balance_current": 3000.0,
     "liability": {"monthly_payment_amount": 150.0}},
]
INVESTMENT_ACCOUNTS = [
    {"total_value_sgd": 80000.0, "concentration_risk": 0.28,
     "asset_allocation": {"equities": 0.62}},
]
INVESTMENT_TXNS = [
    {"type": "deposit", "amount": -900.0, "date": NOW - timedelta(days=5)},
]
POLICIES = [
    {"policy_type": "life", "sum_assured_sgd": 600000.0,
     "annual_premium_sgd": 2400.0},
    {"policy_type": "critical_illness", "sum_assured_sgd": 200000.0,
     "annual_premium_sgd": 1200.0},
    {"policy_type": "hospitalisation", "annual_premium_sgd": 600.0},
]


def run(**overrides):
    args = {"user_doc": USER, "profile": PROFILE,
            "bank_accounts": BANK_ACCOUNTS,
            "investment_accounts": INVESTMENT_ACCOUNTS,
            "investment_transactions": INVESTMENT_TXNS,
            "insurance_policies": POLICIES, "now": NOW}
    args.update(overrides)
    return compute_score_profile(**args)


full = run()
expect("age computed from date_of_birth (never stored)", full["age"], 42)
expect("age band selected for 42", full["age_band"], 3)
expect("age band range reported", full["age_band_range"], "40-49")
expect("no whole-pipeline unavailable reason", full["unavailable_reason"], None)

# --- features re-derived independently, the verify_home_phase2.py way ---------
expected_savings_rate = (MONTHLY_INCOME - MONTHLY_EXPENSES) / MONTHLY_INCOME
expected_ef_months = 25000.0 / MONTHLY_EXPENSES
expected_dti = (900.0 + 150.0) / MONTHLY_INCOME
expected_healthy_debt = 300000.0 / (300000.0 + 3000.0)
expected_investment_rate = 900.0 / MONTHLY_INCOME
expected_risk_age_dev = 0.62 - (scoring.RISK_AGE_EQUITY_BASE - 42) / 100.0
expected_life = 600000.0 / ANNUAL_INCOME
expected_ci = 200000.0 / ANNUAL_INCOME
expected_spend = (2400.0 + 1200.0 + 600.0) / ANNUAL_INCOME

# The pipeline hands these to tier_assignment; rebuild them here by hand and
# assign independently, so a wiring mistake (wrong key, wrong band, features
# crossed between domains) cannot pass.
hand = {
    "banking": {"savings_rate": expected_savings_rate,
                "emergency_fund_months": expected_ef_months,
                "dti": expected_dti,
                "healthy_debt_ratio": expected_healthy_debt},
    "investment": {"investment_rate": expected_investment_rate,
                   "portfolio_concentration": 0.28,
                   "risk_age_deviation": expected_risk_age_dev},
    "insurance": {"life_coverage_ratio": expected_life,
                  "ci_coverage_ratio": expected_ci,
                  "insurance_spend_ratio": expected_spend},
}
hand_tiers = {d: tier_assignment.assign_domain_tier(d, f, 3)
              for d, f in hand.items()}
for domain in score_pipeline.DOMAIN_ORDER:
    got = full["domains"][domain]
    expect(f"{domain:<11} status", got["status"], STATUS_ASSESSED)
    expect(f"{domain:<11} tier matches an independent hand assembly",
           got["tier_label"], hand_tiers[domain])
    expect(f"{domain:<11} tier_index agrees with its label",
           tier_assignment.tier_labels(domain)[got["tier_index"]],
           got["tier_label"])
print(f"INFO  hand-assembled tiers: {hand_tiers}")
# Banking's artifact feature is spelled `dti`; compute_banking_features returns
# `debt_to_income_ratio`. If the pipeline skipped CLUSTERING_FEATURE_KEY_MAP,
# Banking's fourth input would be missing and this domain would be unassessed.
check("Banking's dti/debt_to_income_ratio rename was translated, not assumed",
      full["domains"]["banking"]["status"] == STATUS_ASSESSED
      and not full["domains"]["banking"]["missing_features"],
      f"missing: {full['domains']['banking']['missing_features']}")

# --- overall score, hand-computed from the tiers ------------------------------
b_score = 100.0 if hand_tiers["banking"] == "Stable" else 0.0
i_score = 100.0 if hand_tiers["investment"] == "On Track" else 0.0
indicators = [1.0 if hand_tiers["insurance"] == "Covered" else 0.0,
              0.0,   # income_protection: no income_protection policy above
              1.0]   # hospitalisation: one hospitalisation policy above
cleared = int(sum(indicators))
insurance_third = 100.0 * sum(indicators) / 3.0
option_b = (b_score + i_score + insurance_third) / 3.0
ceiling = {0: 39.0, 1: 69.0, 2: 100.0, 3: 100.0}[cleared]
expected_overall = min(option_b, ceiling)
check("overall_score == min(option_b, ceiling), computed by hand",
      close(full["overall_score"], expected_overall),
      f"{full['overall_score']} vs {expected_overall} "
      f"(option_b={option_b:.2f}, ceiling={ceiling}, cleared={cleared}/3)")
expect("overall_score_reason is None when a score exists",
       full["overall_score_reason"], None)
expect("score_breakdown cached alongside",
       full["score_breakdown"]["insurance_indicators_cleared"], cleared)

# --- layer 3, hand-evaluated --------------------------------------------------
expected_flags = []
if hand_tiers["banking"] == "Stable" and hand_tiers["insurance"] == "Not Covered":
    expected_flags.append(layer3.FLAG_WEALTH_WITHOUT_PROTECTION)
if hand_tiers["insurance"] == "Covered" and hand_tiers["banking"] == "Building":
    expected_flags.append(layer3.FLAG_PROTECTION_FIRST)
if (hand_tiers["banking"] == "Stable"
        and hand_tiers["investment"] == "Getting Started"
        and expected_investment_rate <= layer3.RULE3_INVESTMENT_RATE_MAX
        and expected_savings_rate >= layer3.RULE3_SAVINGS_RATE_MIN
        and expected_ef_months >= layer3.RULE3_EMERGENCY_FUND_MONTHS_MIN):
    expected_flags.append(layer3.FLAG_STRONG_FOUNDATION_ROOM_TO_GROW)
if (hand_tiers["banking"] == "Building"
        and hand_tiers["investment"] == "On Track"
        and expected_investment_rate >= layer3.RULE4_INVESTMENT_RATE_MIN):
    expected_flags.append(layer3.FLAG_AGGRESSIVE_INVESTING_WEAK_FOUNDATION)
expect("layer 3 flags match a hand evaluation of the four rules",
       full["layer3_flags"], expected_flags)
expect("layer3_evaluated True when all three tiers exist",
       full["layer3_evaluated"], True)

# --- layer 1 checklist --------------------------------------------------------
checklist = {f["check"]: f for f in full["checklist_flags"]}
expect("checklist has one entry per scored check",
       sorted(checklist), sorted(overall_score.LAYER1_CHECKS))
expect("income protection check fails (no such policy)",
       checklist["income_protection_coverage"]["passed"], False)
expect("hospitalisation check passes (policy present)",
       checklist["hospitalisation_coverage"]["passed"], True)
# dependants is tone-only: it must change copy and nothing else.
no_dep = run(user_doc={**USER, "dependants": 0})
fail_with = checklist["income_protection_coverage"]["message"]
fail_without = {f["check"]: f for f in no_dep["checklist_flags"]}[
    "income_protection_coverage"]["message"]
check("dependants changes only the failure copy, never an outcome",
      fail_with != fail_without
      and no_dep["overall_score"] == full["overall_score"],
      "same score, different message")

print()
print("=" * 78)
print("A2. Partial data — some domains assessable, overall score withheld")
print("=" * 78)

no_plaid = run(bank_accounts=[], investment_accounts=[],
               investment_transactions=[])
expect("no plaidAccounts -> banking not assessed",
       no_plaid["domains"]["banking"]["status"], STATUS_NOT_ASSESSED)
expect("no investmentAccounts -> investment not assessed",
       no_plaid["domains"]["investment"]["status"], STATUS_NOT_ASSESSED)
expect("insurance still assessed from policies alone",
       no_plaid["domains"]["insurance"]["status"], STATUS_ASSESSED)
expect("banking reason names the problem",
       no_plaid["domains"]["banking"]["reason"], REASON_MISSING_FEATURES)
expect("banking lists exactly which features were missing",
       no_plaid["domains"]["banking"]["missing_features"],
       list(scoring.BANKING_CLUSTERING_FEATURES))
expect("overall_score withheld", no_plaid["overall_score"], None)
expect("overall_score_reason = incomplete_domains",
       no_plaid["overall_score_reason"], REASON_INCOMPLETE_DOMAINS)
expect("layer 3 not evaluated without all three tiers",
       no_plaid["layer3_evaluated"], False)
expect("layer 3 flags empty (and reason recorded, so empty is not ambiguous)",
       (no_plaid["layer3_flags"], no_plaid["layer3_reason"]),
       ([], REASON_INCOMPLETE_DOMAINS))
expect("no whole-pipeline unavailable reason — this is a partial, not a failure",
       no_plaid["unavailable_reason"], None)
check("the checklist still evaluates from insurance data alone",
      all(f["passed"] is not None for f in no_plaid["checklist_flags"]))

bank_only = run(investment_accounts=[], investment_transactions=[])
expect("banking assessed when only Plaid banking exists",
       bank_only["domains"]["banking"]["status"], STATUS_ASSESSED)
expect("two of three domains -> still no overall score",
       bank_only["overall_score"], None)

inv_only = run(bank_accounts=[])
expect("investment assessed when only investmentAccounts exist",
       inv_only["domains"]["investment"]["status"], STATUS_ASSESSED)
expect("banking unassessed in the same result",
       inv_only["domains"]["banking"]["status"], STATUS_NOT_ASSESSED)

# A partial domain: accounts exist, but the derived expense field does not, so
# two of Banking's four features cannot be computed.
partial = run(profile={"monthly_income_sgd": MONTHLY_INCOME,
                       "net_worth_sgd": 150000.0})
expect("banking with no monthly_expenses_sgd -> not assessed",
       partial["domains"]["banking"]["status"], STATUS_NOT_ASSESSED)
expect("…and names only the genuinely missing features",
       partial["domains"]["banking"]["missing_features"],
       ["savings_rate", "emergency_fund_months"])

print()
print("=" * 78)
print("A3. Zero insurance policies — a REAL 'Not Covered', not an unknown")
print("=" * 78)
print("  Locked semantic: no policies means every coverage ratio is a true 0.0")
print("  and both flags a true 0 — a real state. compute_insurance_features()")
print("  is checked directly here so the claim is pinned, not assumed.")

zero_policy_features = scoring.compute_insurance_features([], ANNUAL_INCOME)
expect("compute_insurance_features([]) returns real zeros, not None",
       zero_policy_features,
       {"life_coverage_ratio": 0.0, "ci_coverage_ratio": 0.0,
        "hospitalisation_coverage": 0, "income_protection_coverage": 0,
        "insurance_spend_ratio": 0.0})

zero_policy = run(insurance_policies=[])
expect("zero policies -> Insurance IS assessed",
       zero_policy["domains"]["insurance"]["status"], STATUS_ASSESSED)
expect("zero policies -> 'Not Covered'",
       zero_policy["domains"]["insurance"]["tier_label"], "Not Covered")
expect("zero policies -> a real overall score, not 'unavailable'",
       zero_policy["overall_score"] is not None, True)
expect("zero policies clears 0 of 3 insurance indicators",
       zero_policy["score_breakdown"]["insurance_indicators_cleared"], 0)
expect("…so the ceiling caps the score at 39.0",
       zero_policy["score_breakdown"]["ceiling"], 39.0)
check("…and the cap actually binds for this strong-banking user",
      zero_policy["score_breakdown"]["capped"] is True
      and zero_policy["overall_score"] == 39.0,
      f"option_b={zero_policy['score_breakdown']['option_b_score']:.2f} "
      f"-> {zero_policy['overall_score']}")
check("both checklist checks FAIL (a known gap), rather than reading unknown",
      all(f["passed"] is False for f in zero_policy["checklist_flags"]))

print()
print("=" * 78)
print("A3b. Coverage Gaps — derived from the same ratios, never re-derived")
print("=" * 78)
print("  The Insurance screen shows HOW SHORT a user is, which the binary tier")
print("  cannot say. These checks pin that the published numbers come from")
print("  compute_insurance_features() and the committed benchmark artifacts —")
print("  not from a second implementation that could drift away from the tier.")

coverage = full["insurance_coverage"]
expect("insurance_coverage carries exactly the two continuous ratios",
       sorted(coverage), ["critical_illness", "life"])
expect("life ratio is compute_insurance_features()'s own value, not a re-sum",
       coverage["life"]["ratio"], expected_life)
expect("CI ratio likewise", coverage["critical_illness"]["ratio"], expected_ci)
expect("the life benchmark is band 3's artifact value (LIA 9x)",
       coverage["life"]["benchmark_ratio"], 9.0)
expect("the CI benchmark is the artifact's 3.9x",
       coverage["critical_illness"]["benchmark_ratio"], 3.9)
expect("shortfall is stated in SGD against the same income the ratio used",
       coverage["life"]["shortfall_sgd"],
       max(0.0, 9.0 * ANNUAL_INCOME - expected_life * ANNUAL_INCOME))
check("meets_benchmark agrees with the ratio it was computed from",
      all(gap["meets_benchmark"] == (gap["ratio"] >= gap["benchmark_ratio"])
          for gap in coverage.values()))

# The band-5 drop is the reason the benchmark is resolved per user rather than
# hardcoded once — a 62-year-old must not be measured against 9x.
expect("band 5 (60-70) carries the artifact's 5x life benchmark, not 9x",
       coverage_gaps.coverage_benchmarks(5)["life_coverage_ratio"], 5.0)
expect("…while every other band carries 9x",
       {b: coverage_gaps.coverage_benchmarks(b)["life_coverage_ratio"]
        for b in (1, 2, 3, 4)},
       {1: 9.0, 2: 9.0, 3: 9.0, 4: 9.0})

zero_coverage = zero_policy["insurance_coverage"]
check("zero policies -> a real 0.0 ratio and a full-benchmark shortfall",
      zero_coverage["life"]["ratio"] == 0.0
      and zero_coverage["life"]["meets_benchmark"] is False
      and zero_coverage["life"]["shortfall_sgd"] == 9.0 * ANNUAL_INCOME)

over_covered = run(insurance_policies=[
    {"policy_type": "life", "sum_assured_sgd": 20.0 * ANNUAL_INCOME,
     "annual_premium_sgd": 3000.0}])["insurance_coverage"]
check("an over-covered user reports no negative gap",
      over_covered["life"]["meets_benchmark"] is True
      and over_covered["life"]["shortfall_sgd"] == 0.0)

# personal_accident is excluded from the coverage ratios by
# compute_insurance_features(); because Coverage Gaps READS that result rather
# than re-summing the policies, the exclusion is inherited for free. A Dart-side
# re-implementation is exactly what this check exists to make unnecessary.
pa_only = run(insurance_policies=[
    {"policy_type": "personal_accident", "sum_assured_sgd": 500000.0,
     "annual_premium_sgd": 400.0}])["insurance_coverage"]
expect("a personal_accident payout does not leak into the life gap",
       pa_only["life"]["ratio"], 0.0)

for label, out in [("income = 0",
                    run(profile={**PROFILE, "monthly_income_sgd": 0.0})),
                   ("no profile", run(profile=None))]:
    expect(f"{label:<12} -> insurance_coverage is None, not zeroes",
           out["insurance_coverage"], None)

print()
print("=" * 78)
print("A4. The unavailable states — returned, never raised, never fabricated")
print("=" * 78)

for label, kwargs, want in [
    ("income = 0", {"profile": {**PROFILE, "monthly_income_sgd": 0.0}},
     REASON_INCOME_NOT_SET),
    ("income field absent", {"profile": {"monthly_expenses_sgd": 100.0}},
     REASON_INCOME_NOT_SET),
    ("income negative", {"profile": {**PROFILE, "monthly_income_sgd": -5.0}},
     REASON_INCOME_NOT_SET),
    ("age 19 (under 20)",
     {"user_doc": {**USER, "date_of_birth": date(2007, 1, 1)}},
     REASON_UNSUPPORTED_AGE),
    ("age 80 (over 70)",
     {"user_doc": {**USER, "date_of_birth": date(1946, 1, 1)}},
     REASON_UNSUPPORTED_AGE),
    ("date_of_birth missing", {"user_doc": {"dependants": 0}},
     REASON_DATE_OF_BIRTH_NOT_SET),
    ("users/{uid} document missing", {"user_doc": None},
     REASON_DATE_OF_BIRTH_NOT_SET),
    ("date_of_birth is a string", {"user_doc": {"date_of_birth": "1984-03-10"}},
     REASON_DATE_OF_BIRTH_NOT_SET),
    ("no financialProfile document", {"profile": None},
     REASON_NO_FINANCIAL_PROFILE),
]:
    try:
        out = run(**kwargs)
    except Exception as exc:  # noqa: BLE001 — raising IS the failure here
        check(f"{label:<28} -> {want}", False,
              f"raised {type(exc).__name__}: {exc}")
        continue
    ok = (out["unavailable_reason"] == want
          and out["overall_score"] is None
          and out["overall_score_reason"] == want
          and all(d["status"] == STATUS_NOT_ASSESSED
                  for d in out["domains"].values())
          and all(d["reason"] == want for d in out["domains"].values())
          and out["layer3_flags"] == []
          and out["layer3_evaluated"] is False)
    check(f"{label:<28} -> {want}, no domain assessed, no exception", ok,
          f"reason={out['unavailable_reason']}, "
          f"score={out['overall_score']}")

# The age boundaries themselves: 20 and 70 must be INSIDE.
for age, should_work in [(20, True), (19, False), (70, True), (71, False)]:
    dob = date(NOW.year - age, 1, 1)
    out = run(user_doc={**USER, "date_of_birth": dob})
    check(f"age {age} is {'supported' if should_work else 'unsupported'}",
          (out["unavailable_reason"] is None) == should_work,
          f"reason={out['unavailable_reason']}")

# clamp=False is the point: the pipeline must NOT quietly extrapolate.
band_if_clamped = tier_assignment.age_band_for_date_of_birth(
    date(1946, 1, 1), today=NOW.date(), clamp=True)
check("an out-of-range age is refused rather than clamped to a real band",
      band_if_clamped == 5
      and run(user_doc={**USER, "date_of_birth": date(1946, 1, 1)}
              )["domains"]["banking"]["tier_label"] is None,
      f"clamping would have used band {band_if_clamped}")

print()
print("=" * 78)
print("A5. Result shape and the cached payload")
print("=" * 78)

SHAPE = set(full)
for label, out in [("complete", full), ("partial", no_plaid),
                   ("income_not_set", run(profile={**PROFILE,
                                                   "monthly_income_sgd": 0.0})),
                   ("unsupported_age",
                    run(user_doc={**USER, "date_of_birth": date(1946, 1, 1)})),
                   ("no_profile", run(profile=None))]:
    expect(f"{label:<16} result carries the identical key set", set(out), SHAPE)
    check(f"{label:<16} checklist always present, one row per check",
          len(out["checklist_flags"]) == len(overall_score.LAYER1_CHECKS))

payload = score_profile.cache_payload(full)
expect("cache writes exactly Section 8's four Phase 3 fields", sorted(payload),
       ["checklist_flags", "checklist_last_updated", "ml_profile",
        "overall_score"])
expect("overall_score cached as the float itself",
       payload["overall_score"], full["overall_score"])
expect("checklist_last_updated is the run's own computed_at",
       payload["checklist_last_updated"], full["computed_at"])
expect("checklist_flags is a map of both rule layers",
       sorted(payload["checklist_flags"]),
       ["layer1", "layer3", "layer3_evaluated", "layer3_reason"])
ml = payload["ml_profile"]
for field in ("schema_version", "computed_at", "age_band", "age_band_range",
              "domains", "layer3_flags", "layer3_evaluated", "overall_score",
              "overall_score_reason", "unavailable_reason", "score_breakdown",
              "insurance_coverage", "healthy_debt_ratio", "artifacts"):
    check(f"ml_profile carries {field}", field in ml)
expect("ml_profile caches the coverage gaps verbatim from the result",
       ml["insurance_coverage"], full["insurance_coverage"])
# The second raw-feature exception: the NUMBER, not the feature name.
check("ml_profile caches healthy_debt_ratio as the computed number",
      isinstance(ml["healthy_debt_ratio"], float)
      and abs(ml["healthy_debt_ratio"] - expected_healthy_debt) < 1e-12,
      f"{ml['healthy_debt_ratio']} vs {expected_healthy_debt}")
expect("healthy_debt_ratio is None, not 0.0, when there is no debt to split",
       score_profile.cache_payload(no_plaid)["ml_profile"]["healthy_debt_ratio"],
       None)
# ...but a None ratio no longer withholds the Banking tier. Banking's one
# optional feature (tier_assignment.OPTIONAL_FEATURES_BY_DOMAIN) is scored
# around; savings_rate, dti and emergency_fund_months stay mandatory.
CHECKING_ONLY = [BANK_ACCOUNTS[0]]
debt_free = run(bank_accounts=CHECKING_ONLY)
debt_free_ml = score_profile.cache_payload(debt_free)["ml_profile"]
check("debt-free (checking only): Banking IS assessed on the other three",
      debt_free["domains"]["banking"]["status"] == STATUS_ASSESSED
      and debt_free["domains"]["banking"]["tier_label"]
      in tier_assignment.tier_labels("banking"),
      f"{debt_free['domains']['banking']}")
expect("…while the cached healthy_debt_ratio stays None beside that tier",
       debt_free_ml["healthy_debt_ratio"], None)
zero_expenses = run(profile={**PROFILE, "monthly_expenses_sgd": 0.0})
expect("zero monthly expenses: Banking still NOT assessed (emergency fund "
       "mandatory)",
       (zero_expenses["domains"]["banking"]["status"],
        zero_expenses["domains"]["banking"]["missing_features"]),
       (STATUS_NOT_ASSESSED, ["emergency_fund_months"]))
debt_free_zero_expenses = run(bank_accounts=CHECKING_ONLY,
                              profile={**PROFILE, "monthly_expenses_sgd": 0.0})
expect("debt-free AND zero expenses: still NOT assessed — the ratio is optional, "
       "the emergency fund is not",
       (debt_free_zero_expenses["domains"]["banking"]["status"],
        debt_free_zero_expenses["domains"]["banking"]["missing_features"]),
       (STATUS_NOT_ASSESSED, ["emergency_fund_months", "healthy_debt_ratio"]))
credit_only = run(bank_accounts=BANK_ACCOUNTS[1:])
expect("no depository account (savings_rate None): Banking still NOT assessed",
       (credit_only["domains"]["banking"]["status"],
        credit_only["domains"]["banking"]["missing_features"]),
       (STATUS_NOT_ASSESSED, ["savings_rate", "emergency_fund_months"]))
# Brief Section 5: home equity is healthy debt. Mortgage 300k + home equity
# 50k healthy, card 3k unhealthy -> 350/353 (would be 300/303 if home equity
# were still unclassified and dropped from the ratio).
with_heloc = run(bank_accounts=BANK_ACCOUNTS + [
    {"account_type": "loan", "account_subtype": "home equity",
     "balance_current": 50000.0}])
check("home equity counts as healthy debt",
      abs(with_heloc["healthy_debt_ratio"] - 350000.0 / 353000.0) < 1e-12,
      str(with_heloc["healthy_debt_ratio"]))
expect("ml_profile records WHICH age band produced these tiers",
       ml["age_band"], 3)
expect("ml_profile records both the tier index and the label",
       (ml["domains"]["banking"]["tier_index"],
        ml["domains"]["banking"]["tier_label"]),
       (full["domains"]["banking"]["tier_index"],
        full["domains"]["banking"]["tier_label"]))
check("ml_profile records which artifact build produced the score",
      ml["artifacts"]["generated_utc"].startswith("2026-"),
      str(ml["artifacts"]))
check("score_breakdown's tuple is flattened for Firestore",
      isinstance(ml["score_breakdown"]["insurance_indicators"], list))
check("a partial result caches cleanly too (nulls, not omissions)",
      score_profile.cache_payload(no_plaid)["overall_score"] is None
      and score_profile.cache_payload(no_plaid)["ml_profile"]["domains"]
      ["banking"]["missing_features"] != [])
# Section 14: age is computed from date_of_birth and never stored.
check("no raw age or date of birth is written to the cache",
      "age" not in ml and "date_of_birth" not in str(ml),
      "only age_band / age_band_range")

print()
print("=" * 78)
print("A6. Every integration point is actually wired to the hook")
print("=" * 78)
print("  Section B proves four of these fire for real. /insurance/parse cannot")
print("  be exercised without a real policy PDF and a live Gemini call, so its")
print("  wiring is asserted structurally here rather than left unchecked.")

import inspect  # noqa: E402

import main as backend_main_module  # noqa: E402

HOOKED_ENDPOINTS = {
    "parse_insurance_pdf": "POST /insurance/parse",
    "refresh_financial_profile": "POST /profile/refresh",
    "exchange_plaid_public_token": "POST /plaid/exchange-token",
    "sync_plaid_transactions": "POST /plaid/sync-transactions",
    "sync_plaid_investments": "POST /plaid/sync-investments",
}
for func_name, route in HOOKED_ENDPOINTS.items():
    source = inspect.getsource(getattr(backend_main_module, func_name))
    check(f"{route:<28} calls _refresh_score_nonfatal",
          "_refresh_score_nonfatal(" in source)

# Ordering is load-bearing on the Plaid endpoints and /profile/refresh: Banking
# reads monthly_expenses_sgd / net_worth_sgd, so a score computed before the
# derived-field refresh would be one sync behind.
for func_name in ("exchange_plaid_public_token", "sync_plaid_transactions",
                  "sync_plaid_investments", "refresh_financial_profile"):
    source = inspect.getsource(getattr(backend_main_module, func_name))
    profile_at = source.index("_refresh_profile_nonfatal(") if (
        "_refresh_profile_nonfatal(" in source) else source.index(
        "financial_profile.refresh_financial_profile(")
    check(f"{HOOKED_ENDPOINTS[func_name]:<28} refreshes derived fields BEFORE "
          f"the score", profile_at < source.index("_refresh_score_nonfatal("))

check("the hook delegates to the non-fatal wrapper, not to the raw pipeline",
      "refresh_score_nonfatal" in inspect.getsource(
          backend_main_module._refresh_score_nonfatal))

print()
print(f"section A: {len(passed)} passed, {len(failed)} failed")

if PURE_ONLY:
    print("\n--pure-only: skipping section B (live Firebase + Plaid Sandbox)")
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)

# ===========================================================================
# B. END TO END — real Firestore, real Plaid Sandbox, real endpoints
# ===========================================================================
import json  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from firebase import get_db, init_firebase  # noqa: E402
from firebase_admin import auth as admin_auth  # noqa: E402

from data import plaid_service  # noqa: E402
from plaid.model.products import Products  # noqa: E402
from plaid.model.sandbox_public_token_create_request import (  # noqa: E402
    SandboxPublicTokenCreateRequest,
)

GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
SANDBOX_INSTITUTION = "ins_109508"  # First Platypus Bank
E2E_INCOME = 4500.0
E2E_DOB = date(1990, 6, 15)
# Band 2 (30-39) for a 36-year-old — derived, not hardcoded, so this section
# keeps working as the calendar moves past the DOB above.
E2E_BAND = tier_assignment.age_band_for_date_of_birth(
    E2E_DOB, today=datetime.now(timezone.utc).date())
E2E_DEPENDANTS = 1


def sign_up(email, password):
    req = urllib.request.Request(
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
        method="POST",
        data=json.dumps({"email": email, "password": password,
                         "returnSecureToken": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"\nSign-up failed ({e.code}): {e.read()!r}")
    return body["localId"], body["idToken"]


def delete_collection(ref, batch_size=200):
    deleted = 0
    for doc in ref.list_documents(page_size=batch_size):
        for sub in doc.collections():
            deleted += delete_collection(sub, batch_size)
        doc.delete()
        deleted += 1
    return deleted


print()
print("=" * 78)
print("B1. End to end — throwaway user, real Sandbox Item, real endpoints")
print("=" * 78)

init_firebase()
db = get_db()
stamp = int(time.time())
email = f"e2e.score.{stamp}@unify-test.example"
uid, token = sign_up(email, "E2eScore-12345")
print(f"test user uid={uid}")

from fastapi.testclient import TestClient  # noqa: E402
import main as backend_main  # noqa: E402

client = TestClient(backend_main.app)


def api(method, path, body=None):
    r = client.request(method, path, headers={"Authorization": f"Bearer {token}"},
                       json=body)
    return r.status_code, r.json()


try:
    # --- before onboarding: no financialProfile document at all --------------
    db.collection("users").document(uid).set({
        "name": "E2E Score User", "email": email,
        "date_of_birth": datetime.combine(E2E_DOB, datetime.min.time(),
                                          tzinfo=timezone.utc),
        "dependants": E2E_DEPENDANTS,
    })
    status, body = api("GET", "/score")
    check("GET /score with no financialProfile -> 404, not a 500",
          status == 404, f"{status}: {body.get('detail')}")
    status, body = api("POST", "/score/refresh")
    check("POST /score/refresh with no financialProfile -> 404",
          status == 404, f"{status}: {body.get('detail')}")

    # --- onboarding's exact write, then GET /score before anything computes --
    score_profile.profile_ref(db, uid).set({
        "monthly_income_sgd": E2E_INCOME, "income_source": "user_entered",
        "monthly_expenses_sgd": 0.0, "net_worth_sgd": 0.0,
        "overall_score": None, "checklist_flags": None,
        "checklist_last_updated": None, "ml_profile": None,
    })
    status, body = api("GET", "/score")
    check("GET /score before any computation -> 200 'never computed', "
          "not an error",
          status == 200 and body["computed"] is False
          and body["overall_score"] is None,
          f"{status}: computed={body.get('computed')}")

    # --- a first refresh with no Plaid data and no policies ------------------
    status, body = api("POST", "/score/refresh")
    check("POST /score/refresh with only income + DOB -> 200 partial",
          status == 200
          and body["domains"]["banking"]["status"] == STATUS_NOT_ASSESSED
          and body["domains"]["insurance"]["status"] == STATUS_ASSESSED
          and body["overall_score"] is None
          and body["overall_score_reason"] == REASON_INCOMPLETE_DOMAINS,
          f"insurance={body['domains']['insurance']['tier_label']}, "
          f"reason={body['overall_score_reason']}")
    check("zero uploaded policies really does produce 'Not Covered' end to end",
          body["domains"]["insurance"]["tier_label"] == "Not Covered")

    status, body = api("GET", "/score")
    check("GET /score now reads the cache -> computed true",
          status == 200 and body["computed"] is True
          and body["ml_profile"]["age_band"] == E2E_BAND,
          f"computed={body.get('computed')}, "
          f"age_band={body.get('ml_profile', {}).get('age_band')}")
    cached_ml = body["ml_profile"]
    check("cached ml_profile records the not-assessed domains and why",
          cached_ml["domains"]["banking"]["status"] == STATUS_NOT_ASSESSED
          and cached_ml["domains"]["banking"]["missing_features"] != [],
          str(cached_ml["domains"]["banking"]["missing_features"]))

    # --- connect a real Sandbox Item: the hook must fire on exchange-token ---
    before = score_profile.read_cached_score(db, uid)["computed_at"]
    pt = plaid_service.get_plaid_client().sandbox_public_token_create(
        SandboxPublicTokenCreateRequest(
            institution_id=SANDBOX_INSTITUTION,
            initial_products=[Products("transactions"), Products("investments")],
        )
    ).to_dict()["public_token"]
    status, body = api("POST", "/plaid/exchange-token", {"public_token": pt})
    item_id = body.get("item_id")
    check("POST /plaid/exchange-token still succeeds with the hook attached",
          status == 200 and bool(item_id))
    check("exchange-token response carries the recomputed score",
          isinstance(body.get("score"), dict)
          and body["score"]["age_band"] == E2E_BAND,
          f"score keys: {sorted(body.get('score') or {})[:4]}…")
    after = score_profile.read_cached_score(db, uid)["computed_at"]
    check("exchange-token hook wrote a fresh cache", after > before,
          f"{before} -> {after}")

    deadline = time.time() + 120
    while time.time() < deadline:
        status, body = api("POST", "/plaid/sync-transactions", {"item_id": item_id})
        if status == 200 and not body["results"][0]["added"]:
            break
        time.sleep(5)

    # --- every remaining hook fires and updates the cache -------------------
    for label, method, path, payload in [
        ("/plaid/sync-transactions", "POST", "/plaid/sync-transactions",
         {"item_id": item_id}),
        ("/plaid/sync-investments", "POST", "/plaid/sync-investments",
         {"item_id": item_id}),
        ("/profile/refresh", "POST", "/profile/refresh", None),
    ]:
        before = score_profile.read_cached_score(db, uid)["computed_at"]
        time.sleep(0.05)
        status, body = api(method, path, payload)
        after = score_profile.read_cached_score(db, uid)["computed_at"]
        check(f"{label:<26} fires the score hook and returns it",
              status == 200 and isinstance(body.get("score"), dict)
              and after > before,
              f"{status}, cache {before} -> {after}")

    # --- with real banking + investment data, the score should now exist ----
    status, full_e2e = api("POST", "/score/refresh")
    print(f"INFO  e2e tiers: "
          + ", ".join(f"{d}={full_e2e['domains'][d]['tier_label']}"
                      for d in score_pipeline.DOMAIN_ORDER))
    check("all three domains assessed off real Sandbox data",
          all(full_e2e["domains"][d]["status"] == STATUS_ASSESSED
              for d in score_pipeline.DOMAIN_ORDER),
          f"overall_score={full_e2e['overall_score']}")
    check("overall_score present and in range",
          isinstance(full_e2e["overall_score"], (int, float))
          and 0.0 <= full_e2e["overall_score"] <= 100.0,
          str(full_e2e["overall_score"]))
    check("layer 3 evaluated once all three tiers exist",
          full_e2e["layer3_evaluated"] is True,
          f"flags={full_e2e['layer3_flags']}")

    # The cached copy must equal what the endpoint returned, field for field.
    status, cached = api("GET", "/score")
    check("GET /score returns exactly what the refresh cached",
          cached["overall_score"] == full_e2e["overall_score"]
          and cached["ml_profile"]["domains"] == {
              d: {k: v for k, v in full_e2e["domains"][d].items()}
              for d in score_pipeline.DOMAIN_ORDER}
          and cached["checklist_flags"]["layer3"] == full_e2e["layer3_flags"])

    # --- the non-fatal contract, proved by breaking the pipeline on purpose --
    original = score_profile.refresh_score_profile

    def boom(*args, **kwargs):
        raise RuntimeError("deliberate pipeline failure")

    score_profile.refresh_score_profile = boom
    try:
        status, body = api("POST", "/plaid/sync-transactions", {"item_id": item_id})
        check("a throwing pipeline does NOT break the sync it is attached to",
              status == 200 and body["score"] is None
              and body["items_synced"] == 1,
              f"{status}, score={body.get('score')}, "
              f"items_synced={body.get('items_synced')}")
        status, body = api("POST", "/profile/refresh")
        check("…nor /profile/refresh's own derived fields",
              status == 200 and body["status"] == "refreshed"
              and body["score"] is None,
              f"{status}: {body.get('status')}")
    finally:
        score_profile.refresh_score_profile = original
    still = score_profile.read_cached_score(db, uid)
    check("the previous good cache survives a failed refresh",
          still["overall_score"] == full_e2e["overall_score"])

    # --- income = 0 and an unsupported age, end to end ----------------------
    score_profile.profile_ref(db, uid).update({"monthly_income_sgd": 0.0})
    status, body = api("POST", "/score/refresh")
    check("income = 0 -> 200 with reason income_not_set, no 500",
          status == 200 and body["unavailable_reason"] == REASON_INCOME_NOT_SET
          and body["overall_score"] is None,
          f"{status}: {body.get('unavailable_reason')}")
    status, cached = api("GET", "/score")
    check("the unavailable state is cached, clearing the stale score",
          cached["overall_score"] is None
          and cached["ml_profile"]["unavailable_reason"] == REASON_INCOME_NOT_SET)
    score_profile.profile_ref(db, uid).update({"monthly_income_sgd": E2E_INCOME})

    # A DOB outside 20-70 cannot be produced by signing up (the picker allows
    # it, but nothing else here needs a second account) — writing the field
    # directly is the same document state a real 75-year-old would have.
    db.collection("users").document(uid).update({
        "date_of_birth": datetime(1946, 6, 15, tzinfo=timezone.utc)})
    status, body = api("POST", "/score/refresh")
    check("DOB outside 20-70 -> 200 with reason unsupported_age, no 500",
          status == 200
          and body["unavailable_reason"] == REASON_UNSUPPORTED_AGE
          and body["age_band"] is None,
          f"{status}: {body.get('unavailable_reason')}")
    check("…and no domain was assessed from a model that never saw this age",
          all(body["domains"][d]["tier_label"] is None
              for d in score_pipeline.DOMAIN_ORDER))

finally:
    user_ref = db.collection("users").document(uid)
    n = 0
    for sub in user_ref.collections():
        n += delete_collection(sub)
    user_ref.delete()
    admin_auth.delete_user(uid)
    print(f"cleanup: removed test user and {n} Firestore docs")

print()
print("=" * 78)
print(f"RESULT: {len(passed)} passed, {len(failed)} failed")
if failed:
    for name in failed:
        print(f"  - {name}")
print("=" * 78)
sys.exit(1 if failed else 0)
