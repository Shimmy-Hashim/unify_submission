"""Layer 3 combination-rule verification — backend/ml/layer3.py.

    cd backend && venv\\Scripts\\activate && python scripts/verify_layer3_flags.py

Unlike the other verify_* scripts here, this one needs no Firebase, no Plaid
and no network: detect_combination_flags() is a pure function, so every case
below is a direct call with hand-built inputs. The only file it reads is the
committed production artifact set, to confirm layer3.py's tier-label
constants still match the artifacts' own tier_map.

Coverage:
  - all 8 Banking x Investment x Insurance tier combinations, with the
    continuous features absent, proving rules 1 and 2 are decided by tier
    alone and that rules 3 and 4 cannot fire without their inputs
  - rule 3's three thresholds: exactly at, just inside, just outside
  - rule 4's threshold: exactly at, just inside, just outside
  - both co-occurring pairs (rule 1 + rule 3, rule 2 + rule 4)
  - the "boring middle" case where nothing fires
  - None handling (missing tier, partially missing continuous features)
  - unrecognised / stale tier labels raising ValueError
  - artifact tier_label agreement, including per-band health_rank mapping
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from ml import layer3  # noqa: E402
from ml.layer3 import (  # noqa: E402
    BANKING_BUILDING,
    BANKING_STABLE,
    FLAG_AGGRESSIVE_INVESTING_WEAK_FOUNDATION,
    FLAG_PROTECTION_FIRST,
    FLAG_STRONG_FOUNDATION_ROOM_TO_GROW,
    FLAG_WEALTH_WITHOUT_PROTECTION,
    INSURANCE_COVERED,
    INSURANCE_NOT_COVERED,
    INVESTMENT_GETTING_STARTED,
    INVESTMENT_ON_TRACK,
    detect_combination_flags,
)

R1 = FLAG_WEALTH_WITHOUT_PROTECTION
R2 = FLAG_PROTECTION_FIRST
R3 = FLAG_STRONG_FOUNDATION_ROOM_TO_GROW
R4 = FLAG_AGGRESSIVE_INVESTING_WEAK_FOUNDATION

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}{' — ' + detail if detail else ''}")


def expect(label: str, got: list, want: list) -> None:
    check(label, got == want, f"got {got}, want {want}")


# A conservative_saver-shaped signature: inside all three of rule 3's bounds.
RULE3_MET = dict(investment_rate=0.01, savings_rate=0.35, emergency_fund_months=8.0)
# An over_invested-shaped signature: above rule 4's investment_rate floor.
RULE4_MET = dict(investment_rate=0.30, savings_rate=0.05, emergency_fund_months=0.5)


print("=" * 72)
print("1. All 8 tier combinations (continuous features absent)")
print("=" * 72)
# With every continuous feature None, only the tier-only rules can fire —
# so this table is also the proof that rules 3 and 4 stay silent rather than
# treating missing data as a zero.
tier_only_cases = [
    (BANKING_STABLE, INVESTMENT_ON_TRACK, INSURANCE_COVERED, []),
    (BANKING_STABLE, INVESTMENT_ON_TRACK, INSURANCE_NOT_COVERED, [R1]),
    (BANKING_STABLE, INVESTMENT_GETTING_STARTED, INSURANCE_COVERED, []),
    (BANKING_STABLE, INVESTMENT_GETTING_STARTED, INSURANCE_NOT_COVERED, [R1]),
    (BANKING_BUILDING, INVESTMENT_ON_TRACK, INSURANCE_COVERED, [R2]),
    (BANKING_BUILDING, INVESTMENT_ON_TRACK, INSURANCE_NOT_COVERED, []),
    (BANKING_BUILDING, INVESTMENT_GETTING_STARTED, INSURANCE_COVERED, [R2]),
    (BANKING_BUILDING, INVESTMENT_GETTING_STARTED, INSURANCE_NOT_COVERED, []),
]
for bank, inv, ins, want in tier_only_cases:
    expect(
        f"{bank:<8} / {inv:<15} / {ins:<11} -> {want or 'no flags'}",
        detect_combination_flags(bank, inv, ins),
        want,
    )

print()
print("=" * 72)
print("2. Rule 3 boundaries — investment_rate <= 0.03, savings_rate >= 0.28,")
print("   emergency_fund_months >= 5.0 (Banking=Stable, Investment=Getting Started)")
print("=" * 72)


def rule3(**overrides) -> list:
    features = {**RULE3_MET, **overrides}
    return detect_combination_flags(
        BANKING_STABLE, INVESTMENT_GETTING_STARTED, INSURANCE_COVERED, **features
    )


# investment_rate (a ceiling: <=)
expect("investment_rate exactly 0.03 (at the bound) fires", rule3(investment_rate=0.03), [R3])
expect("investment_rate 0.029 (just inside) fires", rule3(investment_rate=0.029), [R3])
expect("investment_rate 0.031 (just outside) silent", rule3(investment_rate=0.031), [])
# savings_rate (a floor: >=)
expect("savings_rate exactly 0.28 (at the bound) fires", rule3(savings_rate=0.28), [R3])
expect("savings_rate 0.2801 (just inside) fires", rule3(savings_rate=0.2801), [R3])
expect("savings_rate 0.2799 (just outside) silent", rule3(savings_rate=0.2799), [])
# emergency_fund_months (a floor: >=)
expect("emergency_fund_months exactly 5.0 (at the bound) fires",
       rule3(emergency_fund_months=5.0), [R3])
expect("emergency_fund_months 5.01 (just inside) fires",
       rule3(emergency_fund_months=5.01), [R3])
expect("emergency_fund_months 4.99 (just outside) silent",
       rule3(emergency_fund_months=4.99), [])
# All three must hold together.
expect("all three exactly at their bounds fires",
       rule3(investment_rate=0.03, savings_rate=0.28, emergency_fund_months=5.0), [R3])
expect("two bounds met, one missed -> silent",
       rule3(investment_rate=0.03, savings_rate=0.28, emergency_fund_months=4.99), [])
# Tier conditions still required on top of the signature.
expect("rule 3 signature but Banking=Building -> rule 3 silent "
       "(rule 2 fires on tier)",
       detect_combination_flags(BANKING_BUILDING, INVESTMENT_GETTING_STARTED,
                                INSURANCE_COVERED, **RULE3_MET), [R2])
expect("rule 3 signature but Banking=Building, Insurance=Not Covered -> no flags",
       detect_combination_flags(BANKING_BUILDING, INVESTMENT_GETTING_STARTED,
                                INSURANCE_NOT_COVERED, **RULE3_MET), [])
expect("rule 3 signature but Investment=On Track -> silent",
       detect_combination_flags(BANKING_STABLE, INVESTMENT_ON_TRACK,
                                INSURANCE_COVERED, **RULE3_MET), [])

print()
print("=" * 72)
print("3. Rule 4 boundary — investment_rate >= 0.18")
print("   (Banking=Building, Investment=On Track)")
print("=" * 72)


def rule4(investment_rate) -> list:
    return detect_combination_flags(
        BANKING_BUILDING, INVESTMENT_ON_TRACK, INSURANCE_NOT_COVERED,
        investment_rate=investment_rate,
    )


expect("investment_rate exactly 0.18 (at the bound) fires", rule4(0.18), [R4])
expect("investment_rate 0.181 (just inside) fires", rule4(0.181), [R4])
expect("investment_rate 0.179 (just outside) silent", rule4(0.179), [])
expect("investment_rate 0.40 (well inside) fires", rule4(0.40), [R4])
expect("rule 4 signature but Banking=Stable -> rule 4 silent (rule 1 fires on tier)",
       detect_combination_flags(BANKING_STABLE, INVESTMENT_ON_TRACK,
                                INSURANCE_NOT_COVERED, **RULE4_MET), [R1])
expect("rule 4 signature but Investment=Getting Started -> rule 4 silent "
       "(rule 2 fires on tier)",
       detect_combination_flags(BANKING_BUILDING, INVESTMENT_GETTING_STARTED,
                                INSURANCE_COVERED, **RULE4_MET), [R2])

print()
print("=" * 72)
print("4. Two flags at once, and the zero-flag middle")
print("=" * 72)
expect(
    "Stable / Getting Started / Not Covered + rule-3 signature -> rule 1 AND rule 3",
    detect_combination_flags(BANKING_STABLE, INVESTMENT_GETTING_STARTED,
                             INSURANCE_NOT_COVERED, **RULE3_MET),
    [R1, R3],
)
expect(
    "Building / On Track / Covered + rule-4 signature -> rule 2 AND rule 4",
    detect_combination_flags(BANKING_BUILDING, INVESTMENT_ON_TRACK,
                             INSURANCE_COVERED, **RULE4_MET),
    [R2, R4],
)
expect(
    "Building / Getting Started / Not Covered, ordinary features -> no flags",
    detect_combination_flags(BANKING_BUILDING, INVESTMENT_GETTING_STARTED,
                             INSURANCE_NOT_COVERED,
                             investment_rate=0.09, savings_rate=0.22,
                             emergency_fund_months=4.6),
    [],
)
# Three flags must be unreachable: rules 1/2 and 3/4 are each split by
# banking_tier, so no input can produce more than one from either pair.
max_flags = 0
for bank in (BANKING_STABLE, BANKING_BUILDING, None):
    for inv in (INVESTMENT_ON_TRACK, INVESTMENT_GETTING_STARTED, None):
        for ins in (INSURANCE_COVERED, INSURANCE_NOT_COVERED, None):
            for features in (RULE3_MET, RULE4_MET):
                max_flags = max(
                    max_flags, len(detect_combination_flags(bank, inv, ins, **features))
                )
check(f"no input in the full 3x3x3x2 sweep produces more than 2 flags (max={max_flags})",
      max_flags == 2, f"max was {max_flags}")

print()
print("=" * 72)
print("5. Missing data — None is never read as zero")
print("=" * 72)
expect("insurance_tier None -> rule 1 cannot fire",
       detect_combination_flags(BANKING_STABLE, INVESTMENT_GETTING_STARTED, None,
                                **RULE3_MET),
       [R3])
expect("banking_tier None -> only tier-free rules could fire (none exist)",
       detect_combination_flags(None, INVESTMENT_ON_TRACK, INSURANCE_NOT_COVERED,
                                **RULE4_MET), [])
expect("rule 3: savings_rate None -> silent (not treated as 0.0)",
       rule3(savings_rate=None), [])
expect("rule 3: emergency_fund_months None -> silent",
       rule3(emergency_fund_months=None), [])
expect("rule 3: investment_rate None -> silent",
       rule3(investment_rate=None), [])
expect("rule 4: investment_rate None -> silent", rule4(None), [])
expect("no continuous features supplied at all -> tier-only rules still work",
       detect_combination_flags(BANKING_STABLE, INVESTMENT_GETTING_STARTED,
                                INSURANCE_NOT_COVERED), [R1])
# A zero savings_rate is real data (spent everything), not missing data — it
# must be evaluated, and it correctly fails rule 3's floor.
expect("savings_rate 0.0 is evaluated as real data, not skipped",
       rule3(savings_rate=0.0), [])

print()
print("=" * 72)
print("6. Stale / unrecognised tier labels raise, never silently no-op")
print("=" * 72)
bad_cases = [
    # The pre-July-2026 placeholder wording, which the artifacts no longer use.
    ("banking_tier", "Strong", ("Strong", INVESTMENT_ON_TRACK, INSURANCE_COVERED)),
    ("investment_tier", "Weak", (BANKING_STABLE, "Weak", INSURANCE_COVERED)),
    # Case and near-miss wording.
    ("banking_tier", "stable", ("stable", INVESTMENT_ON_TRACK, INSURANCE_COVERED)),
    ("insurance_tier", "Uncovered", (BANKING_STABLE, INVESTMENT_ON_TRACK, "Uncovered")),
]
for arg, bad_value, args in bad_cases:
    try:
        detect_combination_flags(*args)
        check(f"{arg}={bad_value!r} raises ValueError", False, "no exception raised")
    except ValueError:
        check(f"{arg}={bad_value!r} raises ValueError", True)

print()
print("=" * 72)
print("7. Tier-label constants vs the committed production artifacts")
print("=" * 72)
try:
    declared = layer3.verify_tier_labels_against_artifacts()
    for domain, labels in declared.items():
        check(f"{domain:<11} artifact tier_labels == layer3.py constants: {labels}", True)
    check("per-band tier_map labels all match their health_rank "
          "(catches a rebuild pinned to cluster ids)", True)
except AssertionError as exc:
    check("artifact tier labels match layer3.py constants", False, str(exc))

print()
print("=" * 72)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 72)
sys.exit(1 if failed else 0)
