"""Overall-score + Layer 1 checklist verification — backend/ml/overall_score.py.

    cd backend && venv\\Scripts\\activate && python scripts/verify_overall_score.py

Same shape as verify_layer3_flags.py: no Firebase, no Plaid, no network, and no
third-party imports at all. compute_overall_score() and
evaluate_layer1_checklist() are pure functions, so every case below is a direct
call with hand-built inputs.

The one file it reads is the committed, tracked
MLAI/clustering/results/gated_score_validation/per_user_gated_scores.csv — the
recorded output of gated_score_validation.py's artifact tier replay over all
7,000 synthetic users. Section 7 replays the shipped implementation against it
user-by-user. That script is neither imported nor re-run: its per-user tiers are
already committed, so reading them is strictly more faithful than re-deriving
them (and keeps this script free of pandas/numpy/sklearn and of the gitignored
dataset). It is the check that catches an implementation typo which passes every
hand-built case above.

Coverage:
  - all 32 combinations: 8 tier triples x income_protection x hospitalisation,
    against a hardcoded expected table of option_b_score / ceiling / overall
  - the four named patterns: all-strong, all-weak, Wealth Without Protection,
    Protection First
  - the ceiling binding, and NOT binding, at each of the three insurance_third
    levels it can take below 100
  - the specific compensatory-failure check: Banking=Stable + Investment=On
    Track + zero insurance lands under 50, at exactly the 39 ceiling
  - None handling on all five score inputs, and on dependants
  - unrecognised tier labels and non-binary coverage flags raising ValueError
  - Layer 1: both checks pass/fail, message selection by dependants 0 vs 1+,
    not-computable rows, and dependants never affecting pass/fail
  - the 7,000-user replay against the recorded validation figures
"""
import csv
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from ml.layer3 import (  # noqa: E402
    BANKING_BUILDING,
    BANKING_STABLE,
    BANKING_TIERS,
    INSURANCE_COVERED,
    INSURANCE_NOT_COVERED,
    INSURANCE_TIERS,
    INVESTMENT_GETTING_STARTED,
    INVESTMENT_ON_TRACK,
    INVESTMENT_TIERS,
)
from ml.overall_score import (  # noqa: E402
    CEILING_BY_INDICATORS_CLEARED,
    CHECK_HOSPITALISATION,
    CHECK_INCOME_PROTECTION,
    LAYER1_CHECKS,
    compute_overall_score,
    compute_score_breakdown,
    evaluate_layer1_checklist,
)

REPLAY_CSV = (REPO_ROOT / "MLAI" / "clustering" / "results"
              / "gated_score_validation" / "per_user_gated_scores.csv")
REPLAY_COLUMN = "c_optionb_capped[band_ladder]"

TOL = 1e-9

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}{' - ' + detail if detail else ''}")


def close(got, want) -> bool:
    return got is not None and abs(got - want) < TOL


def expect_score(label: str, got, want) -> None:
    check(label, close(got, want), f"got {got}, want {want}")


# ---------------------------------------------------------------------------
# The expected table, derived by hand from the locked formula rather than by
# calling the implementation. Keyed by (banking_score + investment_score,
# insurance indicators cleared) -> (option_b_score, ceiling, overall_score).
#
#   option_b = (banking + investment + 100 * cleared / 3) / 3
#   ceiling  = 39 / 69 / 100 / 100 for cleared 0 / 1 / 2 / 3
# ---------------------------------------------------------------------------
T1, T2 = 100 / 3, 200 / 3          # the two recurring thirds, written once
EXPECTED = {
    #                       option_b    ceiling  overall
    (0, 0):                 (0.0,          39.0,   0.0),
    (0, 1):                 (T1 / 3,       69.0,   T1 / 3),         # 11.11
    (0, 2):                 (T2 / 3,      100.0,   T2 / 3),         # 22.22
    (0, 3):                 (T1,          100.0,   T1),             # 33.33
    (100, 0):               (T1,           39.0,   T1),             # 33.33
    (100, 1):               (400 / 9,      69.0,   400 / 9),        # 44.44
    (100, 2):               (500 / 9,     100.0,   500 / 9),        # 55.56
    (100, 3):               (T2,          100.0,   T2),             # 66.67
    (200, 0):               (T2,           39.0,   39.0),           # CAPPED
    (200, 1):               (700 / 9,      69.0,   69.0),           # CAPPED
    (200, 2):               (800 / 9,     100.0,   800 / 9),        # 88.89
    (200, 3):               (100.0,       100.0, 100.0),
}


print("=" * 78)
print("1. All 32 combinations - 8 tier triples x income_protection x")
print("   hospitalisation, against the hand-derived expected table")
print("=" * 78)
seen_cells = set()
for banking in BANKING_TIERS:
    for investment in INVESTMENT_TIERS:
        for insurance in INSURANCE_TIERS:
            for ip in (0, 1):
                for hosp in (0, 1):
                    bi = (100 * (banking == BANKING_STABLE)
                          + 100 * (investment == INVESTMENT_ON_TRACK))
                    cleared = (insurance == INSURANCE_COVERED) + ip + hosp
                    want_ob, want_ceil, want_overall = EXPECTED[(bi, cleared)]
                    seen_cells.add((bi, cleared))
                    b = compute_score_breakdown(
                        banking, investment, insurance, ip, hosp)
                    ok = (close(b["option_b_score"], want_ob)
                          and close(b["ceiling"], want_ceil)
                          and close(b["overall_score"], want_overall)
                          and b["insurance_indicators_cleared"] == cleared
                          and b["capped"] == (want_ob > want_ceil + TOL)
                          and close(compute_overall_score(
                              banking, investment, insurance, ip, hosp),
                              want_overall))
                    check(
                        f"{banking[:6]:<6}/{investment[:8]:<8}/"
                        f"{insurance[:5]:<5} ip={ip} hosp={hosp} -> "
                        f"ob {b['option_b_score']:6.2f} cap {b['ceiling']:5.1f} "
                        f"= {b['overall_score']:6.2f}",
                        ok,
                        f"want ob {want_ob:.4f} ceiling {want_ceil} "
                        f"overall {want_overall:.4f}")
check(f"all 12 (base, cleared) cells of the expected table were exercised "
      f"({len(seen_cells)}/12)", len(seen_cells) == 12)

print()
print("=" * 78)
print("2. The four named patterns")
print("=" * 78)
all_strong = compute_score_breakdown(
    BANKING_STABLE, INVESTMENT_ON_TRACK, INSURANCE_COVERED, 1, 1)
expect_score("all-strong (Stable / On Track / Covered / both flags) -> 100",
             all_strong["overall_score"], 100.0)
check("all-strong is not capped", all_strong["capped"] is False)

all_weak = compute_score_breakdown(
    BANKING_BUILDING, INVESTMENT_GETTING_STARTED, INSURANCE_NOT_COVERED, 0, 0)
expect_score("all-weak (Building / Getting Started / Not Covered / neither) -> 0",
             all_weak["overall_score"], 0.0)
check("all-weak is not capped (0 is already under the 39 ceiling)",
      all_weak["capped"] is False)

# The pattern the whole formula was designed to fix. Under Option B this user
# scored 66.67 and sat ABOVE the normal population mean.
wwp = compute_score_breakdown(
    BANKING_STABLE, INVESTMENT_ON_TRACK, INSURANCE_NOT_COVERED, 0, 0)
expect_score("Wealth Without Protection: option_b_score is 66.67",
             wwp["option_b_score"], T2)
expect_score("Wealth Without Protection: capped to the 39 ceiling",
             wwp["overall_score"], 39.0)
check("Wealth Without Protection: capped flag set", wwp["capped"] is True)
check("Wealth Without Protection: insurance_third is 0",
      wwp["insurance_third"] == 0.0)

# Protection First: excellent coverage, weak banking. Must be rewarded relative
# to the same banking profile with no coverage, but must NOT top the table -
# an archetype with genuinely weak banking leading would be the mirror-image
# error (brief Section 5, Round 3).
pf = compute_score_breakdown(
    BANKING_BUILDING, INVESTMENT_GETTING_STARTED, INSURANCE_COVERED, 1, 1)
expect_score("Protection First (Building / Getting Started / fully covered) "
             "-> 33.33", pf["overall_score"], T1)
check("Protection First is never capped (full coverage -> ceiling 100)",
      pf["capped"] is False and pf["ceiling"] == 100.0)
check("Protection First scores above the same banking profile with no cover "
      f"({pf['overall_score']:.2f} > {all_weak['overall_score']:.2f})",
      pf["overall_score"] > all_weak["overall_score"])
check("Protection First does not top the table (all-strong still leads)",
      pf["overall_score"] < all_strong["overall_score"])

print()
print("=" * 78)
print("3. The ceiling binds - and does not bind - at each insurance_third")
print("=" * 78)
# cleared = 0 -> ceiling 39
expect_score("third=0, both domains strong: 66.67 -> 39 (binds)",
             compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                                   INSURANCE_NOT_COVERED, 0, 0), 39.0)
expect_score("third=0, one domain strong: 33.33 stays 33.33 (does not bind)",
             compute_overall_score(BANKING_STABLE, INVESTMENT_GETTING_STARTED,
                                   INSURANCE_NOT_COVERED, 0, 0), T1)
# cleared = 1 -> ceiling 69
expect_score("third=33.33, both domains strong: 77.78 -> 69 (binds)",
             compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                                   INSURANCE_NOT_COVERED, 1, 0), 69.0)
expect_score("third=33.33 via hospitalisation instead: 77.78 -> 69 (binds, "
             "the three indicators are interchangeable)",
             compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                                   INSURANCE_NOT_COVERED, 0, 1), 69.0)
expect_score("third=33.33 via the insurance tier instead: 77.78 -> 69 (binds)",
             compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                                   INSURANCE_COVERED, 0, 0), 69.0)
expect_score("third=33.33, one domain strong: 44.44 stays 44.44 (does not bind)",
             compute_overall_score(BANKING_BUILDING, INVESTMENT_ON_TRACK,
                                   INSURANCE_NOT_COVERED, 1, 0), 400 / 9)
# cleared = 2 -> ceiling 100, can never bind (max option_b at 2/3 is 88.89)
expect_score("third=66.67, both domains strong: 88.89 uncapped (ceiling 100)",
             compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                                   INSURANCE_COVERED, 1, 0), 800 / 9)
never_caps = [
    compute_score_breakdown(bank, inv, ins, ip, hosp)["capped"]
    for bank in BANKING_TIERS for inv in INVESTMENT_TIERS
    for ins in INSURANCE_TIERS for ip in (0, 1) for hosp in (0, 1)
    if (ins == INSURANCE_COVERED) + ip + hosp >= 2
]
check("the cap can never fire once 2 of 3 indicators are cleared",
      not any(never_caps))
# Nobody's score ever moves UP - min() only ever removes points.
moved_up = [
    b for b in (compute_score_breakdown(bank, inv, ins, ip, hosp)
                for bank in BANKING_TIERS for inv in INVESTMENT_TIERS
                for ins in INSURANCE_TIERS for ip in (0, 1) for hosp in (0, 1))
    if b["overall_score"] > b["option_b_score"] + TOL
]
check("no combination scores HIGHER than its uncapped option_b_score",
      not moved_up)
check("only users strong in BOTH banking and investment are ever capped",
      all(b["banking_score"] == 100.0 and b["investment_score"] == 100.0
          for b in (compute_score_breakdown(bank, inv, ins, ip, hosp)
                    for bank in BANKING_TIERS for inv in INVESTMENT_TIERS
                    for ins in INSURANCE_TIERS for ip in (0, 1)
                    for hosp in (0, 1))
          if b["capped"]))
check("the ceiling ladder values are exactly the TIER_THRESHOLDS boundaries "
      f"{sorted(set(CEILING_BY_INDICATORS_CLEARED.values()))}",
      sorted(set(CEILING_BY_INDICATORS_CLEARED.values())) == [39.0, 69.0, 100.0])

print()
print("=" * 78)
print("4. The compensatory-failure check - strong Banking + strong Investment")
print("   + zero insurance must land well below a mid-table score")
print("=" * 78)
compensatory = compute_overall_score(
    BANKING_STABLE, INVESTMENT_ON_TRACK, INSURANCE_NOT_COVERED, 0, 0)
check(f"scores below 50 ({compensatory:.2f} < 50)", compensatory < 50)
expect_score("lands on exactly the 39 ceiling, not merely somewhere below 50",
             compensatory, 39.0)
check("39 is the top of the At Risk band - this user can never read 'Building'",
      compensatory <= 39)
# Same user, one indicator cleared: must improve, but still be capped.
one_cleared = compute_overall_score(
    BANKING_STABLE, INVESTMENT_ON_TRACK, INSURANCE_NOT_COVERED, 1, 0)
check(f"clearing one indicator improves the score ({compensatory:.2f} -> "
      f"{one_cleared:.2f}) but is still capped at 69", one_cleared == 69.0)

print()
print("=" * 78)
print("5. Missing data - None is returned, never a fabricated default")
print("=" * 78)
COMPLETE = (BANKING_STABLE, INVESTMENT_ON_TRACK, INSURANCE_COVERED, 1, 1)
for i, name in enumerate(["banking_tier", "investment_tier", "insurance_tier",
                          "income_protection_coverage",
                          "hospitalisation_coverage"]):
    args = list(COMPLETE)
    args[i] = None
    check(f"{name}=None -> compute_overall_score returns None",
          compute_overall_score(*args) is None,
          f"got {compute_overall_score(*args)}")
    check(f"{name}=None -> compute_score_breakdown returns None",
          compute_score_breakdown(*args) is None)
check("all five inputs None -> None",
      compute_overall_score(None, None, None, None, None) is None)
# A cleared=0 user and a not-computable user must NOT be the same result: the
# whole point of returning None rather than defaulting a flag to 0.
check("a missing hospitalisation flag is not scored as a protection gap "
      "(None, not 39.0)",
      compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                            INSURANCE_NOT_COVERED, 0, None) is None)
# A genuine 0 is real data and must be scored, not treated as missing.
expect_score("a genuine 0 coverage flag IS scored (0 is data, None is absence)",
             compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                                   INSURANCE_NOT_COVERED, 0, 0), 39.0)
check("False is accepted as a 0 indicator (Firestore bool round trip)",
      compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                            INSURANCE_NOT_COVERED, False, False) == 39.0)
check("True is accepted as a 1 indicator",
      compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                            INSURANCE_COVERED, True, True) == 100.0)
check("1.0 float is accepted as a 1 indicator",
      compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                            INSURANCE_COVERED, 1.0, 1.0) == 100.0)

print()
print("=" * 78)
print("6. Stale / unrecognised inputs raise, never silently score as 'weak'")
print("=" * 78)
bad_tier_cases = [
    # The pre-July-2026 placeholder wording the artifacts no longer use.
    ("banking_tier", "Strong", ("Strong", INVESTMENT_ON_TRACK, INSURANCE_COVERED)),
    ("investment_tier", "Weak", (BANKING_STABLE, "Weak", INSURANCE_COVERED)),
    ("banking_tier", "stable", ("stable", INVESTMENT_ON_TRACK, INSURANCE_COVERED)),
    ("insurance_tier", "Uncovered", (BANKING_STABLE, INVESTMENT_ON_TRACK,
                                     "Uncovered")),
    # A tier label borrowed from the wrong domain.
    ("banking_tier", "Covered", ("Covered", INVESTMENT_ON_TRACK, INSURANCE_COVERED)),
]
for arg, bad_value, tiers in bad_tier_cases:
    try:
        compute_overall_score(*tiers, 1, 1)
        check(f"{arg}={bad_value!r} raises ValueError", False, "no exception")
    except ValueError:
        check(f"{arg}={bad_value!r} raises ValueError", True)

bad_flag_cases = [2, -1, 0.5, "yes", "1", ""]
for bad in bad_flag_cases:
    try:
        compute_overall_score(BANKING_STABLE, INVESTMENT_ON_TRACK,
                              INSURANCE_COVERED, bad, 1)
        check(f"income_protection_coverage={bad!r} raises ValueError", False,
              "no exception")
    except ValueError:
        check(f"income_protection_coverage={bad!r} raises ValueError", True)
# The bad-tier check must happen even when another input is None, so a stale
# label can't hide behind a missing one.
try:
    compute_overall_score("Strong", None, INSURANCE_COVERED, 1, 1)
    check("a bad tier still raises when another input is None", False,
          "no exception")
except ValueError:
    check("a bad tier still raises when another input is None", True)

print()
print("=" * 78)
print("7. Layer 1 checklist - two scored checks, dependants as tone only")
print("=" * 78)
both_fail_solo = evaluate_layer1_checklist(0, 0, 0)
both_fail_deps = evaluate_layer1_checklist(0, 0, 2)
both_pass = evaluate_layer1_checklist(1, 1, 0)
mixed = evaluate_layer1_checklist(1, 0, 3)

check("always returns one entry per check, in LAYER1_CHECKS order",
      [f["check"] for f in both_fail_solo] == list(LAYER1_CHECKS)
      == [CHECK_INCOME_PROTECTION, CHECK_HOSPITALISATION])
check("both flags fail when both inputs are 0",
      [f["passed"] for f in both_fail_solo] == [False, False])
check("both flags pass when both inputs are 1",
      [f["passed"] for f in both_pass] == [True, True])
check("flags are independent (income protection passes, hospitalisation fails)",
      [f["passed"] for f in mixed] == [True, False])
check("every entry is marked scored=True",
      all(f["scored"] is True for f in both_fail_solo + both_pass))
check("every entry carries a non-empty message",
      all(f["message"] for f in both_fail_solo + both_pass + mixed))

# Message selection by dependants - the locked tone-only behaviour.
for i, name in enumerate(LAYER1_CHECKS):
    solo, deps = both_fail_solo[i]["message"], both_fail_deps[i]["message"]
    check(f"{name}: failure copy differs between dependants 0 and 2",
          solo != deps, "the two variants are identical")
    check(f"{name}: the dependants>0 variant mentions the people affected",
          "depend on you" in deps or "household" in deps, deps)
check("pass copy does NOT branch on dependants (the fact is the same)",
      [f["message"] for f in evaluate_layer1_checklist(1, 1, 0)]
      == [f["message"] for f in evaluate_layer1_checklist(1, 1, 4)])
for n in (1, 2, 5, 17):
    check(f"dependants={n} selects the same copy as any other 1+ value",
          [f["message"] for f in evaluate_layer1_checklist(0, 0, n)]
          == [f["message"] for f in both_fail_deps])
check("dependants=None falls back to the neutral copy",
      [f["message"] for f in evaluate_layer1_checklist(0, 0, None)]
      == [f["message"] for f in both_fail_solo])

# dependants is NEVER scored: it cannot change any pass/fail outcome.
outcomes = {
    tuple(f["passed"] for f in evaluate_layer1_checklist(ip, hosp, d))
    for ip in (0, 1) for hosp in (0, 1)
    for d in (None, 0, 1, 2, 9)
}
check(f"dependants never changes a pass/fail outcome ({len(outcomes)} distinct "
      "outcome tuples across 4 input pairs x 5 dependant values)",
      len(outcomes) == 4)

# Not-computable rows: present, but neither pass nor fail.
unknown = evaluate_layer1_checklist(None, None, 0)
check("None coverage -> passed is None, not False",
      [f["passed"] for f in unknown] == [None, None])
check("None coverage rows are still returned (a dropped row would read as OK)",
      len(unknown) == len(LAYER1_CHECKS))
check("None coverage copy says it cannot be checked, not that cover is missing",
      all("cannot be checked" in f["message"] for f in unknown))
partial = evaluate_layer1_checklist(None, 1, 0)
check("one known, one unknown -> [None, True]",
      [f["passed"] for f in partial] == [None, True])

for bad in (-1, True, 1.5, "2"):
    try:
        evaluate_layer1_checklist(0, 0, bad)
        check(f"dependants={bad!r} raises ValueError", False, "no exception")
    except ValueError:
        check(f"dependants={bad!r} raises ValueError", True)
try:
    evaluate_layer1_checklist(2, 0, 0)
    check("a non-binary coverage flag raises ValueError", False, "no exception")
except ValueError:
    check("a non-binary coverage flag raises ValueError", True)

print()
print("=" * 78)
print("8. Cross-check vs the recorded 7,000-user validation replay")
print("=" * 78)
if not REPLAY_CSV.exists():
    check(f"validation replay CSV present at {REPLAY_CSV}", False, "not found")
else:
    # tier_* columns are cluster health ranks (0 = healthiest), exactly as
    # gated_score_validation.py wrote them; the artifacts' tier_labels are
    # healthiest-first, which is what layer3.py's *_TIERS tuples restate.
    tier_labels = {"banking": BANKING_TIERS, "investment": INVESTMENT_TIERS,
                   "insurance": INSURANCE_TIERS}
    rows, mismatches, worst = 0, 0, 0.0
    scores, capped_n = [], 0
    normal_scores, gap_scores = [], []
    levels = set()
    with REPLAY_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            ip = int(round(float(row["income_protection_coverage"])))
            hosp = int(round(float(row["hospitalisation_coverage"])))
            b = compute_score_breakdown(
                tier_labels["banking"][int(row["tier_banking"])],
                tier_labels["investment"][int(row["tier_investment"])],
                tier_labels["insurance"][int(row["tier_insurance"])],
                ip, hosp,
            )
            recorded = float(row[REPLAY_COLUMN])
            delta = abs(round(b["overall_score"], 2) - recorded)
            worst = max(worst, delta)
            if delta > 0.005:
                mismatches += 1
            scores.append(b["overall_score"])
            levels.add(round(b["overall_score"], 4))
            capped_n += b["capped"]
            if row["profile_type"] == "normal":
                normal_scores.append(b["overall_score"])
            # The key check's group: Banking Stable + Insurance Not Covered +
            # neither Layer 1 flag.
            if (int(row["tier_banking"]) == 0 and int(row["tier_insurance"]) == 1
                    and ip == 0 and hosp == 0):
                gap_scores.append(b["overall_score"])

    mean = sum(scores) / len(scores)
    normal_mean = sum(normal_scores) / len(normal_scores)
    gap_mean = sum(gap_scores) / len(gap_scores)
    pct_capped = 100.0 * capped_n / rows

    check(f"read all 7,000 replayed users ({rows})", rows == 7000)
    check(f"shipped compute_overall_score() reproduces the recorded score for "
          f"every user (mismatches={mismatches}, worst delta={worst:.4f})",
          mismatches == 0)
    check(f"population mean {mean:.2f} matches the recorded 41.94",
          abs(mean - 41.94) < 0.01, f"got {mean:.4f}")
    check(f"reaches the recorded 11 distinct score levels ({len(levels)})",
          len(levels) == 11)
    check(f"zero-insurance/strong-banking group size {len(gap_scores)} matches "
          "the recorded n=1,005", len(gap_scores) == 1005)
    check(f"that group's mean {gap_mean:.2f} matches the recorded 35.45",
          abs(gap_mean - 35.45) < 0.01, f"got {gap_mean:.4f}")
    check(f"normal-population mean {normal_mean:.2f} matches the recorded 41.00",
          abs(normal_mean - 41.00) < 0.01, f"got {normal_mean:.4f}")
    check(f"the group sits {gap_mean - normal_mean:+.2f} points below the normal "
          "mean - the recorded -5.55 finding",
          abs((gap_mean - normal_mean) - (-5.55)) < 0.01,
          f"got {gap_mean - normal_mean:+.4f}")
    check("0% of that group reaches the normal-population mean",
          not any(s >= normal_mean for s in gap_scores))
    check("0% of that group reaches the Stable band (70+)",
          not any(s >= 70 for s in gap_scores))
    check(f"the cap fires on {pct_capped:.2f}% of users - the recorded 11.26%",
          abs(pct_capped - 11.26) < 0.01, f"got {pct_capped:.4f}%")

print()
print("=" * 78)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 78)
sys.exit(1 if failed else 0)
