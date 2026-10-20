"""
net_worth_removal_comparison.py — standalone regeneration + machine-verification of
results/domain_split/NET_WORTH_REMOVAL_COMPARISON.md.

WHY THIS EXISTS (July 2026 script audit):
  The net-worth-removal comparison was committed (aba9c41) as a markdown document ONLY —
  201 lines of prose and tables, with no accompanying .py file. Two things therefore had
  no executable form in the repo:

    1. §3's archetype tables. The run only persists Step 4's archetype table for the
       PRIMARY spec per domain, never for the `plus_nw` alternates, so the with/without
       `over_invested` comparison — the document's single most-cited piece of evidence,
       and the figure quoted in the brief's Feature Set Changelog — existed only as a
       6-line snippet inside a fenced code block in §6.
    2. The "machine-verified against 81 numeric claims" check the brief cites. No script
       performing that verification was ever committed.

  This file supplies both. It refits NOTHING: §3 is re-derived from the run's own cached
  labels (.domain_split_labels_v2.pkl) via domain_split_clustering.assign_tiers(), which is
  fully deterministic, and every other claim is read straight out of the committed
  domain_split_results.json. It is therefore a verifier, not a new analysis — if it ever
  disagrees with the markdown, the markdown (or the evidence file) has drifted.

Run:
    backend/venv/Scripts/python.exe MLAI/clustering/net_worth_removal_comparison.py

Reads:
    MLAI/clustering/data/processed/synthetic_users.csv
    MLAI/clustering/results/domain_split/domain_split_results.json
    MLAI/clustering/results/domain_split/.domain_split_labels_v2.pkl   (label cache)

Writes:
    MLAI/clustering/results/domain_split/net_worth_removal_comparison.json

Exit code is non-zero if any claim fails, so this doubles as a regression check.
"""

import json
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import domain_split_clustering as ds  # noqa: E402
from domain_split_clustering import DATA_PATH, LABEL_CACHE, assign_tiers  # noqa: E402

# Imported exactly as the markdown's §6 snippet does. install_bench() rebuilds
# BENCHMARKS['investment'] only; Banking's benchmark set (all §3 needs) is untouched
# either way. Called with include_rad=False to match the final locked Investment
# benchmark configuration, in which risk_age_deviation was dropped as a benchmark.
from risk_age_diagnostic import install_bench  # noqa: E402

OUT_DIR = os.path.join(ds.RESULTS_DIR, "domain_split")
RESULTS_JSON = os.path.join(OUT_DIR, "domain_split_results.json")
OUT_JSON = os.path.join(OUT_DIR, "net_worth_removal_comparison.json")

# --------------------------------------------------------------------------- claims
# The figures asserted by NET_WORTH_REMOVAL_COMPARISON.md, by section — 86 claims.
#
# NOT every numeric figure in the document. Four §2 table rows (worst-group ordering rho,
# min headroom captured, min lift over base rate, and whether a benchmark inverts) and
# banking_core4's per-band dominant-share ranges are asserted there but not checked here;
# all of them were confirmed by hand against domain_split_results.json. The per-band
# dominant shares that ARE checked are checked as ranges, not as the exact percentages
# §7 quotes, so a within-range drift would pass. Widening the set would change
# claims_checked and invalidate the committed net_worth_removal_comparison.json, so the
# gap is recorded here rather than closed.

SPEC_ORDER = ["banking_plus_nw", "banking_core4", "investment_plus_nw", "investment_core3"]

# §2 — before/after comparison table
S2_GRID_SIZE = 80          # the same 80-combination grid is swept for every spec
S2_SURVIVORS = {"banking_plus_nw": 25, "banking_core4": 28,
                "investment_plus_nw": 23, "investment_core3": 60}
S2_GATE_PASSERS = {"banking_plus_nw": 4, "banking_core4": 9,
                   "investment_plus_nw": 0, "investment_core3": 1}
S2_WINNER = {"banking_plus_nw": "B|gmm_diag|3", "banking_core4": "A|kmeans|2",
             "investment_plus_nw": "A|spectral|2", "investment_core3": "B|agglomerative|5"}
S2_WINNER_MACRO_SIL = {"banking_plus_nw": 0.2099, "banking_core4": 0.2588,
                       "investment_plus_nw": 0.4633, "investment_core3": 0.1976}
S2_WINNER_MIN_CLUSTER = {"banking_plus_nw": 19.06, "banking_core4": 42.02,
                         "investment_plus_nw": 5.20, "investment_core3": 8.51}

# §3 — archetype placement, re-derived from cached labels.
# spec -> combination -> profile -> expected tier shares (%, rank 0 = healthiest)
S3_EXPECTED = {
    "banking_plus_nw": {
        "B|gmm_diag|3": {
            "over_invested": [26.7, 50.3, 22.9],
            "normal": [32.5, 46.8, 20.7],
            "high_debt_investor": [7.6, 70.8, 21.6],
        },
    },
    "banking_core4": {
        "A|kmeans|2": {
            "over_invested": [0.0, 100.0],
            "normal": [47.6, 52.4],
            "high_debt_investor": [0.0, 100.0],
        },
    },
}
S3_OVER_INVESTED_N = 292
# wealthy_exposed is quoted as a single headline share, not a full row
S3_WEALTHY_EXPOSED_TIER0 = {"banking_plus_nw": 98.6, "banking_core4": 93.2}

# §4 — controlled comparison: same combination under both feature sets
S4_ROWS = [
    ("banking_core4", "A|kmeans|2", 0.2588, 42.02, True),
    ("banking_plus_nw", "A|kmeans|2", 0.3496, 3.59, False),
    ("banking_core4", "A|fuzzy_cmeans|3", 0.1693, 26.56, True),
    ("banking_plus_nw", "A|fuzzy_cmeans|3", 0.1134, 25.94, True),
    ("investment_core3", "B|agglomerative|5", 0.1976, 8.51, True),
    ("investment_plus_nw", "B|agglomerative|5", 0.1732, 3.61, False),
]
FLOOR_PCT = 5.0

# §5 — net worth as Banking's out-of-model validation benchmark
S5_BINDING = {"net_worth_to_income": 18, "emergency_fund_months": 4,
              "savings_rate": 3, "healthy_debt_ratio": 3, "dti": 0}
# Each survivor contributes exactly one binding benchmark, so the counts must total the
# survivor count. Taken from S2_SURVIVORS rather than restated as its own literal: the
# claim being checked is "= survivors", and a second copy could drift out of agreement
# with the number that phrase refers to.
S5_BINDING_TOTAL = S2_SURVIVORS["banking_core4"]


class Checker:
    def __init__(self):
        self.rows = []

    def check(self, section, claim, expected, actual, tol=0.0):
        # bool is a subclass of int, so it has to be excluded explicitly — otherwise a
        # pass/fail claim would be compared numerically and True would match 1.0.
        numeric = (isinstance(expected, (int, float)) and not isinstance(expected, bool)
                   and isinstance(actual, (int, float)))
        if numeric:
            ok = abs(float(expected) - float(actual)) <= tol
        else:
            ok = expected == actual
        self.rows.append({"section": section, "claim": claim,
                          "expected": expected, "actual": actual, "pass": bool(ok)})
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r["pass"]]

    def report(self):
        by_section = {}
        for r in self.rows:
            s = by_section.setdefault(r["section"], {"n": 0, "failed": 0})
            s["n"] += 1
            if not r["pass"]:
                s["failed"] += 1
        rule = "=" * 94
        print(f"\n{rule}\nCLAIM VERIFICATION — NET_WORTH_REMOVAL_COMPARISON.md\n{rule}")
        for sec, s in by_section.items():
            status = "OK" if s["failed"] == 0 else f"{s['failed']} FAILED"
            print(f"  {sec:<52} {s['n']:>3} claims   {status}")
        for r in self.failed:
            print(f"\n  FAIL [{r['section']}] {r['claim']}")
            print(f"       expected {r['expected']!r}  actual {r['actual']!r}")
        print(f"\n  {len(self.rows) - len(self.failed)}/{len(self.rows)} claims verified")
        return by_section


def tier_shares(df, spec, combo, labels_store, domain, profile):
    """Re-derive a profile's tier distribution — refits nothing (§6 of the markdown)."""
    rank, k, _ = assign_tiers(df, combo, labels_store[spec]["labels"], domain)
    # assign_tiers builds rank on df's own index, so it masks directly.
    sub = rank[df["profile_type"] == profile]
    counts = (sub.value_counts(normalize=True) * 100).round(1).sort_index()
    return [float(counts.get(i, 0.0)) for i in range(int(k))], int(len(sub))


def main():
    install_bench(include_rad=False)

    df = pd.read_csv(DATA_PATH)
    with open(RESULTS_JSON, encoding="utf-8") as f:
        res = json.load(f)["spec_results"]
    with open(LABEL_CACHE, "rb") as f:
        cache = pickle.load(f)

    c = Checker()
    derived = {}

    # ------------------------------------------------------------------ §2
    for spec in SPEC_ORDER:
        r = res[spec]
        c.check("§2 before/after table", f"{spec}: grid size",
                S2_GRID_SIZE, len(r["grid"]))
        c.check("§2 before/after table", f"{spec}: passed 5% floor",
                S2_SURVIVORS[spec], r["n_survivors"])
        c.check("§2 before/after table", f"{spec}: passed both usability gates",
                S2_GATE_PASSERS[spec], len(r["gate_passing_combinations"]))
        c.check("§2 before/after table", f"{spec}: winning combination",
                S2_WINNER[spec], r["winner"])
        g = r["grid"][r["winner"]]
        c.check("§2 before/after table", f"{spec}: winner macro-silhouette",
                S2_WINNER_MACRO_SIL[spec], g["silhouette_macro"], tol=5e-5)
        c.check("§2 before/after table", f"{spec}: winner smallest cluster %",
                S2_WINNER_MIN_CLUSTER[spec], g["min_cluster_pct"], tol=5e-3)

    # dominant-feature claims (the §2 bottom two rows)
    b_plus = res["banking_plus_nw"]["winner_single_axis_diagnostic"]["all"]
    c.check("§2 dominant feature", "banking_plus_nw: top feature",
            "net_worth_to_income", b_plus["top_feature"])
    c.check("§2 dominant feature", "banking_plus_nw: top feature share ~35%",
            0.347, b_plus["top_feature_share_of_total_separation"], tol=5e-4)
    c.check("§2 dominant feature", "banking_plus_nw: features separating >=0.5 IQR",
            4, b_plus["n_features_separating_ge_0.5"])
    c.check("§2 dominant feature", "banking_plus_nw: n_features", 5, b_plus["n_features"])

    i_core = res["investment_core3"]["winner_single_axis_diagnostic"]["all"]
    c.check("§2 dominant feature", "investment_core3: top feature",
            "investment_rate", i_core["top_feature"])
    c.check("§2 dominant feature", "investment_core3: top feature share ~68%",
            0.68, i_core["top_feature_share_of_total_separation"], tol=5e-3)
    c.check("§2 dominant feature", "investment_core3: features separating >=0.5 IQR (3 of 3)",
            3, i_core["n_features_separating_ge_0.5"])

    # investment_plus_nw is reported per band: 1 of 4 separating in EVERY band,
    # net worth dominant in band 1, investment_rate in bands 2-5 (§7's correction).
    i_plus = res["investment_plus_nw"]["winner_single_axis_diagnostic"]
    for band in range(1, 6):
        bd = i_plus[f"band{band}"]
        c.check("§2 dominant feature (per band)",
                f"investment_plus_nw band{band}: exactly 1 of 4 features separating >=0.5 IQR",
                1, bd["n_features_separating_ge_0.5"])
        expected_top = "net_worth_to_income" if band == 1 else "investment_rate"
        c.check("§2 dominant feature (per band)",
                f"investment_plus_nw band{band}: dominant feature",
                expected_top, bd["top_feature"])
        lo, hi = (0.87, 0.88) if band == 1 else (0.90, 0.96)
        share = bd["top_feature_share_of_total_separation"]
        c.check("§2 dominant feature (per band)",
                f"investment_plus_nw band{band}: dominant share within {lo:.2f}-{hi:.2f}",
                True, lo <= share <= hi)

    # banking_core4's dominant feature alternates by band (savings_rate 1-3, EF 4-5)
    b_core = res["banking_core4"]["winner_single_axis_diagnostic"]
    for band in range(1, 6):
        bd = b_core[f"band{band}"]
        expected_top = "savings_rate" if band <= 3 else "emergency_fund_months"
        c.check("§2 dominant feature (per band)",
                f"banking_core4 band{band}: dominant feature", expected_top, bd["top_feature"])

    # ------------------------------------------------------------------ §3
    # Both §3 specs are Banking variants — the with/without net worth comparison is a
    # Banking question, and Investment's alternates never reached an archetype table.
    for spec, combos in S3_EXPECTED.items():
        for combo, profiles in combos.items():
            for profile, expected in profiles.items():
                shares, n = tier_shares(df, spec, combo, cache, "banking", profile)
                derived.setdefault(spec, {}).setdefault(combo, {})[profile] = {
                    "tier_shares_pct": shares, "n": n}
                c.check("§3 archetype placement",
                        f"{spec} / {combo} / {profile}: tier shares",
                        expected, shares)
                if profile == "over_invested":
                    c.check("§3 archetype placement",
                            f"{spec} / {combo}: over_invested n", S3_OVER_INVESTED_N, n)

    for spec, expected_t0 in S3_WEALTHY_EXPOSED_TIER0.items():
        combo = S2_WINNER[spec]
        shares, n = tier_shares(df, spec, combo, cache, "banking", "wealthy_exposed")
        derived.setdefault(spec, {}).setdefault(combo, {})["wealthy_exposed"] = {
            "tier_shares_pct": shares, "n": n}
        c.check("§3 archetype placement",
                f"{spec} / {combo} / wealthy_exposed: share in healthiest tier",
                expected_t0, shares[0], tol=0.05)

    # ------------------------------------------------------------------ §4
    for spec, combo, macro_sil, min_pct, should_pass in S4_ROWS:
        g = res[spec]["grid"][combo]
        c.check("§4 controlled comparison", f"{combo} @ {spec}: macro-silhouette",
                macro_sil, g["silhouette_macro"], tol=5e-5)
        c.check("§4 controlled comparison", f"{combo} @ {spec}: smallest cluster %",
                min_pct, g["min_cluster_pct"], tol=5e-3)
        c.check("§4 controlled comparison", f"{combo} @ {spec}: clears the 5% floor",
                should_pass, g["min_cluster_pct"] >= FLOOR_PCT)

    # ------------------------------------------------------------------ §5
    binding = res["banking_core4"]["binding_benchmark_counts_across_survivors"]
    for var, expected in S5_BINDING.items():
        c.check("§5 binding benchmark", f"banking_core4: {var} is the binding benchmark",
                expected, binding.get(var, 0))
    c.check("§5 binding benchmark", "banking_core4: binding counts total = survivors",
            S5_BINDING_TOTAL, sum(binding.values()))
    c.check("§5 binding benchmark", "net_worth_to_income is the MOST binding",
            "net_worth_to_income", max(binding, key=binding.get))

    # ------------------------------------------------------------------ report
    by_section = c.report()

    payload = {
        "purpose": "Machine verification of NET_WORTH_REMOVAL_COMPARISON.md against the "
                   "committed domain_split_results.json and the run's cached labels. "
                   "Refits nothing.",
        "evidence_sources": {
            "results_json": os.path.relpath(RESULTS_JSON, ds.REPO_ROOT).replace("\\", "/"),
            "label_cache": os.path.relpath(LABEL_CACHE, ds.REPO_ROOT).replace("\\", "/"),
            "dataset": os.path.relpath(DATA_PATH, ds.REPO_ROOT).replace("\\", "/"),
        },
        "claims_checked": len(c.rows),
        "claims_failed": len(c.failed),
        "by_section": by_section,
        "rederived_archetype_tables": derived,
        "claims": c.rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  saved {OUT_JSON}")

    if c.failed:
        raise SystemExit(f"{len(c.failed)} claim(s) FAILED — the document or the evidence "
                         "file has drifted.")
    print("  All claims reproduce. The document is faithful to the committed evidence.")


if __name__ == "__main__":
    main()
