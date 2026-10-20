"""Layer 2 tier-assignment verification — backend/ml/tier_assignment.py.

    cd backend && venv\\Scripts\\activate && python scripts/verify_tier_assignment.py

Like verify_layer3_flags.py and verify_overall_score.py, this needs no Firebase,
no Plaid and no network. It reads only committed files, all read-only:

    backend/ml/artifacts/domain_split/*                 the production artifacts
    MLAI/clustering/data/processed/synthetic_users.csv  the fitted-on dataset
    MLAI/clustering/results/domain_split/
        domain_split_tier_assignments.csv               per-user ground truth
        domain_split_investment_shipped_tiers.csv       per-user Investment ground truth
        risk_age_final.json                             independent cross-check

THE CENTRAL CLAIM (section 3 below)
    assign_domain_tier(), called one user at a time with raw feature dicts,
    reproduces the per-user tier assignments already committed by the MLAI
    clustering work — exactly, zero mismatches. If that holds, the production
    port is faithful to the validated model rather than a lookalike of it.

⚠️ ONE GROUND-TRUTH SOURCE IS NOT USABLE FOR INVESTMENT, AND WHY.
    domain_split_tier_assignments.csv was written by domain_split_clustering.py
    on 2026-07-26, BEFORE Investment was recalibrated. Its own
    domain_split_results.json records the winners it used:

        banking     A|kmeans|2        == the shipped model
        insurance   B|kmedoids|3      == the shipped model (native, pre-merge)
        investment  B|agglomerative|5 != the shipped A|fuzzy_cmeans|2

    So its tier_investment column holds FIVE tiers from a superseded model and
    cannot be compared to a 2-tier shipped assignment. This is a property of
    the evidence trail (Investment's winner changed later — see
    FINAL_TIER_DECISIONS.md and risk_age_final.json), NOT a discrepancy in the
    artifacts. Banking and Insurance are compared per user against that file;
    Investment is held to two other committed sources instead, both of which
    are per-band and derived from the shipped model's own fitted labels:

      - each band's tier_benchmark_profile ``n`` in investment_model.json,
        i.e. the exact per-band per-tier user counts at build time
      - risk_age_final.json's normal-population tier x age-band crosstab for
        A|fuzzy_cmeans|2, produced by a different script (risk_age_final.py)
        than the one that built the artifacts

    Banking is held to all three, so the two weaker sources are calibrated
    against a source that IS per-user.

    Investment is now ALSO compared per user, against
    domain_split_investment_shipped_tiers.csv, written 2026-10-13 by
    MLAI/clustering/investment_shipped_tier_assignments.py from the
    research-stage fit of A|fuzzy_cmeans|2 alone (benchmark variant V2). That
    script imports nothing from backend/ and reads no artifact, so it is a
    ground truth independent of this module. The two per-band sources above
    are kept as cross-checks.

Coverage:
  1. module constants vs the artifacts and the dataset (feature order, tier
     labels, age-band ranges and midpoints), and the JSON scaler numbers vs the
     committed pickled scalers on all 7,000 rows
  2. every user, every domain: replay through the production function
  3. the replay vs the committed ground truth, plus band x tier coverage
  4. missing / absent / non-finite features -> None, never a fabricated tier
  5. unknown domain, and age_band validation for all three domains
  6. Insurance: age_band is required and genuinely changes the answer
  7. the Banking cluster-0 age-band flip, the trap most likely to look right
  8. age_band_for_date_of_birth, including the out-of-range decision
"""
import json
import math
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd  # noqa: E402

from ml import layer3, scoring, tier_assignment  # noqa: E402
from ml.tier_assignment import (  # noqa: E402
    AGE_BAND_MIDPOINT,
    AGE_BAND_RANGES,
    AGE_BANDS,
    DOMAINS,
    age_band_for_date_of_birth,
    assign_domain_tier,
    assign_domain_tier_index,
)

ART_DIR = BACKEND_DIR / "ml" / "artifacts" / "domain_split"
DATA_PATH = REPO_ROOT / "MLAI/clustering/data/processed/synthetic_users.csv"
RESULTS_DIR = REPO_ROOT / "MLAI/clustering/results/domain_split"
ASSIGNMENTS_PATH = RESULTS_DIR / "domain_split_tier_assignments.csv"
INVESTMENT_SHIPPED_PATH = RESULTS_DIR / "domain_split_investment_shipped_tiers.csv"
RISK_AGE_FINAL_PATH = RESULTS_DIR / "risk_age_final.json"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}{' — ' + detail if detail else ''}")
    else:
        failed += 1
        print(f"  [FAIL] {label}{' — ' + detail if detail else ''}")


def expect(label: str, got, want) -> None:
    check(label, got == want, f"got {got!r}, want {want!r}")


def raises(label: str, exc_types, fn, *args, **kwargs) -> None:
    try:
        result = fn(*args, **kwargs)
    except exc_types as exc:
        check(label, True, f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}")
    except Exception as exc:  # noqa: BLE001 - wrong exception type is a failure
        check(label, False, f"raised {type(exc).__name__}, expected "
                            f"{getattr(exc_types, '__name__', exc_types)}")
    else:
        check(label, False, f"no exception; returned {result!r}")


models = {d: tier_assignment._load_model(d) for d in DOMAINS}
artifacts = {d: json.loads((ART_DIR / f"{d}_model.json").read_text(encoding="utf-8"))
             for d in DOMAINS}

print("=" * 78)
print("1. Module constants vs the committed artifacts and dataset")
print("=" * 78)

# Feature ORDER is positional against each scaler's center_/scale_, so this is
# the check that stops the transform being applied to the wrong column.
try:
    declared = scoring.verify_clustering_features_against_artifacts()
    for domain, feats in declared.items():
        expect(f"{domain:<11} feature order: scoring.py == artifact == module",
               models[domain]["features"], feats)
except AssertionError as exc:
    check("scoring.py feature lists match the artifacts", False, str(exc))

try:
    for domain, labels in layer3.verify_tier_labels_against_artifacts().items():
        expect(f"{domain:<11} tier labels: layer3.py == artifact == module",
               tier_assignment.tier_labels(domain), labels)
except AssertionError as exc:
    check("layer3.py tier labels match the artifacts", False, str(exc))

try:
    ranges = tier_assignment.verify_age_bands_against_artifacts()
    check(f"AGE_BAND_RANGES matches every Approach A artifact's own age_range "
          f"({', '.join(sorted(ranges))})", True)
except AssertionError as exc:
    check("AGE_BAND_RANGES matches the artifacts", False, str(exc))

df = pd.read_csv(DATA_PATH)
# The Insurance model's fourth input is the dataset's age_midpoint, which holds
# exactly one value per band. A wrong midpoint here moves every Insurance user.
observed_midpoints = {int(b): sorted(g["age_midpoint"].unique())
                      for b, g in df.groupby("age_band")}
expect("AGE_BAND_MIDPOINT matches the dataset the models were fitted on",
       {b: v[0] for b, v in observed_midpoints.items()},
       {b: AGE_BAND_MIDPOINT[b] for b in AGE_BANDS})
check("each age band has exactly one age_midpoint in the dataset",
      all(len(v) == 1 for v in observed_midpoints.values()),
      str({b: len(v) for b, v in observed_midpoints.items()}))

# The module deliberately never unpickles (module docstring): it applies the
# JSON's plain center_/scale_. That is only safe if it is numerically identical
# to the committed fitted scaler, on real data, not just on paper.
for domain in DOMAINS:
    model = models[domain]
    with open(ART_DIR / f"{domain}_scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    reference = scaler.transform(df[model["features"]])
    worst = 0.0
    for row_i, (_, row) in enumerate(df[model["features"]].iterrows()):
        for col_i, name in enumerate(model["features"]):
            mine = (float(row[name]) - model["center"][col_i]) / model["scale"][col_i]
            worst = max(worst, abs(mine - float(reference[row_i][col_i])))
    check(f"{domain:<11} JSON center_/scale_ transform == the pickled "
          f"{type(scaler).__name__}, all {len(df):,} rows",
          worst == 0.0, f"max abs difference {worst:.3g}")

    if model["age"] is not None:
        with open(ART_DIR / f"{domain}_age_scaler.pkl", "rb") as fh:
            age_scaler = pickle.load(fh)
        reference = age_scaler.transform(df[["age_midpoint"]])
        mean, scale = model["age"]
        worst = max(
            abs((AGE_BAND_MIDPOINT[int(band)] - mean) / scale - float(ref[0]))
            for band, ref in zip(df["age_band"], reference))
        check(f"{domain:<11} JSON mean_/scale_ age transform == the pickled "
              f"{type(age_scaler).__name__}, all {len(df):,} rows",
              worst == 0.0, f"max abs difference {worst:.3g}")

print()
print("=" * 78)
print("2. Replaying all 7,000 synthetic users through assign_domain_tier()")
print("=" * 78)

replayed = {d: [] for d in DOMAINS}      # tier index per user, in dataset order
clusters = {d: [] for d in DOMAINS}      # native cluster id, for the flip check
margins = {d: math.inf for d in DOMAINS}  # smallest winning distance gap

records = df.to_dict("records")
for row in records:
    band = int(row["age_band"])
    for domain in DOMAINS:
        model = models[domain]
        features = {name: row[name] for name in model["features"]}
        index = assign_domain_tier_index(domain, features, band)
        label = assign_domain_tier(domain, features, band)
        replayed[domain].append(index)
        # Recompute the cluster id and the distance margin through the module's
        # own internals, so the margin quoted below is the real one.
        vector = tier_assignment._feature_vector(model, features, band)
        cluster_id = tier_assignment._nearest_cluster_id(model, vector, band)
        clusters[domain].append(cluster_id)
        key = band if model["approach"] == "A" else None
        d2 = sorted(math.fsum((a - b) ** 2 for a, b in zip(vector, c))
                    for c in model["centroids"][key])
        margins[domain] = min(margins[domain], d2[1] - d2[0])
        # Label and index must be two views of one answer, per user.
        if label != model["tier_labels"][index]:
            raise SystemExit(f"{domain}: label {label!r} != index {index} label")

for domain in DOMAINS:
    n_none = sum(1 for t in replayed[domain] if t is None)
    check(f"{domain:<11} every user got a tier (dataset is complete)",
          n_none == 0, f"{n_none} None of {len(records):,}")
    check(f"{domain:<11} label and index agree for all {len(records):,} users", True)
    check(f"{domain:<11} nearest-centroid decision is not a coin flip",
          margins[domain] > 1e-9,
          f"smallest winning margin over the whole population: "
          f"{margins[domain]:.3e} (squared distance)")

print()
print("=" * 78)
print("3. THE CENTRAL CHECK — replay vs the committed per-user ground truth")
print("=" * 78)

truth = pd.read_csv(ASSIGNMENTS_PATH)
check("ground-truth file is aligned to the dataset, user for user",
      list(truth["user_id"]) == list(df["user_id"]), f"{len(truth):,} rows")

# --- Banking: directly comparable. Same model, same 2 tiers, health-ranked.
mismatches = [(u, g, w) for u, g, w
              in zip(df["user_id"], replayed["banking"], truth["tier_banking"])
              if g != int(w)]
check(f"banking    per-user replay == domain_split_tier_assignments.csv "
      f"(all {len(df):,} users, A|kmeans|2)",
      not mismatches, f"{len(mismatches)} mismatches"
      + (f", first: {mismatches[:3]}" if mismatches else ""))

# --- Insurance: same fitted model, but the committed column predates the merge
# and holds NATIVE health ranks 0-2. Map it forward using the artifact's own
# merge documentation — never a hardcoded {0:0, 1:1, 2:1}.
insurance_art = artifacts["insurance"]
native_order = insurance_art["model"]["native_health_order_best_to_worst"]
shipped_map = insurance_art["tier_merge"]["shipped_tier_map"]
rank_to_shipped = {rank: shipped_map[str(cid)]["shipped_tier"]
                   for rank, cid in enumerate(native_order)}
check("insurance  native health rank -> shipped tier, derived from the "
      "artifact's own merge block",
      rank_to_shipped == {0: 0, 1: 1, 2: 1},
      f"health order {native_order} -> {rank_to_shipped}")
mismatches = [(u, g, w) for u, g, w
              in zip(df["user_id"], replayed["insurance"], truth["tier_insurance"])
              if g != rank_to_shipped[int(w)]]
check(f"insurance  per-user replay == domain_split_tier_assignments.csv through "
      f"that merge (all {len(df):,} users, B|kmedoids|3)",
      not mismatches, f"{len(mismatches)} mismatches"
      + (f", first: {mismatches[:3]}" if mismatches else ""))

# --- Investment: the committed column is a superseded winner (see the module
# docstring). Stated as a finding, then held to the two per-band sources.
truth_investment_tiers = sorted(set(int(t) for t in truth["tier_investment"]))
results = json.loads((RESULTS_DIR / "domain_split_results.json").read_text(encoding="utf-8"))
check("investment the committed per-user column is a SUPERSEDED model, so it is "
      "not comparable — recorded, not papered over",
      results["winners"]["investment"] != artifacts["investment"]["combination"]
      and len(truth_investment_tiers) != artifacts["investment"]["shipped_tiers"],
      f"csv column = {results['winners']['investment']} with tiers "
      f"{truth_investment_tiers}; shipped = "
      f"{artifacts['investment']['combination']} with "
      f"{artifacts['investment']['shipped_tiers']} tiers")

# --- Investment, per user: the shipped-model column written by the research-stage
# script (A|fuzzy_cmeans|2, benchmark V2). It does not import this module or read
# the artifacts, so a match is a real per-user comparison, unlike the column above.
shipped = pd.read_csv(INVESTMENT_SHIPPED_PATH)
check("investment shipped-model ground-truth file is aligned to the dataset, user "
      "for user", list(shipped["user_id"]) == list(df["user_id"]),
      f"{len(shipped):,} rows")
mismatches = [(u, g, w) for u, g, w
              in zip(df["user_id"], replayed["investment"], shipped["tier_investment"])
              if g != int(w)]
check(f"investment per-user replay == domain_split_investment_shipped_tiers.csv "
      f"(all {len(df):,} users, A|fuzzy_cmeans|2)",
      not mismatches, f"{len(mismatches)} mismatches"
      + (f", first: {mismatches[:3]}" if mismatches else ""))

# Per-band per-tier counts, straight off each artifact's own fitted profile.
for domain in ("banking", "investment"):
    art = artifacts[domain]
    bad = []
    for band_key, band in art["bands"].items():
        band_no = int(band_key)
        for tier_key, entry in band["tier_benchmark_profile"].items():
            got = sum(1 for row, t in zip(records, replayed[domain])
                      if int(row["age_band"]) == band_no and t == int(tier_key))
            if got != entry["n"]:
                bad.append(f"band {band_no} tier {tier_key}: {got} vs {entry['n']}")
    check(f"{domain:<11} per-band per-tier counts == the artifact's own fitted "
          f"profile (10 cells)", not bad, "; ".join(bad) if bad else "exact")

# The independent script's crosstab: normal-profile users only, row-normalised
# by age band, 3 dp — 10 cells per domain, from risk_age_final.py.
risk_age = json.loads(RISK_AGE_FINAL_PATH.read_text(encoding="utf-8"))
crosstabs = risk_age["tier_by_age_band_crosstabs"]
for domain, key in [("banking", "banking_validated_A|kmeans|2"),
                    ("investment", "investment_chosen_A|fuzzy_cmeans|2")]:
    want = crosstabs[key]["crosstab_row_normalised"]
    bad = []
    for tier_key, by_band in want.items():
        for band_key, share in by_band.items():
            rows = [t for row, t in zip(records, replayed[domain])
                    if int(row["age_band"]) == int(band_key)
                    and row["profile_type"] == "normal"]
            got = round(sum(1 for t in rows if t == int(tier_key)) / len(rows), 3)
            if got != share:
                bad.append(f"band {band_key} tier {tier_key}: {got} vs {share}")
    check(f"{domain:<11} normal-population tier x age-band crosstab == "
          f"risk_age_final.json ({key.split('_')[-1]}, 10 cells)",
          not bad, "; ".join(bad) if bad else "exact to 3 dp")

# Population shares, the same figure verify_production_artifacts.py checks.
for domain in DOMAINS:
    want = [round(float(x), 1) for x in
            artifacts[domain]["shipped_tiers_summary"]["population_shares_pct"]]
    got = [round(100 * sum(1 for t in replayed[domain] if t == i) / len(records), 1)
           for i in range(len(want))]
    expect(f"{domain:<11} replayed population shares == the artifact's own", got, want)

print("\n  Sample coverage — users replayed per age band and tier:")
for domain in DOMAINS:
    labels = models[domain]["tier_labels"]
    print(f"    {domain}")
    for band in AGE_BANDS:
        counts = [sum(1 for row, t in zip(records, replayed[domain])
                      if int(row["age_band"]) == band and t == i)
                  for i in range(len(labels))]
        print(f"      band {band} ({AGE_BAND_RANGES[band][0]}-{AGE_BAND_RANGES[band][1]}): "
              + "  ".join(f"{labels[i]}={c}" for i, c in enumerate(counts)))
    empty = [(b, i) for b in AGE_BANDS for i in range(len(labels))
             if not sum(1 for row, t in zip(records, replayed[domain])
                        if int(row["age_band"]) == b and t == i)]
    check(f"{domain:<11} every (age band, tier) cell is populated in the replay",
          not empty, f"empty cells {empty}" if empty else "10/10 cells")

print()
print("=" * 78)
print("4. Missing data — None, never a fabricated tier")
print("=" * 78)

complete = {
    "banking": {"savings_rate": 0.25, "emergency_fund_months": 6.0, "dti": 0.20,
                "healthy_debt_ratio": 0.70},
    "investment": {"investment_rate": 0.12, "portfolio_concentration": 0.30,
                   "risk_age_deviation": 0.05},
    "insurance": {"life_coverage_ratio": 9.0, "ci_coverage_ratio": 3.0,
                  "insurance_spend_ratio": 0.06},
}
for domain, features in complete.items():
    baseline = assign_domain_tier(domain, features, 3)
    check(f"{domain:<11} complete features -> a tier ({baseline!r})",
          baseline in models[domain]["tier_labels"], repr(baseline))
    optional = tier_assignment.OPTIONAL_FEATURES_BY_DOMAIN.get(domain, ())
    for name in features:
        if name in optional:
            # Available-case: scored on the remaining dimensions, not refused.
            got = assign_domain_tier(domain, {**features, name: None}, 3)
            check(f"{domain:<11} {name} = None -> a tier from the other dims "
                  f"({got!r})", got in models[domain]["tier_labels"], repr(got))
        else:
            expect(f"{domain:<11} {name} = None -> None",
                   assign_domain_tier(domain, {**features, name: None}, 3), None)
        expect(f"{domain:<11} {name} absent from the dict -> None",
               assign_domain_tier(domain,
                                  {k: v for k, v in features.items() if k != name}, 3),
               None)
        expect(f"{domain:<11} {name} = NaN -> None",
               assign_domain_tier(domain, {**features, name: float('nan')}, 3), None)
    expect(f"{domain:<11} empty features dict -> None",
           assign_domain_tier(domain, {}, 3), None)
    expect(f"{domain:<11} index form returns None on missing data too",
           assign_domain_tier_index(domain, {**features, list(features)[0]: None}, 3),
           None)
    # A zero is real data, not missing data — it must still be scored.
    zeroed = {name: 0.0 for name in features}
    check(f"{domain:<11} all-zero features are scored as real data, not skipped",
          assign_domain_tier(domain, zeroed, 3) in models[domain]["tier_labels"],
          repr(assign_domain_tier(domain, zeroed, 3)))

# Another domain's dict is the realistic way features go absent — it must not
# silently borrow whatever keys happen to overlap (there are none: the three
# feature sets are disjoint, asserted in scoring.py).
expect("passing Investment's features to banking -> None (no partial scoring)",
       assign_domain_tier("banking", complete["investment"], 3), None)

# Available-case distance — Banking's one optional feature only.
expect("only Banking has an optional feature, and only healthy_debt_ratio",
       {d: tuple(v) for d, v in tier_assignment.OPTIONAL_FEATURES_BY_DOMAIN.items()},
       {"banking": ("healthy_debt_ratio",)})
debt_free = {**complete["banking"], "healthy_debt_ratio": None}
check("banking: healthy_debt_ratio None -> tier from the other three",
      assign_domain_tier("banking", debt_free, 3) in models["banking"]["tier_labels"],
      repr(assign_domain_tier("banking", debt_free, 3)))
for mandatory in ("savings_rate", "dti", "emergency_fund_months"):
    expect(f"banking: healthy_debt_ratio None AND {mandatory} None -> None "
           "(mandatory)",
           assign_domain_tier("banking", {**debt_free, mandatory: None}, 3), None)
expect("banking: all four None -> None (nothing connected)",
       assign_domain_tier("banking", {n: None for n in complete["banking"]}, 3),
       None)
expect("banking: the optional feature ABSENT from the dict is still None",
       assign_domain_tier("banking", {k: v for k, v in complete["banking"].items()
                                      if k != "healthy_debt_ratio"}, 3), None)
expect("banking: the optional feature = NaN is still None",
       assign_domain_tier("banking", {**complete["banking"],
                                      "healthy_debt_ratio": float("nan")}, 3), None)


_BANKING_JSON = json.loads(
    (BACKEND_DIR / "ml" / "artifacts" / "domain_split" / "banking_model.json")
    .read_text(encoding="utf-8"))


def _manual_partial_tier(features, band):
    """Independent available-case argmin straight off the raw artifact JSON —
    not through _load_model, so it does not share the code under test. The
    rule is restated here rather than read from OPTIONAL_FEATURES_BY_DOMAIN:
    only healthy_debt_ratio may be dropped; any other None means no tier."""
    if any(features[n] is None for n in features if n != "healthy_debt_ratio"):
        return None
    art = _BANKING_JSON
    names = art["clustering_features"]
    center, scale = art["scaler"]["center_"], art["scaler"]["scale_"]
    band_art = art["bands"][str(band)]
    best = None
    for cid, coords in band_art["centroids_scaled"].items():
        d2 = sum(((features[n] - center[i]) / scale[i] - coords[i]) ** 2
                 for i, n in enumerate(names) if features[n] is not None)
        if best is None or d2 < best[0]:
            best = (d2, cid)
    return band_art["tier_map"][best[1]]["tier_label"]


mismatches = []
for band in AGE_BANDS:
    for sr in (0.02, 0.10, 0.22, 0.35, 0.55):
        for ef in (None, 0.5, 3.0, 9.0):
            for dti in (0.0, 0.15, 0.40):
                for hdr in (None, 0.0, 0.6, 1.0):
                    f = {"savings_rate": sr, "emergency_fund_months": ef,
                         "dti": dti, "healthy_debt_ratio": hdr}
                    got = assign_domain_tier("banking", f, band)
                    if got != _manual_partial_tier(f, band):
                        mismatches.append((band, f, got))
check("banking: available-case tier matches an independent rule + argmin "
      "(1200 cases)",
      not mismatches, f"{len(mismatches)} mismatches, first {mismatches[:1]}")
raises("a non-numeric feature value raises rather than being coerced",
       ValueError, assign_domain_tier, "banking",
       {**complete["banking"], "savings_rate": "0.25"}, 3)

print()
print("=" * 78)
print("5. Unknown domain, and age_band validation")
print("=" * 78)

for bad_domain in ("Banking", "bank", "insurances", "", "overall"):
    raises(f"domain={bad_domain!r} raises ValueError", ValueError,
           assign_domain_tier, bad_domain, complete["banking"], 3)
raises("features not a dict raises TypeError", TypeError,
       assign_domain_tier, "banking", [0.25, 6.0, 0.20, 0.70], 3)

for domain in DOMAINS:
    features = complete[domain]
    raises(f"{domain:<11} age_band=None raises", ValueError,
           assign_domain_tier, domain, features, None)
    raises(f"{domain:<11} age_band omitted entirely raises", ValueError,
           assign_domain_tier, domain, features)
    for bad_band in (0, 6, -1, 99):
        raises(f"{domain:<11} age_band={bad_band} raises", ValueError,
               assign_domain_tier, domain, features, bad_band)
    for bad_type in ("3", 3.0, True, [3]):
        raises(f"{domain:<11} age_band={bad_type!r} raises", ValueError,
               assign_domain_tier, domain, features, bad_type)
    # Validation must happen before the data is inspected, so a bad band is
    # never masked by missing features returning None.
    raises(f"{domain:<11} bad age_band raises even with missing features",
           ValueError, assign_domain_tier, domain, {}, 0)

print()
print("=" * 78)
print("6. Insurance — pooled model, but age_band is an INPUT, not ignored")
print("=" * 78)
print(f"  {tier_assignment.INSURANCE_NEEDS_AGE_TOO}")

check("insurance artifact is Approach B with a single pooled model",
      models["insurance"]["approach"] == "B" and list(models["insurance"]["bands"]) == [None],
      f"approach {models['insurance']['approach']}, "
      f"{len(models['insurance']['bands'])} model(s)")
check("insurance model takes 4 input dimensions (3 features + standardised age)",
      all(len(c) == 4 for c in models["insurance"]["centroids"][None])
      and models["insurance"]["age"] is not None,
      artifacts["insurance"]["model"]["input_dimensions"]
      if isinstance(artifacts["insurance"]["model"].get("input_dimensions"), (int, str))
      else str(artifacts["insurance"]["model"].get("input_dimensions")))

# The decisive test that age is not decorative: find a real user whose Insurance
# tier changes with the band, holding their features fixed.
sensitive = None
for row in records:
    features = {n: row[n] for n in models["insurance"]["features"]}
    seen = {b: assign_domain_tier("insurance", features, b) for b in AGE_BANDS}
    if len(set(seen.values())) > 1:
        sensitive = (row["user_id"], seen)
        break
check("a real user's Insurance tier changes with age_band — so ignoring or "
      "defaulting the band would produce wrong tiers",
      sensitive is not None,
      f"{sensitive[0]}: " + ", ".join(f"band {b}={t}" for b, t in sensitive[1].items())
      if sensitive else "no user found; age would be inert")

n_band_sensitive = sum(
    1 for row in records
    if len({assign_domain_tier("insurance", {n: row[n] for n in
                                             models["insurance"]["features"]}, b)
            for b in AGE_BANDS}) > 1)
print(f"  [note] {n_band_sensitive:,} of {len(records):,} users "
      f"({100 * n_band_sensitive / len(records):.1f}%) would change Insurance tier "
      f"under a wrong age band")
# Insurance must NOT vary by band for a user well inside one medoid's basin —
# a sanity check that the age dimension is one input among four, not dominant.
stable_users = len(records) - n_band_sensitive
check("the age dimension is one input among four, not the dominant one",
      stable_users > n_band_sensitive,
      f"{stable_users:,} users keep the same tier across all five bands")

print()
print("=" * 78)
print("7. The Banking cluster-0 age-band flip")
print("=" * 78)
print("  Artifact README: cluster 0 is 'Building' (weaker) in band 1 and 'Stable'")
print("  (healthy) in bands 2-5. Transcribed here independently of the artifact.")

EXPECTED_CLUSTER_0_LABEL = {1: "Building", 2: "Stable", 3: "Stable",
                            4: "Stable", 5: "Stable"}
banking = models["banking"]
banking_art = artifacts["banking"]

for band in AGE_BANDS:
    # A feature vector that lands in cluster 0 by construction: that band's own
    # cluster-0 centroid, in UNSCALED units — i.e. exactly what a user with
    # those raw values would submit.
    features = dict(banking_art["bands"][str(band)]["centroids_unscaled"]["0"])
    assert sorted(features) == sorted(banking["features"]), features
    label = assign_domain_tier("banking", features, band)
    index = assign_domain_tier_index("banking", features, band)
    landed = tier_assignment._nearest_cluster_id(
        banking, tier_assignment._feature_vector(banking, features, band), band)
    check(f"band {band}: the cluster-0 centroid's own values land in cluster 0",
          landed == 0, f"landed in cluster {landed}")
    expect(f"band {band}: cluster 0 -> {EXPECTED_CLUSTER_0_LABEL[band]!r}",
           label, EXPECTED_CLUSTER_0_LABEL[band])
    expect(f"band {band}: cluster 0 -> tier index "
           f"{0 if EXPECTED_CLUSTER_0_LABEL[band] == 'Stable' else 1}",
           index, 0 if EXPECTED_CLUSTER_0_LABEL[band] == "Stable" else 1)

check("cluster 0 genuinely means different tiers in different bands — the trap "
      "is real, not hypothetical",
      len(set(EXPECTED_CLUSTER_0_LABEL.values())) == 2,
      f"band 1 = {EXPECTED_CLUSTER_0_LABEL[1]}, bands 2-5 = {EXPECTED_CLUSTER_0_LABEL[2]}")

# The same claim on real users, not just on centroids: everyone the model puts
# in cluster 0 gets that band's label for cluster 0, and in band 1 that is the
# opposite of what a fixed "cluster 0 = healthy" assumption would give.
band1_cluster0 = [t for row, c, t in zip(records, clusters["banking"], replayed["banking"])
                  if int(row["age_band"]) == 1 and c == 0]
check("every band-1 user in cluster 0 is 'Building' (index 1), not 'Stable'",
      band1_cluster0 and set(band1_cluster0) == {1},
      f"{len(band1_cluster0):,} users, indices {sorted(set(band1_cluster0))}")
other_cluster0 = [t for row, c, t in zip(records, clusters["banking"], replayed["banking"])
                  if int(row["age_band"]) != 1 and c == 0]
check("every band-2..5 user in cluster 0 is 'Stable' (index 0)",
      other_cluster0 and set(other_cluster0) == {0},
      f"{len(other_cluster0):,} users, indices {sorted(set(other_cluster0))}")
n_would_be_wrong = len(band1_cluster0)
print(f"  [note] a hardcoded 'cluster 0 = healthiest' would mislabel "
      f"{n_would_be_wrong:,} band-1 users ({100 * n_would_be_wrong / len(records):.1f}% "
      f"of the population) — plausibly, and silently")

print()
print("=" * 78)
print("8. age_band_for_date_of_birth")
print("=" * 78)

TODAY = date(2026, 7, 29)
for age, want in [(20, 1), (25, 1), (29, 1), (30, 2), (39, 2), (40, 3), (49, 3),
                  (50, 4), (59, 4), (60, 5), (65, 5), (70, 5)]:
    expect(f"age {age} -> band {want}",
           age_band_for_date_of_birth(date(TODAY.year - age, 1, 1), today=TODAY), want)

# Birthday not yet reached this year: still the younger age.
expect("turns 30 tomorrow -> still band 1",
       age_band_for_date_of_birth(date(1996, 7, 30), today=TODAY), 1)
expect("turned 30 today -> band 2",
       age_band_for_date_of_birth(date(1996, 7, 29), today=TODAY), 2)
expect("turned 30 yesterday -> band 2",
       age_band_for_date_of_birth(date(1996, 7, 28), today=TODAY), 2)
# 29 Feb birthday in a non-leap year: the comparison must not raise.
expect("29 Feb birthday resolves without error",
       age_band_for_date_of_birth(date(1996, 2, 29), today=TODAY), 2)
# What Firestore actually hands back is a tz-aware datetime.
expect("tz-aware datetime (what the Firestore client returns) is accepted",
       age_band_for_date_of_birth(datetime(1996, 3, 10, tzinfo=timezone.utc),
                                  today=TODAY), 2)
expect("naive datetime is accepted",
       age_band_for_date_of_birth(datetime(1975, 3, 10), today=TODAY), 4)

raises("age 19 (under the modelled range) raises by default", ValueError,
       age_band_for_date_of_birth, date(TODAY.year - 19, 1, 1), TODAY)
raises("age 71 (over the modelled range) raises by default", ValueError,
       age_band_for_date_of_birth, date(TODAY.year - 71, 1, 1), TODAY)
expect("age 19 with clamp=True -> band 1, deliberately",
       age_band_for_date_of_birth(date(TODAY.year - 19, 1, 1), TODAY, clamp=True), 1)
expect("age 85 with clamp=True -> band 5, deliberately",
       age_band_for_date_of_birth(date(TODAY.year - 85, 1, 1), TODAY, clamp=True), 5)
raises("a future date of birth raises", ValueError,
       age_band_for_date_of_birth, date(2030, 1, 1), TODAY)
raises("a string date of birth raises TypeError", TypeError,
       age_band_for_date_of_birth, "1996-03-10", TODAY)

# End to end: DOB in, tier out, through the real function pair.
DOB = datetime(1981, 4, 2, tzinfo=timezone.utc)   # age 45 on 2026-07-29 -> band 3
band = age_band_for_date_of_birth(DOB, today=TODAY)
check(f"end to end: DOB {DOB.date()} -> band 3 -> a Banking tier",
      band == 3 and assign_domain_tier("banking", complete["banking"], band)
      in models["banking"]["tier_labels"],
      f"band {band} -> {assign_domain_tier('banking', complete['banking'], band)!r}")

print()
print("=" * 78)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 78)
sys.exit(1 if failed else 0)
