"""
benchmark_core.py — Shared clustering-benchmark methodology + production centroid export.

Extracted July 2026 from the now-archived comprehensive_benchmark.py (the original
5-algorithm cross-domain run, superseded by comprehensive_benchmark_11algo.py). This
module holds ONLY the pieces that are still load-bearing:

  * Shared metric / ranking / k-selection functions — imported by
    comprehensive_benchmark_11algo.py (the canonical 12-algorithm run) and
    ablation_binary_features.py. One definition, same methodology across both.
  * SYNTHETIC_IDEALS (per-band production "ideal user" centroid values) and the
    --export-centroids path that serializes them to the production artifact
    backend/ml/artifacts/synthetic_ideal_centroids.json.

The dead run logic from comprehensive_benchmark.py (run_approach_a/b, the centroid
comparison, per-user scoring, plotting, main()) was NOT carried over — the full original
script is preserved at
results/archive/comprehensive_10feature_5algo/comprehensive_benchmark.py.

Export the production centroid artifact (deliberate; OVERWRITES the artifact — the
artifact re-export is otherwise a separate later task, so do not run this casually):
    python MLAI/clustering/benchmark_core.py --export-centroids
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CENTROID_ARTIFACT_PATH = os.path.join(
    REPO_ROOT, "backend/ml/artifacts/synthetic_ideal_centroids.json")

# The 10-feature set the COMMITTED production artifacts (fitted_scaler.pkl and
# synthetic_ideal_centroids.json) are keyed on. IMPORTANT: this is NOT the current
# Layer 2 clustering set. The binary-feature ablation (July 2026) revised Layer 2 to 11
# CONTINUOUS features — see scoring.py LAYER2_CLUSTERING_FEATURES: hospitalisation_coverage
# was dropped to Layer 1, and healthy_debt_ratio + insurance_spend_ratio were added.
# This 10-feature list and the SYNTHETIC_IDEALS below are retained ONLY so
# --export-centroids keeps reproducing the artifact in its current pre-redesign shape;
# the artifact re-export to the 11-feature design is a separate, later task.
# fit_production_scaler.py fits the scaler on this same 10-feature set for the same reason.
CLUSTERING_FEATURES = [
    "savings_rate", "emergency_fund_months", "dti", "net_worth_to_income",
    "investment_rate", "portfolio_concentration", "risk_age_deviation",
    "life_coverage_ratio", "ci_coverage_ratio", "hospitalisation_coverage",
]

BAND_LABELS = {1: "20-29", 2: "30-39", 3: "40-49", 4: "50-59", 5: "60-70"}

# ======================================================================================
# ⚠️ DEPRECATED AS A SCORING MECHANISM (July 2026) — RETAINED, NOT DELETED
# ======================================================================================
# SYNTHETIC_IDEALS below was built for the OLD architecture: ONE hand-specified "ideal
# user" centroid per age band across all 10 features combined, with a user's score derived
# from their scaled Euclidean distance to it and a tier assigned by absolute score band
# (TIER_THRESHOLDS). Every value in it is a manually chosen literal — a normative statement
# of what a healthy user *should* look like — not a quantity estimated from data.
#
# That architecture no longer applies. Layer 2 now runs THREE independent domain
# clusterings, each with REAL, data-derived cluster centroids and a tier assigned by
# nearest cluster (Insurance additionally merges two of its three native clusters). The
# replacement production artifacts are:
#
#     backend/ml/artifacts/domain_split/{banking,investment,insurance}_model.json
#     backend/ml/artifacts/domain_split/{banking,investment,insurance}_scaler.pkl
#     backend/ml/artifacts/domain_split/insurance_age_scaler.pkl
#     backend/ml/artifacts/domain_split/manifest.json
#
#   produced by MLAI/clustering/build_production_artifacts.py, per the brief's
#   "FINAL ALGORITHM/TIER DECISIONS — LOCKED" (Banking A|kmeans|2, Investment
#   A|fuzzy_cmeans|2, Insurance B|kmedoids|3 merged to 2 tiers).
#
# What is superseded, specifically:
#   * SYNTHETIC_IDEALS as a scoring reference point   — superseded. Nothing measures
#     distance to these vectors any more.
#   * CLUSTERING_FEATURES (the 10-feature list)       — superseded. The three domains use
#     4 / 3 / 3 features and net_worth_to_income is excluded from clustering entirely.
#   * TIER_THRESHOLDS (0-39 / 40-69 / 70-100)         — superseded AS A MECHANISM: tiers
#     come from cluster membership now, not from score bands. The *rule* it encoded —
#     tiers must be absolute, never a relative ranking within the sample — still binds and
#     is honoured by the new design (cluster membership is fixed by fitted centroids, not
#     by the user's percentile).
#   * export_ideal_centroids() / --export-centroids   — deprecated; now requires an
#     explicit --force flag so it cannot regenerate a dead architecture's artifact by
#     accident. The committed backend/ml/artifacts/synthetic_ideal_centroids.json is
#     LEFT AS-IS on purpose: it is the historical record of what the old architecture
#     actually shipped, and re-exporting would silently replace that record with fresher
#     numbers for a mechanism nothing uses.
#
# ✅ WHAT IS *NOT* SUPERSEDED — one part of SYNTHETIC_IDEALS is still load-bearing.
# Three of its values are read as the per-age-band thresholds of the OUT-OF-MODEL
# validation benchmarks that the domain-split selection was judged against
# (domain_split_clustering.py, NW_IDEAL_BY_BAND / LIFE_IDEAL_BY_BAND / CONC_IDEAL_BY_BAND):
#
#     net_worth_to_income      1/3/6/8/10   -> Banking's benchmark ladder. LIVE. This is
#                                             Banking's single most binding benchmark
#                                             (lowest-headroom variable in 18 of 28
#                                             survivors), and it is strictly out-of-model
#                                             (excluded from all three feature sets).
#     life_coverage_ratio      9/9/9/9/5    -> Insurance's benchmark. LIVE. The band-5
#                                             value of 5.0 is why the domain-split run
#                                             uses a band-appropriate threshold instead of
#                                             the flat 9x that over-flagged band 5.
#     portfolio_concentration  0.20 -> 0.08 -> Investment's ORIGINAL benchmark. SUPERSEDED
#                                             (see the divergence note below), but still
#                                             read by domain_split_clustering.py's default
#                                             benchmark table.
#
# In that role these are threshold constants, not a scoring centroid, so they survive the
# architecture change. This block therefore must NOT be deleted — deleting it would break
# the ability to reproduce the locked selection evidence.
#
# ⚠️ KNOWN, DELIBERATE DIVERGENCE — portfolio_concentration (brief Section 5 /
# FINAL_TIER_DECISIONS.md §8 item 1, resolved here as DOCUMENTATION, not as a value edit):
# the age-tapering ladder below (0.20/0.17/0.13/0.10/0.08) was diagnosed as a category
# error — it is private-bank advice for HNW clients, not a description of real retail
# investor behaviour, which Goetzmann & Kumar (2008, Review of Finance, >40,000 real
# brokerage accounts) shows is systematically under-diversified and more so when younger.
# The AUTHORITATIVE Investment benchmark is now the flat, structurally-reasoned
#     investment_recalibration.CONC_THRESHOLD = 0.33
# (a largest holding of at most a third implies at least three meaningful positions;
# verified insensitive across 0.25-0.40). The ladder below is NOT edited to match, on
# purpose: it belongs to the deprecated centroid artifact and changing it would rewrite
# the historical record of what that artifact contained without changing any live result.
# Any code needing the Investment concentration benchmark must import CONC_THRESHOLD, not
# read this table. (risk_age_diagnostic.install_bench already does exactly that, which is
# why the locked Investment winner is unaffected by this divergence.)
# ======================================================================================
#
# Synthetic ideal centroids — the per-band "ideal user" vectors the OLD production score
# measured distance to. Citation coverage is MIXED, not uniform. Revised July 2026
# after a manual citation-verification session across MAS/MoneySense/SingStat/DBS/
# MOF/OCBC sources; per value:
#
#   Cleanly traceable to a named SG source (3):
#   - emergency_fund_months 6.0    MoneySense 3-6 month emergency fund benchmark,
#                                  ALL BANDS. Band 5 corrected July 2026 from an
#                                  uncited 12.0 that the official guidance directly
#                                  contradicts: the MoneySense/MAS "Golden Years"
#                                  (65+) guide's own worked example for a 67-year-old
#                                  uses 6 months ("more than the recommended 6
#                                  months' worth of expenses ($7,200)"), and the
#                                  Fresh Entrant and Golden Years guides state the
#                                  same 3-6 month range — the benchmark does not
#                                  vary by age.
#   - life_coverage_ratio 9.0      LIA Protection Gap Study 2022 recommendation (9-10x)
#   - ci_coverage_ratio 3.9        LIA PGS 2022 recommendation. Precision caveat:
#                                  MoneySense's public Guide rounds this to 4x; the
#                                  3.9x decimal traces to the underlying LIA study,
#                                  which corroborates but was not independently
#                                  re-verified at this exact decimal precision.
#
#   Anchored to an official minimum, upper value unsourced (1):
#   - investment_rate 0.15         the official MoneySense/MAS Basic Financial
#                                  Planning Guide states "invest at least 10% of
#                                  income" — a MINIMUM, confirmed across three
#                                  official documents; NO official upper bound
#                                  exists. 0.15 is an independently-chosen round
#                                  number above that floor — it coincidentally
#                                  matches the same Guide's unrelated 15% INSURANCE
#                                  protection spending cap, and does not derive
#                                  from it (an earlier "10-15% guidance" citation
#                                  here conflated the two).
#
#   Ideal by construction, no external source needed (2):
#   - risk_age_deviation 0.0       zero deviation from the age-appropriate equity
#                                  target is the definition of ideal. The target
#                                  itself is the "100 minus age" rule — a widely-
#                                  used heuristic with no single traceable origin
#                                  (1980s-90s planning practice/media), no academic
#                                  or institutional credit, and no SG-specific
#                                  basis; generator EQUITY_TARGETS and scoring.py's
#                                  RISK_AGE_EQUITY_BASE both standardized on 100
#                                  (July 2026; scoring.py previously used 110).
#   - hospitalisation_coverage 1.0 binary flag; ideal = has coverage (Integrated
#                                  Shield Plan)
#
#   Reasoned estimates, disclosed sourcing state (4):
#   - savings_rate 0.30            sits above the OCBC FWI 2024 ~25% national
#                                  average, but no source names 30% as "ideal"
#   - dti 0.15 (0.10/0.05 in bands 4/5)  conservative relative to DBS's recommended
#                                  ideal of <=35% ("7 financial ratios to gauge your
#                                  financial health": "Ideally, a ratio of 35% or
#                                  below means you have ample income to fulfil your
#                                  monthly debt repayments") and well below MAS's
#                                  55% TDSR regulatory maximum (confirmed at
#                                  mas.gov.sg/regulation/explainers/tdsr-for-
#                                  property-loans). The specific 0.15/0.10/0.05
#                                  values remain reasoned choices. (An earlier
#                                  "mortgage loans ~82% of annual PDI" claim was
#                                  removed — confirmed absent from MAS FSR 2025.)
#   - net_worth_to_income 1/3/6/8/10  revised July 2026 (was 1/4/8/12/15, pegged to
#                                  population means now understood to be too high).
#                                  Based on a widely-cited personal-finance heuristic
#                                  (~1x income by 30, rising with age) with NO
#                                  official SG source — confirmed absent after
#                                  checking MAS FSR, SingStat's Household Sector
#                                  Balance Sheet Dashboard, and MOF's Feb 2026
#                                  Occasional Paper (all quintile-only, never by
#                                  age). The ladder sits at the upper-middle of the
#                                  revised population marginals (aspirational, not
#                                  extreme).
#   - portfolio_concentration 0.20/0.17/0.13/0.10/0.08  revised July 2026 (was 0.20
#                                  flat — no life-cycle logic, inconsistent with the
#                                  age-tapering principle used elsewhere, e.g.
#                                  risk_age_deviation). No SG-specific source exists
#                                  (confirmed absent after checking CPF Board/CPFIS
#                                  materials — product/performance data only).
#                                  Grounded in general international wealth-
#                                  management guidance: J.P. Morgan Private Bank
#                                  cites 10-20% as the accepted range for a properly
#                                  diversified portfolio; Fidelity, Calamos Wealth
#                                  Management, and HDFC each suggest a tighter 5-10%
#                                  ideal for lower-risk-tolerance investors. The
#                                  taper (permissive young, tighter old) mirrors the
#                                  life-cycle risk-reduction principle already in
#                                  this feature set.
#
#   The band-4/5 adjustments (dti 0.10/0.05, investment_rate 0.12/0.08, band 5's
#   savings_rate 0.25 and life_coverage_ratio 5.0) are reasoned life-stage estimates
#   with no specific SG source.
#
#   NOTE (July 2026): these values are AHEAD of the committed artifact —
#   backend/ml/artifacts/synthetic_ideal_centroids.json still holds the pre-revision
#   ideals and is deliberately untouched until the post-revision benchmark task
#   re-runs and re-exports (--export-centroids). Do not export as a side effect.
# --------------------------------------------------------------------------------------
SYNTHETIC_IDEALS = {
    1: {"savings_rate": 0.30, "emergency_fund_months": 6.0, "dti": 0.15,
        "net_worth_to_income": 1.0, "investment_rate": 0.15, "portfolio_concentration": 0.20,
        "risk_age_deviation": 0.0, "life_coverage_ratio": 9.0, "ci_coverage_ratio": 3.9,
        "hospitalisation_coverage": 1.0},
    2: {"savings_rate": 0.30, "emergency_fund_months": 6.0, "dti": 0.15,
        "net_worth_to_income": 3.0, "investment_rate": 0.15, "portfolio_concentration": 0.17,
        "risk_age_deviation": 0.0, "life_coverage_ratio": 9.0, "ci_coverage_ratio": 3.9,
        "hospitalisation_coverage": 1.0},
    3: {"savings_rate": 0.30, "emergency_fund_months": 6.0, "dti": 0.15,
        "net_worth_to_income": 6.0, "investment_rate": 0.15, "portfolio_concentration": 0.13,
        "risk_age_deviation": 0.0, "life_coverage_ratio": 9.0, "ci_coverage_ratio": 3.9,
        "hospitalisation_coverage": 1.0},
    4: {"savings_rate": 0.30, "emergency_fund_months": 6.0, "dti": 0.10,
        "net_worth_to_income": 8.0, "investment_rate": 0.12, "portfolio_concentration": 0.10,
        "risk_age_deviation": 0.0, "life_coverage_ratio": 9.0, "ci_coverage_ratio": 3.9,
        "hospitalisation_coverage": 1.0},
    5: {"savings_rate": 0.25, "emergency_fund_months": 6.0, "dti": 0.05,
        "net_worth_to_income": 10.0, "investment_rate": 0.08, "portfolio_concentration": 0.08,
        "risk_age_deviation": 0.0, "life_coverage_ratio": 5.0, "ci_coverage_ratio": 3.9,
        "hospitalisation_coverage": 1.0},
}

# Absolute tier boundaries for the DEPRECATED distance-to-ideal score (see the banner
# above). Superseded as a mechanism — the domain-split architecture assigns tiers by
# cluster membership, not by score band — but the constraint it encoded (absolute, never
# relative ranking within the sample) still binds project-wide.
TIER_THRESHOLDS = {"At Risk": [0, 39], "Building": [40, 69], "Stable": [70, 100]}


# ======================================================================================
# Shared metric / ranking / k-selection methodology
# (imported by comprehensive_benchmark_11algo.py and ablation_binary_features.py)
# ======================================================================================
def compute_metrics(X, labels):
    if len(np.unique(labels)) < 2:
        return {"silhouette_score": 0.0, "davies_bouldin_index": 99.0,
                "calinski_harabasz_index": 0.0}
    return {
        "silhouette_score": round(float(silhouette_score(X, labels)), 4),
        "davies_bouldin_index": round(float(davies_bouldin_score(X, labels)), 4),
        "calinski_harabasz_index": round(float(calinski_harabasz_score(X, labels)), 4),
    }


def cluster_sizes_and_balance(labels):
    _, counts = np.unique(labels, return_counts=True)
    return counts.tolist(), round(float(min(counts) / max(counts)), 4)


def _rank(algos, values, higher_better):
    """Rank algos 1..n on `values`; rank 1 is best. Ties resolve by input order."""
    order = sorted(algos, key=lambda a: values[a], reverse=higher_better)
    return {a: r + 1 for r, a in enumerate(order)}


def composite_rank(results_dict):
    algos = list(results_dict.keys())
    sil = {a: results_dict[a]["silhouette_score"] for a in algos}
    db = {a: results_dict[a]["davies_bouldin_index"] for a in algos}
    ch = {a: results_dict[a]["calinski_harabasz_index"] for a in algos}
    rank_sil = _rank(algos, sil, True)
    rank_db = _rank(algos, db, False)
    rank_ch = _rank(algos, ch, True)
    return {a: rank_sil[a] + rank_db[a] + rank_ch[a] for a in algos}


def selection_score(metrics_by_k):
    ks = list(metrics_by_k.keys())
    sil = {k: metrics_by_k[k]["silhouette_score"] for k in ks}
    db = {k: metrics_by_k[k]["davies_bouldin_index"] for k in ks}
    ch = {k: metrics_by_k[k]["calinski_harabasz_index"] for k in ks}
    bal = {k: metrics_by_k[k]["size_balance"] for k in ks}

    def norm(d, higher_better=True):
        mn, mx = min(d.values()), max(d.values())
        if mx == mn:
            return {k: 0.5 for k in d}
        if higher_better:
            return {k: (d[k] - mn) / (mx - mn) for k in d}
        return {k: 1 - (d[k] - mn) / (mx - mn) for k in d}

    n_sil, n_db, n_ch, n_bal = norm(sil, True), norm(db, False), norm(ch, True), norm(bal, True)
    return {k: round((n_sil[k] + n_db[k] + n_ch[k] + n_bal[k]) / 4, 4) for k in ks}


# ======================================================================================
# Production centroid artifact export
# ======================================================================================
def export_ideal_centroids(path: str | None = None) -> str:
    """⚠️ DEPRECATED — serializes SYNTHETIC_IDEALS to the OLD single-model production
    artifact. See the deprecation banner above SYNTHETIC_IDEALS.

    `path` falls back to CENTROID_ARTIFACT_PATH, resolved HERE rather than bound as an
    argument default. Binding it in the signature captured the value at import time, so
    reassigning the module global had no effect and the call still wrote to the real
    production artifact — which is exactly how this function overwrote
    synthetic_ideal_centroids.json during the July 2026 reproducibility audit. Redirecting
    via the module global now works; passing path= explicitly remains the clearest option.

    Nothing consumes the resulting artifact any more: the domain-split architecture scores
    against three fitted models in backend/ml/artifacts/domain_split/, produced by
    MLAI/clustering/build_production_artifacts.py. The committed
    synthetic_ideal_centroids.json is retained as the historical record of what the old
    architecture shipped, and running this would overwrite that record with newer numbers
    for a mechanism that is no longer used — so the CLI now requires an explicit --force:

        python MLAI/clustering/benchmark_core.py --export-centroids --force

    Kept (not deleted) because the artifact would otherwise have no producing script at
    all, and because the report cites this as the regeneration path for the historical
    artifact. (Extracted from the archived comprehensive_benchmark.py July 2026.)
    """
    path = CENTROID_ARTIFACT_PATH if path is None else path
    artifact = {
        "STATUS": (
            "DEPRECATED (July 2026). This artifact belongs to the superseded "
            "single-combined-model architecture: one hand-specified ideal centroid per age "
            "band across 10 features, scored by distance and tiered by absolute score band. "
            "Layer 2 now runs three independent domain clusterings with real, data-derived "
            "centroids and nearest-cluster tier assignment. The live production artifact "
            "set is backend/ml/artifacts/domain_split/ (manifest.json indexes it). This "
            "file is retained for historical and report reference ONLY and must not be used "
            "for scoring. NOTE: the numbers in the committed copy of this file predate the "
            "July 2026 citation revisions (net-worth ladder, portfolio-concentration "
            "gradient, emergency-fund band 5) that SYNTHETIC_IDEALS itself now holds; that "
            "staleness is deliberate and is part of what makes it a historical record. One "
            "part of SYNTHETIC_IDEALS is NOT deprecated - its net_worth_to_income and "
            "life_coverage_ratio ladders are still read as per-band out-of-model validation "
            "benchmark thresholds by the domain-split selection; its "
            "portfolio_concentration ladder was superseded there by a flat <=0.33."
        ),
        "description": (
            "Production artifacts for the Unify Financial Health Score. Per-age-band "
            "synthetic ideal centroids (values revised July 2026 after a manual "
            "citation-verification session). Citation state is MIXED, not uniform - per "
            "feature: "
            "CITED to a named source - emergency_fund_months 6.0 all bands (MoneySense "
            "3-6 month fund; band 5 corrected from an uncited 12.0 that the official "
            "Golden Years guide's own 6-month worked example contradicts), "
            "life_coverage_ratio (LIA PGS 2022, 9-10x income), ci_coverage_ratio (LIA "
            "PGS 2022 ~3.9x; MoneySense's public Guide rounds to 4x - the 3.9 decimal "
            "was not independently re-verified), savings_rate 0.30 (above MoneySense's "
            "20% minimum recommendation, aligned with SingStat's observed ~31-37% "
            "national personal savings rate). "
            "ANCHORED TO AN OFFICIAL MINIMUM, upper value unsourced - investment_rate "
            "0.15 (MoneySense/MAS Basic Financial Planning Guide says 'invest at least "
            "10% of income', a floor with NO official upper bound; 0.15 is an "
            "independently-chosen round number that coincidentally matches the Guide's "
            "unrelated 15% insurance-spend cap and does not derive from it). "
            "REASONED, not cited - dti 0.15/0.10/0.05 (conservative vs DBS's "
            "recommended ideal of <=35% and MAS's 55% TDSR regulatory ceiling; the "
            "specific values are not cited figures), net_worth_to_income ladder "
            "1/3/6/8/10 (widely-cited ~1x-by-30 age heuristic; no official SG "
            "wealth-by-age table exists - MAS FSR, SingStat dashboard and MOF's 2026 "
            "wealth paper are all quintile-only; revised down from 1/4/8/12/15), "
            "portfolio_concentration 0.20/0.17/0.13/0.10/0.08 (no SG source exists, "
            "CPF/CPFIS checked; age taper grounded in international wealth-management "
            "guidance - J.P. Morgan Private Bank 10-20% diversified range; "
            "Fidelity/Calamos/HDFC 5-10% ideal; revised from a flat 0.20). "
            "IDEAL BY CONSTRUCTION, for distinct reasons - risk_age_deviation 0.0 is "
            "the structural zero-point of a DERIVED quantity (equity_pct minus the "
            "'100 minus age' target - itself an uncredited 1980s-90s planning "
            "heuristic with no SG basis, standardized across generator and scoring "
            "July 2026); hospitalisation_coverage 1.0 is the 'yes' endpoint of a "
            "directly-generated binary flag, whose generation probability is anchored "
            "to ~71% Integrated Shield Plan prevalence, not a derivation artifact. "
            "HISTORICAL decisions this artifact was built for, now SUPERSEDED: K-Means++, "
            "k=3, age-stratified (Approach A), RobustScaler (fitted_scaler.pkl), synthetic "
            "centroid. The current locked decisions are per-domain - see STATUS above and "
            "backend/ml/artifacts/domain_split/manifest.json."
        ),
        "source": (
            "Exported from SYNTHETIC_IDEALS in MLAI/clustering/benchmark_core.py"
        ),
        "clustering_features": CLUSTERING_FEATURES,
        "age_band_labels": {str(b): label for b, label in BAND_LABELS.items()},
        "synthetic_ideal_centroids": {
            str(b): ideals for b, ideals in SYNTHETIC_IDEALS.items()
        },
        "tier_thresholds": TIER_THRESHOLDS,
    }
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Exported synthetic ideal centroids -> {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="benchmark_core — shared methodology + DEPRECATED centroid export")
    parser.add_argument(
        "--export-centroids", action="store_true",
        help="DEPRECATED. Serialize SYNTHETIC_IDEALS to "
             "backend/ml/artifacts/synthetic_ideal_centroids.json (OVERWRITES it). The "
             "artifact belongs to the superseded single-model architecture; requires "
             "--force.")
    parser.add_argument(
        "--force", action="store_true",
        help="Acknowledge the deprecation and overwrite the historical artifact anyway.")
    args = parser.parse_args()
    if args.export_centroids and not args.force:
        raise SystemExit(
            "REFUSED: --export-centroids is DEPRECATED (July 2026).\n"
            "  backend/ml/artifacts/synthetic_ideal_centroids.json belongs to the "
            "superseded single-combined-model\n"
            "  architecture and is retained as the historical record of what that "
            "architecture shipped. Nothing\n"
            "  reads it. Overwriting it would replace that record with newer numbers for a "
            "dead mechanism.\n\n"
            "  The live production artifacts are backend/ml/artifacts/domain_split/, built "
            "by:\n"
            "      backend/venv/Scripts/python.exe "
            "MLAI/clustering/build_production_artifacts.py\n\n"
            "  If you genuinely intend to regenerate the historical artifact, re-run with "
            "--force.")
    if args.export_centroids:
        print("WARNING: exporting a DEPRECATED artifact (--force given).")
        export_ideal_centroids()
    else:
        parser.print_help()
