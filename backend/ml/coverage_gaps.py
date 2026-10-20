"""
coverage_gaps.py — the Insurance screen's Coverage Gaps, from the SAME features
that decide the Insurance tier.

WHY THIS MODULE EXISTS AT ALL
The Insurance tier is a binary verdict ("Covered" / "Not Covered"). A gap
message needs the continuous value behind it — a user 2.9x covered against
LIA's 9x and a user 8.9x covered are both "Not Covered", and telling them the
same thing is useless. compute_insurance_features() already produces those two
continuous ratios; nothing published them anywhere a client could read.

WHAT IT ADDS, AND WHAT IT DELIBERATELY DOES NOT
It adds ONE derived, read-only output: each coverage ratio next to the
benchmark it is measured against, plus the arithmetic between them. It changes
no feature, no tier, no threshold and no score — score_pipeline calls it after
every scoring decision has already been made, and nothing downstream of the
score reads its result.

EVERY NUMBER HERE COMES OUT OF A COMMITTED ARTIFACT
No benchmark multiple is written down in this file. The life multiple is the
per-age-band ``life_coverage_ratio`` from
artifacts/synthetic_ideal_centroids.json (9x for bands 1-4, 5x for band 5) and
the CI multiple is its ``ci_coverage_ratio`` (3.9x, every band) — the values
the brief cites to LIA Protection Gap Study 2022, and the same ones the
synthetic training population was generated against.

The production Insurance artifact declares the same two benchmarks in its own
``out_of_model_benchmarks`` block, with the citations attached. Both are read,
and _benchmarks() asserts they agree: if SYNTHETIC_IDEALS is ever revised, the
two artifacts diverge and this raises at load time rather than quietly showing
a user a gap measured against a superseded benchmark.

⚠️ THE BAND-5 DROP IS REAL, NOT A TYPO
Band 5 (60-70) carries a 5x life benchmark, not 9x — dependants are typically
grown and the income being replaced has a shorter run left. A 62-year-old must
not be told they are 4x short of a benchmark that was never theirs. This is
exactly why the benchmark is resolved per band here and cached per user,
instead of being a single constant a client could hardcode.
"""

import json
import numbers
from functools import lru_cache
from pathlib import Path

_ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# ---------------------------------------------------------------------------
# The benchmarked features
# ---------------------------------------------------------------------------
# The two continuous insurance features that have a benchmark to be short of.
# hospitalisation_coverage and income_protection_coverage are binary and are
# already published as Layer 1 checklist entries with their own copy — they are
# deliberately not duplicated here.
LIFE = "life_coverage_ratio"
CRITICAL_ILLNESS = "ci_coverage_ratio"

# The cached map's keys. Short, because they name a kind of cover rather than a
# feature; the feature name each one is computed from is the value.
GAP_KEY_BY_FEATURE = {LIFE: "life", CRITICAL_ILLNESS: "critical_illness"}


# ---------------------------------------------------------------------------
# Benchmark loading (artifact cross-check)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _benchmarks(artifact_dir: str | None = None) -> dict:
    """
    {age_band: {feature: benchmark multiple}}, read from the artifacts and
    cross-checked between them. Parsed once per process.

    Raises AssertionError when the two artifacts disagree — see the module
    docstring.
    """
    directory = Path(artifact_dir) if artifact_dir else _ARTIFACT_DIR

    centroids = json.loads(
        (directory / "synthetic_ideal_centroids.json").read_text(encoding="utf-8")
    )["synthetic_ideal_centroids"]
    by_band = {
        int(band): {feature: float(values[feature])
                    for feature in GAP_KEY_BY_FEATURE}
        for band, values in centroids.items()
    }

    insurance = json.loads(
        (directory / "domain_split" / "insurance_model.json")
        .read_text(encoding="utf-8"))
    declared = {entry["variable"]: entry
                for entry in insurance["out_of_model_benchmarks"]}

    for feature in GAP_KEY_BY_FEATURE:
        entry = declared.get(feature)
        if entry is None:
            raise AssertionError(
                f"insurance_model.json declares no out-of-model benchmark for "
                f"{feature!r} — this module has no other source for it.")
        if entry["operator"] != ">=":
            raise AssertionError(
                f"{feature}: the shipped benchmark operator is "
                f"{entry['operator']!r}, not '>=' — 'more cover is better' is "
                "the only direction the gap arithmetic below is valid for.")

        per_band = [by_band[band][feature] for band in sorted(by_band)]
        threshold = entry["threshold"]
        if isinstance(threshold, numbers.Real):
            # A single number: every band's ideal must equal it.
            if any(value != float(threshold) for value in per_band):
                raise AssertionError(
                    f"{feature}: insurance_model.json declares a flat "
                    f"{threshold} benchmark, but synthetic_ideal_centroids.json "
                    f"has per-band values {per_band}. The two artifacts "
                    "disagree; do not measure a user against either until "
                    "they are reconciled.")
        else:
            # A prose threshold naming the per-band ladder, e.g.
            # "per-band life-coverage ideal from SYNTHETIC_IDEALS (9/9/9/9/5)".
            ladder = "/".join(_compact(value) for value in per_band)
            if ladder not in str(threshold):
                raise AssertionError(
                    f"{feature}: insurance_model.json's benchmark reads "
                    f"{threshold!r}, which does not contain the per-band ladder "
                    f"{ladder} from synthetic_ideal_centroids.json. The two "
                    "artifacts disagree; do not measure a user against either "
                    "until they are reconciled.")

    return by_band


def _compact(value: float) -> str:
    """9.0 -> '9', 3.9 -> '3.9' — the way the artifacts write their ladders."""
    return str(int(value)) if float(value).is_integer() else str(value)


def clear_benchmark_cache() -> None:
    """Drops the parsed-artifact cache (tests that point at a temp dir)."""
    _benchmarks.cache_clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def coverage_benchmarks(age_band: int, artifact_dir: str | None = None) -> dict:
    """
    The life and CI coverage multiples that apply to one age band.

    Args:
        age_band: 1-5, from tier_assignment.age_band_for_date_of_birth().

    Returns:
        {"life_coverage_ratio": 9.0, "ci_coverage_ratio": 3.9} — multiples of
        ANNUAL income, matching the ratios compute_insurance_features() returns.

    Raises:
        ValueError: on a band the artifacts do not cover.
    """
    by_band = _benchmarks(artifact_dir)
    if age_band not in by_band:
        raise ValueError(
            f"age_band {age_band!r} is not one of {sorted(by_band)} — the "
            "benchmark ladder is defined per fitted age band only.")
    return dict(by_band[age_band])


def compute_coverage_gaps(
    insurance_features: dict,
    annual_income_sgd: float,
    age_band: int,
    artifact_dir: str | None = None,
) -> dict:
    """
    Each continuous coverage ratio next to the benchmark it is measured
    against, for the Insurance screen's Coverage Gaps section.

    Pure function, no I/O beyond the two committed artifacts. Derived output
    only: it reads compute_insurance_features()'s result and never recomputes a
    ratio, so a gap message and the Insurance tier can never disagree about how
    much cover a user has.

    Args:
        insurance_features: compute_insurance_features()'s dict.
        annual_income_sgd:  The SAME denominator those ratios were computed
                            against — passing a different one would produce SGD
                            figures that contradict the ratios beside them.
        age_band:           1-5. Selects the life benchmark (see the band-5
                            note in the module docstring).

    Returns:
        {"life": {...}, "critical_illness": {...}}, each entry:

            feature          which compute_insurance_features() key this is
            ratio            multiples of annual income actually covered
            benchmark_ratio  the benchmark multiple for this age band
            meets_benchmark  ratio >= benchmark_ratio
            covered_sgd      ratio * annual income
            recommended_sgd  benchmark_ratio * annual income
            shortfall_sgd    the difference, floored at 0.0 when the benchmark
                             is met — a user who is over-covered has no
                             "negative gap", and reporting one would invite a
                             recommendation this project does not make

        The SGD figures are computed here rather than client-side so that the
        multiple and the dollar amount on the same card can never be derived
        from two different incomes.

    Raises:
        ValueError: on a non-positive income (same contract as
        compute_insurance_features), an unknown age band, or a ratio that is
        absent or not a number.
    """
    if annual_income_sgd <= 0:
        raise ValueError("annual_income_sgd must be positive")

    benchmarks = coverage_benchmarks(age_band, artifact_dir)
    gaps = {}
    for feature, key in GAP_KEY_BY_FEATURE.items():
        ratio = insurance_features.get(feature)
        if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real):
            # compute_insurance_features() always returns both as floats, so
            # this means the wrong dict arrived — loud, not a fabricated 0.0.
            raise ValueError(
                f"insurance_features[{feature!r}] = {ratio!r} is not a number. "
                "Pass compute_insurance_features()'s own output.")
        ratio = float(ratio)
        benchmark = benchmarks[feature]
        covered = ratio * annual_income_sgd
        recommended = benchmark * annual_income_sgd
        gaps[key] = {
            "feature": feature,
            "ratio": ratio,
            "benchmark_ratio": benchmark,
            "meets_benchmark": ratio >= benchmark,
            "covered_sgd": covered,
            "recommended_sgd": recommended,
            "shortfall_sgd": max(0.0, recommended - covered),
        }
    return gaps


if __name__ == "__main__":
    print("Coverage benchmarks, per age band (LIA PGS 2022, via the "
          "committed artifacts)\n")
    for band in sorted(_benchmarks()):
        values = coverage_benchmarks(band)
        print(f"  band {band}:  life {_compact(values[LIFE])}x   "
              f"critical illness {_compact(values[CRITICAL_ILLNESS])}x")

    print("\nOne worked example — S$60,000 a year, band 2, S$120,000 of life "
          "cover, no CI cover:")
    FEATURES = {LIFE: 2.0, CRITICAL_ILLNESS: 0.0}
    for key, gap in compute_coverage_gaps(FEATURES, 60000.0, 2).items():
        state = "meets" if gap["meets_benchmark"] else "short by"
        amount = "" if gap["meets_benchmark"] else f" S${gap['shortfall_sgd']:,.0f}"
        print(f"  {key:<17} {gap['ratio']:.2f}x of "
              f"{_compact(gap['benchmark_ratio'])}x — {state}{amount}")
