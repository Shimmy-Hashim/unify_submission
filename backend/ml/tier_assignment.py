"""
tier_assignment.py — Layer 2 inference: a user's raw feature values -> a domain tier.

Step 2 of the scoring pipeline — what score_pipeline.py calls between
scoring.py's feature assembly and layer3.py's cross-domain rules, and the path
verify_production_artifacts.py proves is replayable from the artifacts alone:

    1. assemble the user's features (scoring.compute_{banking,investment,
       insurance}_features)
    2. per domain: scale -> nearest centroid in the user's age band -> tier
       label via that band's tier_map          <- THIS MODULE
    3. detect_combination_flags(...)           <- layer3.py
    4. cache onto users/{uid}/financialProfile/profile

Scope is deliberately step 2 only. No overall score, no Layer 1 checklist, no
FastAPI endpoint, no Firestore access — those are Phase 3b. This module reads
three committed artifact files and does arithmetic.

WHAT IT PORTS, AND FROM WHERE
  The algorithm is not re-derived here. It is the procedure written down in
  backend/ml/artifacts/domain_split/README.md ("How a consumer applies these")
  and already executed against the whole 7,000-user dataset by
  MLAI/clustering/verify_production_artifacts.py::assign(). That script's
  vectorised replay and this module's per-user path are checked against each
  other, and against the committed per-user assignments, by
  backend/scripts/verify_tier_assignment.py.

THE TWO APPROACHES ARE NOT INTERCHANGEABLE
  Banking and Investment are Approach A: FIVE independently fitted models per
  domain, one per age band. The age band selects the model — a wrong band
  reads the wrong centroids and (because of the flip below) can return a
  confidently wrong label.

  Insurance is Approach B: ONE pooled model — but age is not absent from it,
  it is an INPUT. The 3 insurance features are RobustScaler-scaled and the
  user's age_midpoint is appended, StandardScaler-standardised, as a fourth
  dimension. So age_band is required for all three domains; what differs is
  what it is used FOR (model selection vs. a model input). "Insurance is
  pooled" must not be read as "Insurance ignores age" — see
  ``INSURANCE_NEEDS_AGE_TOO`` below and insurance_model.json's ``age_scaler``.

⚠️ THE CLUSTER-ID TRAP (brief Section 5; artifact README's loudest warning)
  Cluster id does NOT mean the same tier in every age band. For Banking,
  cluster 0 is "Building" (the weaker tier) in band 1 and "Stable" (the
  healthy tier) in bands 2-5. Every lookup here goes through the artifact's
  own tier_map / shipped_tier_map; there is no cluster-id -> label mapping
  written in this file, and _load_model() refuses to load an artifact whose
  map does not agree with its own health ranks. A hardcoded assumption would
  produce a plausible, entirely wrong tier for one fifth of Banking users.

WHY THE SCALERS ARE READ FROM JSON AND NOT UNPICKLED
  Each *_model.json carries its scaler's center_/scale_ (and, for Insurance,
  the age scaler's mean_/scale_) as plain numbers precisely so a consumer can
  apply the transform without unpickling — the artifact README says so. Doing
  that keeps this module's runtime dependency surface at "json + math": no
  scikit-learn import on the request path, and no risk that an unpickle
  silently drifts when the backend's sklearn version moves away from the
  1.8.0 the manifest records. The committed .pkl files remain the reference:
  verify_tier_assignment.py loads them and asserts, numerically, that this
  module's transform reproduces scaler.transform() on real data.
"""

import json
import logging
import math
import numbers
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "domain_split"

# Artifact filename stems, same three keys and spelling as
# scoring.LAYER2_CLUSTERING_FEATURES_BY_DOMAIN and layer3.py's verification.
DOMAINS = ("banking", "investment", "insurance")

# ---------------------------------------------------------------------------
# Age bands
# ---------------------------------------------------------------------------
# The five bands the models were fitted over, spanning ages 20-70. Bands 1-4
# are decades; band 5 is eleven years wide (60-70), which is why the midpoints
# below are NOT (lo + hi) / 2 for every band — they are the dataset's own
# age_midpoint values, and the Insurance model was fitted on exactly those five
# numbers. Do not "correct" 25 to 24.5: it would move every Insurance user's
# fourth input dimension away from the values the medoids were fitted on.
#
# Sources, both asserted by verify_tier_assignment.py rather than trusted:
#   ranges    banking_model.json / investment_model.json bands[b].age_range,
#             and MLAI/clustering/benchmark_core.py BAND_LABELS
#   midpoints the age_midpoint column of
#             MLAI/clustering/data/processed/synthetic_users.csv, which holds
#             exactly one distinct value per band
AGE_BANDS = (1, 2, 3, 4, 5)
AGE_BAND_RANGES = {1: (20, 29), 2: (30, 39), 3: (40, 49), 4: (50, 59), 5: (60, 70)}
AGE_BAND_MIDPOINT = {1: 25.0, 2: 35.0, 3: 45.0, 4: 55.0, 5: 65.0}

MIN_MODELLED_AGE = AGE_BAND_RANGES[1][0]      # 20
MAX_MODELLED_AGE = AGE_BAND_RANGES[5][1]      # 70

# Read by callers that want to state the dependency explicitly; also the thing
# this module most wants a future reader to notice.
INSURANCE_NEEDS_AGE_TOO = (
    "Insurance is Approach B — one pooled model, so age_band does not select a "
    "model. It is still REQUIRED: the user's age_midpoint enters the model as "
    "its fourth input dimension (insurance_model.json.age_scaler). Passing the "
    "wrong band moves the user in that dimension and can change the tier."
)

# ---------------------------------------------------------------------------
# Available-case features
# ---------------------------------------------------------------------------
# Features that may be None while a tier is still assigned, by nearest centroid
# over the remaining dimensions only. Banking only, and only one:
#
#   healthy_debt_ratio     None for a debt-free user (a proportion of zero
#                          classified debt)
#
# It goes None in a state the locked model never saw: the generator samples a
# 0-1 healthy_debt_ratio for every synthetic user, debt-free or not
# (generate_synthetic_data.py's "Simplification" note). No substitute value is
# defensible — tested against the Banking centroids, 0.0 pushes a middling
# debt-free user to Building in every band, 1.0 to Stable in every band, and
# the scaler centre splits by band — so the dimension is dropped rather than
# filled. The same dimensions are dropped for every centroid compared, so the
# comparison stays like-for-like.
#
# Everything else stays MANDATORY:
#   savings_rate   None with no depository account (credit/loan only); a tier
#                  read off debt data alone would be a different claim.
#   dti            never None once income is valid.
#   emergency_fund_months
#                  None when monthly_expenses_sgd is 0. Tried as optional and
#                  reverted: zero expenses also forces savings_rate to 1.0,
#                  far past the training data's 0.68 maximum, and that one
#                  out-of-range value put strong, middling and weak savers
#                  alike in Stable in every band. In practice a 0 usually means
#                  no transactions synced, not zero spending — so no tier.
# Investment and Insurance have no optional features.
OPTIONAL_FEATURES_BY_DOMAIN = {
    "banking": ("healthy_debt_ratio",),
}


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load_model(domain: str, artifact_dir: str | None = None) -> dict:
    """
    Loads and structurally validates one domain artifact, once per process.

    Returns a small dict of exactly what inference needs, so the hot path never
    re-parses JSON or re-walks the artifact's nested shape:

        features    ordered feature names (the scaler's positional contract)
        center      RobustScaler center_, aligned to features
        scale       RobustScaler scale_, aligned to features
        age         None, or (mean, scale) for the appended age dimension
        bands       {age_band or None: {cluster_id: (tier_index, tier_label)}}
                    keyed by band for Approach A, by the single key None for B
        centroids   {age_band or None: [coords per cluster id, in id order]}

    The validation here is the load-time half of the cluster-id trap guard: an
    artifact whose tier map disagrees with its own health ranks, or whose
    centroids do not match its declared feature count, is rejected loudly
    instead of quietly producing wrong labels.
    """
    directory = Path(artifact_dir) if artifact_dir else _ARTIFACT_DIR
    art = json.loads((directory / f"{domain}_model.json").read_text(encoding="utf-8"))

    features = art["clustering_features"]
    scaler = art["scaler"]
    if scaler["features"] != features:
        raise AssertionError(
            f"{domain}: artifact scaler.features {scaler['features']} != "
            f"clustering_features {features} — the artifact is internally "
            "inconsistent and the scaler would be applied to the wrong columns.")
    optional = OPTIONAL_FEATURES_BY_DOMAIN.get(domain, ())
    unknown_optional = [name for name in optional if name not in features]
    if unknown_optional:
        # A renamed feature would otherwise silently turn available-case
        # scoring off (or leave a stale name nothing ever matches).
        raise AssertionError(
            f"{domain}: OPTIONAL_FEATURES_BY_DOMAIN names {unknown_optional}, "
            f"which are not among the artifact's clustering_features {features}.")
    if len(optional) >= len(features):
        raise AssertionError(
            f"{domain}: every feature is optional — a tier could be assigned "
            "from no data at all.")

    labels = art["shipped_tiers_summary"]["tier_labels"]
    k_native = art["k_native"]
    approach = art["approach"]
    n_dims = len(features)

    age = None
    if approach == "B":
        age_scaler = art["age_scaler"]
        if age_scaler["feature"] != "age_midpoint":
            raise AssertionError(
                f"{domain}: age_scaler is on {age_scaler['feature']!r}, not "
                "'age_midpoint' — this module standardises the band midpoint.")
        age = (float(age_scaler["mean_"][0]), float(age_scaler["scale_"][0]))
        n_dims += 1

    bands: dict[int | None, dict[int, tuple[int, str]]] = {}
    centroids: dict[int | None, list[list[float]]] = {}

    if approach == "A":
        # Approach A — one fitted model per age band; tier_map is per band and
        # is the ONLY thing that may decide what a cluster id means.
        for band_key, band in art["bands"].items():
            band_no = int(band_key)
            tier_map = {
                int(cid): (int(e["health_rank"]), e["tier_label"])
                for cid, e in band["tier_map"].items()
            }
            bands[band_no] = tier_map
            centroids[band_no] = [
                [float(x) for x in band["centroids_scaled"][str(cid)]]
                for cid in range(k_native)
            ]
        if set(bands) != set(AGE_BANDS):
            raise AssertionError(
                f"{domain}: artifact has bands {sorted(bands)}, expected "
                f"{list(AGE_BANDS)} — this module's age-band definition and the "
                "fitted model disagree.")
    elif approach == "B":
        # Approach B — one pooled model. The native partition may be wider than
        # the shipped tiers (Insurance: 3 native -> 2 shipped), so the map that
        # matters is the merge's shipped_tier_map, not the native ordering.
        merge = art.get("tier_merge") or {}
        if not merge.get("applied"):
            raise AssertionError(
                f"{domain}: Approach B artifact has no applied tier_merge — "
                "this module has no other documented way to turn its native "
                "cluster ids into shipped tiers.")
        bands[None] = {
            int(cid): (int(e["shipped_tier"]), e["tier_label"])
            for cid, e in merge["shipped_tier_map"].items()
        }
        centroids[None] = [
            [float(x) for x in art["model"]["centroids_scaled"][str(cid)]]
            for cid in range(k_native)
        ]
    else:
        raise AssertionError(
            f"{domain}: unknown approach {approach!r} — only 'A' "
            "(per-age-band models) and 'B' (pooled) are implemented.")

    # Structural guards. Each one is a bug this module could otherwise ship
    # silently, because a wrong-but-well-formed lookup still returns a label.
    for key, tier_map in bands.items():
        where = f"{domain}" + (f" band {key}" if key is not None else "")
        if sorted(tier_map) != list(range(k_native)):
            raise AssertionError(
                f"{where}: tier map covers cluster ids {sorted(tier_map)}, "
                f"expected every id in 0..{k_native - 1}.")
        for cid, (index, label) in tier_map.items():
            if not 0 <= index < art["shipped_tiers"]:
                raise AssertionError(
                    f"{where} cluster {cid}: tier index {index} outside "
                    f"0..{art['shipped_tiers'] - 1}.")
            if label != labels[index]:
                raise AssertionError(
                    f"{where} cluster {cid}: label {label!r} does not match "
                    f"tier index {index} -> {labels[index]!r}. The artifact "
                    "pinned a label to a cluster id; that is invalid because "
                    "cluster ids do not carry a fixed meaning across bands.")
        for cid, coords in enumerate(centroids[key]):
            if len(coords) != n_dims:
                raise AssertionError(
                    f"{where} cluster {cid}: centroid has {len(coords)} "
                    f"coordinates, expected {n_dims}.")

    return {
        "domain": domain,
        "approach": approach,
        "features": list(features),
        "optional": frozenset(optional),
        "center": [float(x) for x in scaler["center_"]],
        "scale": [float(x) for x in scaler["scale_"]],
        "age": age,
        "bands": bands,
        "centroids": centroids,
        "tier_labels": list(labels),
    }


def clear_artifact_cache() -> None:
    """Drops the parsed-artifact cache (tests that point at a temp dir)."""
    _load_model.cache_clear()


# ---------------------------------------------------------------------------
# Age band
# ---------------------------------------------------------------------------
def age_band_for_date_of_birth(
    date_of_birth: date | datetime,
    today: date | None = None,
    clamp: bool = False,
) -> int:
    """
    Maps a date of birth to one of the five modelled age bands (1-5).

    Age is computed from date_of_birth and never stored (Section 14), matching
    AppUser.age in lib/models/app_user.dart — same has-the-birthday-passed
    adjustment, so backend and client agree on the band for the same user on
    the same day.

    Args:
        date_of_birth: users/{uid}.date_of_birth. A tz-aware datetime (what the
                       Firestore client returns) is read in its own calendar
                       date; a plain date works too.
        today:         Injectable clock for tests; defaults to the local date.
        clamp:         What to do about an age the models never saw. False (the
                       default) raises. True maps ages under 20 to band 1 and
                       over 70 to band 5 and logs a warning.

    Returns:
        int in 1..5.

    Raises:
        ValueError: on a future date of birth, or — unless clamp=True — an age
        outside 20-70.

    ⚠️ On the out-of-range decision. Onboarding's date picker allows any DOB in
    the last 100 years (lib/screens/auth/onboarding_screen.dart), so ages under
    20 and over 70 are reachable in production, while all three models were
    fitted on 20-70 only. Neither answer is free: raising means a 72-year-old
    gets no tier, clamping means they get one from a model that never saw their
    age. The default is the loud one, and clamping is opt-in AT THE CALL SITE
    so that Phase 3b's choice is visible in Phase 3b's code rather than buried
    in a default here. Whichever it picks, the extrapolation is recorded (this
    warning) instead of being indistinguishable from a normal assignment.
    """
    if isinstance(date_of_birth, datetime):
        dob = date_of_birth.date()
    elif isinstance(date_of_birth, date):
        dob = date_of_birth
    else:
        raise TypeError(
            f"date_of_birth must be a date or datetime, got "
            f"{type(date_of_birth).__name__}")

    today = today or date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    if age < 0:
        raise ValueError(f"date_of_birth {dob} is in the future (age {age})")

    if age < MIN_MODELLED_AGE or age > MAX_MODELLED_AGE:
        if not clamp:
            raise ValueError(
                f"age {age} is outside the modelled range "
                f"{MIN_MODELLED_AGE}-{MAX_MODELLED_AGE}; all three Layer 2 "
                "models were fitted on that range only. Pass clamp=True to "
                "extrapolate to the nearest band, deliberately.")
        band = 1 if age < MIN_MODELLED_AGE else 5
        logger.warning(
            "age %d is outside the modelled range %d-%d; clamping to band %d — "
            "this user's tier comes from a model fitted on a different age.",
            age, MIN_MODELLED_AGE, MAX_MODELLED_AGE, band)
        return band

    for band, (lo, hi) in AGE_BAND_RANGES.items():
        if lo <= age <= hi:
            return band
    raise AssertionError(  # unreachable while the bands stay contiguous
        f"age {age} is in range but matched no band — AGE_BAND_RANGES has a gap")


def _validate_age_band(domain: str, age_band: object) -> int:
    """Rejects a missing or out-of-range band for any domain. See module docstring."""
    if age_band is None:
        raise ValueError(
            f"age_band is required for {domain}. "
            + (INSURANCE_NEEDS_AGE_TOO if domain == "insurance" else
               "Banking and Investment are Approach A: the band selects which "
               "of the five fitted models to use."))
    if isinstance(age_band, bool) or not isinstance(age_band, int):
        raise ValueError(
            f"age_band must be an int in {AGE_BANDS}, got "
            f"{age_band!r} ({type(age_band).__name__})")
    if age_band not in AGE_BANDS:
        raise ValueError(
            f"age_band {age_band} is not one of {AGE_BANDS} "
            f"(1: 20-29, 2: 30-39, 3: 40-49, 4: 50-59, 5: 60-70)")
    return age_band


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _feature_vector(
    model: dict, features: dict, age_band: int,
) -> list[float | None] | None:
    """
    Builds the scaled input vector, or None when the user's data cannot support
    one. Missing data is never imputed — see assign_domain_tier's docstring.

    An OPTIONAL_FEATURES_BY_DOMAIN feature that is None keeps its slot as None,
    so _nearest_cluster_id can drop that dimension; any other missing feature
    still means no vector at all.
    """
    values: list[float | None] = []
    for i, name in enumerate(model["features"]):
        if name not in features:
            # Distinguished from a present-but-None value on purpose: the
            # compute_* functions in scoring.py always return every key, so an
            # ABSENT key usually means the caller assembled the wrong domain's
            # dict. The contract still returns None (a tier is genuinely not
            # computable), but this leaves a trace that the value-is-None path
            # would not.
            logger.warning(
                "%s: feature %r absent from the features dict (keys: %s) — no "
                "tier assigned. Expected this domain's %d features: %s",
                model["domain"], name, sorted(features), len(model["features"]),
                model["features"])
            return None
        value = features[name]
        if value is None:
            if name in model["optional"]:
                # Only an explicit None qualifies: it is what scoring.py returns
                # for "not computable". An absent key (above) and NaN/inf (below)
                # still mean something went wrong upstream.
                logger.debug("%s: %s is None — scored on the remaining "
                             "dimensions", model["domain"], name)
                values.append(None)
                continue
            logger.debug("%s: %s is None — no tier assigned", model["domain"], name)
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            # numbers.Real, not (int, float), so a numpy scalar out of a
            # DataFrame is accepted — but deliberately strict about the rest: a
            # numeric STRING is not silently coerced. The compute_* functions
            # return floats or None, so a string arriving here means the value
            # took a detour through a document field or a request body without
            # being parsed — and a coerced "0.25" would score perfectly happily
            # while the next value along, "", would not. bool is excluded
            # explicitly because it IS a numbers.Real, and a binary Layer 1
            # flag reaching a continuous Layer 2 slot is a real bug, not a 1.0.
            raise ValueError(
                f"{model['domain']}: feature {name!r} = {value!r} is a "
                f"{type(value).__name__}, expected a number or None. Parse it "
                "before assigning a tier; None is the way to say 'not "
                "computable'.")
        value = float(value)
        if not math.isfinite(value):
            # NaN/inf reach the distance calculation as a silently unbeatable
            # or meaningless coordinate; treat them as missing data instead.
            logger.warning("%s: %s is %r — treated as missing, no tier assigned",
                           model["domain"], name, value)
            return None
        # RobustScaler: (x - center_) / scale_, positionally against features.
        values.append((value - model["center"][i]) / model["scale"][i])

    if model["age"] is not None:
        # Approach B only. StandardScaler on age_midpoint, appended LAST —
        # the order the model was fitted in (artifact README, step 2).
        mean, scale = model["age"]
        values.append((AGE_BAND_MIDPOINT[age_band] - mean) / scale)

    return values


def _nearest_cluster_id(
    model: dict, vector: list[float | None], age_band: int,
) -> int:
    """
    Nearest centroid by squared Euclidean distance — the artifact's declared
    assignment_rule, and the same argmin the validated replay uses. Ties go to
    the lowest cluster id (numpy argmin's behaviour); they do not occur on real
    data, and verify_tier_assignment.py reports the smallest observed margin.

    A None coordinate (an optional feature that was not computable) is left out
    of the sum for every centroid alike — available-case distance over the
    dimensions the user actually has. A complete vector sums exactly the terms
    it always did.
    """
    key = age_band if model["approach"] == "A" else None
    best_id, best_d2 = 0, math.inf
    for cluster_id, centroid in enumerate(model["centroids"][key]):
        d2 = math.fsum((a - b) ** 2 for a, b in zip(vector, centroid)
                       if a is not None)
        if d2 < best_d2:
            best_id, best_d2 = cluster_id, d2
    return best_id


def _assign(
    domain: str,
    features: dict,
    age_band: object,
    artifact_dir: str | None,
) -> tuple[int, str] | None:
    """Shared body of the two public entry points."""
    if domain not in DOMAINS:
        raise ValueError(
            f"unknown domain {domain!r} — must be one of {DOMAINS} (the "
            "artifact filename stems in backend/ml/artifacts/domain_split/)")
    if not isinstance(features, dict):
        raise TypeError(
            f"features must be a dict keyed by {domain}'s clustering feature "
            f"names, got {type(features).__name__}")

    age_band = _validate_age_band(domain, age_band)
    model = _load_model(domain, artifact_dir)

    vector = _feature_vector(model, features, age_band)
    if vector is None:
        return None

    cluster_id = _nearest_cluster_id(model, vector, age_band)
    key = age_band if model["approach"] == "A" else None
    # THE lookup. Never a literal, never an index into tier_labels by cluster
    # id — always this band's own map (module docstring, cluster-id trap).
    return model["bands"][key][cluster_id]


def assign_domain_tier(
    domain: str,
    features: dict,
    age_band: int | None = None,
    artifact_dir: str | None = None,
) -> str | None:
    """
    Assigns one domain tier for one user, from the committed production
    artifacts alone.

    Args:
        domain:    "banking", "investment" or "insurance".
        features:  Raw, UNSCALED values keyed by that domain's clustering
                   feature names — banking: savings_rate, emergency_fund_months,
                   dti, healthy_debt_ratio; investment: investment_rate,
                   portfolio_concentration, risk_age_deviation; insurance:
                   life_coverage_ratio, ci_coverage_ratio, insurance_spend_ratio.
                   Extra keys are ignored, so a domain's own compute_* output
                   can be passed straight in. Note that Banking's ``dti`` is
                   spelled ``debt_to_income_ratio`` by
                   scoring.compute_banking_features() — translate through
                   scoring.CLUSTERING_FEATURE_KEY_MAP, do not rename by hand.
        age_band:  1-5. REQUIRED for all three domains, including Insurance —
                   see INSURANCE_NEEDS_AGE_TOO. Use
                   age_band_for_date_of_birth().
        artifact_dir: Override the artifact directory (tests only).

    Returns:
        The tier label ("Stable" / "Building", "On Track" / "Getting Started",
        "Covered" / "Not Covered"), ready to hand to
        layer3.detect_combination_flags() — or None when the user's data cannot
        support an assignment.

        The label is a display string; the artifacts state that the tier INDEX
        is the authoritative key for downstream rules. Use
        assign_domain_tier_index() where an index is wanted, rather than
        comparing labels or looking one up by position.

    Missing data (None):
        None is returned whenever ANY of that domain's features is missing from
        the dict or non-finite, or is None — with one narrow exception below.
        Nothing is ever imputed: a zero for an unknown savings rate is not a
        neutral choice, it is a specific and usually wrong claim about the
        user, and it would place them at a real point in feature space with a
        real tier attached. Same contract as scoring.py's compute_* functions
        and layer3.py's rules.

        The exception is OPTIONAL_FEATURES_BY_DOMAIN — Banking's
        healthy_debt_ratio only. When it is None (and present in the dict), the
        tier comes from the nearest centroid over the other three dimensions,
        the same subset for every centroid. savings_rate, dti and
        emergency_fund_months are never optional, so a Banking tier always
        rests on at least those three real values.

    Raises:
        ValueError: unknown domain; missing / out-of-range / non-int age_band;
                    a feature value that is neither a number nor None (strings
                    are not coerced, deliberately).
        TypeError:  features is not a dict.
        AssertionError: the artifact is internally inconsistent (raised at load
                    time by _load_model).
    """
    result = _assign(domain, features, age_band, artifact_dir)
    return None if result is None else result[1]


def assign_domain_tier_index(
    domain: str,
    features: dict,
    age_band: int | None = None,
    artifact_dir: str | None = None,
) -> int | None:
    """
    Same call as assign_domain_tier(), returning the tier INDEX instead of the
    label — 0 is always the healthiest, ascending.

    The artifacts are explicit that the index, not the display string, is the
    authoritative key any downstream rule should be written against (a label is
    display wording that can be re-worded without touching the model). Provided
    so scoring code never has to recover an index by searching the label list.
    """
    result = _assign(domain, features, age_band, artifact_dir)
    return None if result is None else result[0]


def tier_labels(domain: str, artifact_dir: str | None = None) -> list[str]:
    """That domain's shipped tier labels, healthiest first, read from its artifact."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r} — must be one of {DOMAINS}")
    return list(_load_model(domain, artifact_dir)["tier_labels"])


def verify_age_bands_against_artifacts(artifact_dir: str | None = None) -> dict:
    """
    Re-reads the two Approach A artifacts and asserts AGE_BAND_RANGES still
    matches the bands the models were actually fitted over — same five bands,
    same age_range strings.

    Not called at import (this module does its file I/O on first inference,
    not on import). Direct analogue of layer3.verify_tier_labels_against_
    artifacts() and scoring.verify_clustering_features_against_artifacts(),
    and it guards the same class of bug: a band definition that drifts from the
    fitted models still returns a perfectly plausible tier, from the wrong
    model. Run by backend/scripts/verify_tier_assignment.py, which additionally
    checks AGE_BAND_MIDPOINT against the dataset the models were fitted on.

    Returns:
        Dict mapping domain -> {band: age_range string} as declared.
    """
    directory = Path(artifact_dir) if artifact_dir else _ARTIFACT_DIR
    declared = {}
    for domain in DOMAINS:
        art = json.loads(
            (directory / f"{domain}_model.json").read_text(encoding="utf-8"))
        if art["approach"] != "A":
            continue
        ranges = {int(b): v["age_range"] for b, v in art["bands"].items()}
        expected = {b: f"{lo}-{hi}" for b, (lo, hi) in AGE_BAND_RANGES.items()}
        if ranges != expected:
            raise AssertionError(
                f"{domain}: artifact age ranges {ranges} != AGE_BAND_RANGES "
                f"{expected} — this module would select the wrong per-band "
                "model. Fix before shipping.")
        declared[domain] = ranges
    if not declared:
        raise AssertionError(
            "no Approach A artifact found — age bands went unverified, which "
            "is not the same as verified. Check the artifact directory.")
    return declared


if __name__ == "__main__":
    print("Layer 2 tier assignment — production artifacts "
          f"({_ARTIFACT_DIR.relative_to(Path(__file__).parents[2])})\n")
    for domain in DOMAINS:
        model = _load_model(domain)
        n_models = len(model["bands"])
        print(f"  {domain:<11} approach {model['approach']}  "
              f"{n_models} fitted model(s)  features {model['features']}"
              + ("  + standardised age_midpoint" if model["age"] else ""))
        print(f"  {'':<11} tiers {model['tier_labels']}")
    print("\nBanking's cluster 0, band by band (the flip):")
    banking = _load_model("banking")
    for band in AGE_BANDS:
        index, label = banking["bands"][band][0]
        print(f"  band {band} ({AGE_BAND_RANGES[band][0]}-{AGE_BAND_RANGES[band][1]}):"
              f"  cluster 0 -> tier {index} {label!r}")
    print("\nAge-band definition vs the fitted artifacts:")
    for domain, ranges in verify_age_bands_against_artifacts().items():
        print(f"  {domain:<11} {ranges}")
    print("\nFull test suite: python scripts/verify_tier_assignment.py")
