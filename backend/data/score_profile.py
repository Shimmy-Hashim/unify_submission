"""Phase 3 score outputs on financialProfile — the I/O half of the pipeline.

    ml/score_pipeline.py    pure: documents in, result dict out
    data/score_profile.py   this file: reads the documents, writes the cache

The split follows the one already in the codebase — everything in backend/ml/ is
I/O-free and everything that touches Firestore lives in backend/data/ — and it
is what lets verify_score_pipeline.py exercise every pipeline branch with no
Firebase, no Plaid and no network.

WHICH FIELDS THIS MODULE OWNS
financial_profile.py records the split by writer, and this module claims the
four fields it listed as unowned:

    overall_score            float | None
    checklist_flags          map    (Layer 1 checks + Layer 3 flags)
    checklist_last_updated   timestamp
    ml_profile               map    (Section 8's "deferred until ML model
                                     selected" — no longer deferred; the schema
                                     is documented on _ml_profile() below)

It never touches monthly_income_sgd / income_source (client, at onboarding) or
monthly_expenses_sgd / net_worth_sgd (financial_profile.py). It READS the latter
two, so it must run AFTER the derived-field refresh on any shared trigger —
Banking's savings_rate and emergency_fund_months are computed from them.

It also owns users/{uid}/scoreHistory — one document per day, written from the
same result that produces the cache. See snapshot_payload() below.

NO PROFILE DOCUMENT -> NO WRITE, same rule as refresh_financial_profile():
the document's existence is the signal that the user has completed onboarding,
and a score-only document would confuse the client's "needs income setup"
routing check.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ml import score_pipeline
from notifications import rules as notification_rules
from notifications import service as notification_service

from .financial_profile import profile_ref

logger = logging.getLogger(__name__)

# Bumped when the cached SHAPE changes, so a client (or a later migration) can
# tell a v1 cache from a v2 one without guessing from which keys are present.
ML_PROFILE_SCHEMA_VERSION = score_pipeline.SCHEMA_VERSION


def fetch_score_inputs(db, uid: str) -> dict:
    """
    Every document the pipeline reads, in one place.

    Deliberately the same set of reads as verify_home_phase2.py's fetch_inputs()
    — that script was written as the dry run for this assembler.
    """
    user_ref = db.collection("users").document(uid)
    user_snap = user_ref.get()
    profile_snap = profile_ref(db, uid).get()
    investment_account_snaps = list(
        user_ref.collection("investmentAccounts").stream())
    return {
        "user_doc": user_snap.to_dict() if user_snap.exists else None,
        "profile": profile_snap.to_dict() if profile_snap.exists else None,
        "bank_accounts": [
            s.to_dict() for s in user_ref.collection("plaidAccounts").stream()
        ],
        "investment_accounts": [s.to_dict() for s in investment_account_snaps],
        "investment_transactions": [
            t.to_dict()
            for s in investment_account_snaps
            for t in s.reference.collection("transactions").stream()
        ],
        "insurance_policies": [
            s.to_dict()
            for s in user_ref.collection("insurancePolicies").stream()
        ],
    }


def refresh_score_profile(db, uid: str, now: datetime | None = None,
                          notify: bool = True, snapshot: bool = True) -> dict:
    """
    Runs the pipeline for one user and caches the result.

    Returns the full pipeline result either way. A result carrying
    ``unavailable_reason`` is a legitimate outcome, not a failure — the cache is
    still updated with it, because "we checked, and here is why there is no
    score" is exactly what the client needs to show, and leaving a stale score
    behind after the data that produced it changed would be worse.

    The one case that writes nothing is no financialProfile document at all
    (see the module docstring); the result still comes back, with
    ``unavailable_reason == "no_financial_profile"``.

    ``notify`` runs the Settings Pass 2 push hooks (Score Change Alerts and
    Risk Alerts) against the difference between the cache being overwritten and
    the result overwriting it. This is THE hook point for both, and deliberately
    the only one: every trigger the task names — all three Plaid sync endpoints
    and /insurance/parse — already funnels through here via
    _refresh_score_nonfatal, so hooking the cache write covers all four without
    four separate call sites that could drift apart. Set notify=False in tests
    and backfills.

    ``snapshot`` writes the day's scoreHistory document from the same result,
    and hangs off the same hook for the same reason — the cache write is the one
    moment a computed score exists, so anything that wants to observe a score
    belongs here rather than at a sixth call site. Suppressed the same way as
    ``notify``, for tests and backfills.
    """
    inputs = fetch_score_inputs(db, uid)
    # Captured BEFORE the write below overwrites it — this is the "previous"
    # half of every change comparison the notification rules make.
    previous_profile = inputs["profile"] or {}
    result = score_pipeline.compute_score_profile(**inputs, now=now)

    if inputs["profile"] is None:
        logger.info("no financialProfile for %s — score computed but not cached",
                    uid)
        return result

    profile_ref(db, uid).update(cache_payload(result))

    # Strictly after the cache write: the score is the thing that must land, and
    # neither a notification nor a history problem must cost the user their
    # refreshed score. Both are wrapped whole for that reason.
    if snapshot:
        _write_snapshot_nonfatal(db, uid, result)
    if notify:
        _notify_score_change_nonfatal(
            db, uid, previous_profile, result, inputs["insurance_policies"])
    return result


def _notify_score_change_nonfatal(db, uid: str, previous_profile: dict,
                                  result: dict, policies: list) -> None:
    """Score Change Alerts + Risk Alerts, from one before/after comparison.

    Wrapped whole rather than per-rule: this function is called from inside an
    operation the user is waiting on, and the entire notification concern is
    optional to that operation's success.
    """
    try:
        new_gaps = notification_rules.coverage_gaps_of(policies)
        notes = notification_rules.detect_score_change(
            previous_profile.get("overall_score"),
            result["overall_score"],
            computed_at=result["computed_at"],
        )
        notes += notification_rules.detect_risk_alerts(
            previous_profile.get("ml_profile"),
            result,
            previous_gaps=notification_service.read_known_gaps(db, uid),
            new_gaps=new_gaps,
        )
        if notes:
            notification_service.deliver(db, uid, notes)
        # Recorded whether or not anything fired, and whether or not delivery
        # succeeded: this is "the gaps as they now stand", the baseline the next
        # refresh compares against.
        notification_service.record_known_gaps(db, uid, new_gaps)
    except Exception:
        logger.exception("score-change notification hook failed for %s", uid)


def cache_payload(result: dict) -> dict:
    """
    The exact four Section 8 fields, built from a pipeline result.

    Separate from the write so the mapping is testable without Firestore.

    ``checklist_last_updated`` is the pipeline's own computed_at rather than
    firestore.SERVER_TIMESTAMP: the API returns the written result directly, and
    a sentinel cannot be echoed to the caller without a follow-up read. Using
    one value also keeps checklist_last_updated and ml_profile.computed_at
    identical, which they must be — they describe the same computation.
    """
    return {
        "overall_score": result["overall_score"],
        "checklist_flags": _checklist_flags(result),
        "checklist_last_updated": result["computed_at"],
        "ml_profile": _ml_profile(result),
    }


def _checklist_flags(result: dict) -> dict:
    """
    Section 8: "stored results of all rule-based checks". Both rule layers land
    here, keyed by layer — overall_score.py's integration note says Layer 1's
    checks and Layer 3's flags share this field, and keeping them in one flat
    list would lose which is which (Layer 1 entries are pass/fail checks with
    copy; Layer 3 entries are pattern names that fired).

    ``layer3_evaluated`` is stored alongside because an empty layer3 list is
    ambiguous on its own: "all three tiers known, no pattern fired" and "not all
    three tiers known, so nothing was tested" are different things to show.
    """
    return {
        "layer1": result["checklist_flags"],
        "layer3": result["layer3_flags"],
        "layer3_evaluated": result["layer3_evaluated"],
        "layer3_reason": result["layer3_reason"],
    }


def _ml_profile(result: dict) -> dict:
    """
    ml_profile — Section 8's "deferred until ML model selected" block, now
    filled in. Every field earns its place; the reasoning, field by field:

      schema_version / computed_at
          A cached score is only interpretable against the shape and the moment
          that produced it.

      age_band / age_band_range
          THE most important field here. The same feature vector yields
          different tiers in different bands, and Banking's cluster 0 is
          "Building" in band 1 but "Stable" in bands 2-5. Without the band, a
          cached tier cannot be explained, reproduced or audited. Stored as the
          band number plus its human range so nothing has to re-derive it.
          The user's AGE is deliberately NOT stored — Section 14 requires age to
          be computed from date_of_birth and never stored, and the band is the
          coarse model input, not a personal detail.

      domains[*].tier_index / tier_label
          Both, on purpose. The artifacts state the INDEX is the authoritative
          key for downstream rules (0 = healthiest), while the label is display
          wording that can be re-worded without touching the model. Storing both
          means a re-wording does not require recomputing every user, and no
          renderer has to look a label up by position.

      domains[*].status / reason / missing_features
          Turns "no Banking tier" into "connect a bank account" without the
          client re-deriving which inputs were missing. This is the partial-
          result contract made legible.

      layer3_flags / layer3_evaluated
          Duplicated from checklist_flags deliberately: checklist_flags is the
          user-facing checklist payload, ml_profile is the model-provenance
          record of the same run, and a consumer of either should not have to
          read the other to know what fired.

      overall_score / overall_score_reason / score_breakdown
          The score is duplicated from its own top-level Section 8 field so that
          ml_profile is a self-contained record of one run. The breakdown is
          what makes the score explainable — ``capped`` answers "would closing
          my protection gap raise this score?" and insurance_indicators_cleared
          says how far from the next ceiling the user is, both without
          recomputing anything. That is the brief's "critical gaps surface
          regardless of overall score" requirement, cached.

      artifacts
          The artifact set's own schema_version and generated_utc. If the models
          are ever rebuilt, every cached score silently becomes stale; recording
          which artifact build produced this one makes that detectable instead
          of invisible.

      insurance_coverage
          The ONE narrow exception to the rule below, added deliberately rather
          than by drift, so the reasoning is recorded here next to it.

          It is not the raw feature vector: it is the two CONTINUOUS coverage
          ratios (life, critical illness) beside the age-band benchmark each is
          measured against, and nothing else. The Insurance screen's Coverage
          Gaps section needs "how far short", and the cached Insurance tier is
          binary — "Not Covered" is the same word for a user at 2.9x of LIA's 9x
          and one at 8.9x.

          Each of the three reasons below was re-checked against this block
          before it was added, and none of them holds for it:
            - "recomputable client-side" — only by re-implementing
              compute_insurance_features()'s policy_type guards in Dart (life
              counts ONLY policy_type == "life"; personal_accident and other are
              excluded, a rule that exists because a real accidental-death
              benefit once inflated a life ratio during gold-set testing). A
              second implementation of that rule is exactly the drift this
              project avoids everywhere else, and it would let a gap message
              contradict the tier printed beside it.
            - "the most sensitive thing in the pipeline" — a coverage ratio is
              the sum assured the Insurance screen already prints, over the
              income the user typed into Settings themselves. It exposes
              nothing the two screens do not already show.
            - "a stale second source of truth" — every input to it already
              triggers this exact cache write: /insurance/parse recomputes on
              upload, and Edit Profile posts /score/refresh on an income
              change. It cannot go stale without the cached SCORE beside it
              going stale in the same breath, which is one state, not two.

      healthy_debt_ratio
          The SECOND narrow exception, and the only raw Banking feature that
          reaches the cache. Added for Home's Wealth Breakdown, which needs to
          say how much of a user's debt is structured (mortgage, student loan,
          auto, home equity) versus revolving or unsecured.

          The same three reasons were re-checked against it, and none holds:
            - "recomputable client-side" — only by copying scoring.py's
              _HEALTHY_DEBT_SUBTYPES / _UNHEALTHY_DEBT_SUBTYPES tables into
              Dart, including the unclassified-subtype exclusion and the
              None-when-no-classified-debt rule. Those tables have already
              drifted once from the brief (home equity), in Python alone; a
              second copy is a second place to drift, and a Dart ratio that
              disagreed would sit beside the Banking tier this exact number
              was fed into.
            - "the most sensitive thing in the pipeline" — it is a proportion
              of balances the Savings screen already lists account by account.
              It exposes nothing the client cannot already read.
            - "a stale second source of truth" — it is derived only from
              plaidAccounts, and every plaidAccounts change (all three Plaid
              sync endpoints and disconnect) already triggers this cache write.
              It goes stale only together with the Banking tier beside it.

          A number, not a flag: None when not computable (no classifiable debt
          or no bank accounts), which is different from 0.0 (every classified
          dollar is unhealthy debt). Present whether or not Banking was
          assessed — a user missing savings_rate still has a real debt split.
          The reverse also holds: None here does not mean Banking went
          unassessed. A debt-free user with savings_rate, dti and
          emergency_fund_months is tiered on those three (tier_assignment's
          OPTIONAL_FEATURES_BY_DOMAIN), so a cached tier can sit beside a None
          ratio.

          Still deliberately NOT stored: everything else. Banking's other three
          features and all of Investment's remain out, on all three of the
          original grounds — the Home screen reads those numbers from
          financialProfile directly, so a cached copy really would be a second
          source of truth.
    """
    return {
        "schema_version": ML_PROFILE_SCHEMA_VERSION,
        "computed_at": result["computed_at"],
        "age_band": result["age_band"],
        "age_band_range": result["age_band_range"],
        "domains": _domains_for_cache(result),
        "layer3_flags": result["layer3_flags"],
        "layer3_evaluated": result["layer3_evaluated"],
        "overall_score": result["overall_score"],
        "overall_score_reason": result["overall_score_reason"],
        "unavailable_reason": result["unavailable_reason"],
        "score_breakdown": _breakdown_for_cache(result["score_breakdown"]),
        "insurance_coverage": result["insurance_coverage"],
        "healthy_debt_ratio": result["healthy_debt_ratio"],
        "artifacts": _artifact_provenance(),
    }


def _domains_for_cache(result: dict) -> dict:
    """The per-domain block, projected to exactly the five fields that are
    cached.

    Shared by _ml_profile() and snapshot_payload() rather than written out
    twice: a snapshot whose domain shape had drifted from the live cache's
    would be unreadable by the same code that renders the live one, which is
    the whole point of storing it.
    """
    return {
        domain: {
            "status": value["status"],
            "tier_index": value["tier_index"],
            "tier_label": value["tier_label"],
            "reason": value["reason"],
            "missing_features": value["missing_features"],
        }
        for domain, value in result["domains"].items()
    }


def _breakdown_for_cache(breakdown: dict | None) -> dict | None:
    """compute_score_breakdown()'s dict, with its tuple flattened to a list —
    Firestore has no tuple type, and a silent conversion is worse than an
    explicit one."""
    if breakdown is None:
        return None
    return {**breakdown,
            "insurance_indicators": list(breakdown["insurance_indicators"])}


_ARTIFACTS: dict | None = None


def _artifact_provenance() -> dict:
    """Which artifact build produced this score. Read once per process, and
    returned as a copy so a caller cannot mutate the cached values."""
    global _ARTIFACTS
    if _ARTIFACTS is None:
        manifest = json.loads(
            (Path(__file__).resolve().parents[1] / "ml" / "artifacts"
             / "domain_split" / "manifest.json").read_text(encoding="utf-8"))
        _ARTIFACTS = {
            "schema_version": manifest["schema_version"],
            "generated_utc": manifest["generated_utc"],
        }
    return dict(_ARTIFACTS)


# ---------------------------------------------------------------------------
# scoreHistory — one document per day
# ---------------------------------------------------------------------------
# financialProfile holds only the LATEST score: every refresh overwrites the
# same four fields, so before this collection existed the project genuinely
# stored no history at all (brief Section 11's Home row records that as the
# reason Home shows no trend). This collection is that gap closed.
HISTORY_COLLECTION = "scoreHistory"

# The document ID, and also stored as a field so the collection can be ordered
# and range-queried without relying on __name__.
HISTORY_DATE_FORMAT = "%Y-%m-%d"


def history_collection(db, uid: str):
    """users/{uid}/scoreHistory. Same accessor shape as notifications/service.py's
    events_collection()."""
    return db.collection("users").document(uid).collection(HISTORY_COLLECTION)


def snapshot_date(computed_at: datetime) -> str:
    """The YYYY-MM-DD document ID for a run, in UTC.

    UTC, not Asia/Singapore, deliberately: every timestamp this backend
    produces is tz-aware UTC (score_pipeline's ``now``, and therefore
    ``computed_at``), and notifications/rules.py's _stamp() already derives its
    document IDs the same way. A Singapore-local day boundary here would be the
    only local-time arithmetic in the backend, and would disagree with the
    ``computed_at`` stored inside the very document it keys.

    The consequence, stated rather than buried: SGT is UTC+8, so a refresh
    between 08:00 and 23:59 SGT lands on that Singapore day, while one between
    00:00 and 07:59 SGT lands on the previous one. Worth revisiting if the chart
    is ever shown with day labels a user would compare against their own clock.
    """
    return computed_at.astimezone(timezone.utc).strftime(HISTORY_DATE_FORMAT)


def snapshot_payload(result: dict) -> dict:
    """One day's snapshot, built from a pipeline result.

    Separate from the write for the same reason cache_payload() is: the mapping
    is the part worth testing, and it should be testable without Firestore.

    WHAT IS STORED, AND WHY EXACTLY THIS
    Enough to redraw the Home score card as it stood that day, and nothing that
    would need a second source of truth to interpret:

      date / computed_at
          The document ID repeated as a field (so the collection can be ordered
          and range-queried directly) plus the exact instant it came from. The
          two are not redundant: the ID is a UTC day, computed_at is the moment
          inside it that actually produced these numbers, and after a same-day
          overwrite it says which run survived.

      overall_score / overall_score_reason
          Copied verbatim, INCLUDING None. A null score is a real, meaningful
          day — the user's income was cleared, or only one domain was connected
          — and the reason code says which. Coercing it to 0.0 would draw a
          crash on the chart where the truth is a gap, so it is preserved as
          None and the reason travels with it.

      schema_version
          The shape these fields were written in, stamped at write time. A
          collection that accumulates for months will outlive at least one
          schema change, and a reader that cannot tell v1 rows from v2 rows has
          to guess from which keys are present.

      age_band
          Required, not optional decoration. banking_model.json maps cluster 0
          to "Building" in band 1 and to "Stable" in bands 2-5, so a tier index
          cannot be interpreted without knowing which band's model produced it.
          A snapshot carrying tier_label alone would be unreadable the moment
          the labels were re-worded — which the artifacts explicitly permit,
          since they declare the INDEX authoritative and the label display copy.

      domains
          Copied verbatim from the same projection the live cache uses
          (_domains_for_cache), so one renderer reads both.

      score_breakdown
          The whole map, verbatim. This is deliberately stored INSTEAD OF a
          "capping reason" enum, because no such enum exists anywhere in this
          system: the live Home caption "Score capped by your insurance gap" is
          Flutter copy assembled from the `capped` boolean, not backend data.
          Inventing a reason string here would create a second vocabulary that
          the live path does not use and could silently disagree with. Storing
          `capped`, `ceiling`, `insurance_indicators` and
          `insurance_indicators_cleared` instead lets a reader reconstruct
          exactly the same explanation, months later, with the same logic Home
          runs live. None on a day with no score, same as the live cache.

      net_worth_sgd
          financialProfile's own net_worth_sgd as the pipeline read it for this
          run — the figure Home's Net Worth card showed at that moment. Copied,
          never recomputed: data/financial_profile.py's compute_net_worth() is
          its only calculation. It is as fresh as the last derived-field
          refresh, which every Plaid sync and disconnect runs before the score;
          POST /score/refresh does not re-derive it first, so on that path it is
          the value already cached. Recorded even on a day with no score, since
          net worth needs neither income nor age.
    """
    return {
        "date": snapshot_date(result["computed_at"]),
        "computed_at": result["computed_at"],
        "overall_score": result["overall_score"],
        "overall_score_reason": result["overall_score_reason"],
        "schema_version": ML_PROFILE_SCHEMA_VERSION,
        "age_band": result["age_band"],
        "domains": _domains_for_cache(result),
        "score_breakdown": _breakdown_for_cache(result["score_breakdown"]),
        "net_worth_sgd": result["net_worth_sgd"],
    }


def write_snapshot(db, uid: str, result: dict) -> str:
    """Writes (or replaces) today's snapshot. Returns the document ID.

    LAST-WRITE-WINS, AND THIS IS THE ONE DELIBERATE DIVERGENCE FROM THE
    notificationEvents PRECEDENT — a product decision, not an oversight.
    notifications/service.py claims its event document create-if-absent, because
    that collection is an already-sent ledger where the FIRST write is the
    truth and a re-run must not send twice. This collection answers a different
    question: "what did this user's score look like on this day?" The honest
    answer is the day's LATEST computation, so a second refresh on the same day
    must replace the first rather than be discarded.

    Hence set() — a full overwrite, not update() and not create-if-absent.
    set() rather than update() so a snapshot never ends up half old and half
    new: if a future schema drops a field, the row must lose it too, not keep a
    stale copy alongside the new shape.
    """
    document_id = snapshot_date(result["computed_at"])
    history_collection(db, uid).document(document_id).set(snapshot_payload(result))
    return document_id


def _write_snapshot_nonfatal(db, uid: str, result: dict) -> None:
    """write_snapshot() with every exception swallowed.

    Wrapped whole, and for the same reason _notify_score_change_nonfatal is:
    this runs inside an operation the user is waiting on, and recording history
    is optional to that operation's success. A failed snapshot costs one day's
    point on a chart; a failed refresh costs the user their score.
    """
    try:
        write_snapshot(db, uid, result)
    except Exception:
        logger.exception("score history snapshot failed for %s", uid)


def read_score_history(db, uid: str, limit: int) -> list[dict]:
    """The most recent ``limit`` daily snapshots, newest first.

    Ordered on the ``date`` field rather than the document ID: they hold the
    same value, but ordering on a real field keeps the query readable and does
    not depend on __name__ ordering semantics. YYYY-MM-DD sorts
    lexicographically in calendar order, so a string sort is a date sort.
    """
    snaps = (history_collection(db, uid)
             .order_by("date", direction="DESCENDING")
             .limit(limit)
             .stream())
    return [s.to_dict() for s in snaps]


def read_cached_score(db, uid: str) -> dict | None:
    """
    The cached score, without recomputing — the fast path for an app open.

    Returns None when the user has no financialProfile document. Otherwise
    returns the four cached fields plus ``computed``: False means the pipeline
    has never run for this user, which is a normal state for a brand-new
    account, not an error.
    """
    snap = profile_ref(db, uid).get()
    if not snap.exists:
        return None
    profile = snap.to_dict()
    computed_at = profile.get("checklist_last_updated")
    return {
        "computed": computed_at is not None,
        "computed_at": computed_at,
        "overall_score": profile.get("overall_score"),
        "checklist_flags": profile.get("checklist_flags"),
        "ml_profile": profile.get("ml_profile"),
    }


def refresh_score_nonfatal(db, uid: str) -> dict | None:
    """
    refresh_score_profile() with every exception swallowed — the form the sync
    hooks use.

    Same contract and same reasoning as main.py's _refresh_profile_nonfatal:
    a score refresh is a derived-data update attached to somebody else's
    operation, and it must never be the reason a Plaid sync or a policy upload
    reports failure. The score catches up on the next trigger.

    Note this is broader than the pipeline's own "unavailable" handling: user
    data problems (no income, unsupported age) are already returned as normal
    results, so anything reaching this except block is a genuine fault —
    unreadable artifacts, a Firestore outage — and is logged with a traceback
    rather than silently discarded.
    """
    try:
        return refresh_score_profile(db, uid)
    except Exception:
        logger.exception("score refresh failed for %s", uid)
        return None
