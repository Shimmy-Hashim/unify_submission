"""Trigger rules — pure. No Firestore, no FCM, no clock of its own.

Every function here takes already-fetched documents (or an already-computed
pipeline result) and returns a list of Notification records describing what
SHOULD be sent. Nothing here decides whether the user wants it, whether it was
already sent, or how to deliver it — service.py owns all three.

Keeping it pure is deliberate: "does the premium reminder correctly identify a
policy 7 days from renewal" is then a table test over plain dicts, not an
end-to-end push with a real device attached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Notification kinds — the toggle keys they are gated on
# ---------------------------------------------------------------------------
# These four strings are the contract between the three surfaces that use them:
# the Flutter toggle rows, the notificationSettings document, and service.py's
# gating check. KIND_* is the machine key; never branch on the display copy.
KIND_RISK_ALERT = "risk_alerts"
KIND_PREMIUM_DUE = "premium_due_reminders"
# Displayed as "Score Change Alerts". The key keeps its original spelling on
# purpose: it is a stored Firestore field name, and renaming it would read as
# absent on every document already written and silently reset that user's
# choice back to the default. The display copy is free to change; this is not.
KIND_MONTHLY_SCORE = "monthly_score_update"

# The master switch. Not a kind that ever fires on its own — service.py refuses
# to send anything at all when it is off.
KIND_PUSH_MASTER = "push_notifications"

ALL_KINDS = (KIND_RISK_ALERT, KIND_PREMIUM_DUE, KIND_MONTHLY_SCORE)

# Default when a user has no notificationSettings document yet: master ON so a
# permission grant is not silently inert, every category ON. Nothing is
# delivered regardless until the OS permission is granted AND a device token
# exists, so "on by default" cannot produce a push the user never allowed.
DEFAULT_SETTINGS = {
    KIND_PUSH_MASTER: True,
    KIND_RISK_ALERT: True,
    KIND_PREMIUM_DUE: True,
    KIND_MONTHLY_SCORE: True,
}

# How far ahead of renewal_date a premium reminder fires.
PREMIUM_LEAD_DAYS = 7

# A score has to move by at least this much to be worth a push. Below it the
# change is pipeline noise (a cent of expenses shifting a ratio), and a phone
# buzzing for a 0.4-point move trains the user to turn the category off.
SCORE_CHANGE_EPSILON = 1.0


@dataclass(frozen=True)
class Notification:
    """One push that should be delivered, fully rendered.

    event_key is the idempotency key AND the notificationEvents document ID:
    deterministic, so re-running a job (or two syncs racing) writes the same
    document rather than sending twice. See service.py's create-if-absent.
    """

    kind: str
    title: str
    body: str
    event_key: str
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Score Change Alerts  (stored key: monthly_score_update)
# ---------------------------------------------------------------------------


def detect_score_change(
    previous_score: float | None,
    new_score: float | None,
    *,
    computed_at: datetime,
    epsilon: float = SCORE_CHANGE_EPSILON,
) -> list[Notification]:
    """Fires when overall_score actually changes value on a refresh.

    Deliberately NOT fired on the first score a user ever gets (previous is
    None): that is "your score is ready", a different message, and firing a
    "your score changed" push at a user who has never seen a score is wrong.
    Equally, a score going away (new is None — income cleared, artifacts
    unreadable) is not a score change worth a push.

    The trigger is the score cache write — the only moment the backend can
    observe a change — so notifications follow real activity (a sync or an
    upload recomputing the score) rather than any schedule. The toggle is
    called "Score Change Alerts" in the UI to match. It was originally
    "Monthly Score Update", which implied a calendar cadence nothing here
    provides; the display copy was corrected. The stored key is still
    ``monthly_score_update`` — see KIND_MONTHLY_SCORE.
    """
    if previous_score is None or new_score is None:
        return []
    delta = new_score - previous_score
    if abs(delta) < epsilon:
        return []

    direction = "up" if delta > 0 else "down"
    arrow = "↑" if delta > 0 else "↓"
    # Rounded for copy only; the exact values ride along in data for the client.
    return [
        Notification(
            kind=KIND_MONTHLY_SCORE,
            title=f"Your Financial Health Score moved {direction}",
            body=(
                f"{arrow} {abs(delta):.0f} point{'s' if round(abs(delta)) != 1 else ''} "
                f"— now {new_score:.0f}/100 (was {previous_score:.0f})."
            ),
            # One event per computed_at: two refreshes in the same second are
            # the same observation, a later one is a genuinely new change.
            event_key=f"score_change_{_stamp(computed_at)}",
            data={
                "type": KIND_MONTHLY_SCORE,
                "previous_score": f"{previous_score:.2f}",
                "new_score": f"{new_score:.2f}",
                "delta": f"{delta:.2f}",
            },
        )
    ]


# ---------------------------------------------------------------------------
# Risk Alerts
# ---------------------------------------------------------------------------
# Two kinds of meaningful negative change, per the task's definition:
#   1. a domain tier moving to the weaker tier (Building / Getting Started /
#      Not Covered)
#   2. a new coverage gap appearing
#
# tier_index is the comparator, not the label: the artifacts state the index is
# authoritative (0 = healthiest) and the labels are display wording that can be
# re-worded without a model change. Comparing labels would make a copy edit look
# like a tier movement.


def detect_risk_alerts(
    previous_ml_profile: dict | None,
    new_result: dict,
    *,
    previous_gaps: set[str] | None = None,
    new_gaps: set[str] | None = None,
) -> list[Notification]:
    """Tier degradations and newly-appeared coverage gaps, as pushes.

    previous_ml_profile is the cached ml_profile map from before this refresh
    (None for a user who has never been scored). new_result is the pipeline
    result dict from this refresh.

    A first-ever assessment never alerts. "Your Insurance is Not Covered" is
    true on day one for almost every new user, and delivering it as a *risk
    alert* would frame a starting state as a regression. The Home screen is
    where a standing state belongs; this channel is for changes.
    """
    alerts: list[Notification] = []
    computed_at = new_result.get("computed_at")
    previous_domains = ((previous_ml_profile or {}).get("domains") or {})

    for domain, new_domain in (new_result.get("domains") or {}).items():
        previous_index = (previous_domains.get(domain) or {}).get("tier_index")
        new_index = new_domain.get("tier_index")
        # Only a real move between two known tiers counts. Going from assessed
        # to not_assessed (new_index None) is a data-availability change — a
        # disconnected bank, not a financial decline — and must not alert.
        if previous_index is None or new_index is None:
            continue
        if new_index <= previous_index:
            continue

        label = new_domain.get("tier_label") or "a weaker tier"
        previous_label = (previous_domains.get(domain) or {}).get("tier_label")
        alerts.append(
            Notification(
                kind=KIND_RISK_ALERT,
                title=f"{domain.capitalize()} needs attention",
                body=(
                    f"Your {domain} tier moved from {previous_label} to {label}. "
                    "Open Unify to see what changed."
                ),
                event_key=(
                    f"tier_drop_{domain}_{previous_index}_{new_index}_"
                    f"{_stamp(computed_at)}"
                ),
                data={
                    "type": KIND_RISK_ALERT,
                    "subtype": "tier_drop",
                    "domain": domain,
                    "previous_tier_index": str(previous_index),
                    "new_tier_index": str(new_index),
                    "new_tier_label": str(label),
                },
            )
        )

    # Coverage gaps are computed by the insurance parser onto each policy, so a
    # "new gap" is a gap string present now and absent before. Compared as sets
    # of strings because that is exactly how the field is stored; both sides
    # default to empty rather than None so a first upload (previous empty) does
    # surface its gaps — unlike tiers, a gap IS news the first time, since it
    # only exists once the user has uploaded a policy for it to be computed
    # against.
    previous_gaps = previous_gaps or set()
    new_gaps = new_gaps or set()
    appeared = sorted(new_gaps - previous_gaps)
    if appeared:
        alerts.append(
            Notification(
                kind=KIND_RISK_ALERT,
                title="New coverage gap found",
                body=(
                    appeared[0]
                    if len(appeared) == 1
                    else f"{appeared[0]} (+{len(appeared) - 1} more)"
                ),
                event_key=f"coverage_gap_{_digest(appeared)}",
                data={
                    "type": KIND_RISK_ALERT,
                    "subtype": "coverage_gap",
                    "gaps": " | ".join(appeared),
                },
            )
        )
    return alerts


def coverage_gaps_of(policies: list[dict] | None) -> set[str]:
    """Every coverage_gaps entry across a user's policies, flattened.

    Tolerant on purpose: the field is written by the Gemini parser, so a
    missing key, a null, or a stray non-string entry are all real possibilities
    and none of them should crash a notification check attached to somebody
    else's upload.
    """
    gaps: set[str] = set()
    for policy in policies or []:
        for gap in (policy or {}).get("coverage_gaps") or []:
            if isinstance(gap, str) and gap.strip():
                gaps.add(gap.strip())
    return gaps


# ---------------------------------------------------------------------------
# Premium Due Reminders
# ---------------------------------------------------------------------------


def find_due_premium_reminders(
    policies: list[dict],
    *,
    now: datetime,
    lead_days: int = PREMIUM_LEAD_DAYS,
) -> list[Notification]:
    """Policies renewing within `lead_days`, as reminder pushes.

    WINDOW CHOICE — "within 7 days", not "exactly 7 days out". The literal
    reading (fire on the day renewal_date - 7 falls) only works if the job runs
    every single day. This project has no always-on scheduler (see
    run_premium_reminders.py) — the backend is a laptop-run uvicorn process, so
    a day WILL be missed, and a strict equality check would silently drop that
    policy's only reminder for the year. A range window plus the per-renewal
    dedupe marker in service.py gives the same one-reminder-per-renewal
    behaviour while surviving a missed day: the first run inside the window
    sends, later runs find the marker and skip.

    Already-past renewals are excluded — a reminder for a date that has gone is
    noise, not a reminder.
    """
    today = now.astimezone(timezone.utc).date()
    horizon = today + timedelta(days=lead_days)
    due: list[Notification] = []

    for policy in policies or []:
        renewal = _as_date(policy.get("renewal_date"))
        if renewal is None:
            continue  # parser returns null renewal_date often enough to expect it
        if not (today <= renewal <= horizon):
            continue

        days_left = (renewal - today).days
        name = policy.get("policy_name") or "Your policy"
        insurer = policy.get("insurer")
        premium = policy.get("annual_premium_sgd")

        due.append(
            Notification(
                kind=KIND_PREMIUM_DUE,
                title="Premium due soon",
                body=_premium_body(name, insurer, premium, days_left, renewal),
                # Keyed by policy AND renewal date: next year's renewal of the
                # same policy is a different event and must be allowed to fire.
                event_key=(
                    f"premium_due_{policy.get('policy_id') or name}_"
                    f"{renewal.isoformat()}"
                ),
                data={
                    "type": KIND_PREMIUM_DUE,
                    "policy_id": str(policy.get("policy_id") or ""),
                    "policy_name": str(name),
                    "renewal_date": renewal.isoformat(),
                    "days_left": str(days_left),
                },
            )
        )
    return due


def _premium_body(
    name: str,
    insurer: str | None,
    premium: float | None,
    days_left: int,
    renewal: date,
) -> str:
    if days_left == 0:
        when = "renews today"
    elif days_left == 1:
        when = "renews tomorrow"
    else:
        when = f"renews in {days_left} days"
    who = f" ({insurer})" if insurer else ""
    amount = ""
    if isinstance(premium, (int, float)) and premium > 0 and math.isfinite(premium):
        amount = f" — S${premium:,.0f} annual premium"
    return f"{name}{who} {when}, on {renewal.strftime('%d %b %Y')}{amount}."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_date(value: object) -> date | None:
    """renewal_date as a plain UTC date.

    Firestore hands back a DatetimeWithNanoseconds (a datetime subclass); the
    verification scripts and pure tests pass a datetime, a date, or an ISO
    string. All four are accepted rather than requiring callers to normalise,
    because the same rule is exercised from all four directions.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return _as_date(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _stamp(moment: object) -> str:
    """A computed_at as a compact, filename-and-document-ID-safe key part."""
    parsed = _as_datetime(moment)
    if parsed is None:
        return "unknown"
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return _as_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _digest(parts: list[str]) -> str:
    """Short stable hash of a gap list — the gap strings themselves are far too
    long (and contain characters Firestore rejects) to key a document by."""
    import hashlib

    joined = "|".join(parts).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:16]
