"""Delivery — the I/O half of notifications. Firestore in, FCM out.

WHERE THINGS LIVE (all three are new, all three are subcollections of
users/{uid} and therefore already covered by the existing owner-only Firestore
rule — no rules change was needed for this pass):

    users/{uid}/notificationSettings/preferences   the four toggles
    users/{uid}/notificationTokens/{token}         one doc per device
    users/{uid}/notificationEvents/{event_key}     what was already sent

WHY A TOKEN SUBCOLLECTION AND NOT users/{uid}.fcmToken
A single field holds one device. Installing Unify on a second device would
overwrite the first device's token, and the first device would then silently
stop receiving anything — with nothing anywhere to show why. Tokens also rotate
(reinstall, app-data clear, FCM's own refresh), so the field would need pruning
logic regardless. A subcollection keyed BY the token gives multi-device support,
free deduplication (re-registering the same token rewrites the same document),
per-device metadata for debugging, and per-token pruning when FCM reports one
dead. It also keeps users/{uid} exactly the seven-field profile document
Section 8 specifies, which is the same reason AppPreferences kept theme and
balance-masking out of it.

WHY THE TOGGLES ARE IN FIRESTORE AND NOT shared_preferences
All four gate sends that are decided on the SERVER — the premium-due job, the
score-change hook, and the risk-alert hook all run in the backend and must read
the user's choice. shared_preferences is device-local and unreadable from
Python, so it can only ever hold a purely client-side preference. None of these
four is one. (Theme and Hide Balances, which ARE purely client-side, correctly
stay in shared_preferences — see services/app_preferences.dart.)
"""
from __future__ import annotations

import logging

from firebase_admin import exceptions as fb_exceptions
from firebase_admin import messaging
from google.api_core import exceptions as gcloud_exceptions
from google.cloud import firestore as gcloud_firestore

from . import rules

logger = logging.getLogger(__name__)

SETTINGS_COLLECTION = "notificationSettings"
SETTINGS_DOC_ID = "preferences"
STATE_DOC_ID = "state"
TOKENS_COLLECTION = "notificationTokens"
EVENTS_COLLECTION = "notificationEvents"

# Android needs a channel that exists on the device; the Flutter side creates
# one with this exact ID and declares it as the manifest default.
ANDROID_CHANNEL_ID = "unify_alerts"


# ---------------------------------------------------------------------------
# Document references
# ---------------------------------------------------------------------------


def settings_ref(db, uid: str):
    return (db.collection("users").document(uid)
            .collection(SETTINGS_COLLECTION).document(SETTINGS_DOC_ID))


def state_ref(db, uid: str):
    """Backend-owned scratch state, alongside the client-owned preferences doc.

    OWNERSHIP SPLIT inside notificationSettings, worth stating because both
    documents sit in one collection: `preferences` is written by the Flutter
    toggles and only ever read here; `state` is written only here and never by
    the client. (The Firestore rule grants the owner write access to every
    subcollection, so this is a convention the code keeps, not something the
    rules enforce — the same posture as every other backend-written collection
    outside plaidItems.)
    """
    return (db.collection("users").document(uid)
            .collection(SETTINGS_COLLECTION).document(STATE_DOC_ID))


def tokens_collection(db, uid: str):
    return db.collection("users").document(uid).collection(TOKENS_COLLECTION)


def events_collection(db, uid: str):
    return db.collection("users").document(uid).collection(EVENTS_COLLECTION)


# ---------------------------------------------------------------------------
# Settings + tokens
# ---------------------------------------------------------------------------


def read_settings(db, uid: str) -> dict:
    """The user's four toggles, with defaults filled in for anything absent.

    A user who has never opened the Notifications section has no document; that
    is the DEFAULT_SETTINGS case, not an error. Merging per-key (rather than
    returning defaults wholesale when the doc is missing) also means a document
    written by an older client that only knew three toggles still gets a sane
    value for the fourth.
    """
    snap = settings_ref(db, uid).get()
    stored = snap.to_dict() if snap.exists else {}
    return {
        key: bool(stored.get(key, default))
        for key, default in rules.DEFAULT_SETTINGS.items()
    }


def is_enabled(settings: dict, kind: str) -> bool:
    """Master switch AND the per-category switch. Both must be on.

    The master is checked here rather than at each call site so there is exactly
    one place that can get it wrong.
    """
    return bool(settings.get(rules.KIND_PUSH_MASTER)) and bool(settings.get(kind))


def device_tokens(db, uid: str) -> list[str]:
    """Every registered device token for the user. Document ID IS the token."""
    return [snap.id for snap in tokens_collection(db, uid).stream()]


def register_token(db, uid: str, token: str, platform: str = "unknown") -> None:
    """Upsert one device token. Used by the verification script; the Flutter
    client writes the identical shape directly (it owns the document under the
    existing owner-write rule, so no endpoint sits in between)."""
    tokens_collection(db, uid).document(token).set(
        {
            "token": token,
            "platform": platform,
            "last_seen_at": gcloud_firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )


def read_known_gaps(db, uid: str) -> set[str]:
    """Coverage gaps the user has already been alerted about.

    Needed because "a NEW coverage gap appeared" cannot be derived from the
    policy documents alone — one score refresh reads the current gaps, and the
    current gaps are all it can see. The previous set has to have been recorded
    at the time, so it lives here.
    """
    snap = state_ref(db, uid).get()
    stored = (snap.to_dict() or {}) if snap.exists else {}
    return {g for g in stored.get("known_coverage_gaps") or [] if isinstance(g, str)}


def record_known_gaps(db, uid: str, gaps: set[str]) -> None:
    """Replace (not union) the recorded gap set with what is true right now.

    Replacing matters: a gap the user has since closed must drop out, so that
    if it ever reappears it is genuinely new again and alerts a second time.
    """
    state_ref(db, uid).set(
        {"known_coverage_gaps": sorted(gaps),
         "updated_at": gcloud_firestore.SERVER_TIMESTAMP},
        merge=True,
    )


def _prune_tokens(db, uid: str, dead: list[str]) -> None:
    """Drop tokens FCM has told us are dead. Best-effort — a failure to prune
    only costs a wasted send next time."""
    for token in dead:
        try:
            tokens_collection(db, uid).document(token).delete()
        except Exception:
            logger.warning("could not prune dead token for %s", uid, exc_info=True)


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------


def deliver(
    db,
    uid: str,
    notifications: list[rules.Notification] | None,
    *,
    dry_run: bool = False,
) -> dict:
    """Gate, deduplicate, send, record. The single entry point for every trigger.

    Returns a summary rather than raising, because every caller is a hook
    hanging off somebody else's operation (a Plaid sync, a policy upload, a
    nightly job) and none of them may fail because a push did not go out.

    ORDER — claim, then send, then record the outcome:
      1. create() the event document. create() fails if it already exists, so
         two racing syncs cannot both claim the same event_key; the loser skips.
         Claiming BEFORE sending is what makes a double-send impossible.
      2. send. If it raises, the claim is deleted again so the next run can
         retry — a claim left behind after a failed send would silently
         suppress the notification forever.
      3. update the claim with the delivery counts, for audit.
    """
    summary = {"sent": 0, "skipped_disabled": 0, "skipped_duplicate": 0,
               "skipped_no_token": 0, "failed": 0, "events": []}
    notifications = list(notifications or [])
    if not notifications:
        return summary

    settings = read_settings(db, uid)
    tokens = device_tokens(db, uid)

    for note in notifications:
        if not is_enabled(settings, note.kind):
            summary["skipped_disabled"] += 1
            continue
        if not tokens:
            # No device has registered — nothing to deliver to. Deliberately NOT
            # recorded as an event: the user may register a device tomorrow, and
            # a claim written now would suppress the retry.
            summary["skipped_no_token"] += 1
            continue

        claim = events_collection(db, uid).document(note.event_key)
        try:
            claim.create({
                "kind": note.kind,
                "title": note.title,
                "body": note.body,
                "data": note.data,
                "sent_at": gcloud_firestore.SERVER_TIMESTAMP,
                "status": "sending",
            })
        except gcloud_exceptions.AlreadyExists:
            summary["skipped_duplicate"] += 1
            continue

        try:
            delivered, dead = _send(tokens, note, dry_run=dry_run)
        except Exception:
            logger.exception("FCM send failed for %s (%s)", uid, note.event_key)
            # Release the claim so this is retried rather than lost.
            try:
                claim.delete()
            except Exception:
                logger.warning("could not release claim %s", note.event_key,
                               exc_info=True)
            summary["failed"] += 1
            continue

        if dead:
            _prune_tokens(db, uid, dead)
            tokens = [t for t in tokens if t not in dead]

        claim.update({"status": "sent", "delivered_count": delivered,
                      "pruned_tokens": len(dead)})
        summary["sent"] += 1
        summary["events"].append(note.event_key)

    return summary


def _send(
    tokens: list[str], note: rules.Notification, *, dry_run: bool
) -> tuple[int, list[str]]:
    """Fan one Notification out to every device token.

    send_each (not send_all/multicast) so one dead token cannot fail the batch
    for the user's other devices, and so each token's own error is visible —
    that per-token error is what drives pruning.
    """
    messages = [
        messaging.Message(
            token=token,  # registration token; `fid` in the SDK's newer API
            notification=messaging.Notification(title=note.title, body=note.body),
            data={k: str(v) for k, v in (note.data or {}).items()},
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id=ANDROID_CHANNEL_ID),
            ),
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(sound="default"))),
        )
        for token in tokens
    ]
    batch = messaging.send_each(messages, dry_run=dry_run)

    dead: list[str] = []
    for token, response in zip(tokens, batch.responses):
        if response.success:
            continue
        exc = response.exception
        # UnregisteredError  — app uninstalled or token rotated: prune.
        # SenderIdMismatch   — token belongs to another Firebase project: prune.
        # InvalidArgument    — malformed token string: prune.
        # Anything else (quota, transient 5xx) is NOT a reason to drop a
        # perfectly good token.
        if isinstance(exc, (messaging.UnregisteredError,
                            messaging.SenderIdMismatchError,
                            fb_exceptions.InvalidArgumentError)):
            dead.append(token)
        else:
            logger.warning("FCM send to one device failed: %s", exc)
    return batch.success_count, dead
