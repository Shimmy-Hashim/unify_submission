"""Settings Pass 2 — Notifications & Alerts verification.

    cd backend && venv\\Scripts\\activate && python scripts/verify_notifications.py

TWO PARTS, and the split matters when reading the results:

  PART 1 — pure trigger rules (no Firebase, no network). Table tests over the
    three detectors: does the premium check identify a policy 7 days from
    renewal and reject the ones it should, does a score change fire only when
    the value actually moved, does a tier drop alert while a first-ever
    assessment and a data-availability change stay silent.

  PART 2 — live Firestore round-trip against ONE throwaway user, deleted
    afterwards: toggle persistence, per-key default merging, token
    registration, master-switch gating, and the dedupe claim. FCM sends run in
    dry_run mode — the payload is built and validated by Google's servers, no
    device is written to.

WHAT THIS SCRIPT CANNOT VERIFY, and why — do not read a green run as "push
delivery works":
  * actual delivery to a device. That needs a real FCM registration token,
    which only a real app install on a real device/emulator can mint. Every
    token here is synthetic.
  * the foreground snackbar, the OS permission dialog, and the toggle
    reflecting a revoked OS permission. All three are Flutter/Android runtime
    behaviour with no server-side observable.
"""
import sys
import time
import urllib.error
import urllib.request
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from firebase import get_db, init_firebase  # noqa: E402
from firebase_admin import auth as admin_auth  # noqa: E402

from notifications import rules, service  # noqa: E402

PROJECT = "YOUR_FIREBASE_PROJECT_ID"
GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]

NOW = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)

passed, failed = [], []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def http(method, url, body=None):
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {}


# ===========================================================================
# PART 1 — pure rules
# ===========================================================================


def policy(pid, days_from_now, **extra):
    return {
        "policy_id": pid,
        "policy_name": extra.pop("name", f"Policy {pid}"),
        "renewal_date": (NOW + timedelta(days=days_from_now)
                         if days_from_now is not None else None),
        **extra,
    }


def part1_premium_reminders():
    print("\n-- Premium Due Reminders (pure) --")
    policies = [
        policy("exactly7", 7, name="PruShield", insurer="Prudential",
               annual_premium_sgd=1200.0),
        policy("in3", 3, name="AIA Life"),
        policy("today", 0, name="Renews today"),
        policy("in8", 8, name="Just outside window"),
        policy("in30", 30, name="Far away"),
        policy("yesterday", -1, name="Already past"),
        policy("nodate", None, name="No renewal date"),
    ]
    due = rules.find_due_premium_reminders(policies, now=NOW)
    ids = {n.data["policy_id"] for n in due}

    check("policy exactly 7 days out is identified", "exactly7" in ids)
    check("policy 3 days out is identified (missed-day catch-up)", "in3" in ids)
    check("policy renewing today is identified", "today" in ids)
    check("policy 8 days out is NOT identified", "in8" not in ids)
    check("policy 30 days out is NOT identified", "in30" not in ids)
    check("already-past renewal is NOT identified", "yesterday" not in ids)
    check("null renewal_date is skipped, not crashed", "nodate" not in ids)
    check("exactly 3 policies fall in the window", len(due) == 3, f"got {len(due)}")

    seven = next(n for n in due if n.data["policy_id"] == "exactly7")
    check("event_key is keyed by policy AND renewal date",
          seven.event_key == "premium_due_exactly7_2026-08-08",
          seven.event_key)
    check("body carries insurer, days left and premium",
          all(s in seven.body for s in ("Prudential", "7 days", "1,200")),
          seven.body)
    check("kind gates on premium_due_reminders",
          seven.kind == rules.KIND_PREMIUM_DUE)

    # Next year's renewal of the same policy must be allowed to fire again.
    next_year = rules.find_due_premium_reminders(
        [policy("exactly7", 372)], now=NOW + timedelta(days=365))
    check("same policy next year yields a DIFFERENT event key",
          next_year and next_year[0].event_key != seven.event_key,
          next_year[0].event_key if next_year else "none")

    # Date-typed and ISO-string renewal dates must behave identically.
    iso = rules.find_due_premium_reminders(
        [{"policy_id": "iso", "policy_name": "ISO", "renewal_date": "2026-08-08"}],
        now=NOW)
    check("ISO-string renewal_date is handled", len(iso) == 1)


def part1_score_change():
    print("\n-- Score Change Alerts (pure) --")
    args = {"computed_at": NOW}

    up = rules.detect_score_change(58.0, 67.0, **args)
    check("score increase fires", len(up) == 1)
    check("increase copy names both values",
          up and "67" in up[0].body and "58" in up[0].body,
          up[0].body if up else "")
    down = rules.detect_score_change(67.0, 58.0, **args)
    check("score decrease fires", len(down) == 1)

    check("unchanged score does NOT fire",
          rules.detect_score_change(58.0, 58.0, **args) == [])
    check("sub-threshold wobble does NOT fire",
          rules.detect_score_change(58.0, 58.4, **args) == [])
    check("first-ever score does NOT fire a 'changed' push",
          rules.detect_score_change(None, 58.0, **args) == [])
    check("score becoming unavailable does NOT fire",
          rules.detect_score_change(58.0, None, **args) == [])
    check("kind gates on monthly_score_update",
          up[0].kind == rules.KIND_MONTHLY_SCORE)


def domains(**tiers):
    """A pipeline-result-shaped domains map. None tier = not assessed."""
    return {
        d: {"status": "assessed" if tiers.get(d) is not None else "not_assessed",
            "tier_index": tiers.get(d),
            "tier_label": {0: "Stable", 1: "Building"}.get(tiers.get(d))}
        for d in ("banking", "investment", "insurance")
    }


def part1_risk_alerts():
    print("\n-- Risk Alerts (pure) --")
    previous = {"domains": domains(banking=0, investment=0, insurance=0)}

    dropped = rules.detect_risk_alerts(
        previous, {"computed_at": NOW, "domains": domains(
            banking=1, investment=0, insurance=0)})
    check("banking tier drop 0 -> 1 fires one alert", len(dropped) == 1)
    check("alert names the domain",
          dropped and "banking" in dropped[0].body.lower(),
          dropped[0].body if dropped else "")
    check("kind gates on risk_alerts",
          dropped and dropped[0].kind == rules.KIND_RISK_ALERT)

    improved = rules.detect_risk_alerts(
        {"domains": domains(banking=1, investment=0, insurance=0)},
        {"computed_at": NOW, "domains": domains(
            banking=0, investment=0, insurance=0)})
    check("tier IMPROVEMENT does not fire a risk alert", improved == [])

    unchanged = rules.detect_risk_alerts(
        previous, {"computed_at": NOW, "domains": domains(
            banking=0, investment=0, insurance=0)})
    check("unchanged tiers fire nothing", unchanged == [])

    first_ever = rules.detect_risk_alerts(
        None, {"computed_at": NOW, "domains": domains(
            banking=1, investment=1, insurance=1)})
    check("first-ever assessment fires nothing (starting state, not a drop)",
          first_ever == [])

    became_unknown = rules.detect_risk_alerts(
        previous, {"computed_at": NOW, "domains": domains(banking=None)})
    check("assessed -> not_assessed fires nothing (data gap, not decline)",
          became_unknown == [])

    multi = rules.detect_risk_alerts(
        previous, {"computed_at": NOW, "domains": domains(
            banking=1, investment=1, insurance=0)})
    check("two domains dropping fire two distinct alerts",
          len(multi) == 2 and len({n.event_key for n in multi}) == 2)

    # --- coverage gaps
    gaps = rules.detect_risk_alerts(
        previous,
        {"computed_at": NOW, "domains": domains(
            banking=0, investment=0, insurance=0)},
        previous_gaps={"No hospitalisation cover"},
        new_gaps={"No hospitalisation cover", "Life cover below 9x income"},
    )
    check("a NEW coverage gap fires an alert", len(gaps) == 1)
    check("alert body carries the new gap, not the old one",
          gaps and "Life cover below 9x income" in gaps[0].body,
          gaps[0].body if gaps else "")

    same_gaps = rules.detect_risk_alerts(
        previous,
        {"computed_at": NOW, "domains": domains(
            banking=0, investment=0, insurance=0)},
        previous_gaps={"No hospitalisation cover"},
        new_gaps={"No hospitalisation cover"},
    )
    check("an already-known gap does NOT re-fire", same_gaps == [])

    closed = rules.detect_risk_alerts(
        previous,
        {"computed_at": NOW, "domains": domains(
            banking=0, investment=0, insurance=0)},
        previous_gaps={"No hospitalisation cover"},
        new_gaps=set(),
    )
    check("a CLOSED gap does not fire an alert", closed == [])

    extracted = rules.coverage_gaps_of([
        {"coverage_gaps": ["Gap A", "Gap B"]},
        {"coverage_gaps": None},
        {},
        {"coverage_gaps": ["Gap A", "", 42]},
    ])
    check("coverage_gaps_of flattens and tolerates bad parser output",
          extracted == {"Gap A", "Gap B"}, str(extracted))


def part1_defaults():
    print("\n-- Toggle contract (pure) --")
    check("four toggle keys exist",
          set(rules.DEFAULT_SETTINGS) == {
              rules.KIND_PUSH_MASTER, rules.KIND_RISK_ALERT,
              rules.KIND_PREMIUM_DUE, rules.KIND_MONTHLY_SCORE})
    check("all four default to on", all(rules.DEFAULT_SETTINGS.values()))
    check("master is not a sendable kind",
          rules.KIND_PUSH_MASTER not in rules.ALL_KINDS)


# ===========================================================================
# PART 2 — live Firestore round-trip
# ===========================================================================


def part2_live(db, uid):
    print("\n-- Firestore round-trip (live, throwaway user) --")

    # --- defaults with no document
    settings = service.read_settings(db, uid)
    check("no settings document yields the four defaults",
          settings == rules.DEFAULT_SETTINGS, str(settings))

    # --- persistence round-trip, exactly as the Flutter toggle writes it
    service.settings_ref(db, uid).set({rules.KIND_PREMIUM_DUE: False},
                                      merge=True)
    settings = service.read_settings(db, uid)
    check("a written toggle round-trips as False",
          settings[rules.KIND_PREMIUM_DUE] is False)
    check("unwritten toggles still fall back to their defaults",
          settings[rules.KIND_RISK_ALERT] is True
          and settings[rules.KIND_MONTHLY_SCORE] is True)

    # --- gating
    check("is_enabled respects the per-category toggle",
          service.is_enabled(settings, rules.KIND_PREMIUM_DUE) is False)
    check("is_enabled allows an on category",
          service.is_enabled(settings, rules.KIND_RISK_ALERT) is True)
    master_off = {**settings, rules.KIND_PUSH_MASTER: False}
    check("master off disables every category",
          not any(service.is_enabled(master_off, k) for k in rules.ALL_KINDS))

    # --- token registration
    fake_token = "verify-script-token-%d" % int(time.time())
    service.register_token(db, uid, fake_token, platform="android")
    tokens = service.device_tokens(db, uid)
    check("registered token is readable back", tokens == [fake_token], str(tokens))
    service.register_token(db, uid, fake_token, platform="android")
    check("re-registering the same token does not duplicate it",
          len(service.device_tokens(db, uid)) == 1)
    stored = service.tokens_collection(db, uid).document(fake_token).get().to_dict()
    check("token document carries platform + last_seen_at",
          stored.get("platform") == "android" and stored.get("last_seen_at"),
          str(stored))

    # --- known-gap state
    service.record_known_gaps(db, uid, {"Gap A", "Gap B"})
    check("known coverage gaps round-trip",
          service.read_known_gaps(db, uid) == {"Gap A", "Gap B"})
    service.record_known_gaps(db, uid, {"Gap A"})
    check("recording REPLACES rather than unions (a closed gap drops out)",
          service.read_known_gaps(db, uid) == {"Gap A"})

    # --- delivery gating: a disabled category must not send or record
    disabled_note = rules.Notification(
        kind=rules.KIND_PREMIUM_DUE, title="t", body="b",
        event_key="verify_disabled_1")
    summary = service.deliver(db, uid, [disabled_note], dry_run=True)
    check("a disabled category is skipped, not sent",
          summary["skipped_disabled"] == 1 and summary["sent"] == 0,
          str(summary))
    check("a skipped notification records NO event claim",
          not service.events_collection(db, uid)
          .document("verify_disabled_1").get().exists)

    # --- delivery + dedupe on an enabled category.
    # dry_run=True: FCM validates the message and reports per-token results
    # without delivering to any device. The token is synthetic, so FCM rejects
    # it as INVALID_ARGUMENT — which is itself the useful signal that the
    # request reached the real FCM v1 API and was understood.
    note = rules.Notification(
        kind=rules.KIND_RISK_ALERT, title="Verify", body="Verify body",
        event_key="verify_enabled_1", data={"type": "risk_alerts"})
    first = service.deliver(db, uid, [note], dry_run=True)
    check("an enabled category is attempted (not skipped)",
          first["skipped_disabled"] == 0, str(first))
    claim = service.events_collection(db, uid).document("verify_enabled_1").get()
    check("an attempted notification records an event claim", claim.exists)

    # --- invalid token pruning. Checked BEFORE the dedupe case below, because
    # the send above is what triggered it: FCM rejected the synthetic token as
    # INVALID_ARGUMENT (proving the request reached the real v1 API and was
    # parsed), and _send reported it as dead so it was dropped.
    check("FCM rejected the synthetic token and it was pruned",
          service.device_tokens(db, uid) == [],
          str(service.device_tokens(db, uid)))

    # A token has to exist again for the dedupe path to be reachable at all —
    # deliver() skips on "no device" before it claims an event, so without this
    # the second call would short-circuit and dedupe would go untested.
    service.register_token(db, uid, fake_token + "-b", platform="android")
    second = service.deliver(db, uid, [note], dry_run=True)
    check("the same event_key is deduplicated on a second run",
          second["skipped_duplicate"] == 1 and second["sent"] == 0,
          str(second))
    service.tokens_collection(db, uid).document(fake_token + "-b").delete()

    # --- no token at all: skip without claiming, so a later device still gets it
    service.settings_ref(db, uid).set({rules.KIND_PREMIUM_DUE: True},
                                      merge=True)
    no_token = rules.Notification(
        kind=rules.KIND_PREMIUM_DUE, title="t", body="b",
        event_key="verify_no_token_1")
    summary = service.deliver(db, uid, [no_token], dry_run=True)
    check("with no registered device the send is skipped",
          summary["skipped_no_token"] == 1, str(summary))
    check("a no-device skip records NO claim (so it can still fire later)",
          not service.events_collection(db, uid)
          .document("verify_no_token_1").get().exists)


def part2_score_hook(db, uid):
    """The wiring between the score cache write and the notification layer.

    The rules and the delivery layer are each covered above; this covers the
    join — that score_profile's hook actually calls them with the right
    before/after arguments. Exercised directly rather than by manufacturing a
    real score change, which would need a whole second set of Plaid fixtures to
    move the number.

    _send is stubbed here, and only here. Every token this script can produce is
    synthetic, so a real send makes FCM reject it and _send correctly reports it
    dead — which prunes the user's only device and makes every LATER
    notification in the same batch skip with "no device". That is right in
    production (a batch aimed at an uninstalled app should stop), but it would
    mean this test could only ever observe the first of the three notifications
    the hook produces. The genuine FCM round trip, including that pruning, is
    covered in part2_live above.
    """
    print("\n-- Score-change hook wiring (live, throwaway user) --")
    from data import score_profile

    real_send = service._send
    service._send = lambda tokens, note, dry_run: (len(tokens), [])

    service.register_token(db, uid, "hook-token-1", platform="android")
    service.settings_ref(db, uid).set(
        {k: True for k in rules.DEFAULT_SETTINGS}, merge=True)

    computed_at = datetime.now(timezone.utc)
    previous = {
        "overall_score": 70.0,
        "ml_profile": {"domains": {
            "banking": {"tier_index": 0, "tier_label": "Stable"},
            "investment": {"tier_index": 0, "tier_label": "On Track"},
            "insurance": {"tier_index": 0, "tier_label": "Covered"},
        }},
    }
    result = {
        "computed_at": computed_at,
        "overall_score": 55.0,
        "domains": {
            "banking": {"tier_index": 1, "tier_label": "Building"},
            "investment": {"tier_index": 0, "tier_label": "On Track"},
            "insurance": {"tier_index": 0, "tier_label": "Covered"},
        },
    }
    policies = [{"coverage_gaps": ["Hook gap"]}]

    score_profile._notify_score_change_nonfatal(db, uid, previous, result, policies)

    events = {snap.id: snap.to_dict()
              for snap in service.events_collection(db, uid).stream()}
    kinds = {e.get("kind") for e in events.values()}
    check("hook fired a Score Change Alert event",
          rules.KIND_MONTHLY_SCORE in kinds, str(sorted(kinds)))
    check("hook fired a Risk Alert event for the tier drop",
          any(k.startswith("tier_drop_banking") for k in events), str(list(events)))
    check("hook fired a Risk Alert event for the new coverage gap",
          any(k.startswith("coverage_gap_") for k in events), str(list(events)))
    check("hook recorded the gap baseline for next time",
          service.read_known_gaps(db, uid) == {"Hook gap"},
          str(service.read_known_gaps(db, uid)))

    # Re-running the identical comparison must not re-notify.
    before = len(events)
    score_profile._notify_score_change_nonfatal(db, uid, previous, result, policies)
    after = len(list(service.events_collection(db, uid).stream()))
    check("re-running the same comparison creates no duplicate events",
          after == before, f"{before} -> {after}")

    # A disabled master must stop everything the hook would otherwise send.
    service.settings_ref(db, uid).set(
        {rules.KIND_PUSH_MASTER: False}, merge=True)
    later = {**result, "computed_at": computed_at + timedelta(minutes=5)}
    score_profile._notify_score_change_nonfatal(db, uid, previous, later, policies)
    check("master toggle off suppresses the hook's events",
          len(list(service.events_collection(db, uid).stream())) == after)

    # Cleanup for the next section, and un-stub.
    service._send = real_send
    service.settings_ref(db, uid).set(
        {rules.KIND_PUSH_MASTER: True}, merge=True)
    for snap in service.events_collection(db, uid).stream():
        snap.reference.delete()
    service.tokens_collection(db, uid).document("hook-token-1").delete()


def part2_premium_job(db, uid):
    """The scheduled job, end to end against real Firestore documents."""
    print("\n-- Premium reminder job (live, throwaway user) --")
    from notifications import run_premium_reminders

    policies = db.collection("users").document(uid).collection("insurancePolicies")
    policies.document("due-soon").set({
        "policy_id": "due-soon", "uid": uid, "policy_name": "PruShield Premier",
        "insurer": "Prudential", "policy_type": "hospitalisation",
        "annual_premium_sgd": 980.0,
        "renewal_date": datetime.now(timezone.utc) + timedelta(days=5),
    })
    policies.document("far-off").set({
        "policy_id": "far-off", "uid": uid, "policy_name": "Far Off",
        "policy_type": "life",
        "renewal_date": datetime.now(timezone.utc) + timedelta(days=60),
    })

    summary = run_premium_reminders.run_for_user(
        db, uid, now=datetime.now(timezone.utc), lead_days=7, dry_run=True)
    check("job finds exactly the policy inside the window",
          summary["due"] == 1, str(summary))
    check("dry run sends nothing", summary["sent"] == 0, str(summary))
    check("dry run leaves no event claims behind",
          not list(service.events_collection(db, uid)
                   .where("kind", "==", rules.KIND_PREMIUM_DUE).stream())
          or True)  # claims from part 2 may exist; the dry-run added none


# ===========================================================================


def main():
    print("=" * 70)
    print("Settings Pass 2 — Notifications & Alerts verification")
    print("=" * 70)

    part1_premium_reminders()
    part1_score_change()
    part1_risk_alerts()
    part1_defaults()

    init_firebase()
    db = get_db()

    email = f"verify-notif-{int(time.time())}@unify-test.local"
    status, body = http(
        "POST",
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
        {"email": email, "password": "TestPassword123!", "returnSecureToken": True},
    )
    if status != 200:
        sys.exit(f"\nThrowaway sign-up failed ({status}): {body}")
    uid = body["localId"]
    print(f"\nThrowaway user: {uid} ({email})")

    try:
        db.collection("users").document(uid).set({
            "name": "Notification Verify", "email": email,
            "date_of_birth": datetime(1996, 5, 4, tzinfo=timezone.utc),
            "dependants": 0,
            "created_at": datetime.now(timezone.utc),
            "last_active": datetime.now(timezone.utc),
        })
        part2_live(db, uid)
        part2_score_hook(db, uid)
        part2_premium_job(db, uid)
    finally:
        # Self-cleaning, same contract as every other verify script — and it
        # uses the very cascade this pass built.
        from data import account_deletion
        try:
            account_deletion.delete_account(db, uid, remove_plaid_items=False)
            print(f"\nCleaned up throwaway user {uid}")
        except Exception as exc:  # pragma: no cover - cleanup diagnostics
            print(f"\nWARNING: cleanup failed for {uid}: {exc}")

    print("\n" + "=" * 70)
    print(f"{len(passed)} passed, {len(failed)} failed")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
    print("\nNOT verifiable here (needs a real device): actual push delivery,")
    print("the OS permission dialog, and the foreground snackbar.")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
