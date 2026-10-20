"""Delete one insurance policy — DELETE /insurance/{policy_id} verification.

    cd backend && venv\\Scripts\\activate && python scripts/verify_policy_deletion.py

⚠️ SAFETY: this script creates its OWN throwaway Firebase users and touches
nothing else. Every uid it handles is one it signed up seconds earlier via the
Identity Toolkit REST API; every policy it deletes is one it wrote itself.
There is no code path here that accepts, discovers or enumerates an existing
uid or policy.

WHAT IS ACTUALLY PROVED, AND HOW

  1. THE DOCUMENT IS REALLY GONE — read back with the Admin SDK after the
     endpoint answers, not inferred from a 200. The user's other policy is
     re-read and must be byte-for-byte unchanged.

  2. THE SCORE MOVES WITH IT — the user holds a life + CI policy big enough to
     reach the "Covered" Insurance tier. Deleting it must drop the CACHED
     financialProfile's Insurance tier to "Not Covered" and its cached life
     coverage ratio to zero, and the endpoint's own score must agree with the
     cache. financialProfile's derived-field refresh must NOT run.

  3. CROSS-USER REJECTION — user B calls the real endpoint with user A's
     policy_id and a genuine ID token of their own. Must 404, and user A's
     policy must still be there afterwards (checked, not assumed — a 404
     returned after a delete would still be a breach).

  4. NOT-FOUND SHAPES — a made-up id, a Firestore-reserved id, and a repeat
     delete of an already-deleted policy all 404 rather than 500 or 200.
     No token at all is a 401.

  5. A SURVIVING DOCUMENT IS A FAILURE, NOT A SUCCESS — a delete that leaves
     the document behind must raise, and the endpoint must answer 500 with the
     policy still listed.

  6. PREMIUM-REMINDER MARKERS CANNOT COLLIDE — a "sent" marker left behind by
     a deleted policy must not suppress the reminder for a re-upload of the
     same policy (same name, same renewal date).
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from firebase import get_db, init_firebase  # noqa: E402
from firebase_admin import auth as admin_auth  # noqa: E402

from data import financial_profile, policy_deletion, score_profile  # noqa: E402
from notifications import rules, service  # noqa: E402

GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
PASSWORD = "TestPassword123!"

MONTHLY_INCOME = 5000.0          # 60k/yr
DOB = datetime(1994, 3, 15, tzinfo=timezone.utc)

passed, failed = [], []
created_uids: list[str] = []


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


def sign_up(label):
    """A brand-new throwaway account. The ONLY way this script obtains a uid."""
    email = f"verify-policy-delete-{label}-{int(time.time() * 1000)}@unify-test.local"
    status, body = http(
        "POST",
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
        {"email": email, "password": PASSWORD, "returnSecureToken": True},
    )
    if status != 200:
        sys.exit(f"Throwaway sign-up failed ({status}): {body}")
    created_uids.append(body["localId"])
    return body["localId"], body["idToken"], email


def seed_user(db, uid, email):
    """Income + DOB, and nothing from Plaid: Insurance is then the one domain
    that can be assessed, which is exactly the one a policy delete moves."""
    db.collection("users").document(uid).set({
        "name": "Policy Delete Verify", "email": email,
        "date_of_birth": DOB, "dependants": 1,
    })
    score_profile.profile_ref(db, uid).set({
        "monthly_income_sgd": MONTHLY_INCOME, "income_source": "user_entered",
        "overall_score": None, "checklist_flags": None,
        "checklist_last_updated": None, "ml_profile": None,
    })


def write_policy(db, uid, **fields):
    """A policy exactly as /insurance/parse writes one: an auto-ID document
    whose policy_id field repeats its own ID."""
    ref = (db.collection("users").document(uid)
           .collection("insurancePolicies").document())
    ref.set({"policy_id": ref.id, "uid": uid, "exclusions": [],
             "coverage_gaps": [], "original_filename": "test.pdf",
             "uploaded_at": datetime.now(timezone.utc), **fields})
    return ref


def cached_insurance(db, uid):
    profile = score_profile.profile_ref(db, uid).get().to_dict() or {}
    ml = profile.get("ml_profile") or {}
    domain = (ml.get("domains") or {}).get("insurance") or {}
    coverage = ml.get("insurance_coverage") or {}
    return domain.get("tier_label"), coverage


def life_ratio(coverage):
    life = (coverage or {}).get("life") or {}
    return life.get("ratio")


def auth(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------


def test_delete(db, client):
    print("\n--- delete one policy: gone, sibling intact, score recomputed")
    uid, token, email = sign_up("a")
    seed_user(db, uid, email)

    life = write_policy(db, uid, policy_name="Verify Term Life",
                        insurer="Test Insurer", policy_type="life",
                        sum_assured_sgd=600000.0, annual_premium_sgd=1200.0)
    ci = write_policy(db, uid, policy_name="Verify CI Cover",
                      insurer="Test Insurer", policy_type="critical_illness",
                      sum_assured_sgd=250000.0, annual_premium_sgd=900.0)
    hosp = write_policy(db, uid, policy_name="Verify Shield",
                        insurer="Test Insurer", policy_type="hospitalisation",
                        annual_premium_sgd=600.0)
    score_profile.refresh_score_profile(db, uid, notify=False, snapshot=False)

    tier_before, coverage_before = cached_insurance(db, uid)
    print(f"      before: insurance tier={tier_before!r}, "
          f"life ratio={life_ratio(coverage_before)}")
    check("baseline: the seeded user is Insurance 'Covered'",
          tier_before == "Covered", f"tier={tier_before!r}")
    check("baseline: cached life ratio reflects the 600k policy (10x)",
          life_ratio(coverage_before) is not None
          and abs(life_ratio(coverage_before) - 10.0) < 1e-9,
          f"ratio={life_ratio(coverage_before)}")

    ci_before = ci.get().to_dict()
    hosp_before = hosp.get().to_dict()

    # financialProfile's derived-field refresh must not run on this path.
    profile_refresh_calls = []
    original_refresh = financial_profile.refresh_financial_profile

    def spy(*args, **kwargs):
        profile_refresh_calls.append(args)
        return original_refresh(*args, **kwargs)

    financial_profile.refresh_financial_profile = spy
    try:
        r = client.delete(f"/insurance/{life.id}", headers=auth(token))
    finally:
        financial_profile.refresh_financial_profile = original_refresh
    body = r.json()

    check("owner delete returns 200", r.status_code == 200,
          f"HTTP {r.status_code}" + ("" if r.status_code == 200 else f" {body}"))
    check("  response says deleted and verified",
          body.get("status") == "deleted"
          and body.get("firestore_verified_deleted") is True
          and body.get("policy_id") == life.id, str(body.get("status")))
    check("  response names the policy that was deleted",
          body.get("policy_name") == "Verify Term Life",
          str(body.get("policy_name")))
    check("the document is gone when read back with the Admin SDK",
          not life.get().exists)
    check("the user's other policies are byte-for-byte unchanged",
          ci.get().to_dict() == ci_before and hosp.get().to_dict() == hosp_before)
    check("financialProfile's derived-field refresh did NOT run",
          profile_refresh_calls == [], f"{len(profile_refresh_calls)} calls")

    tier_after, coverage_after = cached_insurance(db, uid)
    print(f"      after:  insurance tier={tier_after!r}, "
          f"life ratio={life_ratio(coverage_after)}")
    check("the CACHED Insurance tier dropped to 'Not Covered'",
          tier_after == "Not Covered", f"tier={tier_after!r}")
    check("the cached life ratio is now 0.0 — the deleted policy is not counted",
          life_ratio(coverage_after) == 0.0, f"ratio={life_ratio(coverage_after)}")
    score = body.get("score") or {}
    check("the endpoint's returned score agrees with the cache",
          (score.get("domains") or {}).get("insurance", {}).get("tier_label")
          == tier_after,
          str((score.get("domains") or {}).get("insurance", {}).get("tier_label")))

    r = client.delete(f"/insurance/{life.id}", headers=auth(token))
    check("deleting the same policy again is a 404, not a second 200",
          r.status_code == 404, f"HTTP {r.status_code}")

    return uid, token, ci


def test_cross_user(db, client, uid_a, policy_a):
    print("\n--- cross-user: user B cannot delete user A's policy")
    uid_b, token_b, email_b = sign_up("b")
    seed_user(db, uid_b, email_b)
    before = policy_a.get().to_dict()

    r = client.delete(f"/insurance/{policy_a.id}", headers=auth(token_b))
    check("user B deleting user A's policy_id gets 404 (not 403)",
          r.status_code == 404, f"HTTP {r.status_code}")
    check("  the 404 body does not confirm the policy exists elsewhere",
          r.json().get("detail") == "No policy with that ID for this user",
          r.json().get("detail"))
    check("user A's policy is still there afterwards, unchanged",
          policy_a.get().exists and policy_a.get().to_dict() == before)


def test_not_found_shapes(db, client, token):
    print("\n--- not-found and unauthenticated shapes")
    r = client.delete("/insurance/doesNotExist12345", headers=auth(token))
    check("a made-up policy_id is a 404", r.status_code == 404,
          f"HTTP {r.status_code}")
    r = client.delete("/insurance/__reserved__", headers=auth(token))
    check("a Firestore-reserved id is a 404, not a 500",
          r.status_code == 404, f"HTTP {r.status_code}")
    r = client.delete("/insurance/anything")
    check("no ID token at all is a 401", r.status_code == 401,
          f"HTTP {r.status_code}")


class _StubbornSnap:
    exists = True

    def to_dict(self):
        return {"policy_id": "stubborn", "policy_name": "Stubborn"}


class _StubbornDb:
    """Stands in for every hop of db.collection().document()... and ends on a
    document whose delete() quietly does nothing — the case a trusting
    implementation would report as success."""

    def collection(self, _name):
        return self

    def document(self, _id):
        return self

    def get(self):
        return _StubbornSnap()

    def delete(self):
        pass


def test_survivor_is_failure(client, token, policy):
    print("\n--- a document that survives the delete is a failure")
    try:
        policy_deletion.delete_policy(_StubbornDb(), "uid", "stubborn")
        raised = False
    except RuntimeError:
        raised = True
    check("delete_policy raises when the document is still there afterwards",
          raised)

    original = policy_deletion.delete_policy

    def failing(*_args, **_kwargs):
        raise RuntimeError("simulated survivor")

    policy_deletion.delete_policy = failing
    try:
        r = client.delete(f"/insurance/{policy.id}", headers=auth(token))
    finally:
        policy_deletion.delete_policy = original
    check("the endpoint answers 500 on a surviving document",
          r.status_code == 500, f"HTTP {r.status_code}")
    check("  and the policy is still listed, so the user can retry",
          policy.get().exists)


def test_premium_marker(db, uid):
    print("\n--- premium-reminder markers cannot collide across delete + re-upload")
    now = datetime.now(timezone.utc)
    renewal = now + timedelta(days=3)
    fields = dict(policy_name="Verify Renewing Plan", insurer="Test Insurer",
                  policy_type="life", sum_assured_sgd=100000.0,
                  annual_premium_sgd=500.0, renewal_date=renewal)

    old = write_policy(db, uid, **fields)
    (old_note,) = rules.find_due_premium_reminders([old.get().to_dict()], now=now)
    print(f"      original key:   {old_note.event_key}")
    check("the event key is built from policy_id, not policy_name",
          old.id in old_note.event_key, old_note.event_key)

    # What deliver() leaves behind after a successful send.
    events = service.events_collection(db, uid)
    events.document(old_note.event_key).set(
        {"kind": old_note.kind, "status": "sent"})

    policy_deletion.delete_policy(db, uid, old.id)
    check("the marker for the deleted policy is left behind (inert)",
          events.document(old_note.event_key).get().exists)

    # The same PDF uploaded again: same name, same renewal date, new auto-ID.
    new = write_policy(db, uid, **fields)
    (new_note,) = rules.find_due_premium_reminders([new.get().to_dict()], now=now)
    print(f"      re-upload key:  {new_note.event_key}")
    check("the re-uploaded policy gets a different policy_id",
          new.id != old.id)
    check("…and therefore a different event key",
          new_note.event_key != old_note.event_key)
    check("no claim exists at the new key, so deliver()'s create() would succeed",
          not events.document(new_note.event_key).get().exists)


def main():
    init_firebase()
    db = get_db()

    from fastapi.testclient import TestClient
    import main as backend_main
    client = TestClient(backend_main.app, raise_server_exceptions=False)

    try:
        uid, token, surviving_policy = test_delete(db, client)
        test_cross_user(db, client, uid, surviving_policy)
        test_not_found_shapes(db, client, token)
        test_survivor_is_failure(client, token, surviving_policy)
        test_premium_marker(db, uid)
    finally:
        for dead_uid in created_uids:
            try:
                from data import account_deletion
                account_deletion.delete_account(db, dead_uid,
                                                remove_plaid_items=False)
            except Exception:
                try:
                    admin_auth.delete_user(dead_uid)
                except Exception:
                    pass
        print(f"cleanup: removed {len(created_uids)} test users")

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
