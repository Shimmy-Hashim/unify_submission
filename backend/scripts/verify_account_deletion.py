"""Settings Pass 2 — Delete Account cascade verification.

    cd backend && venv\\Scripts\\activate && python scripts/verify_account_deletion.py

⚠️ SAFETY: this script creates its OWN throwaway Firebase users and deletes only
those. Every uid it touches is one it just signed up seconds earlier via the
Identity Toolkit REST API; there is no code path here that accepts, discovers or
guesses an existing uid. It never enumerates the users collection. Nothing in it
can reach a real account.

THREE USERS, three different things to prove:

  user A — the full cascade. Populated with a document in every collection
    Section 8 defines, including two levels of nesting (plaidAccounts ->
    transactions), an undeclared collection that is not in the deletion list,
    and a "missing" parent document that holds a subcollection without carrying
    fields of its own — the case stream() silently skips and list_documents()
    catches. Then deleted, and every collection is re-read to confirm it is
    genuinely empty and the Auth record is genuinely gone.

  user B — THE ORDERING PROOF. A Firestore delete is made to fail partway
    through, and the script asserts that the Auth account SURVIVES, still has
    its data, and can still authenticate. This is the scenario the
    Firestore-then-Auth order exists to prevent, and the reason the endpoint
    returns 500-with-account-intact rather than half-deleting.

  user C — retry after that partial failure. The same user B, with the failure
    removed, must delete cleanly on a second attempt. A safety property that
    left the account permanently undeletable would not be much of one.
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

from data import account_deletion  # noqa: E402

PROJECT = "YOUR_FIREBASE_PROJECT_ID"
GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
PASSWORD = "TestPassword123!"

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
    email = f"verify-delete-{label}-{int(time.time() * 1000)}@unify-test.local"
    status, body = http(
        "POST",
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
        {"email": email, "password": PASSWORD, "returnSecureToken": True},
    )
    if status != 200:
        sys.exit(f"Throwaway sign-up failed ({status}): {body}")
    created_uids.append(body["localId"])
    return body["localId"], email


def can_sign_in(email):
    status, _ = http(
        "POST",
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
        f"?key={API_KEY}",
        {"email": email, "password": PASSWORD, "returnSecureToken": True},
    )
    return status == 200


def auth_user_exists(uid):
    try:
        admin_auth.get_user(uid)
        return True
    except admin_auth.UserNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Fixture — one document in every collection the cascade must handle
# ---------------------------------------------------------------------------

EXPECTED_COLLECTIONS = [
    "financialProfile", "plaidItems", "plaidAccounts", "manualAccounts",
    "investmentAccounts", "insurancePolicies", "notificationSettings",
    "notificationTokens", "notificationEvents", "scoreHistory",
    "legacyExperiment",
]


def populate(db, uid, email):
    now = datetime.now(timezone.utc)
    user_ref = db.collection("users").document(uid)
    user_ref.set({
        "name": "Deletion Verify", "email": email,
        "date_of_birth": datetime(1995, 2, 2, tzinfo=timezone.utc),
        "dependants": 1,
        "created_at": now, "last_active": now,
    })
    user_ref.collection("financialProfile").document("profile").set({
        "monthly_income_sgd": 5200.0, "income_source": "user_entered",
        "monthly_expenses_sgd": 2100.0, "net_worth_sgd": 48000.0,
        "overall_score": 63.0, "checklist_flags": {}, "ml_profile": {},
        "checklist_last_updated": now,
    })
    user_ref.collection("plaidItems").document("item-1").set({
        "item_id": "item-1", "uid": uid, "access_token": "access-sandbox-FAKE",
        "institution_name": "First Platypus Bank", "plaid_cursor": "cursor",
    })

    # plaidAccounts + their transactions — two levels of nesting.
    for account_id in ("acct-1", "acct-2"):
        account = user_ref.collection("plaidAccounts").document(account_id)
        account.set({"account_id": account_id, "uid": uid,
                     "account_type": "depository", "balance_current": 1000.0})
        for i in range(3):
            account.collection("transactions").document(f"txn-{account_id}-{i}").set(
                {"transaction_id": f"txn-{account_id}-{i}", "amount": 12.5 + i,
                 "date": now})

    # manualAccounts + transactions. DELIBERATELY FORWARD-COMPATIBLE COVERAGE:
    # manual account entry is not built — no production code path in lib/ or
    # backend/ ever writes this collection, so a real user's tree never contains
    # it (see Section 8 of the brief, and the note in firestore.rules). The
    # fixture creates it synthetically because account_deletion.py lists it in
    # NESTED_SUBCOLLECTIONS, and this proves the cascade would clear it
    # correctly if the feature shipped. Passing this assertion is evidence about
    # the cascade, NOT evidence that manual accounts exist.
    manual = user_ref.collection("manualAccounts").document("manual-1")
    manual.set({"account_id": "manual-1", "uid": uid, "balance_current": 500.0})
    manual.collection("transactions").document("mtxn-1").set(
        {"transaction_id": "mtxn-1", "amount": 30.0, "date": now})

    # investmentAccounts + transactions.
    invest = user_ref.collection("investmentAccounts").document("inv-1")
    invest.set({"account_id": "inv-1", "uid": uid, "total_value_sgd": 20000.0,
                "holdings": [{"ticker": "VOO", "quantity": 10}]})
    invest.collection("transactions").document("itxn-1").set(
        {"transaction_id": "itxn-1", "type": "deposit", "amount": -500.0,
         "date": now})

    # A "missing" parent: a document path that holds a subcollection but has no
    # fields of its own. stream() does not return it; list_documents() does.
    # If the cascade used stream(), this account's transaction would survive.
    ghost = user_ref.collection("investmentAccounts").document("ghost-parent")
    ghost.collection("transactions").document("ghost-txn").set(
        {"transaction_id": "ghost-txn", "amount": 1.0})

    user_ref.collection("insurancePolicies").document("policy-1").set({
        "policy_id": "policy-1", "uid": uid, "policy_name": "PruShield",
        "policy_type": "hospitalisation", "annual_premium_sgd": 900.0,
        "renewal_date": now + timedelta(days=200), "coverage_gaps": ["A gap"],
    })
    user_ref.collection("notificationSettings").document("preferences").set(
        {"push_notifications": True, "risk_alerts": True})
    user_ref.collection("notificationSettings").document("state").set(
        {"known_coverage_gaps": ["A gap"]})
    user_ref.collection("notificationTokens").document("token-abc").set(
        {"token": "token-abc", "platform": "android"})
    user_ref.collection("notificationEvents").document("event-1").set(
        {"kind": "risk_alerts", "status": "sent"})
    # Two days, so the cascade is proved to clear a MULTI-document collection
    # here and not just a single row.
    for day, score in (("2026-09-07", 61.0), ("2026-09-08", None)):
        user_ref.collection("scoreHistory").document(day).set(
            {"date": day, "overall_score": score})

    # Not in FLAT_SUBCOLLECTIONS — the discovery pass must still catch it.
    user_ref.collection("legacyExperiment").document("leftover").set({"x": 1})


def document_count(db, uid):
    """Every document under users/{uid}, at any depth."""
    return len(account_deletion.surviving_paths(db, uid))


# ---------------------------------------------------------------------------
# User A — the full cascade
# ---------------------------------------------------------------------------


def test_full_cascade(db):
    print("\n-- User A: full cascade --")
    uid, email = sign_up("a")
    populate(db, uid, email)

    user_ref = db.collection("users").document(uid)
    present = {c.id for c in user_ref.collections()}
    check("fixture populated every expected collection",
          set(EXPECTED_COLLECTIONS).issubset(present),
          f"missing {set(EXPECTED_COLLECTIONS) - present}")
    before = document_count(db, uid)
    check("fixture has a non-trivial document tree", before >= 18, f"{before} docs")
    check("Auth user exists before deletion", auth_user_exists(uid))
    check("Auth user can sign in before deletion", can_sign_in(email))

    result = account_deletion.delete_account(db, uid, remove_plaid_items=False)

    # --- Firestore genuinely empty
    surviving = account_deletion.surviving_paths(db, uid)
    check("NO document survives anywhere under users/{uid}",
          surviving == [], f"{len(surviving)} survived: {surviving[:5]}")
    check("users/{uid} itself is gone", not user_ref.get().exists)
    check("no subcollection survives the parent document",
          list(user_ref.collections()) == [])

    # Per-collection, explicitly — a single aggregate check would let one
    # collection hide behind the others.
    for collection in EXPECTED_COLLECTIONS:
        docs = list(user_ref.collection(collection).list_documents())
        check(f"  {collection} is empty", docs == [], f"{len(docs)} left")

    # The nested levels specifically.
    for account_id in ("acct-1", "acct-2"):
        txns = list(user_ref.collection("plaidAccounts").document(account_id)
                    .collection("transactions").list_documents())
        check(f"  plaidAccounts/{account_id}/transactions is empty", txns == [],
              f"{len(txns)} left")
    ghost_txns = list(user_ref.collection("investmentAccounts")
                      .document("ghost-parent").collection("transactions")
                      .list_documents())
    check("  fieldless parent's transactions are deleted too "
          "(list_documents, not stream)", ghost_txns == [],
          f"{len(ghost_txns)} left")

    # --- Auth genuinely gone
    check("Auth user record is deleted", not auth_user_exists(uid))
    check("deleted user can no longer sign in", not can_sign_in(email))

    # --- the endpoint's own report
    check("result reports firestore verified empty",
          result["firestore_verified_empty"] is True)
    check("result reports the Auth user deleted",
          result["auth_user_deleted"] is True)
    check("result counts what it deleted",
          sum(result["firestore_documents_deleted"].values()) >= before,
          str(result["firestore_documents_deleted"]))


# ---------------------------------------------------------------------------
# User B — the ordering proof
# ---------------------------------------------------------------------------


def test_partial_failure_protects_auth(db):
    """THE test this ordering exists for.

    A Firestore delete is forced to fail midway. The assertion is not merely
    "an exception was raised" — it is that the account is left completely
    intact and usable: Auth record alive, sign-in working, data still readable.
    """
    print("\n-- User B: simulated Firestore failure must NOT orphan the account --")
    uid, email = sign_up("b")
    populate(db, uid, email)
    before = document_count(db, uid)

    # Fail partway through, AFTER some deletes have already committed — a clean
    # "failed before touching anything" would not test the dangerous case.
    real_delete_refs = account_deletion._delete_refs
    state = {"calls": 0}

    def exploding_delete_refs(db_, refs):
        state["calls"] += 1
        if state["calls"] >= 3:
            raise RuntimeError("simulated Firestore outage mid-cascade")
        return real_delete_refs(db_, refs)

    account_deletion._delete_refs = exploding_delete_refs
    raised = None
    try:
        account_deletion.delete_account(db, uid, remove_plaid_items=False)
    except Exception as exc:
        raised = exc
    finally:
        account_deletion._delete_refs = real_delete_refs

    check("a mid-cascade Firestore failure propagates (not swallowed)",
          raised is not None, type(raised).__name__ if raised else "no exception")
    check("the failure happened after some deletes had committed",
          state["calls"] >= 3, f"{state['calls']} batch calls")

    # --- the actual safety property
    check("Auth user SURVIVES a failed Firestore cascade", auth_user_exists(uid))
    check("the user can still sign in after the failed deletion",
          can_sign_in(email))

    remaining = document_count(db, uid)
    check("the user still has data (not silently emptied)", remaining > 0,
          f"{remaining} of {before} documents remain")
    check("the account is therefore NOT an orphaned Auth record with no data",
          auth_user_exists(uid) and remaining > 0)

    # And the reverse of the dangerous ordering: because Auth is deleted last
    # and only after verification, there is no state where Auth is gone while
    # data remains.
    check("no orphaned-data state exists (Auth gone + data left) — impossible "
          "by construction", not (not auth_user_exists(uid) and remaining > 0))

    return uid, email, before


def test_retry_after_failure(db, uid, email, before):
    """A safety property that made the account permanently undeletable would be
    a bug of its own."""
    print("\n-- User B (retry): deletion must succeed on a second attempt --")
    result = account_deletion.delete_account(db, uid, remove_plaid_items=False)

    check("retry deletes everything that was left",
          account_deletion.surviving_paths(db, uid) == [],
          str(account_deletion.surviving_paths(db, uid)[:5]))
    check("retry removes the Auth user", not auth_user_exists(uid))
    check("retry leaves the account unable to sign in", not can_sign_in(email))
    check("retry reports success", result["auth_user_deleted"] is True)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_delete_is_idempotent(db):
    print("\n-- Idempotency: deleting an already-deleted account --")
    uid, email = sign_up("c")
    populate(db, uid, email)
    account_deletion.delete_account(db, uid, remove_plaid_items=False)

    try:
        result = account_deletion.delete_account(db, uid, remove_plaid_items=False)
        check("a second delete does not raise", True)
        check("a second delete still reports the Auth user gone",
              result["auth_user_deleted"] is True)
    except Exception as exc:
        check("a second delete does not raise", False, repr(exc))


# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Settings Pass 2 — Delete Account cascade verification")
    print("THROWAWAY ACCOUNTS ONLY — every uid here is created by this script")
    print("=" * 70)

    init_firebase()
    db = get_db()

    try:
        test_full_cascade(db)
        uid_b, email_b, before_b = test_partial_failure_protects_auth(db)
        test_retry_after_failure(db, uid_b, email_b, before_b)
        test_delete_is_idempotent(db)
    finally:
        # Belt and braces: anything a failed assertion left behind goes now.
        leftovers = 0
        for uid in created_uids:
            try:
                if (db.collection("users").document(uid).get().exists
                        or auth_user_exists(uid)):
                    account_deletion.delete_account(db, uid,
                                                    remove_plaid_items=False)
                    leftovers += 1
            except Exception as exc:  # pragma: no cover - cleanup diagnostics
                print(f"WARNING: cleanup failed for {uid}: {exc}")
        if leftovers:
            print(f"\nCleanup removed {leftovers} leftover throwaway account(s)")

    print("\n" + "=" * 70)
    print(f"{len(passed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"  FAILED: {name}")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
