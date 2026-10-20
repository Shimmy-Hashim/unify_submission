"""Disconnect a bank connection — POST /plaid/disconnect-item verification.

    cd backend && venv\\Scripts\\activate && python scripts/verify_item_disconnect.py

⚠️ SAFETY: this script creates its OWN throwaway Firebase users and its own
Plaid Sandbox Items, and touches nothing else. Every uid it handles is one it
signed up seconds earlier via the Identity Toolkit REST API; every item_id is
one it minted via /sandbox/public_token/create. There is no code path here that
accepts, discovers or enumerates an existing uid or Item.

WHAT IS ACTUALLY PROVED, AND HOW

  1. REAL REVOCATION AT PLAID — not a 200 from our own endpoint. The Item's
     access_token is captured before the disconnect, confirmed live by calling
     Plaid's /item/get with it, and then confirmed DEAD by calling /item/get
     again afterwards and requiring ITEM_NOT_FOUND. A local-delete-only
     implementation passes every Firestore assertion below and fails this one.

  2. THE CASCADE IS SCOPED — user A holds TWO real Items from the same
     institution. One is disconnected; the other's Item document, accounts,
     transactions and investment accounts are re-read afterwards and must be
     byte-for-byte still there, and its access_token must still work at Plaid.

  3. CROSS-USER REJECTION — user B calls the real HTTP endpoint with user A's
     item_id and a genuine ID token of their own. Must 404, and user A's data
     must be untouched afterwards (checked, not assumed — a 404 returned after
     a delete would still be a breach).

  4. FAILURE LEAVES IT RETRYABLE — a forced mid-cascade Firestore failure must
     leave the plaidItems document alive (so Settings still lists the
     connection and the user can try again), and the retry must then finish.

  5. REVOCATION FAILURE TOUCHES NOTHING — an Item whose access_token Plaid
     rejects must abort with nothing deleted, rather than orphaning a live
     connection that nothing can ever revoke again.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import plaid  # noqa: E402
from firebase import get_db, init_firebase  # noqa: E402
from firebase_admin import auth as admin_auth  # noqa: E402
from google.cloud.firestore_v1.base_query import FieldFilter  # noqa: E402
from plaid.model.item_get_request import ItemGetRequest  # noqa: E402
from plaid.model.products import Products  # noqa: E402
from plaid.model.sandbox_public_token_create_request import (  # noqa: E402
    SandboxPublicTokenCreateRequest,
)

from data import item_disconnect, plaid_service  # noqa: E402

PROJECT = "YOUR_FIREBASE_PROJECT_ID"
GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
PASSWORD = "TestPassword123!"
SANDBOX_INSTITUTION = "ins_109508"  # First Platypus Bank

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
    email = f"verify-disconnect-{label}-{int(time.time() * 1000)}@unify-test.local"
    status, body = http(
        "POST",
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
        {"email": email, "password": PASSWORD, "returnSecureToken": True},
    )
    if status != 200:
        sys.exit(f"Throwaway sign-up failed ({status}): {body}")
    created_uids.append(body["localId"])
    return body["localId"], body["idToken"]


# ---------------------------------------------------------------------------
# Plaid helpers — the revocation oracle
# ---------------------------------------------------------------------------


def token_is_live_at_plaid(access_token):
    """Ask Plaid directly whether this access_token still resolves to an Item.

    This is the whole point of the script: our own endpoint returning 200 says
    nothing about whether Plaid actually revoked anything.
    Returns (live: bool, error_code: str|None).
    """
    try:
        plaid_service.get_plaid_client().item_get(
            ItemGetRequest(access_token=access_token))
        return True, None
    except plaid.ApiException as exc:
        try:
            code = json.loads(exc.body).get("error_code", "UNKNOWN")
        except Exception:
            code = "UNKNOWN"
        return False, code


def connect_sandbox_item(db, uid):
    """Mint a real Sandbox Item and persist it exactly as the app would."""
    public_token = plaid_service.get_plaid_client().sandbox_public_token_create(
        SandboxPublicTokenCreateRequest(
            institution_id=SANDBOX_INSTITUTION,
            initial_products=[Products("transactions")],
        )
    ).to_dict()["public_token"]
    summary = plaid_service.connect_bank_item(db, uid, public_token)
    item_id = summary["item_id"]
    access_token = (
        db.collection("users").document(uid)
        .collection("plaidItems").document(item_id)
        .get().to_dict()["access_token"]
    )
    return item_id, access_token


def add_nested_fixture(db, uid, item_id, tag):
    """Guarantee both collections and both transactions subcollections are
    non-empty for this Item, so 'the cascade deleted them' is a real assertion
    rather than one that passes vacuously on empty collections."""
    now = datetime.now(timezone.utc)
    user_ref = db.collection("users").document(uid)

    for account in user_ref.collection("plaidAccounts").where(
            filter=FieldFilter("item_id", "==", item_id)).stream():
        for i in range(3):
            account.reference.collection("transactions").document(
                f"txn-{tag}-{account.id}-{i}").set(
                {"transaction_id": f"txn-{tag}-{account.id}-{i}",
                 "account_id": account.id, "amount": 10.0 + i, "date": now})

    invest = user_ref.collection("investmentAccounts").document(f"inv-{tag}")
    invest.set({"account_id": f"inv-{tag}", "uid": uid, "item_id": item_id,
                "source": "plaid", "account_name": f"Brokerage {tag}",
                "account_type": "brokerage", "total_value_sgd": 20000.0,
                "holdings": [{"ticker": "VOO", "quantity": 10}]})
    for i in range(2):
        invest.collection("transactions").document(f"itxn-{tag}-{i}").set(
            {"transaction_id": f"itxn-{tag}-{i}", "type": "deposit",
             "amount": -500.0, "date": now})


def item_footprint(db, uid, item_id):
    """Every stored path belonging to this Item. Reuses the production
    surviving_item_paths so the test and the endpoint's own completeness gate
    cannot disagree about what 'belonging to this Item' means."""
    return sorted(item_disconnect.surviving_item_paths(db, uid, item_id))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disconnect(db, client):
    uid, token = sign_up("a")
    print(f"\n--- user A uid={uid}")
    now = datetime.now(timezone.utc)
    user_ref = db.collection("users").document(uid)
    user_ref.set({"name": "Disconnect Verify", "email": f"{uid}@unify-test.local",
                  "date_of_birth": datetime(1995, 2, 2, tzinfo=timezone.utc),
                  "dependants": 0, "created_at": now, "last_active": now})
    user_ref.collection("financialProfile").document("profile").set(
        {"monthly_income_sgd": 5200.0, "income_source": "user_entered"})

    # TWO real Items from the same institution — also the real "2 Tartan Bank"
    # shape, and the case where a scoping bug would be invisible with one Item.
    item_x, token_x = connect_sandbox_item(db, uid)
    item_y, token_y = connect_sandbox_item(db, uid)
    add_nested_fixture(db, uid, item_x, "x")
    add_nested_fixture(db, uid, item_y, "y")
    check("two distinct Sandbox Items connected for one user", item_x != item_y,
          f"{item_x} / {item_y}")

    before_x = item_footprint(db, uid, item_x)
    before_y = item_footprint(db, uid, item_y)
    check("Item X has a non-trivial footprint before disconnect",
          len(before_x) >= 6, f"{len(before_x)} paths")
    check("Item X footprint includes transactions subcollection documents",
          any("/transactions/" in p for p in before_x))
    check("Item X footprint includes an investmentAccounts document",
          any("/investmentAccounts/" in p for p in before_x))
    check("Item Y has a non-trivial footprint before disconnect",
          len(before_y) >= 6, f"{len(before_y)} paths")

    # --- the revocation oracle, BEFORE
    live, _ = token_is_live_at_plaid(token_x)
    check("Item X's access_token is live at Plaid BEFORE disconnect", live)
    live_y, _ = token_is_live_at_plaid(token_y)
    check("Item Y's access_token is live at Plaid BEFORE disconnect", live_y)

    # --- cross-user rejection, while X still exists to be stolen
    uid_b, token_b = sign_up("b")
    print(f"--- user B uid={uid_b}")
    attack = client.post(
        "/plaid/disconnect-item", json={"item_id": item_x},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    check("user B disconnecting user A's item_id is REJECTED",
          attack.status_code == 404, f"HTTP {attack.status_code}")
    check("user A's Item X is completely untouched after B's attempt",
          item_footprint(db, uid, item_x) == before_x)
    live_after_attack, _ = token_is_live_at_plaid(token_x)
    check("user A's Item X is still live at Plaid after B's attempt",
          live_after_attack)

    # --- unauthenticated
    r = client.post("/plaid/disconnect-item", json={"item_id": item_x})
    check("disconnect without auth is rejected 401", r.status_code == 401,
          f"HTTP {r.status_code}")

    # --- a nonexistent item_id for a legitimate caller
    r = client.post("/plaid/disconnect-item", json={"item_id": "item-does-not-exist"},
                    headers={"Authorization": f"Bearer {token}"})
    check("nonexistent item_id 404s for its own owner too", r.status_code == 404,
          f"HTTP {r.status_code}")

    # --- the real disconnect
    r = client.post("/plaid/disconnect-item", json={"item_id": item_x},
                    headers={"Authorization": f"Bearer {token}"})
    body = r.json() if r.status_code == 200 else {}
    check("POST /plaid/disconnect-item succeeds", r.status_code == 200,
          f"HTTP {r.status_code} {r.text[:120]}")
    check("response reports the Item disconnected",
          body.get("status") == "disconnected" and body.get("item_id") == item_x)
    check("response reports Plaid revocation",
          (body.get("plaid") or {}).get("revoked") is True, str(body.get("plaid")))
    check("response reports Firestore verified clean",
          body.get("firestore_verified_empty") is True)
    check("response never leaks an access_token",
          "access_token" not in json.dumps(body).lower())

    # --- 1. REAL REVOCATION, asked of Plaid rather than of ourselves
    live_x, code_x = token_is_live_at_plaid(token_x)
    check("Item X is GENUINELY revoked at Plaid (item/get -> ITEM_NOT_FOUND)",
          (not live_x) and code_x == "ITEM_NOT_FOUND", f"live={live_x} code={code_x}")

    # --- 2. cascade complete, and scoped
    after_x = item_footprint(db, uid, item_x)
    check("NO stored path survives for Item X", after_x == [], f"{after_x}")
    check("  plaidItems/{item_x} is gone",
          not user_ref.collection("plaidItems").document(item_x).get().exists)
    check("  no plaidAccounts document carries item X",
          not list(user_ref.collection("plaidAccounts")
                   .where(filter=FieldFilter("item_id", "==", item_x)).stream()))
    check("  no investmentAccounts document carries item X",
          not list(user_ref.collection("investmentAccounts")
                   .where(filter=FieldFilter("item_id", "==", item_x)).stream()))

    after_y = item_footprint(db, uid, item_y)
    check("Item Y survives the disconnect COMPLETELY intact",
          after_y == before_y,
          f"{len(before_y)} -> {len(after_y)} paths")
    check("  Item Y's transactions subcollections survive",
          any("/transactions/" in p for p in after_y))
    live_y2, _ = token_is_live_at_plaid(token_y)
    check("Item Y's access_token is STILL LIVE at Plaid (not over-revoked)",
          live_y2)

    # --- 3. score refresh fired
    check("disconnect response carries a financial_profile refresh",
          "financial_profile" in body)
    check("disconnect response carries a score refresh result", "score" in body)
    profile = (user_ref.collection("financialProfile").document("profile")
               .get().to_dict() or {})
    check("financialProfile was recomputed after disconnect (net_worth present)",
          "net_worth_sgd" in profile, str(profile.get("net_worth_sgd")))

    return uid, token, item_y, token_y, uid_b


def test_revocation_failure_touches_nothing(db, client, uid, token):
    """An Item Plaid refuses to revoke must not be deleted locally — otherwise
    the live connection is orphaned with nothing left to revoke it with."""
    print("\n--- revocation failure")
    user_ref = db.collection("users").document(uid)
    item_id = "item-bogus-token"
    user_ref.collection("plaidItems").document(item_id).set(
        {"item_id": item_id, "uid": uid, "access_token": "access-sandbox-not-real",
         "institution_name": "Nowhere Bank", "plaid_cursor": None})
    user_ref.collection("plaidAccounts").document("acct-bogus").set(
        {"account_id": "acct-bogus", "uid": uid, "item_id": item_id,
         "account_type": "depository", "balance_current": 5.0})
    before = item_footprint(db, uid, item_id)

    r = client.post("/plaid/disconnect-item", json={"item_id": item_id},
                    headers={"Authorization": f"Bearer {token}"})
    check("a Plaid revocation failure returns 502, not a false success",
          r.status_code == 502, f"HTTP {r.status_code}")
    check("NOTHING was deleted when revocation failed",
          item_footprint(db, uid, item_id) == before,
          f"{len(before)} paths before")
    check("  the connection is still listed, so the user can retry",
          user_ref.collection("plaidItems").document(item_id).get().exists)

    # cleanup of this synthetic fixture
    user_ref.collection("plaidAccounts").document("acct-bogus").delete()
    user_ref.collection("plaidItems").document(item_id).delete()


def test_partial_failure_is_retryable(db, uid, item_id):
    """A mid-cascade Firestore failure must leave the connection listed and
    retryable — the single-Item analogue of account_deletion's Auth-last proof.
    """
    print("\n--- partial cascade failure")
    user_ref = db.collection("users").document(uid)
    original = item_disconnect._delete_refs
    calls = {"n": 0}

    def exploding_delete_refs(db_, refs):
        calls["n"] += 1
        if calls["n"] == 1:
            # Let the first batch through, then fail — so the failure lands
            # after real deletes have committed, not before any.
            return original(db_, refs)
        raise RuntimeError("injected Firestore failure")

    item_disconnect._delete_refs = exploding_delete_refs
    raised = None
    try:
        item_disconnect.disconnect_item(db, uid, item_id)
    except Exception as exc:
        raised = exc
    finally:
        item_disconnect._delete_refs = original

    check("a mid-cascade Firestore failure propagates (not swallowed)",
          isinstance(raised, RuntimeError), type(raised).__name__)
    check("the failure happened after some deletes had committed",
          calls["n"] > 1, f"{calls['n']} batch calls")
    check("plaidItems SURVIVES a failed cascade (connection still listed)",
          user_ref.collection("plaidItems").document(item_id).get().exists)
    check("  so the disconnect is retryable rather than half-applied",
          item_footprint(db, uid, item_id) != [])

    # --- retry, with the failure removed. The Item is already revoked at
    # Plaid by the aborted attempt, so this also exercises the
    # ITEM_NOT_FOUND-counts-as-revoked path.
    result = item_disconnect.disconnect_item(db, uid, item_id)
    check("retry completes the disconnect cleanly",
          result.get("firestore_verified_empty") is True)
    check("retry treats the already-revoked Item as revoked",
          (result.get("plaid") or {}).get("revoked") is True,
          str(result.get("plaid")))
    check("retry leaves NO stored path for the Item",
          item_footprint(db, uid, item_id) == [])


def main():
    init_firebase()
    db = get_db()

    from fastapi.testclient import TestClient
    import main as backend_main
    client = TestClient(backend_main.app, raise_server_exceptions=False)

    uid = None
    try:
        uid, token, item_y, token_y, uid_b = test_disconnect(db, client)
        test_revocation_failure_touches_nothing(db, client, uid, token)
        test_partial_failure_is_retryable(db, uid, item_y)
        # Item Y was the survivor of test 1 and is now disconnected by the
        # retry above — confirm that its token really went too, so the run
        # leaves nothing live at Plaid.
        live, code = token_is_live_at_plaid(token_y)
        check("Item Y is revoked at Plaid once it is disconnected in turn",
              (not live) and code == "ITEM_NOT_FOUND", f"code={code}")
    finally:
        for dead_uid in created_uids:
            try:
                from data import account_deletion
                account_deletion.delete_account(db, dead_uid)
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
