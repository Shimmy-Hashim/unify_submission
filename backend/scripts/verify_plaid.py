"""End-to-end Plaid Phase 1 verification (banking only). Requires real
Sandbox keys in backend/.env and Firebase Auth Email/Password enabled:

    cd backend && venv\\Scripts\\activate && python scripts/verify_plaid.py

Exercises the real chain with one throwaway Firebase user and one throwaway
Sandbox Item (First Platypus Bank):
  1. sign up a test user (Firebase Auth REST, as the Flutter app would)
  2. POST /plaid/link-token with the real ID token
  3. mint a Sandbox public_token via /sandbox/public_token/create (stands in
     for the Phase 2 Link UI) and POST /plaid/exchange-token
  4. confirm plaidItems + plaidAccounts documents match Section 8, with
     banking-only account types and access_token stored on the Item
  5. poll POST /plaid/sync-transactions until Plaid's initial pull lands,
     confirm the transactions subcollection populates per schema
  6. re-run sync -> zero new writes, no duplicates
  7. confirm no response body anywhere contained an access_token
  8. clean up the test user and all Firestore test documents
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from firebase import get_db, init_firebase  # noqa: E402
from firebase_admin import auth as admin_auth  # noqa: E402

from data import plaid_service  # noqa: E402
from plaid.model.products import Products  # noqa: E402
from plaid.model.sandbox_public_token_create_request import (  # noqa: E402
    SandboxPublicTokenCreateRequest,
)

PROJECT = "YOUR_FIREBASE_PROJECT_ID"
GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
FIRESTORE = f"https://firestore.googleapis.com/v1/projects/{PROJECT}/databases/(default)/documents"

# First Platypus Bank — the standard Sandbox test institution.
SANDBOX_INSTITUTION = "ins_109508"

passed, failed = [], []
responses_seen = []  # every API response body, for the access_token leak check


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def http(method, url, body=None, token=None):
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {}


def sign_up(email, password):
    status, body = http(
        "POST",
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
        {"email": email, "password": password, "returnSecureToken": True},
    )
    if status != 200:
        sys.exit(f"\nSign-up failed ({status}): {body.get('error', {}).get('message')}")
    return body["localId"], body["idToken"]


def api(client, method, path, token, body=None):
    r = client.request(method, path, headers={"Authorization": f"Bearer {token}"},
                       json=body)
    try:
        parsed = r.json()
    except Exception:
        parsed = {}
    responses_seen.append((path, parsed))
    return r.status_code, parsed


def delete_collection(ref, batch_size=200):
    """Recursively deletes a collection (docs + their subcollections)."""
    deleted = 0
    for doc in ref.list_documents(page_size=batch_size):
        for sub in doc.collections():
            deleted += delete_collection(sub, batch_size)
        doc.delete()
        deleted += 1
    return deleted


def main():
    init_firebase()
    db = get_db()
    stamp = int(time.time())
    uid, token = sign_up(f"e2e.plaid.{stamp}@unify-test.example", "E2ePlaid-12345")
    uid_b = None
    print(f"test user uid={uid}")

    from fastapi.testclient import TestClient
    import main as backend_main
    client = TestClient(backend_main.app)

    try:
        # --- link token -----------------------------------------------------
        status, body = api(client, "POST", "/plaid/link-token", token)
        check("POST /plaid/link-token returns a link_token",
              status == 200 and str(body.get("link_token", "")).startswith("link-sandbox-"),
              f"HTTP {status}")

        r = client.post("/plaid/link-token")
        check("link-token without auth is rejected 401", r.status_code == 401,
              f"HTTP {r.status_code}")

        # --- sandbox public token stands in for the Phase 2 Link UI ---------
        pt = plaid_service.get_plaid_client().sandbox_public_token_create(
            SandboxPublicTokenCreateRequest(
                institution_id=SANDBOX_INSTITUTION,
                initial_products=[Products("transactions")],
            )
        ).to_dict()["public_token"]

        status, body = api(client, "POST", "/plaid/exchange-token", token,
                           {"public_token": pt})
        item_id = body.get("item_id")
        check("POST /plaid/exchange-token connects the Item",
              status == 200 and body.get("status") == "connected" and item_id,
              f"HTTP {status}, institution={body.get('institution_name')}")
        # Confirmed live: Sandbox's default user includes investment
        # accounts (IRA/401k) in /accounts/get even with only the
        # "transactions" product requested — they must be skipped, and the
        # banking-only Firestore check below proves none were written.
        skipped = body.get("skipped_non_banking_accounts", [])
        check("investment-type accounts skipped, never written to plaidAccounts",
              all(a["account_type"] not in ("depository", "credit", "loan")
                  for a in skipped),
              f"skipped {len(skipped)}: "
              f"{sorted({a['account_type'] for a in skipped})}")

        # --- Firestore documents match Section 8 ----------------------------
        item_doc = (db.collection("users").document(uid)
                    .collection("plaidItems").document(item_id).get().to_dict())
        check("plaidItems doc stored with access_token + null cursor start",
              bool(item_doc) and item_doc.get("uid") == uid
              and str(item_doc.get("access_token", "")).startswith("access-sandbox-")
              and "plaid_cursor" in item_doc,
              f"institution={item_doc.get('institution_name')}")

        accounts = list(db.collection("users").document(uid)
                        .collection("plaidAccounts").stream())
        acct_dicts = [a.to_dict() for a in accounts]
        check("plaidAccounts populated, banking types only",
              len(acct_dicts) > 0 and all(
                  a["account_type"] in ("depository", "credit", "loan")
                  for a in acct_dicts),
              f"{len(acct_dicts)} accounts: "
              f"{sorted({a['account_type'] for a in acct_dicts})}")
        required = {"account_id", "item_id", "uid", "account_name",
                    "account_type", "account_subtype", "balance_current",
                    "balance_available", "iso_currency_code", "liability",
                    "mask", "institution_name", "last_synced"}
        check("plaidAccounts docs carry all Section 8 fields",
              all(required <= set(a) for a in acct_dicts))

        # --- liabilities merged into plaidAccounts ---------------------------
        # This Item was linked with initial_products=["transactions"] only
        # (see the sandbox_public_token_create call above), so liability
        # data landing on it doubles as proof that no re-link is needed
        # for pre-Liabilities Phase 1 Items.
        with_liab = {a["account_subtype"]: a["liability"]
                     for a in acct_dicts if a.get("liability")}
        without_liab = [a for a in acct_dicts if not a.get("liability")]
        check("liability merged on exactly the 3 covered accounts "
              "(on a transactions-only Item — no re-link needed)",
              set(with_liab) == {"credit card", "mortgage", "student"},
              f"covered={sorted(with_liab)}")
        check("uncovered accounts carry explicit null liability, "
              "not an empty map",
              len(without_liab) == len(acct_dicts) - 3
              and all(a["liability"] is None for a in without_liab),
              f"{len(without_liab)} accounts null")
        liab_fields = {"monthly_payment_amount", "next_payment_due_date",
                       "interest_rate_percentage", "is_overdue"}
        check("liability maps carry exactly the schema fields",
              all(set(m) == liab_fields for m in with_liab.values()))
        # Normalization: minimum payment for credit/student, actual
        # installment for mortgage (known Sandbox fixture values).
        expected_payment = {"credit card": 20.0, "mortgage": 3141.54,
                            "student": 25.0}
        check("monthly_payment_amount normalized per account kind",
              all(with_liab[k]["monthly_payment_amount"] == v
                  for k, v in expected_payment.items()),
              str({k: m["monthly_payment_amount"]
                   for k, m in with_liab.items()}))

        # --- Firestore rules: who can read plaidItems? -----------------------
        uid_b, token_b = sign_up(
            f"e2e.plaid.b.{stamp}@unify-test.example", "E2ePlaidB-12345")
        status, _ = http(
            "GET", f"{FIRESTORE}/users/{uid}/plaidItems/{item_id}",
            token=token_b)
        check("second user blocked from reading plaidItems (rules deny)",
              status == 403, f"HTTP {status}")
        status, _ = http(
            "GET", f"{FIRESTORE}/users/{uid}/plaidItems/{item_id}",
            token=token)
        # Not a pass/fail check: documents the known gap flagged in
        # firestore.rules — the OWNER's client can currently read their own
        # plaidItems doc, which contains access_token. Backend-only reads
        # should be enforced by carving plaidItems out of the blanket rule.
        print(f"INFO  owner-client read of own plaidItems: HTTP {status} "
              + ("— GAP CONFIRMED: access_token is client-readable by owner"
                 if status == 200 else "— owner blocked (gap closed)"))

        # Phase 2 reads bank data from the client — the plaidItems lockdown
        # must not spill over onto plaidAccounts.
        first_account_id = acct_dicts[0]["account_id"]
        status, _ = http(
            "GET",
            f"{FIRESTORE}/users/{uid}/plaidAccounts/{first_account_id}",
            token=token)
        check("owner's client can still read own plaidAccounts (rules allow)",
              status == 200, f"HTTP {status}")

        # --- transactions sync (poll until Plaid's initial pull is ready) ---
        def count_txns():
            return sum(
                len(list(a.reference.collection("transactions").stream()))
                for a in db.collection("users").document(uid)
                .collection("plaidAccounts").stream())

        total = count_txns()  # exchange already ran the initial sync
        deadline = time.time() + 120
        while total == 0 and time.time() < deadline:
            time.sleep(5)
            status, body = api(client, "POST", "/plaid/sync-transactions",
                               token, {"item_id": item_id})
            if status == 200 and body["results"][0]["written"]:
                total = count_txns()
        # Sandbox delivers the initial (30-day) and historical transaction
        # updates as separate waves. Keep syncing until a pass returns
        # nothing new so the idempotency check below runs against a
        # quiesced Item instead of racing the second wave — a latent race
        # here surfaced in Phase 3 when the (slightly slower) exchange call
        # started landing the first wave before this poll loop began.
        while time.time() < deadline:
            status, body = api(client, "POST", "/plaid/sync-transactions",
                               token, {"item_id": item_id})
            if status == 200 and not body["results"][0]["added"]:
                break
            time.sleep(5)
        total = count_txns()
        check("transactions subcollection populated after sync", total > 0,
              f"{total} transactions")

        sample = next(
            (t.to_dict() for a in db.collection("users").document(uid)
             .collection("plaidAccounts").stream()
             for t in a.reference.collection("transactions").limit(1).stream()),
            None)
        txn_fields = {"transaction_id", "account_id", "amount",
                      "iso_currency_code", "date", "name", "category",
                      "pending"}
        check("transaction docs carry all Section 8 fields",
              sample is not None and txn_fields <= set(sample),
              f"sample name={sample.get('name') if sample else None}")

        item_doc = (db.collection("users").document(uid)
                    .collection("plaidItems").document(item_id).get().to_dict())
        check("plaid_cursor advanced on the Item after sync",
              bool(item_doc.get("plaid_cursor")))

        # --- idempotency: re-sync must not duplicate or error ----------------
        status, body = api(client, "POST", "/plaid/sync-transactions", token, {})
        again = count_txns()
        check("re-sync (all items) is idempotent — no dupes, no error",
              status == 200 and body["items_synced"] == 1 and again == total,
              f"before={total} after={again}, "
              f"added={body['results'][0]['added']}")
        check("re-sync refreshed liability data on the 3 covered accounts",
              body["results"][0].get("liabilities_updated") == 3,
              f"liabilities_updated="
              f"{body['results'][0].get('liabilities_updated')}")

        status, body = api(client, "POST", "/plaid/sync-transactions", token,
                           {"item_id": "item-does-not-exist"})
        check("sync with unknown item_id returns 404", status == 404,
              f"HTTP {status}")

        # --- security: no access_token in any response body ------------------
        leaks = [p for p, b in responses_seen
                 if "access-sandbox-" in json.dumps(b) or
                 '"access_token"' in json.dumps(b)]
        check("no API response body contained an access_token", not leaks,
              f"checked {len(responses_seen)} responses"
              + (f", LEAKED IN {leaks}" if leaks else ""))

    finally:
        user_ref = db.collection("users").document(uid)
        n = 0
        for sub in user_ref.collections():
            n += delete_collection(sub)
        user_ref.delete()
        admin_auth.delete_user(uid)
        if uid_b:
            admin_auth.delete_user(uid_b)
        print(f"cleanup: removed test users and {n} Firestore docs")

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
