"""End-to-end Plaid Phase 3 verification (investments). Requires real
Sandbox keys in backend/.env and Firebase Auth Email/Password enabled:

    cd backend && venv\\Scripts\\activate && python scripts/verify_plaid_investments.py

Exercises the real chain with one throwaway Firebase user and one throwaway
Sandbox Item (First Platypus Bank — its default user carries IRA and 401k
investment accounts with holdings):
  1. sign up a test user (Firebase Auth REST, as the Flutter app would)
  2. POST /plaid/link-token (now requests transactions + investments)
  3. mint a Sandbox public_token with both products and POST
     /plaid/exchange-token — investment accounts must still be skipped out
     of plaidAccounts (Phase 1 filter unchanged) AND land in
     investmentAccounts (the other half of that filter)
  4. confirm investmentAccounts docs match Section 8, spot-check the
     concentration_risk and asset_allocation math against the stored
     holdings array
  5. poll POST /plaid/sync-investments until investment transactions land,
     confirm the subcollection matches Section 8 with the strict 5-value
     type enum, and that deposit rows are identifiable/summable per month
     (the investment-rate feature's actual downstream need)
  6. re-run sync -> identical doc counts, no duplicates
  7. Firestore rules: cross-user read of investmentAccounts -> 403,
     owner-client read -> 200 (Phase 4 reads it from the client)
  8. confirm no response body anywhere contained an access_token
  9. clean up the test user and all Firestore test documents
"""
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
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

# First Platypus Bank — the standard Sandbox test institution; its default
# user includes investment accounts (IRA, 401k) with holdings.
SANDBOX_INSTITUTION = "ins_109508"

ACCOUNT_TYPE_ENUM = {"brokerage", "srs", "cpfis", "other"}
TXN_TYPE_ENUM = {"buy", "sell", "dividend", "deposit", "withdrawal"}

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
    uid, token = sign_up(f"e2e.inv.{stamp}@unify-test.example", "E2eInvest-12345")
    uid_b = None
    print(f"test user uid={uid}")

    from fastapi.testclient import TestClient
    import main as backend_main
    client = TestClient(backend_main.app)

    try:
        # --- mapping helpers (pure unit checks, no fixtures needed) ---------
        check("txn mapping: deposit/contribution -> deposit",
              plaid_service.map_investment_txn_type(
                  {"type": "cash", "subtype": "deposit"}) == "deposit"
              and plaid_service.map_investment_txn_type(
                  {"type": "cash", "subtype": "contribution"}) == "deposit")
        check("txn mapping: buy/sell/dividend/withdrawal enum paths",
              plaid_service.map_investment_txn_type(
                  {"type": "buy", "subtype": "buy"}) == "buy"
              and plaid_service.map_investment_txn_type(
                  {"type": "sell", "subtype": "sell"}) == "sell"
              and plaid_service.map_investment_txn_type(
                  {"type": "cash", "subtype": "dividend"}) == "dividend"
              and plaid_service.map_investment_txn_type(
                  {"type": "cash", "subtype": "withdrawal"}) == "withdrawal")
        check("txn mapping: fees/transfers refuse to map (skipped, not guessed)",
              plaid_service.map_investment_txn_type(
                  {"type": "fee", "subtype": "management fee"}) is None
              and plaid_service.map_investment_txn_type(
                  {"type": "transfer", "subtype": "transfer"}) is None)
        check("account mapping: brokerage/401k/ira/529 land per the documented rules",
              plaid_service.map_investment_account_type("brokerage") == "brokerage"
              and plaid_service.map_investment_account_type("401k") == "srs"
              and plaid_service.map_investment_account_type("ira") == "srs"
              and plaid_service.map_investment_account_type("529") == "other")

        # --- link token ------------------------------------------------------
        status, body = api(client, "POST", "/plaid/link-token", token)
        check("POST /plaid/link-token succeeds with investments in products",
              status == 200 and str(body.get("link_token", "")).startswith("link-sandbox-"),
              f"HTTP {status}")

        # --- connect (public token minted with both products, as Link would) -
        pt = plaid_service.get_plaid_client().sandbox_public_token_create(
            SandboxPublicTokenCreateRequest(
                institution_id=SANDBOX_INSTITUTION,
                initial_products=[Products("transactions"),
                                  Products("investments")],
            )
        ).to_dict()["public_token"]

        status, body = api(client, "POST", "/plaid/exchange-token", token,
                           {"public_token": pt})
        item_id = body.get("item_id")
        check("POST /plaid/exchange-token connects the Item",
              status == 200 and body.get("status") == "connected" and item_id,
              f"HTTP {status}, institution={body.get('institution_name')}")

        # Phase 1 filter unchanged: investment accounts still skipped out of
        # plaidAccounts...
        skipped = body.get("skipped_non_banking_accounts", [])
        check("investment accounts still skipped out of plaidAccounts (Phase 1 filter)",
              len(skipped) > 0 and all(
                  a["account_type"] not in ("depository", "credit", "loan")
                  for a in skipped),
              f"skipped {len(skipped)}: {sorted({a['account_type'] for a in skipped})}")
        banking = list(db.collection("users").document(uid)
                       .collection("plaidAccounts").stream())
        check("plaidAccounts still banking-only",
              all(a.to_dict()["account_type"] in ("depository", "credit", "loan")
                  for a in banking),
              f"{len(banking)} accounts")

        # ...and the same accounts now land in investmentAccounts.
        inv_docs = {d.id: d.to_dict()
                    for d in db.collection("users").document(uid)
                    .collection("investmentAccounts").stream()}
        skipped_investment_ids = {a["account_id"] for a in skipped
                                  if a["account_type"] in ("investment", "brokerage")}
        check("skipped investment accounts all landed in investmentAccounts",
              skipped_investment_ids and
              skipped_investment_ids == set(inv_docs),
              f"{len(inv_docs)} investmentAccounts docs")

        # --- Section 8 schema on investmentAccounts --------------------------
        required = {"account_id", "uid", "item_id", "source", "account_name",
                    "account_type", "total_value_sgd", "holdings",
                    "asset_allocation", "concentration_risk",
                    "iso_currency_code", "last_synced"}
        check("investmentAccounts docs carry all Section 8 fields",
              all(required <= set(d) for d in inv_docs.values()))
        check("account_type mapped into the schema enum, source=plaid, item_id set",
              all(d["account_type"] in ACCOUNT_TYPE_ENUM
                  and d["source"] == "plaid" and d["item_id"] == item_id
                  and d["uid"] == uid for d in inv_docs.values()),
              str({d["account_subtype_raw"]: d["account_type"]
                   for d in inv_docs.values()}))
        holding_fields = {"ticker", "name", "quantity", "market_value",
                          "asset_type", "cost_basis"}
        with_holdings = [d for d in inv_docs.values() if d["holdings"]]
        check("holdings arrays populated with the Section 8 element shape",
              len(with_holdings) > 0 and all(
                  holding_fields <= set(h)
                  for d in with_holdings for h in d["holdings"]),
              f"{sum(len(d['holdings']) for d in inv_docs.values())} holdings "
              f"across {len(with_holdings)} accounts")
        # cost_basis is nullable per holding, but Sandbox's First Platypus
        # fixtures populate it 13/13 — at least one non-null value proves the
        # Phase 4 follow-on pass-through, without over-asserting on fixtures.
        check("cost_basis persisted (nullable, non-null on Sandbox fixtures)",
              any(h["cost_basis"] is not None
                  for d in with_holdings for h in d["holdings"]),
              str([f"{h['ticker']}={h['cost_basis']}"
                   for d in with_holdings for h in d["holdings"]][:5]))

        # --- cached analytics: recompute independently and compare -----------
        math_ok, detail = True, []
        for d in with_holdings:
            values = [h["market_value"] or 0 for h in d["holdings"]]
            total = sum(values)
            if abs(d["total_value_sgd"] - total) > 0.01:
                math_ok = False
            expected_conc = round(max(values) / total, 4) if total else None
            if d["concentration_risk"] != expected_conc:
                math_ok = False
            alloc_sum = sum(d["asset_allocation"].values())
            if d["asset_allocation"] and abs(alloc_sum - 1.0) > 0.01:
                math_ok = False
            detail.append(f"{d['account_subtype_raw']}: total={total:.2f} "
                          f"conc={d['concentration_risk']} alloc={d['asset_allocation']}")
        check("total_value/concentration_risk/asset_allocation recompute exactly",
              math_ok and with_holdings, "; ".join(detail))

        # --- investment transactions (poll until Sandbox data is ready) ------
        def count_inv_txns():
            return sum(
                len(list(a.reference.collection("transactions").stream()))
                for a in db.collection("users").document(uid)
                .collection("investmentAccounts").stream())

        total = count_inv_txns()  # exchange already ran the initial sync
        deadline = time.time() + 120
        while total == 0 and time.time() < deadline:
            time.sleep(5)
            status, body = api(client, "POST", "/plaid/sync-investments",
                               token, {"item_id": item_id})
            if status == 200 and body["results"][0]["transactions"]["written"]:
                total = count_inv_txns()
        check("investment transactions subcollection populated", total > 0,
              f"{total} transactions")

        txns = [t.to_dict()
                for a in db.collection("users").document(uid)
                .collection("investmentAccounts").stream()
                for t in a.reference.collection("transactions").stream()]
        txn_fields = {"transaction_id", "account_id", "type", "amount",
                      "quantity", "price", "ticker", "iso_currency_code",
                      "date"}
        check("transaction docs carry all Section 8 fields",
              len(txns) > 0 and all(txn_fields <= set(t) for t in txns))
        check("every stored type is in the strict 5-value enum",
              all(t["type"] in TXN_TYPE_ENUM for t in txns),
              str(sorted({t["type"] for t in txns})))

        # The downstream need (Section 5 investment rate): deposit rows must
        # be identifiable and summable per month. Sandbox fixtures may not
        # include deposit rows — the mapping itself is covered by the unit
        # checks above; here we report what the fixture actually provided.
        monthly = defaultdict(float)
        for t in txns:
            if t["type"] == "deposit" and t["date"]:
                monthly[t["date"].strftime("%Y-%m")] += abs(t["amount"] or 0)
        if monthly:
            check("deposit rows identifiable and summable per month",
                  all(v > 0 for v in monthly.values()), dict(monthly))
        else:
            print("INFO  Sandbox fixture returned no deposit-type rows in the "
                  "window — monthly summing exercised structurally: "
                  f"types present {sorted({t['type'] for t in txns})}; the "
                  "deposit/contribution mapping is covered by the unit checks")

        # --- idempotency ------------------------------------------------------
        accounts_before, txns_before = len(inv_docs), count_inv_txns()
        status, body = api(client, "POST", "/plaid/sync-investments", token, {})
        inv_after = len(list(db.collection("users").document(uid)
                             .collection("investmentAccounts").stream()))
        check("re-sync (all items) is idempotent — no dupes, no error",
              status == 200 and body["items_synced"] == 1
              and inv_after == accounts_before
              and count_inv_txns() == txns_before,
              f"accounts {accounts_before}->{inv_after}, "
              f"txns {txns_before}->{count_inv_txns()}")

        status, _ = api(client, "POST", "/plaid/sync-investments", token,
                        {"item_id": "item-does-not-exist"})
        check("sync-investments with unknown item_id returns 404", status == 404,
              f"HTTP {status}")

        r = client.post("/plaid/sync-investments", json={})
        check("sync-investments without auth is rejected 401",
              r.status_code == 401, f"HTTP {r.status_code}")

        # --- Firestore rules: investmentAccounts scoping ----------------------
        first_inv_id = next(iter(inv_docs))
        uid_b, token_b = sign_up(
            f"e2e.inv.b.{stamp}@unify-test.example", "E2eInvestB-12345")
        status, _ = http(
            "GET", f"{FIRESTORE}/users/{uid}/investmentAccounts/{first_inv_id}",
            token=token_b)
        check("second user blocked from reading investmentAccounts (403)",
              status == 403, f"HTTP {status}")
        status, _ = http(
            "GET", f"{FIRESTORE}/users/{uid}/investmentAccounts/{first_inv_id}",
            token=token)
        check("owner's client can read own investmentAccounts (Phase 4 needs this)",
              status == 200, f"HTTP {status}")
        # plaidItems must stay locked even for the owner (Phase 1 fix intact).
        status, _ = http(
            "GET", f"{FIRESTORE}/users/{uid}/plaidItems/{item_id}",
            token=token)
        check("owner-client read of plaidItems still blocked (403)",
              status == 403, f"HTTP {status}")

        # --- security: no access_token in any response body -------------------
        leaks = [p for p, b in responses_seen
                 if "access-sandbox-" in json.dumps(b, default=str) or
                 '"access_token"' in json.dumps(b, default=str)]
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
