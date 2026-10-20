"""Refresh connected data — POST /plaid/refresh verification.

    cd backend && venv\\Scripts\\activate && python scripts/verify_plaid_refresh.py

⚠️ SAFETY: this script creates its OWN throwaway Firebase users and its own
Plaid Sandbox Items, and touches nothing else. Every uid it handles is one it
signed up seconds earlier via the Identity Toolkit REST API; every item_id is
one it minted via /sandbox/public_token/create. Everything is removed at the
end through the production account-deletion cascade (which revokes each Item
at Plaid), and the cleanup is then checked, not assumed.

HOW IT RUNS
  A real uvicorn process serves main:app on a local port and every endpoint
  call is real HTTP with a genuine Firebase ID token. The script talks to
  Firestore and Plaid directly only to set up fixtures, to tamper, and to
  read the ground truth it compares against.

WHY TAMPER-AND-RESTORE
  Sandbox data is static: a refresh of an untouched Item writes the values
  that are already there, so "the balance is correct afterwards" proves
  nothing. Each write path is proved by overwriting the stored value with a
  sentinel the refresh cannot produce (-1.23), then requiring the refresh to
  replace it with what Plaid itself reports for that account, fetched here
  with the Item's own access_token.
"""
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import plaid  # noqa: E402
from firebase import get_db, init_firebase  # noqa: E402
from firebase_admin import auth as admin_auth  # noqa: E402
from google.cloud.firestore_v1.base_query import FieldFilter  # noqa: E402
from plaid.model.accounts_get_request import AccountsGetRequest  # noqa: E402
from plaid.model.investments_holdings_get_request import (  # noqa: E402
    InvestmentsHoldingsGetRequest,
)
from plaid.model.item_get_request import ItemGetRequest  # noqa: E402
from plaid.model.products import Products  # noqa: E402
from plaid.model.sandbox_item_reset_login_request import (  # noqa: E402
    SandboxItemResetLoginRequest,
)
from plaid.model.sandbox_public_token_create_request import (  # noqa: E402
    SandboxPublicTokenCreateRequest,
)

from data import account_deletion, financial_profile, plaid_service  # noqa: E402

GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
PASSWORD = "TestPassword123!"
SANDBOX_INSTITUTION = "ins_109508"  # First Platypus Bank
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
SENTINEL = -1.23
LOG_PATH = Path(tempfile.gettempdir()) / "verify_plaid_refresh_server.log"

passed, failed = [], []
created_uids: list[str] = []
access_tokens: list[str] = []
responses_seen: list[str] = []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def http(method, url, body=None, token=None, raw_body=False):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = body if raw_body else json.dumps(body).encode()
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            text = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        text = e.read().decode()
        status = e.code
    if url.startswith(BASE):
        responses_seen.append(text)
    try:
        return status, json.loads(text) if text else {}
    except ValueError:
        return status, {"_raw": text}


def refresh(token, body=None, raw_body=False):
    return http("POST", f"{BASE}/plaid/refresh", body, token, raw_body)


def sign_up(label):
    """A brand-new throwaway account. The ONLY way this script obtains a uid."""
    email = f"verify-refresh-{label}-{int(time.time() * 1000)}@unify-test.local"
    status, body = http(
        "POST",
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
        {"email": email, "password": PASSWORD, "returnSecureToken": True},
    )
    if status != 200:
        sys.exit(f"Throwaway sign-up failed ({status}): {body}")
    created_uids.append(body["localId"])
    return body["localId"], body["idToken"]


def onboard(db, uid):
    now = datetime.now(timezone.utc)
    user_ref = db.collection("users").document(uid)
    user_ref.set({"name": "Refresh Verify", "email": f"{uid}@unify-test.local",
                  "date_of_birth": datetime(1994, 3, 3, tzinfo=timezone.utc),
                  "dependants": 0, "created_at": now, "last_active": now})
    user_ref.collection("financialProfile").document("profile").set(
        {"monthly_income_sgd": 6000.0, "income_source": "user_entered"})
    return user_ref


def plaid_code(exc):
    try:
        return json.loads(exc.body).get("error_code", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def connect_via_endpoint(db, uid, token):
    """Mint a Sandbox public_token and exchange it through the real HTTP
    endpoint, exactly as the app does after Plaid Link."""
    public_token = plaid_service.get_plaid_client().sandbox_public_token_create(
        SandboxPublicTokenCreateRequest(
            institution_id=SANDBOX_INSTITUTION,
            initial_products=[Products("transactions"), Products("investments")],
        )
    ).to_dict()["public_token"]
    status, body = http("POST", f"{BASE}/plaid/exchange-token",
                        {"public_token": public_token}, token)
    if status != 200:
        sys.exit(f"exchange-token failed ({status}): {body}")
    item_id = body["item_id"]
    access_token = (db.collection("users").document(uid)
                    .collection("plaidItems").document(item_id)
                    .get().to_dict()["access_token"])
    access_tokens.append(access_token)
    return item_id, access_token


def quiesce(db, uid, item_id, deadline_s=150):
    """Sandbox delivers transactions in waves after an Item is created. Sync
    until a pass adds nothing, so later count comparisons are not racing a
    wave Plaid was always going to deliver."""
    deadline = time.time() + deadline_s
    seen_any = False
    while time.time() < deadline:
        try:
            r = plaid_service.sync_item_transactions(db, uid, item_id)
        except plaid.ApiException as exc:
            if plaid_code(exc) != "PRODUCT_NOT_READY":
                raise
            time.sleep(5)
            continue
        seen_any = seen_any or r["written"] > 0 or txn_count(db, uid, item_id) > 0
        if seen_any and r["added"] == 0:
            return True
        time.sleep(5)
    return False


def accounts_of(db, uid, item_id, collection="plaidAccounts"):
    return list(db.collection("users").document(uid).collection(collection)
                .where(filter=FieldFilter("item_id", "==", item_id)).stream())


def txn_count(db, uid, item_id):
    return sum(len(list(a.reference.collection("transactions").stream()))
               for a in accounts_of(db, uid, item_id))


def item_doc(db, uid, item_id):
    return (db.collection("users").document(uid).collection("plaidItems")
            .document(item_id).get().to_dict())


def backdate(db, uid, item_id, minutes=10):
    (db.collection("users").document(uid).collection("plaidItems")
     .document(item_id).update(
        {"last_synced": datetime.now(timezone.utc) - timedelta(minutes=minutes)}))


def plaid_accounts(access_token):
    resp = plaid_service.get_plaid_client().accounts_get(
        AccountsGetRequest(access_token=access_token)).to_dict()
    return {a["account_id"]: a for a in resp["accounts"]}


def token_is_live(access_token):
    try:
        plaid_service.get_plaid_client().item_get(
            ItemGetRequest(access_token=access_token))
        return True, None
    except plaid.ApiException as exc:
        return False, plaid_code(exc)


def start_server():
    log = open(LOG_PATH, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(BACKEND_DIR), stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if r.status == 200:
                    return proc, log
        except Exception:
            time.sleep(1)
    stop_server(proc)
    sys.exit("uvicorn did not come up")


def stop_server(proc):
    """On Windows a venv's python.exe is a launcher that spawns the real
    interpreter, so terminating proc alone orphans uvicorn. Kill the tree."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_connections(db):
    print("\n--- 1. no connections")
    uid, token = sign_up("empty")
    onboard(db, uid)
    status, body = refresh(token, {})
    check("no connections: 200 with status no_connections",
          status == 200 and body.get("status") == "no_connections",
          f"HTTP {status} {body.get('status')}")
    check("no connections: no error, items empty, score refresh ran",
          body.get("items") == [] and "score" in body and "detail" not in body)
    status, body = refresh(token)  # no body at all
    check("no connections: a request with NO body is accepted too",
          status == 200 and body.get("status") == "no_connections",
          f"HTTP {status}")
    return uid, token


def test_two_items(db, uid, token, item_x, item_y):
    print("\n--- 2. two Items, one institution")
    before_txns = {i: txn_count(db, uid, i) for i in (item_x, item_y)}
    for i in (item_x, item_y):
        backdate(db, uid, i)
    before_item_sync = {i: item_doc(db, uid, i)["last_synced"] for i in (item_x, item_y)}
    before_acct_sync = {a.id: a.to_dict()["last_synced"]
                        for i in (item_x, item_y) for a in accounts_of(db, uid, i)}

    t0 = time.perf_counter()
    status, body = refresh(token, {})
    elapsed = time.perf_counter() - t0
    entries = {e["item_id"]: e for e in body.get("items", [])}
    check("two-Item refresh returns 200 status refreshed",
          status == 200 and body.get("status") == "refreshed",
          f"HTTP {status} {body.get('status')} {str(body)[:200] if status != 200 else ''}")
    check("both Items refreshed, each with an entry",
          set(entries) == {item_x, item_y}
          and all(e["status"] == "refreshed" and e["error_code"] is None
                  for e in entries.values()))
    check("each entry carries institution_name, counts, accounts_updated",
          all(e["institution_name"] and isinstance(e["transactions"], dict)
              and {"added", "modified", "removed"} <= set(e["transactions"])
              and e["accounts_updated"] > 0 for e in entries.values()),
          str({k: (v["institution_name"], v["transactions"], v["accounts_updated"])
               for k, v in entries.items()}))
    check("investments synced for both (each Item has investment accounts)",
          all(e["investments_synced"] for e in entries.values()))
    check("response carries refreshed_at, financial_profile and score",
          body.get("refreshed_at") and body.get("financial_profile") is not None
          and "score" in body)

    for i in (item_x, item_y):
        e = entries.get(i, {})
        t = e.get("transactions") or {}
        expected = before_txns[i] + t.get("added", 0) - t.get("removed", 0)
        after = txn_count(db, uid, i)
        check(f"transaction count consistent, no duplicates ({i[:8]})",
              after == expected and after == before_txns[i],
              f"before={before_txns[i]} after={after} counts={t}")

    after_item_sync = {i: item_doc(db, uid, i)["last_synced"] for i in (item_x, item_y)}
    check("both plaidItems.last_synced advanced",
          all(after_item_sync[i] > before_item_sync[i] for i in (item_x, item_y)))
    after_acct = {a.id: a.to_dict()["last_synced"]
                  for i in (item_x, item_y) for a in accounts_of(db, uid, i)}
    check("every plaidAccounts.last_synced advanced",
          after_acct.keys() == before_acct_sync.keys()
          and all(after_acct[k] > before_acct_sync[k] for k in after_acct),
          f"{len(after_acct)} accounts")
    return elapsed


def test_cooldown(db, uid, token, item_x, item_y):
    print("\n--- 6. cooldown")
    before = {i: item_doc(db, uid, i)["last_synced"] for i in (item_x, item_y)}
    before_acct = {a.id: a.to_dict()["last_synced"]
                   for i in (item_x, item_y) for a in accounts_of(db, uid, i)}
    status, body = refresh(token, {})
    check("immediate second call returns skipped_recent",
          status == 200 and body.get("status") == "skipped_recent"
          and body.get("last_synced"), f"{body}")
    after = {i: item_doc(db, uid, i)["last_synced"] for i in (item_x, item_y)}
    after_acct = {a.id: a.to_dict()["last_synced"]
                  for i in (item_x, item_y) for a in accounts_of(db, uid, i)}
    check("skipped call changed no timestamp",
          before == after and before_acct == after_acct)
    status, body = refresh(token, {"item_id": item_x, "force": True})
    check("no request field bypasses the cooldown",
          status == 200 and body.get("status") == "skipped_recent", f"{body.get('status')}")


def tamper(db, uid, item_x):
    """Sentinel one depository account (both balances), one credit account
    (both balances) and one investment account's total_value_sgd."""
    user_ref = db.collection("users").document(uid)
    accts = {a.id: a.to_dict() for a in accounts_of(db, uid, item_x)}
    dep = next(k for k, v in accts.items() if v["account_type"] == "depository")
    cred = next(k for k, v in accts.items() if v["account_type"] == "credit")
    for acct_id in (dep, cred):
        user_ref.collection("plaidAccounts").document(acct_id).update(
            {"balance_current": SENTINEL, "balance_available": SENTINEL})
    inv = accounts_of(db, uid, item_x, "investmentAccounts")[0]
    original_inv_total = inv.to_dict()["total_value_sgd"]
    inv.reference.update({"total_value_sgd": SENTINEL})
    backdate(db, uid, item_x)
    return dep, cred, inv.id, original_inv_total


def test_balances_and_investments(db, uid, token, item_x, token_x):
    print("\n--- 3/4. tamper-and-restore: balances and investments")
    user_ref = db.collection("users").document(uid)
    dep, cred, inv_id, original_inv_total = tamper(db, uid, item_x)
    check("sentinels are in place before the refresh",
          user_ref.collection("plaidAccounts").document(dep).get()
          .to_dict()["balance_current"] == SENTINEL
          and user_ref.collection("investmentAccounts").document(inv_id).get()
          .to_dict()["total_value_sgd"] == SENTINEL)

    status, body = refresh(token, {"item_id": item_x})
    check("single-Item refresh (item_id given) succeeds",
          status == 200 and body.get("status") == "refreshed"
          and [e["item_id"] for e in body.get("items", [])] == [item_x],
          f"HTTP {status} {body.get('status')}")

    truth = plaid_accounts(token_x)  # ground truth, straight from Plaid
    d = user_ref.collection("plaidAccounts").document(dep).get().to_dict()
    check("depository balance_current restored to Plaid's value",
          d["balance_current"] == truth[dep]["balances"]["current"]
          and d["balance_current"] != SENTINEL,
          f"stored={d['balance_current']} plaid={truth[dep]['balances']['current']}")
    check("depository balance_available restored to Plaid's value",
          d["balance_available"] == truth[dep]["balances"]["available"]
          and d["balance_available"] != SENTINEL,
          f"stored={d['balance_available']} plaid={truth[dep]['balances']['available']}")
    c = user_ref.collection("plaidAccounts").document(cred).get().to_dict()
    check("credit balance_current restored to the OWED amount (Plaid current)",
          c["balance_current"] == truth[cred]["balances"]["current"]
          and c["balance_current"] != SENTINEL,
          f"stored={c['balance_current']} plaid current="
          f"{truth[cred]['balances']['current']} available="
          f"{truth[cred]['balances']['available']}")
    check("credit balance_available is Plaid's available (no fallback to current)",
          c["balance_available"] == truth[cred]["balances"]["available"],
          f"stored={c['balance_available']}")
    check("refresh wrote no field beyond balances/last_synced on the account "
          "(type and subtype intact)",
          c["account_type"] == "credit" and d["account_type"] == "depository"
          and "liability" in c)

    holdings = plaid_service.get_plaid_client().investments_holdings_get(
        InvestmentsHoldingsGetRequest(access_token=token_x)).to_dict()
    plaid_total = sum(h.get("institution_value") or 0 for h in holdings["holdings"]
                      if h["account_id"] == inv_id)
    inv = user_ref.collection("investmentAccounts").document(inv_id).get().to_dict()
    check("investmentAccounts total_value_sgd restored",
          inv["total_value_sgd"] != SENTINEL
          and inv["total_value_sgd"] == original_inv_total
          and abs(inv["total_value_sgd"] - plaid_total) < 1e-6,
          f"stored={inv['total_value_sgd']} before_tamper={original_inv_total} "
          f"plaid_sum={plaid_total}")


def test_derived(db, uid):
    print("\n--- 5. derived figures")
    profile = (db.collection("users").document(uid).collection("financialProfile")
               .document("profile").get().to_dict())
    nw, assets, debts = (profile.get("net_worth_sgd"), profile.get("total_assets_sgd"),
                         profile.get("total_debts_sgd"))
    check("net_worth_sgd == total_assets_sgd - total_debts_sgd to the cent",
          None not in (nw, assets, debts) and round(assets - debts, 2) == nw,
          f"{assets} - {debts} = {nw}")
    # Independent recomputation from the restored stored values.
    user_ref = db.collection("users").document(uid)
    a = d = 0.0
    for s in user_ref.collection("plaidAccounts").stream():
        v = s.to_dict()
        if v["account_type"] == "depository":
            a += v.get("balance_current") or 0
        else:
            d += v.get("balance_current") or 0
    for s in user_ref.collection("investmentAccounts").stream():
        a += s.to_dict().get("total_value_sgd") or 0
    check("stored figures match a fresh computation from restored balances",
          round(a, 2) == assets and round(d, 2) == debts
          and financial_profile.compute_balance_totals(db, uid)
          == {"total_assets_sgd": assets, "total_debts_sgd": debts, "net_worth_sgd": nw},
          f"fresh assets={round(a, 2)} debts={round(d, 2)}")
    check("no sentinel leaked into the derived figures",
          SENTINEL not in (nw, assets, debts))


def test_isolation(db, uid_a, item_x):
    print("\n--- 7. isolation")
    user_ref = db.collection("users").document(uid_a)
    dep = next(a for a in accounts_of(db, uid_a, item_x)
               if a.to_dict()["account_type"] == "depository")
    dep.reference.update({"balance_current": SENTINEL, "balance_available": SENTINEL})
    backdate(db, uid_a, item_x)
    a_sync_before = item_doc(db, uid_a, item_x)["last_synced"]

    uid_b, token_b = sign_up("b")
    onboard(db, uid_b)
    status, body = refresh(token_b, {})
    check("user B's own refresh is no_connections",
          status == 200 and body.get("status") == "no_connections")
    status, body = refresh(token_b, {"item_id": item_x})
    check("user B passing user A's item_id gets 404",
          status == 404, f"HTTP {status}")
    d = user_ref.collection("plaidAccounts").document(dep.id).get().to_dict()
    check("user A's tampered balances untouched by B's calls",
          d["balance_current"] == SENTINEL and d["balance_available"] == SENTINEL)
    check("user A's plaidItems.last_synced untouched by B's calls",
          item_doc(db, uid_a, item_x)["last_synced"] == a_sync_before)
    return uid_b, token_b


def test_bank_only_item(db, uid, token, item_y, token_y):
    """An Item with no investment accounts takes the bank-only path: balances
    are still refreshed, the investments step is skipped, and that skip is
    not an error. Item Y is made bank-only by deleting its investmentAccounts
    documents (with their transactions) through the Admin SDK."""
    print("\n--- 7b. bank-only Item (no investment accounts)")
    user_ref = db.collection("users").document(uid)
    for inv in accounts_of(db, uid, item_y, "investmentAccounts"):
        for sub in inv.reference.collections():
            for doc in sub.list_documents():
                doc.delete()
        inv.reference.delete()
    accts = {a.id: a.to_dict() for a in accounts_of(db, uid, item_y)}
    dep = next(k for k, v in accts.items() if v["account_type"] == "depository")
    cred = next(k for k, v in accts.items() if v["account_type"] == "credit")
    for acct_id in (dep, cred):
        user_ref.collection("plaidAccounts").document(acct_id).update(
            {"balance_current": SENTINEL, "balance_available": SENTINEL})
    backdate(db, uid, item_y)
    check("bank-only setup: Item Y has no investmentAccounts left",
          not accounts_of(db, uid, item_y, "investmentAccounts"))

    status, body = refresh(token, {"item_id": item_y})
    entry = (body.get("items") or [{}])[0]
    check("bank-only Item refresh returns status refreshed",
          status == 200 and body.get("status") == "refreshed"
          and entry.get("item_id") == item_y and entry.get("status") == "refreshed",
          f"HTTP {status} {body.get('status')} {entry.get('status')}")
    check("bank-only Item: investments_synced false, no error_code",
          entry.get("investments_synced") is False and entry.get("error_code") is None,
          f"investments_synced={entry.get('investments_synced')} "
          f"error_code={entry.get('error_code')}")
    truth = plaid_accounts(token_y)
    stored = {k: user_ref.collection("plaidAccounts").document(k).get().to_dict()
              for k in (dep, cred)}
    check("bank-only Item: depository and credit balances restored to Plaid's",
          all(stored[k]["balance_current"] == truth[k]["balances"]["current"]
              and stored[k]["balance_available"] == truth[k]["balances"]["available"]
              and stored[k]["balance_current"] != SENTINEL for k in (dep, cred)),
          str({k[:8]: (stored[k]["balance_current"], truth[k]["balances"]["current"])
               for k in (dep, cred)}))


def test_partial_failure(db, uid, token, item_x, item_y, token_y):
    print("\n--- 8. partial failure (sandbox/item/reset_login)")
    try:
        plaid_service.get_plaid_client().sandbox_item_reset_login(
            SandboxItemResetLoginRequest(access_token=token_y))
        method = "real"
    except plaid.ApiException as exc:
        method = f"unavailable ({plaid_code(exc)})"
    check("sandbox/item/reset_login put Item Y into ITEM_LOGIN_REQUIRED",
          method == "real", method)
    for i in (item_x, item_y):
        backdate(db, uid, i)
    x_acct_before = {a.id: a.to_dict()["last_synced"] for a in accounts_of(db, uid, item_x)}
    y_acct_before = {a.id: a.to_dict()["last_synced"] for a in accounts_of(db, uid, item_y)}

    status, body = refresh(token, {})
    entries = {e["item_id"]: e for e in body.get("items", [])}
    check("refresh with one broken Item returns 200 status partial",
          status == 200 and body.get("status") == "partial",
          f"HTTP {status} {body.get('status')}")
    y = entries.get(item_y, {})
    check("broken Item reported failed with ITEM_LOGIN_REQUIRED",
          y.get("status") == "failed" and y.get("error_code") == "ITEM_LOGIN_REQUIRED",
          f"{y.get('status')} {y.get('error_code')}")
    x = entries.get(item_x, {})
    check("the other Item still refreshed",
          x.get("status") == "refreshed" and x.get("error_code") is None)
    x_after = {a.id: a.to_dict() for a in accounts_of(db, uid, item_x)}
    check("the other Item's accounts were written (last_synced advanced, "
          "isolation sentinel cleared)",
          all(x_after[k]["last_synced"] > x_acct_before[k] for k in x_after)
          and all(v["balance_current"] != SENTINEL for v in x_after.values()))
    check("the broken Item's accounts were not written",
          {a.id: a.to_dict()["last_synced"] for a in accounts_of(db, uid, item_y)}
          == y_acct_before)
    check("derived refresh still ran after a partial result",
          body.get("financial_profile") is not None and "score" in body)


def test_auth_and_leaks(token):
    print("\n--- 9. auth and leaks")
    status, _ = refresh(None, {})
    check("refresh without a token is rejected 401", status == 401, f"HTTP {status}")
    status, _ = refresh(token, {"item_id": "item-does-not-exist"})
    check("unknown item_id 404s for a legitimate caller", status == 404,
          f"HTTP {status}")
    leaks = [t for t in responses_seen
             if "access-sandbox-" in t or '"access_token"' in t
             or any(tok in t for tok in access_tokens)]
    check("no response body contained an access_token",
          not leaks, f"{len(responses_seen)} responses scanned")


def verify_cleanup(db):
    print("\n--- cleanup")
    for dead_uid in created_uids:
        try:
            account_deletion.delete_account(db, dead_uid)
        except Exception as exc:
            print(f"  account_deletion failed for {dead_uid}: {exc}")
            try:
                admin_auth.delete_user(dead_uid)
            except Exception:
                pass
    gone_auth = all_gone_fs = True
    for dead_uid in created_uids:
        try:
            admin_auth.get_user(dead_uid)
            gone_auth = False
        except admin_auth.UserNotFoundError:
            pass
        ref = db.collection("users").document(dead_uid)
        if ref.get().exists or any(
                list(c.limit(1).stream()) for c in ref.collections()):
            all_gone_fs = False
    dead = [token_is_live(t) for t in access_tokens]
    check("cleanup: every throwaway Auth user deleted", gone_auth,
          f"{len(created_uids)} users")
    check("cleanup: no Firestore data left under any throwaway uid", all_gone_fs)
    check("cleanup: every Sandbox Item revoked at Plaid (ITEM_NOT_FOUND)",
          all((not live) and code == "ITEM_NOT_FOUND" for live, code in dead),
          str(dead))


def main():
    init_firebase()
    db = get_db()
    proc, log = start_server()
    elapsed = None
    try:
        test_no_connections(db)

        uid, token = sign_up("a")
        onboard(db, uid)
        print(f"\n--- user A uid={uid}: connecting two Items via /plaid/exchange-token")
        item_x, token_x = connect_via_endpoint(db, uid, token)
        item_y, token_y = connect_via_endpoint(db, uid, token)
        check("two distinct Sandbox Items from one institution",
              item_x != item_y
              and item_doc(db, uid, item_x)["institution_id"]
              == item_doc(db, uid, item_y)["institution_id"])
        q = [quiesce(db, uid, i) for i in (item_x, item_y)]
        check("both Items quiesced (no further Sandbox transaction waves)", all(q))
        check("both Items have investment accounts to exercise that path",
              all(accounts_of(db, uid, i, "investmentAccounts") for i in (item_x, item_y)))

        elapsed = test_two_items(db, uid, token, item_x, item_y)
        test_cooldown(db, uid, token, item_x, item_y)
        test_balances_and_investments(db, uid, token, item_x, token_x)
        test_derived(db, uid)
        test_isolation(db, uid, item_x)
        test_bank_only_item(db, uid, token, item_y, token_y)
        test_partial_failure(db, uid, token, item_x, item_y, token_y)
        test_auth_and_leaks(token)
    finally:
        # Cleanup first, so nothing below can strand throwaway users or live
        # Sandbox Items.
        verify_cleanup(db)
        stop_server(proc)
        log.close()
        server_log = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        check("server log contains no access_token",
              "access-sandbox-" not in server_log
              and not any(t in server_log for t in access_tokens),
              f"{len(server_log.splitlines())} log lines")
        try:
            LOG_PATH.unlink()
        except OSError:
            pass

    if elapsed is not None:
        print(f"\nwall-clock for one 2-Item refresh: {elapsed:.1f}s")
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    for name in failed:
        print(f"  FAILED: {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
