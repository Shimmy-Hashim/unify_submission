"""End-to-end Home Phase 1 verification — financialProfile collection,
derived fields, security rules, and the compute_insurance_features()
dependency that income data unblocks (brief Section 5, Home phase plan).

    cd backend && venv\\Scripts\\activate && python scripts/verify_financial_profile.py

Exercises the real chain with three throwaway Firebase users and one
throwaway Sandbox Item (First Platypus Bank):
  user A — fresh-signup shape: users/{uid} + financialProfile created via
           the client REST API in exactly the shape the Flutter onboarding
           transaction writes; rules + document shape + empty-data refresh
  user B — adversary: cross-user reads/writes of A's financialProfile must
           all be denied by the rules
  user C — robustness against the doc-missing state (test/dev accounts
           only — the app itself always creates financialProfile inside
           the onboarding transaction, and there is deliberately no
           retrofit UI): users/{uid} doc only, connects a real Sandbox
           Item BEFORE financialProfile exists (sync refresh must no-op,
           never invent the doc), then the doc is created and the
           on-demand refresh must pick up real balances/transactions; net
           worth and expenses are re-derived independently and compared;
           real insurancePolicies are then attached and
           compute_insurance_features() runs end-to-end off Firestore data
Everything is deleted afterwards.
"""
import json
import math
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

from data import financial_profile, plaid_service  # noqa: E402
from ml import scoring  # noqa: E402
from plaid.model.products import Products  # noqa: E402
from plaid.model.sandbox_public_token_create_request import (  # noqa: E402
    SandboxPublicTokenCreateRequest,
)

PROJECT = "YOUR_FIREBASE_PROJECT_ID"
GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
FIRESTORE = f"https://firestore.googleapis.com/v1/projects/{PROJECT}/databases/(default)/documents"

SANDBOX_INSTITUTION = "ins_109508"  # First Platypus Bank
MONTHLY_INCOME = 4500.0

passed, failed = [], []


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


def rest_user_doc(email):
    return {"fields": {
        "name": {"stringValue": "E2E Profile User"},
        "email": {"stringValue": email},
        "date_of_birth": {"timestampValue": "1998-03-10T00:00:00Z"},
        "dependants": {"integerValue": "0"},
        "created_at": {"timestampValue": "2026-07-07T00:00:00Z"},
        "last_active": {"timestampValue": "2026-07-07T00:00:00Z"},
    }}


def rest_financial_profile(income):
    """Exactly UserFinancialProfile.createMap()'s shape (8 fields)."""
    return {"fields": {
        "monthly_income_sgd": {"doubleValue": income},
        "income_source": {"stringValue": "user_entered"},
        "monthly_expenses_sgd": {"doubleValue": 0.0},
        "net_worth_sgd": {"doubleValue": 0.0},
        "overall_score": {"nullValue": None},
        "checklist_flags": {"nullValue": None},
        "checklist_last_updated": {"nullValue": None},
        "ml_profile": {"nullValue": None},
    }}


def profile_url(uid):
    return f"{FIRESTORE}/users/{uid}/financialProfile/profile"


def savings_screen_avg_monthly_spent(db, uid, now=None):
    """Line-for-line mirror of the Savings screen's Emergency Fund figure
    (SavingsSummary.avgMonthlySpent in lib/models/plaid_account.dart):
    per depository account, the 100 most recent transactions date-desc,
    keep those inside 30 days with amount > 0 (Plaid positive = outflow),
    sum. Credit accounts excluded (repayment double-count)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    user_ref = db.collection("users").document(uid)
    total = 0.0
    for acct in user_ref.collection("plaidAccounts").stream():
        if acct.to_dict().get("account_type") != "depository":
            continue
        for snap in (user_ref.collection("plaidAccounts").document(acct.id)
                     .collection("transactions")
                     .order_by("date", direction="DESCENDING")
                     .limit(100).stream()):
            txn = snap.to_dict()
            if (txn.get("date") and txn.get("amount") is not None
                    and txn["date"] >= cutoff and txn["amount"] > 0):
                total += float(txn["amount"])
    return round(total, 2)


def manual_net_worth_parts(db, uid):
    """Independent recomputation of the Section 8 net-worth definition,
    straight off the Firestore documents."""
    user_ref = db.collection("users").document(uid)
    depository = credit_loan = investments = 0.0
    for snap in user_ref.collection("plaidAccounts").stream():
        account = snap.to_dict()
        balance = float(account.get("balance_current") or 0.0)
        if account.get("account_type") == "depository":
            depository += balance
        elif account.get("account_type") in ("credit", "loan"):
            credit_loan += balance
    for snap in user_ref.collection("investmentAccounts").stream():
        investments += float(snap.to_dict().get("total_value_sgd") or 0.0)
    return depository, investments, credit_loan


def find_real_insurance_policies(db, exclude_uids):
    """Returns (uid, [policy dicts]) for a real (non-test) user with parsed
    insurancePolicies, or (None, []). Read-only — the real user's data is
    never modified."""
    by_user = {}
    for snap in db.collection_group("insurancePolicies").limit(50).stream():
        uid = snap.reference.parent.parent.id
        if uid in exclude_uids:
            continue
        by_user.setdefault(uid, []).append(snap.to_dict())
    if not by_user:
        return None, []
    uid = sorted(by_user, key=lambda u: len(by_user[u]), reverse=True)[0]
    return uid, by_user[uid]


def delete_collection(ref, batch_size=200):
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
    uid_a, token_a = sign_up(f"e2e.fp.a.{stamp}@unify-test.example", "E2eFpA-12345")
    uid_b, token_b = sign_up(f"e2e.fp.b.{stamp}@unify-test.example", "E2eFpB-12345")
    uid_c, token_c = sign_up(f"e2e.fp.c.{stamp}@unify-test.example", "E2eFpC-12345")
    print(f"test users: A={uid_a} B={uid_b} C={uid_c}")

    from fastapi.testclient import TestClient
    import main as backend_main
    client = TestClient(backend_main.app)

    def api(method, path, token, body=None):
        r = client.request(method, path,
                           headers={"Authorization": f"Bearer {token}"},
                           json=body)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {}

    try:
        # --- user A: fresh-signup shape + rules -----------------------------
        status, _ = http("PATCH", f"{FIRESTORE}/users/{uid_a}",
                         rest_user_doc(f"e2e.fp.a.{stamp}@unify-test.example"),
                         token_a)
        check("A creates own users/{uid} (rules allow)", status == 200,
              f"HTTP {status}")
        status, _ = http("PATCH", profile_url(uid_a),
                         rest_financial_profile(MONTHLY_INCOME), token_a)
        check("A creates own financialProfile/profile (rules allow)",
              status == 200, f"HTTP {status}")

        doc = financial_profile.profile_ref(db, uid_a).get().to_dict()
        schema_fields = {"monthly_income_sgd", "income_source",
                         "monthly_expenses_sgd", "net_worth_sgd",
                         "overall_score", "checklist_flags",
                         "checklist_last_updated", "ml_profile"}
        check("financialProfile doc carries exactly the Section 8 fields",
              doc is not None and set(doc) == schema_fields,
              f"fields={sorted(doc or {})}")
        check("income stored as float, source user_entered, Phase 3 fields null",
              isinstance(doc.get("monthly_income_sgd"), float)
              and doc["monthly_income_sgd"] == MONTHLY_INCOME
              and doc["income_source"] == "user_entered"
              and doc["overall_score"] is None
              and doc["checklist_flags"] is None
              and doc["checklist_last_updated"] is None
              and doc["ml_profile"] is None)

        # --- rules: cross-user + anonymous access denied ----------------------
        status, _ = http("GET", profile_url(uid_a), token=token_b)
        check("B blocked from reading A's financialProfile (rules deny)",
              status == 403, f"HTTP {status}")
        status, _ = http("PATCH", profile_url(uid_a),
                         rest_financial_profile(9999.0), token_b)
        check("B blocked from writing A's financialProfile (rules deny)",
              status == 403, f"HTTP {status}")
        status, _ = http("GET", profile_url(uid_a))
        check("anonymous blocked from reading A's financialProfile",
              status == 403, f"HTTP {status}")
        status, _ = http("GET", profile_url(uid_a), token=token_a)
        check("owner's client can read own financialProfile (rules allow)",
              status == 200, f"HTTP {status}")

        # --- refresh with no Plaid data: honest zeros -------------------------
        status, body = api("POST", "/profile/refresh", token_a)
        check("POST /profile/refresh with no Plaid data returns 0.0 / 0.0",
              status == 200 and body.get("monthly_expenses_sgd") == 0.0
              and body.get("net_worth_sgd") == 0.0, f"HTTP {status} {body}")

        r = client.post("/profile/refresh")
        check("refresh without auth is rejected 401", r.status_code == 401,
              f"HTTP {r.status_code}")

        # --- user C: financialProfile-missing state (test/dev-only shape) ----
        status, _ = http("PATCH", f"{FIRESTORE}/users/{uid_c}",
                         rest_user_doc(f"e2e.fp.c.{stamp}@unify-test.example"),
                         token_c)
        check("C has users/{uid} but no financialProfile (doc-missing state)",
              status == 200
              and not financial_profile.profile_ref(db, uid_c).get().exists)

        status, _ = api("POST", "/profile/refresh", token_c)
        check("refresh before income setup returns 404 (doc never created)",
              status == 404
              and not financial_profile.profile_ref(db, uid_c).get().exists,
              f"HTTP {status}")

        # --- C connects a real Sandbox Item BEFORE income setup ---------------
        pt = plaid_service.get_plaid_client().sandbox_public_token_create(
            SandboxPublicTokenCreateRequest(
                institution_id=SANDBOX_INSTITUTION,
                initial_products=[Products("transactions"),
                                  Products("investments")],
            )
        ).to_dict()["public_token"]
        status, body = api("POST", "/plaid/exchange-token", token_c,
                           {"public_token": pt})
        item_id = body.get("item_id")
        check("C connects Sandbox Item", status == 200 and bool(item_id),
              f"institution={body.get('institution_name')}")
        check("sync-triggered refresh no-ops while financialProfile missing",
              body.get("financial_profile") is None
              and not financial_profile.profile_ref(db, uid_c).get().exists)

        # Quiesce the initial transaction waves (same pattern as
        # verify_plaid.py) so the expenses figures are stable.
        deadline = time.time() + 120
        while time.time() < deadline:
            status, body = api("POST", "/plaid/sync-transactions", token_c,
                               {"item_id": item_id})
            if status == 200 and not body["results"][0]["added"]:
                break
            time.sleep(5)

        # --- C's financialProfile created after the fact ----------------------
        status, _ = http("PATCH", profile_url(uid_c),
                         rest_financial_profile(MONTHLY_INCOME), token_c)
        check("C creates own financialProfile (rules allow)",
              status == 200, f"HTTP {status}")

        status, body = api("POST", "/profile/refresh", token_c)
        check("POST /profile/refresh succeeds with real Plaid data",
              status == 200, f"HTTP {status} {body}")

        stored = financial_profile.profile_ref(db, uid_c).get().to_dict()

        # --- net worth vs independently recomputed components -----------------
        depository, investments, credit_loan = manual_net_worth_parts(db, uid_c)
        expected_net_worth = round(depository + investments - credit_loan, 2)
        check("net_worth_sgd = depository + investments - credit/loan balances",
              stored.get("net_worth_sgd") == expected_net_worth,
              f"stored={stored.get('net_worth_sgd')}, manual: "
              f"{depository} + {investments} - {credit_loan} "
              f"= {expected_net_worth}")
        check("total_assets_sgd = depository + investments (independent recompute)",
              stored.get("total_assets_sgd")
              == round(depository + investments, 2),
              f"stored={stored.get('total_assets_sgd')}, manual="
              f"{round(depository + investments, 2)}")
        check("total_debts_sgd = credit/loan balances (independent recompute)",
              stored.get("total_debts_sgd") == round(credit_loan, 2),
              f"stored={stored.get('total_debts_sgd')}, manual="
              f"{round(credit_loan, 2)}")
        check("stored assets - stored debts reconciles to stored net worth, "
              "to the cent",
              round(stored.get("total_assets_sgd", 0)
                    - stored.get("total_debts_sgd", 0), 2)
              == stored.get("net_worth_sgd"),
              f"{stored.get('total_assets_sgd')} - "
              f"{stored.get('total_debts_sgd')} vs {stored.get('net_worth_sgd')}")
        check("net worth actually includes real balances (non-zero components)",
              depository > 0 and credit_loan > 0,
              f"depository={depository}, investments={investments}, "
              f"debt={credit_loan}")

        # --- expenses parity with the Savings screen's estimator --------------
        savings_screen_value = savings_screen_avg_monthly_spent(db, uid_c)
        check("monthly_expenses_sgd equals Savings screen avgMonthlySpent",
              stored.get("monthly_expenses_sgd") == savings_screen_value,
              f"stored={stored.get('monthly_expenses_sgd')}, "
              f"savings-screen mirror={savings_screen_value}")

        # --- sync trigger refreshes derived fields ----------------------------
        status, body = api("POST", "/plaid/sync-transactions", token_c, {})
        fp = body.get("financial_profile") or {}
        check("sync-transactions response now carries refreshed derived fields",
              status == 200
              and fp.get("net_worth_sgd") == expected_net_worth
              and "monthly_expenses_sgd" in fp, f"financial_profile={fp}")

        # --- Step 4: compute_insurance_features() end-to-end ------------------
        real_uid, policies = find_real_insurance_policies(
            db, {uid_a, uid_b, uid_c})
        if real_uid:
            print(f"INFO  using {len(policies)} real parsed policies from "
                  f"user {real_uid} (read-only)")
            # Attach copies to C so one user carries BOTH collections; the
            # copies are deleted with the rest of C in cleanup.
            for policy in policies:
                ref = (db.collection("users").document(uid_c)
                       .collection("insurancePolicies").document())
                ref.set({**policy, "policy_id": ref.id, "uid": uid_c})
        else:
            print("INFO  no real insurancePolicies found — seeding one "
                  "parser-shaped policy")
            ref = (db.collection("users").document(uid_c)
                   .collection("insurancePolicies").document())
            ref.set({
                "policy_id": ref.id, "uid": uid_c,
                "policy_name": "Term Life Cover", "insurer": "E2E Insurer",
                "policy_type": "life", "policy_type_raw": "Term Life Plan",
                "sum_assured_sgd": 500000.0, "annual_premium_sgd": 1200.0,
                "renewal_date": None, "exclusions": [], "coverage_gaps": [],
                "original_filename": "policy.pdf", "uploaded_at": None,
            })

        # The exact reads Phase 3's inference endpoint will do.
        profile_doc = financial_profile.profile_ref(db, uid_c).get().to_dict()
        stored_policies = [
            s.to_dict() for s in db.collection("users").document(uid_c)
            .collection("insurancePolicies").stream()
        ]
        annual = scoring.annual_income_from_financial_profile(profile_doc)
        check("annual income derived from financialProfile (monthly x 12)",
              annual == MONTHLY_INCOME * 12, f"annual={annual}")

        features = scoring.compute_insurance_features(stored_policies, annual)
        expected_keys = {"life_coverage_ratio", "ci_coverage_ratio",
                         "hospitalisation_coverage",
                         "income_protection_coverage",
                         "insurance_spend_ratio"}
        check("compute_insurance_features runs end-to-end off Firestore data",
              set(features) == expected_keys
              and all(isinstance(v, (int, float)) and math.isfinite(v)
                      and v >= 0 for v in features.values()),
              f"features={features} "
              f"(from {len(stored_policies)} policies, income={annual})")

        # Zero/missing income contracts (documented, deliberate):
        try:
            scoring.compute_insurance_features(
                stored_policies,
                scoring.annual_income_from_financial_profile(
                    {"monthly_income_sgd": 0.0}))
            zero_ok = False
        except ValueError:
            zero_ok = True
        check("zero income raises ValueError (Phase 3 shows 'unavailable', "
              "never a fabricated ratio)", zero_ok)
        try:
            scoring.annual_income_from_financial_profile({})
            missing_ok = False
        except ValueError:
            missing_ok = True
        check("missing monthly_income_sgd raises ValueError", missing_ok)

    finally:
        n = 0
        for uid in (uid_a, uid_b, uid_c):
            user_ref = db.collection("users").document(uid)
            for sub in user_ref.collections():
                n += delete_collection(sub)
            user_ref.delete()
            admin_auth.delete_user(uid)
        print(f"cleanup: removed 3 test users and {n} Firestore docs")

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()


