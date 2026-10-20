"""End-to-end Home Phase 2 verification — compute_banking_features() and
compute_investment_features() against a real Sandbox-connected account
(brief Section 5 feature table, Home phased plan).

    cd backend && venv\\Scripts\\activate && python scripts/verify_home_phase2.py

One throwaway Firebase user gets a users/{uid} doc (with date_of_birth +
dependants), a financialProfile, and a real Sandbox Item (banking +
investments). Every feature both functions return is then re-derived
independently from the raw Firestore documents and compared, followed by
the contract cases: income=0 raises, income=None keeps income-independent
features computing, empty plaidAccounts / investmentAccounts return None
(never fabricated zeros), dependants is readable for Phase 3's assembler,
and compute_insurance_features() is regression-checked. Everything is
deleted afterwards.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
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

SANDBOX_INSTITUTION = "ins_109508"  # First Platypus Bank
MONTHLY_INCOME = 4500.0
DATE_OF_BIRTH = date(1998, 3, 10)
DEPENDANTS = 2

passed, failed = [], []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def sign_up(email, password):
    req = urllib.request.Request(
        f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
        method="POST",
        data=json.dumps({"email": email, "password": password,
                         "returnSecureToken": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"\nSign-up failed ({e.code}): {e.read()!r}")
    return body["localId"], body["idToken"]


def age_from_dob(dob: date, on: date) -> int:
    """Same rule as the Flutter AppUser.age getter — age is always computed
    from date_of_birth, never stored (Section 14)."""
    years = on.year - dob.year
    if (on.month, on.day) < (dob.month, dob.day):
        years -= 1
    return years


def fetch_inputs(db, uid):
    """The reads Phase 3's feature-vector assembler will do."""
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get().to_dict()
    profile = financial_profile.profile_ref(db, uid).get().to_dict()
    bank_accounts = [
        s.to_dict() for s in user_ref.collection("plaidAccounts").stream()
    ]
    inv_accounts = [
        s.to_dict()
        for s in user_ref.collection("investmentAccounts").stream()
    ]
    inv_txns = [
        t.to_dict()
        for s in user_ref.collection("investmentAccounts").stream()
        for t in s.reference.collection("transactions").stream()
    ]
    return user_doc, profile, bank_accounts, inv_accounts, inv_txns


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
    email = f"e2e.hp2.{stamp}@unify-test.example"
    uid, token = sign_up(email, "E2eHp2-12345")
    print(f"test user uid={uid}")

    from fastapi.testclient import TestClient
    import main as backend_main
    client = TestClient(backend_main.app)

    def api(method, path, body=None):
        r = client.request(method, path,
                           headers={"Authorization": f"Bearer {token}"},
                           json=body)
        return r.status_code, r.json()

    try:
        # --- seed the exact documents onboarding creates ----------------------
        db.collection("users").document(uid).set({
            "name": "E2E HP2 User", "email": email,
            "date_of_birth": datetime.combine(
                DATE_OF_BIRTH, datetime.min.time(), tzinfo=timezone.utc),
            "dependants": DEPENDANTS,
        })
        financial_profile.profile_ref(db, uid).set({
            "monthly_income_sgd": MONTHLY_INCOME,
            "income_source": "user_entered",
            "monthly_expenses_sgd": 0.0, "net_worth_sgd": 0.0,
            "overall_score": None, "checklist_flags": None,
            "checklist_last_updated": None, "ml_profile": None,
        })

        # --- real Sandbox connection (banking + investments) ------------------
        pt = plaid_service.get_plaid_client().sandbox_public_token_create(
            SandboxPublicTokenCreateRequest(
                institution_id=SANDBOX_INSTITUTION,
                initial_products=[Products("transactions"),
                                  Products("investments")],
            )
        ).to_dict()["public_token"]
        status, body = api("POST", "/plaid/exchange-token",
                           {"public_token": pt})
        item_id = body.get("item_id")
        check("Sandbox Item connected (banking + investments)",
              status == 200 and bool(item_id),
              f"institution={body.get('institution_name')}")

        deadline = time.time() + 120
        while time.time() < deadline:
            status, body = api("POST", "/plaid/sync-transactions",
                               {"item_id": item_id})
            if status == 200 and not body["results"][0]["added"]:
                break
            time.sleep(5)
        api("POST", "/profile/refresh")

        # --- assemble inputs the way Phase 3 will -----------------------------
        user_doc, profile, bank_accounts, inv_accounts, inv_txns = (
            fetch_inputs(db, uid)
        )
        today = datetime.now(timezone.utc).date()
        age = age_from_dob(DATE_OF_BIRTH, today)
        annual = scoring.annual_income_from_financial_profile(profile)
        check("inputs assembled from Firestore",
              bool(bank_accounts) and bool(inv_accounts)
              and annual == MONTHLY_INCOME * 12,
              f"{len(bank_accounts)} bank accounts, {len(inv_accounts)} "
              f"investment accounts, {len(inv_txns)} inv txns, age={age}")

        # --- Step 3: dependants readable for the Phase 3 assembler ------------
        check("dependants readable from users/{uid} alongside the features",
              isinstance(user_doc.get("dependants"), int)
              and user_doc["dependants"] == DEPENDANTS,
              f"dependants={user_doc.get('dependants')}")

        # --- banking features vs independent recomputation --------------------
        banking = scoring.compute_banking_features(
            profile, bank_accounts, annual)
        print(f"INFO  banking features: {banking}")

        monthly_income = MONTHLY_INCOME
        expenses = profile["monthly_expenses_sgd"]
        liquid = sum(a["balance_current"] or 0.0 for a in bank_accounts
                     if a["account_type"] == "depository")
        payments = sum((a.get("liability") or {})
                       .get("monthly_payment_amount") or 0.0
                       for a in bank_accounts)
        # Lists, not dicts keyed by subtype — the Sandbox fixture carries
        # duplicate subtypes (e.g. two "credit card" accounts).
        healthy_bal = [(a["account_subtype"], a["balance_current"])
                       for a in bank_accounts
                       if (a["account_subtype"] or "").lower()
                       in ("mortgage", "student", "auto", "home equity")]
        unhealthy_bal = [(a["account_subtype"], a["balance_current"])
                         for a in bank_accounts
                         if (a["account_subtype"] or "").lower()
                         in ("credit card", "paypal", "consumer", "loan",
                             "line of credit", "overdraft")]
        unclassified_debt = [(a["account_subtype"], a["balance_current"])
                             for a in bank_accounts
                             if a["account_type"] in ("credit", "loan")]
        unclassified_debt = [x for x in unclassified_debt
                             if x not in healthy_bal + unhealthy_bal]
        if unclassified_debt:
            print(f"INFO  unclassified debt subtypes in fixture: "
                  f"{unclassified_debt}")
        healthy_sum = sum(b for _, b in healthy_bal)
        debt_sum = healthy_sum + sum(b for _, b in unhealthy_bal)

        check("savings_rate = (income - expenses) / income",
              close(banking["savings_rate"],
                    (monthly_income - expenses) / monthly_income),
              f"{banking['savings_rate']:.4f} "
              f"(income={monthly_income}, expenses={expenses})")
        check("emergency_fund_months = depository liquid / monthly expenses",
              close(banking["emergency_fund_months"], liquid / expenses),
              f"{banking['emergency_fund_months']:.2f} months "
              f"(liquid={liquid}, expenses={expenses})")
        check("debt_to_income_ratio = liability payments / monthly income "
              "(monthly basis both sides)",
              close(banking["debt_to_income_ratio"],
                    payments / monthly_income),
              f"{banking['debt_to_income_ratio']:.4f} "
              f"(payments={payments}/mo)")
        check("healthy_debt_ratio = structured debt / classified debt",
              debt_sum > 0
              and close(banking["healthy_debt_ratio"],
                        healthy_sum / debt_sum),
              f"{banking['healthy_debt_ratio']:.4f} "
              f"(healthy={healthy_bal}, unhealthy={unhealthy_bal})")
        check("net_worth_to_income_ratio = net_worth_sgd / annual income",
              close(banking["net_worth_to_income_ratio"],
                    profile["net_worth_sgd"] / annual),
              f"{banking['net_worth_to_income_ratio']:.4f} "
              f"(net_worth={profile['net_worth_sgd']})")

        # --- investment features vs independent recomputation -----------------
        investment = scoring.compute_investment_features(
            inv_accounts, inv_txns, annual, age)
        print(f"INFO  investment features: {investment}")

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        deposits = sum(abs(t["amount"] or 0.0) for t in inv_txns
                       if t.get("type") == "deposit"
                       and t.get("date") and t["date"] >= cutoff)
        cached_conc = [a["concentration_risk"] for a in inv_accounts
                       if a.get("concentration_risk") is not None]
        total_value = sum(a["total_value_sgd"] or 0.0 for a in inv_accounts)
        equities_value = sum(
            (a.get("asset_allocation") or {}).get("equities", 0.0)
            * (a["total_value_sgd"] or 0.0) for a in inv_accounts)

        check("has_investments is True with real investmentAccounts",
              investment["has_investments"] is True)
        check("investment_rate = 30-day deposits / monthly income",
              close(investment["investment_rate"],
                    deposits / monthly_income, tol=1e-6),
              f"{investment['investment_rate']:.4f} "
              f"(deposits={deposits} in 30d)")
        check("portfolio_concentration = max cached concentration_risk",
              bool(cached_conc)
              and investment["portfolio_concentration"] == max(cached_conc),
              f"{investment['portfolio_concentration']} "
              f"(per-account={cached_conc})")
        expected_dev = (equities_value / total_value) - min(
            1.0, max(0.0, (scoring.RISK_AGE_EQUITY_BASE - age) / 100.0))
        # Label is derived from the constant, not hardcoded: the generator and scoring.py
        # were standardised on 100-minus-age in July 2026, but this string still said
        # "110 - age" and so contradicted the value it was asserting. Deriving it means
        # the two cannot drift apart again.
        check(f"risk_age_deviation = actual equity - "
              f"({scoring.RISK_AGE_EQUITY_BASE} - age)/100 target",
              close(investment["risk_age_deviation"], expected_dev, tol=1e-6),
              f"{investment['risk_age_deviation']:.4f} "
              f"(actual={equities_value / total_value:.4f}, "
              f"target={(scoring.RISK_AGE_EQUITY_BASE - age) / 100.0:.2f})")

        # --- contract cases ----------------------------------------------------
        for fn, args in (
            (scoring.compute_banking_features, (profile, bank_accounts, 0.0)),
            (scoring.compute_investment_features,
             (inv_accounts, inv_txns, 0.0, age)),
        ):
            try:
                fn(*args)
                ok = False
            except ValueError:
                ok = True
            check(f"income=0 raises ValueError ({fn.__name__})", ok)

        b_no_income = scoring.compute_banking_features(
            profile, bank_accounts, None)
        check("income=None: banking income features None, others compute",
              b_no_income["savings_rate"] is None
              and b_no_income["debt_to_income_ratio"] is None
              and b_no_income["net_worth_to_income_ratio"] is None
              and close(b_no_income["emergency_fund_months"],
                        liquid / expenses)
              and b_no_income["healthy_debt_ratio"] is not None)
        i_no_income = scoring.compute_investment_features(
            inv_accounts, inv_txns, None, age)
        check("income=None: has_investments/concentration/risk-age still "
              "compute, investment_rate None",
              i_no_income["has_investments"] is True
              and i_no_income["investment_rate"] is None
              and i_no_income["portfolio_concentration"]
              == max(cached_conc)
              and i_no_income["risk_age_deviation"] is not None)

        b_empty = scoring.compute_banking_features(profile, [], annual)
        check("no plaidAccounts: all banking features None (not zeros)",
              all(v is None for v in b_empty.values()),
              f"{b_empty}")
        i_empty = scoring.compute_investment_features([], [], annual, age)
        check("no investmentAccounts: has_investments False, rest None",
              i_empty["has_investments"] is False
              and i_empty["investment_rate"] is None
              and i_empty["portfolio_concentration"] is None
              and i_empty["risk_age_deviation"] is None)

        # --- insurance regression ----------------------------------------------
        policies = [
            {"policy_type": "life", "sum_assured_sgd": 500000.0,
             "annual_premium_sgd": 1200.0},
            {"policy_type": "personal_accident", "sum_assured_sgd": 100000.0,
             "annual_premium_sgd": 240.0},
        ]
        ins = scoring.compute_insurance_features(policies, annual)
        check("compute_insurance_features unchanged (regression)",
              close(ins["life_coverage_ratio"], 500000.0 / annual)
              and close(ins["insurance_spend_ratio"], 1440.0 / annual)
              and ins["hospitalisation_coverage"] == 0,
              f"life={ins['life_coverage_ratio']:.4f}")

    finally:
        user_ref = db.collection("users").document(uid)
        n = 0
        for sub in user_ref.collections():
            n += delete_collection(sub)
        user_ref.delete()
        admin_auth.delete_user(uid)
        print(f"cleanup: removed test user and {n} Firestore docs")

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
