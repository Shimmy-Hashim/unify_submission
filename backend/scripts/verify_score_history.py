"""Score History verification — the snapshot writer, its rules, its cascade,
and GET /score/history.

    cd backend && venv\\Scripts\\activate && python scripts/verify_score_history.py
    python scripts/verify_score_history.py --pure-only     # skip section B

Two sections, split the same way verify_score_pipeline.py splits its own, and
for the same reason: section A runs anywhere with no credentials, section B
needs a live Firebase project.

  A. THE PAYLOAD — no Firebase, no network. snapshot_payload() and
     snapshot_date() driven with constructed pipeline results, including the
     null-score cases, plus the structural wiring checks (is the hook actually
     the shared one? does every trigger reach it? is scoreHistory actually in
     the deletion list and the rules?).

  B. END TO END — one throwaway Firebase user, the real endpoints over
     TestClient, real Firestore writes. Proves the behaviours that cannot be
     proved from source: a snapshot really is written on a refresh, a second
     refresh on the same day really does replace the first, a refresh that
     cannot produce a score really does still write a row, the endpoint really
     orders as claimed, and the deletion cascade really does clear the
     collection. Everything is deleted afterwards.

⚠️ SECTION B6 TESTS THE DEPLOYED RULESET, NOT THIS REPO'S firestore.rules.
Firestore rules are enforced by whatever is deployed to the project; editing
the file in this repo changes nothing until it is deployed. B6 therefore FAILS
by design until `firebase deploy --only firestore:rules` has been run, and that
failure is the intended signal — a rule that is committed but not deployed is
not a rule. Section A's rule checks are source checks only and must never be
read as enforcement: A6 proves the text is right, B6 proves it is live.
"""
import inspect
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from data import account_deletion, score_profile  # noqa: E402
from ml import score_pipeline  # noqa: E402
from ml.score_pipeline import compute_score_profile  # noqa: E402

PURE_ONLY = "--pure-only" in sys.argv

passed, failed = [], []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def expect(name, got, want):
    check(name, got == want, f"got {got!r}, want {want!r}")


# ===========================================================================
# A. THE PAYLOAD
# ===========================================================================
print("=" * 78)
print("A1. The snapshot payload — exactly the agreed field set")
print("=" * 78)

# A frozen clock, so the expected document ID below is a constant rather than
# arithmetic that could drift with this script's own bug.
NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
DOB = date(1984, 3, 10)  # age 42 on that clock -> band 3 (40-49)
USER = {"date_of_birth": DOB, "dependants": 2}
PROFILE = {"monthly_income_sgd": 5000.0, "monthly_expenses_sgd": 2500.0,
           "net_worth_sgd": 120000.0}
ACCOUNTS = [
    {"account_type": "depository", "account_subtype": "checking",
     "balance_current": 30000.0},
    {"account_type": "credit", "account_subtype": "credit card",
     "balance_current": 4000.0, "liability": {"monthly_payment_amount": 300.0}},
]
INVESTMENTS = [{"total_value_sgd": 50000.0, "concentration_risk": 0.25,
                "asset_allocation": {"equities": 0.6}}]
TXNS = [{"type": "deposit", "amount": -600.0, "date": NOW}]
POLICIES = [{"policy_type": "life", "sum_assured_sgd": 500000.0,
             "annual_premium_sgd": 1800.0},
            {"policy_type": "hospitalisation", "annual_premium_sgd": 600.0}]


def run(**overrides):
    args = {"user_doc": USER, "profile": PROFILE, "bank_accounts": ACCOUNTS,
            "investment_accounts": INVESTMENTS, "investment_transactions": TXNS,
            "insurance_policies": POLICIES, "now": NOW}
    args.update(overrides)
    return compute_score_profile(**args)


full = run()
snap = score_profile.snapshot_payload(full)

expect("snapshot carries exactly the agreed field set", sorted(snap),
       ["age_band", "computed_at", "date", "domains", "net_worth_sgd",
        "overall_score", "overall_score_reason", "schema_version",
        "score_breakdown"])
expect("date is the UTC calendar day of computed_at", snap["date"], "2026-09-08")
expect("the document ID is that same date string",
       score_profile.snapshot_date(full["computed_at"]), snap["date"])
expect("computed_at is the run's own instant, not re-clocked",
       snap["computed_at"], full["computed_at"])
expect("overall_score copied verbatim from the result",
       snap["overall_score"], full["overall_score"])
expect("schema_version is stamped at write time",
       snap["schema_version"], score_profile.ML_PROFILE_SCHEMA_VERSION)
expect("schema_version tracks the pipeline's own constant",
       snap["schema_version"], score_pipeline.SCHEMA_VERSION)
expect("age_band is stored — a tier index is uninterpretable without it",
       snap["age_band"], 3)
expect("net_worth_sgd is financialProfile's own value, copied not recomputed",
       snap["net_worth_sgd"], PROFILE["net_worth_sgd"])
expect("net_worth_sgd survives a day with no score (needs no income)",
       score_profile.snapshot_payload(
           run(profile={**PROFILE, "monthly_income_sgd": 0.0}))["net_worth_sgd"],
       PROFILE["net_worth_sgd"])
expect("net_worth_sgd is None only when there is no profile to copy from",
       score_profile.snapshot_payload(run(profile=None))["net_worth_sgd"], None)

# Section 14: age is computed from date_of_birth and never stored. The snapshot
# is a new persistence surface, so the same rule has to hold on it too.
check("no raw age or date of birth reaches the snapshot",
      "age" not in snap and "date_of_birth" not in str(snap),
      "only age_band")

print()
print("=" * 78)
print("A2. Domains and breakdown — copied verbatim, not re-derived")
print("=" * 78)

expect("all three domains present, in pipeline order",
       list(snap["domains"]), list(score_pipeline.DOMAIN_ORDER))
for domain in score_pipeline.DOMAIN_ORDER:
    expect(f"  {domain:<11} carries the five cached fields",
           sorted(snap["domains"][domain]),
           ["missing_features", "reason", "status", "tier_index", "tier_label"])
    expect(f"  {domain:<11} copied verbatim from the result",
           snap["domains"][domain], full["domains"][domain])

# The live cache and the snapshot must agree field for field, or a chart built
# on history would disagree with the card built on the cache.
cached = score_profile.cache_payload(full)
expect("snapshot domains are byte-identical to the live cache's",
       snap["domains"], cached["ml_profile"]["domains"])
expect("snapshot score_breakdown is byte-identical to the live cache's",
       snap["score_breakdown"], cached["ml_profile"]["score_breakdown"])
expect("snapshot overall_score agrees with the live cache's",
       snap["overall_score"], cached["overall_score"])

expect("score_breakdown carries the whole map",
       sorted(snap["score_breakdown"]),
       ["banking_score", "capped", "ceiling", "insurance_indicators",
        "insurance_indicators_cleared", "insurance_third", "investment_score",
        "option_b_score", "overall_score"])
check("score_breakdown's tuple is flattened for Firestore",
      isinstance(snap["score_breakdown"]["insurance_indicators"], list))

# The whole reason a breakdown is stored instead of a "capping reason" enum:
# the explanation must be reconstructable from the raw fields, months later.
b = snap["score_breakdown"]
check("this fixture is genuinely a capped case (so the next check means "
      "something)", b["capped"] is True,
      f"option_b {b['option_b_score']:.2f} vs ceiling {b['ceiling']}")
check("'capped by the insurance gap' is reconstructable without an enum",
      b["capped"] and b["ceiling"] < 100.0
      and b["insurance_indicators_cleared"] < 3
      and b["overall_score"] == b["ceiling"] < b["option_b_score"],
      f"cleared {b['insurance_indicators_cleared']}/3 -> ceiling {b['ceiling']}")

# No invented vocabulary anywhere in the row.
check("no fabricated reason/caption string is stored",
      not any(k in snap for k in
              ("capping_reason", "capped_by", "caption", "reason_text")))

print()
print("=" * 78)
print("A3. The null-score days — preserved as gaps, never coerced")
print("=" * 78)

GAP_CASES = [
    ("income_not_set", {"profile": {**PROFILE, "monthly_income_sgd": 0.0}},
     score_pipeline.REASON_INCOME_NOT_SET),
    ("unsupported_age", {"user_doc": {**USER, "date_of_birth": date(1946, 1, 1)}},
     score_pipeline.REASON_UNSUPPORTED_AGE),
    ("date_of_birth_not_set", {"user_doc": {"dependants": 0}},
     score_pipeline.REASON_DATE_OF_BIRTH_NOT_SET),
    ("incomplete_domains", {"bank_accounts": []},
     score_pipeline.REASON_INCOMPLETE_DOMAINS),
]
for label, overrides, reason in GAP_CASES:
    result = run(**overrides)
    row = score_profile.snapshot_payload(result)
    check(f"{label:<21} snapshot still builds", set(row) == set(snap))
    check(f"{label:<21} overall_score is None, not 0.0",
          row["overall_score"] is None, repr(row["overall_score"]))
    expect(f"{label:<21} the reason travels with the gap",
           row["overall_score_reason"], reason)
    check(f"{label:<21} still carries a date and a computed_at",
          row["date"] == "2026-09-08" and row["computed_at"] == NOW)

# A whole-pipeline failure has no age band; the row must still be writable.
no_band = score_profile.snapshot_payload(run(user_doc={"dependants": 0}))
check("a row with no age band is still a valid row (None, not omitted)",
      "age_band" in no_band and no_band["age_band"] is None)
check("a row with no score has score_breakdown None, not an empty map",
      no_band["score_breakdown"] is None, repr(no_band["score_breakdown"]))
check("a row with no score still carries all three domains",
      list(no_band["domains"]) == list(score_pipeline.DOMAIN_ORDER))

print()
print("=" * 78)
print("A4. snapshot_date — UTC, and one document per day")
print("=" * 78)

DAY = datetime(2026, 9, 8, tzinfo=timezone.utc)
for hour in (0, 1, 12, 23):
    expect(f"  {hour:02d}:00 UTC keys to 2026-09-08",
           score_profile.snapshot_date(DAY.replace(hour=hour)), "2026-09-08")
expect("  the next midnight rolls the key over",
       score_profile.snapshot_date(DAY + timedelta(days=1)), "2026-09-09")
# The documented consequence of keying on UTC, asserted rather than assumed:
# 03:00 SGT is still the previous UTC day.
SGT = timezone(timedelta(hours=8))
expect("  03:00 SGT lands on the previous UTC day (the documented tradeoff)",
       score_profile.snapshot_date(datetime(2026, 9, 9, 3, 0, tzinfo=SGT)),
       "2026-09-08")
expect("  a non-UTC instant is normalised, not formatted as-is",
       score_profile.snapshot_date(datetime(2026, 9, 9, 12, 0, tzinfo=SGT)),
       "2026-09-09")
check("YYYY-MM-DD sorts lexicographically in calendar order",
      sorted(["2026-01-09", "2026-09-01", "2025-12-31"])
      == ["2025-12-31", "2026-01-09", "2026-09-01"])

print()
print("=" * 78)
print("A5. The write path — one hook, last-write-wins, suppressible")
print("=" * 78)

source = inspect.getsource(score_profile.refresh_score_profile)
check("the snapshot hangs off refresh_score_profile, the shared hook point",
      "_write_snapshot_nonfatal(" in source)
check("it fires strictly AFTER the cache write, not before",
      source.index("profile_ref(db, uid).update(")
      < source.index("_write_snapshot_nonfatal("))
check("it sits beside the notification hook, not at a separate call site",
      abs(source.index("_write_snapshot_nonfatal(")
          - source.index("_notify_score_change_nonfatal(")) < 400)
check("a snapshot=False escape hatch exists, mirroring notify=False",
      "snapshot" in inspect.signature(
          score_profile.refresh_score_profile).parameters)
expect("snapshot defaults to True, so every trigger records history",
       inspect.signature(
           score_profile.refresh_score_profile).parameters["snapshot"].default,
       True)
check("the snapshot write is non-fatal to the refresh",
      "except Exception" in inspect.getsource(
          score_profile._write_snapshot_nonfatal))
check("write_snapshot uses set() — the deliberate last-write-wins divergence",
      ".set(" in inspect.getsource(score_profile.write_snapshot)
      and ".update(" not in inspect.getsource(score_profile.write_snapshot))
check("that divergence from notificationEvents is documented in the code",
      "create-if-absent" in inspect.getsource(score_profile.write_snapshot)
      and "notificationEvents" in inspect.getsource(score_profile.write_snapshot))
check("the non-fatal wrapper the sync hooks use passes snapshot through "
      "by default",
      "refresh_score_profile(db, uid)" in inspect.getsource(
          score_profile.refresh_score_nonfatal))

print()
print("=" * 78)
print("A6. Every trigger reaches the hook, and both enforcement points")
print("=" * 78)

import main as backend_main_module  # noqa: E402

# The four sync triggers funnel through main's non-fatal wrapper; /score/refresh
# calls refresh_score_profile directly. All five therefore snapshot.
HOOKED_ENDPOINTS = {
    "parse_insurance_pdf": "POST /insurance/parse",
    "exchange_plaid_public_token": "POST /plaid/exchange-token",
    "sync_plaid_transactions": "POST /plaid/sync-transactions",
    "sync_plaid_investments": "POST /plaid/sync-investments",
}
for func_name, route in HOOKED_ENDPOINTS.items():
    src = inspect.getsource(getattr(backend_main_module, func_name))
    check(f"{route:<28} reaches the snapshot via _refresh_score_nonfatal",
          "_refresh_score_nonfatal(" in src)
check("POST /score/refresh              reaches it directly",
      "refresh_score_profile(" in inspect.getsource(
          backend_main_module.refresh_score))
check("neither wrapper suppresses the snapshot",
      "snapshot=False" not in inspect.getsource(backend_main_module)
      and "snapshot=False" not in inspect.getsource(
          score_profile.refresh_score_nonfatal))

check("scoreHistory is in the deletion cascade's flat list",
      score_profile.HISTORY_COLLECTION in account_deletion.FLAT_SUBCOLLECTIONS,
      str(account_deletion.FLAT_SUBCOLLECTIONS))
check("it is NOT in the nested list (it owns no subcollection)",
      score_profile.HISTORY_COLLECTION
      not in account_deletion.NESTED_SUBCOLLECTIONS)

RULES = (BACKEND_DIR.parent / "firestore.rules").read_text(encoding="utf-8")
HISTORY_MATCH = "match /users/{uid}/scoreHistory/{document=**} {"
check("firestore.rules declares a scoreHistory match block",
      HISTORY_MATCH in RULES)
# The grants inside that block, read as lines rather than by splitting on "}"
# — the block's own closing brace is not the first one in the file after it.
history_block = [
    line.strip()
    for line in RULES.split(HISTORY_MATCH)[1].split("}")[0].splitlines()
    if line.strip()
]
expect("it grants the owner READ and nothing else", history_block,
       ["allow read: if isOwner(uid);"])
check("scoreHistory is excluded from the read+write catch-all",
      "'scoreHistory'" in RULES.split("match /users/{uid}/{collection}")[1])
check("it is NOT sealed like plaidItems (the owner must be able to read it)",
      "allow read, write: if false" not in
      RULES.split(HISTORY_MATCH)[1].split("match ")[0])

print()
print(f"section A: {len(passed)} passed, {len(failed)} failed")

if PURE_ONLY:
    print("\n--pure-only: skipping section B (live Firebase)")
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)

# ===========================================================================
# B. END TO END — real Firestore, real endpoints
# ===========================================================================
import json  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from firebase import get_db, init_firebase  # noqa: E402
from firebase_admin import auth as admin_auth  # noqa: E402

PROJECT = "YOUR_FIREBASE_PROJECT_ID"
GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
FIRESTORE = (f"https://firestore.googleapis.com/v1/projects/{PROJECT}"
             "/databases/(default)/documents")

E2E_INCOME = 4500.0
E2E_DOB = date(1990, 6, 15)
E2E_DEPENDANTS = 1


def http(method, url, body=None, token=None):
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
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
        {"email": email, "password": password, "returnSecureToken": True})
    if status != 200:
        sys.exit(f"\nSign-up failed ({status}): "
                 f"{body.get('error', {}).get('message', '')}")
    return body["localId"], body["idToken"]


print()
print("=" * 78)
print("B1. A real refresh writes a real snapshot")
print("=" * 78)

init_firebase()
db = get_db()
stamp = int(time.time())
email = f"e2e.history.{stamp}@unify-test.example"
uid, id_token = sign_up(email, "E2eHistory-12345")
print(f"test user uid={uid}")

from fastapi.testclient import TestClient  # noqa: E402
import main as backend_main  # noqa: E402

client = TestClient(backend_main.app)


def api(method, path, body=None):
    r = client.request(method, path,
                       headers={"Authorization": f"Bearer {id_token}"}, json=body)
    return r.status_code, r.json()


def history_docs():
    return sorted(score_profile.history_collection(db, uid).list_documents(),
                  key=lambda d: d.id)


try:
    user_ref = db.collection("users").document(uid)
    user_ref.set({
        "name": "E2E History User", "email": email,
        "date_of_birth": datetime.combine(E2E_DOB, datetime.min.time(),
                                          tzinfo=timezone.utc),
        "dependants": E2E_DEPENDANTS,
    })
    score_profile.profile_ref(db, uid).set({
        "monthly_income_sgd": E2E_INCOME, "income_source": "user_entered",
        "monthly_expenses_sgd": 2200.0, "net_worth_sgd": 60000.0,
        "overall_score": None, "checklist_flags": None,
        "checklist_last_updated": None, "ml_profile": None,
    })

    # All three domains seeded, so this user reaches a REAL overall score.
    # Written with the Admin SDK rather than through Plaid: the pipeline reads
    # these collections and does not care who wrote them, and B3 needs a day
    # that genuinely holds a score before it can prove one turning into a gap.
    user_ref.collection("plaidAccounts").document("acct-dep").set({
        "account_id": "acct-dep", "uid": uid, "account_type": "depository",
        "account_subtype": "savings", "balance_current": 26000.0,
    })
    user_ref.collection("plaidAccounts").document("acct-credit").set({
        "account_id": "acct-credit", "uid": uid, "account_type": "credit",
        "account_subtype": "credit card", "balance_current": 1800.0,
        "liability": {"monthly_payment_amount": 150.0},
    })
    inv = user_ref.collection("investmentAccounts").document("inv-1")
    inv.set({
        "account_id": "inv-1", "uid": uid, "source": "plaid",
        "account_type": "brokerage", "total_value_sgd": 42000.0,
        "concentration_risk": 0.22,
        "asset_allocation": {"equities": 0.65, "bonds": 0.25, "cash": 0.1},
    })
    inv.collection("transactions").document("itxn-1").set({
        "transaction_id": "itxn-1", "account_id": "inv-1", "type": "deposit",
        "amount": -550.0, "date": datetime.now(timezone.utc) - timedelta(days=5),
    })
    user_ref.collection("insurancePolicies").document("policy-life").set({
        "policy_id": "policy-life", "uid": uid, "policy_name": "Term Life",
        "policy_type": "life", "sum_assured_sgd": 550000.0,
        "annual_premium_sgd": 1500.0,
    })
    user_ref.collection("insurancePolicies").document("policy-hosp").set({
        "policy_id": "policy-hosp", "uid": uid, "policy_name": "Shield",
        "policy_type": "hospitalisation", "annual_premium_sgd": 700.0,
    })

    check("no history exists before the first refresh", history_docs() == [])

    status, body = api("POST", "/score/refresh")
    check("POST /score/refresh -> 200", status == 200, str(status))
    today = score_profile.snapshot_date(
        datetime.now(timezone.utc))
    docs = history_docs()
    check("exactly one snapshot document now exists", len(docs) == 1,
          f"{[d.id for d in docs]}")
    check("it is keyed by today's UTC date",
          docs and docs[0].id == today, f"{docs[0].id if docs else None} vs {today}")

    stored = docs[0].get().to_dict()
    expect("stored row carries exactly the agreed field set", sorted(stored),
           ["age_band", "computed_at", "date", "domains", "net_worth_sgd",
            "overall_score", "overall_score_reason", "schema_version",
            "score_breakdown"])
    expect("stored date field matches the document ID", stored["date"], docs[0].id)
    expect("stored overall_score matches what the endpoint returned",
           stored["overall_score"], body["overall_score"])
    expect("stored overall_score matches the live cache",
           stored["overall_score"],
           score_profile.profile_ref(db, uid).get().to_dict()["overall_score"])
    expect("stored age_band matches the run", stored["age_band"], body["age_band"])
    expect("stored net_worth_sgd matches the live financialProfile",
           stored["net_worth_sgd"],
           score_profile.profile_ref(db, uid).get().to_dict()["net_worth_sgd"])
    expect("stored domains match the run's domains",
           {d: stored["domains"][d]["tier_index"]
            for d in score_pipeline.DOMAIN_ORDER},
           {d: body["domains"][d]["tier_index"]
            for d in score_pipeline.DOMAIN_ORDER})
    check("computed_at survived the Firestore round trip as a timestamp",
          isinstance(stored["computed_at"], datetime),
          type(stored["computed_at"]).__name__)

    print()
    print("=" * 78)
    print("B2. Two refreshes on the same day — last write wins")
    print("=" * 78)

    first_computed_at = stored["computed_at"]
    first_score = stored["overall_score"]
    time.sleep(1.1)  # so the second computed_at is distinguishable
    status, body2 = api("POST", "/score/refresh")
    check("second POST /score/refresh -> 200", status == 200, str(status))

    docs = history_docs()
    check("STILL exactly one document — not two", len(docs) == 1,
          f"{[d.id for d in docs]}")
    second = docs[0].get().to_dict()
    check("the surviving row is the SECOND run, not the first",
          second["computed_at"] > first_computed_at,
          f"{second['computed_at']} vs {first_computed_at}")
    expect("the second run's score is what is stored",
           second["overall_score"], body2["overall_score"])
    check("this is genuinely last-write-wins, not create-if-absent",
          second["computed_at"] != first_computed_at)

    print()
    print("=" * 78)
    print("B3. A refresh with no computable score still writes a row")
    print("=" * 78)

    check("precondition: the day currently holds a real score",
          first_score is not None, repr(first_score))
    # Income to 0 — the income_not_set path, via the real endpoint.
    score_profile.profile_ref(db, uid).update({"monthly_income_sgd": 0.0})
    status, body3 = api("POST", "/score/refresh")
    check("POST /score/refresh with income 0 -> 200, not an error",
          status == 200, str(status))
    expect("the pipeline reports income_not_set", body3["unavailable_reason"],
           score_pipeline.REASON_INCOME_NOT_SET)

    docs = history_docs()
    check("still one document for today", len(docs) == 1,
          f"{[d.id for d in docs]}")
    gap = docs[0].get().to_dict()
    check("the row was REWRITTEN as a gap — score is null",
          gap["overall_score"] is None, repr(gap["overall_score"]))
    expect("the reason is stored alongside the null",
           gap["overall_score_reason"], score_pipeline.REASON_INCOME_NOT_SET)
    check("a gap row still carries date and computed_at",
          gap["date"] == today and isinstance(gap["computed_at"], datetime))
    check("a gap row carries all three domains as not_assessed",
          all(gap["domains"][d]["status"] == score_pipeline.STATUS_NOT_ASSESSED
              for d in score_pipeline.DOMAIN_ORDER))
    check("a gap row's score_breakdown is null, not fabricated",
          gap["score_breakdown"] is None, repr(gap["score_breakdown"]))

    score_profile.profile_ref(db, uid).update({"monthly_income_sgd": E2E_INCOME})

    print()
    print("=" * 78)
    print("B4. snapshot=False genuinely suppresses the write")
    print("=" * 78)

    suppress_day = datetime.now(timezone.utc) - timedelta(days=40)
    before = {d.id for d in history_docs()}
    score_profile.refresh_score_profile(db, uid, now=suppress_day,
                                        notify=False, snapshot=False)
    after = {d.id for d in history_docs()}
    expect("snapshot=False wrote no new document", after, before)
    score_profile.refresh_score_profile(db, uid, now=suppress_day, notify=False)
    check("the same call with the default DOES write it",
          score_profile.snapshot_date(suppress_day) in
          {d.id for d in history_docs()})
    score_profile.history_collection(db, uid).document(
        score_profile.snapshot_date(suppress_day)).delete()

    print()
    print("=" * 78)
    print("B5. GET /score/history — ordering, shape and the limit")
    print("=" * 78)

    # Seed a run of past days through the real writer, using the pipeline's own
    # injectable clock rather than hand-built documents: this proves the
    # endpoint against rows the production path actually produces.
    seeded = []
    for days_ago in range(1, 6):
        when = datetime.now(timezone.utc) - timedelta(days=days_ago)
        score_profile.refresh_score_profile(db, uid, now=when, notify=False)
        seeded.append(score_profile.snapshot_date(when))

    status, hist = api("GET", "/score/history")
    check("GET /score/history -> 200", status == 200, str(status))
    expect("it returns every seeded day plus today",
           hist["count"], len(seeded) + 1)
    dates = [s["date"] for s in hist["snapshots"]]
    expect("newest first", dates, sorted(dates, reverse=True))
    expect("no duplicate days", len(set(dates)), len(dates))
    expect("today is the first row", dates[0], today)
    expect("the response reports the window it used",
           hist["limit"], backend_main.HISTORY_DEFAULT_LIMIT)
    expect("each row is the full snapshot shape", sorted(hist["snapshots"][0]),
           ["age_band", "computed_at", "date", "domains", "net_worth_sgd",
            "overall_score", "overall_score_reason", "schema_version",
            "score_breakdown"])

    status, capped = api("GET", "/score/history?limit=2")
    check("an explicit limit is honoured",
          capped["count"] == 2 and capped["limit"] == 2, str(capped["count"]))
    expect("a limited window still returns the NEWEST rows",
           [s["date"] for s in capped["snapshots"]], dates[:2])
    status, _ = api("GET", "/score/history?limit=0")
    check("limit=0 is rejected, not silently treated as unlimited",
          status == 422, str(status))
    status, _ = api("GET", f"/score/history?limit={backend_main.HISTORY_MAX_LIMIT + 1}")
    check("a limit above the ceiling is rejected", status == 422, str(status))

    r = client.get("/score/history")
    check("the endpoint requires auth", r.status_code in (401, 403),
          str(r.status_code))

    print()
    print("=" * 78)
    print("B6. The security rule, as actually deployed")
    print("=" * 78)

    status, _ = http("GET", f"{FIRESTORE}/users/{uid}/scoreHistory/{today}",
                     token=id_token)
    check("the owner's client CAN read its own snapshot (the chart needs this)",
          status == 200, f"HTTP {status}")

    forged = {"fields": {"date": {"stringValue": today},
                         "overall_score": {"doubleValue": 100.0}}}
    status, _ = http("PATCH", f"{FIRESTORE}/users/{uid}/scoreHistory/{today}",
                     forged, id_token)
    check("the owner's client CANNOT overwrite a snapshot", status == 403,
          f"HTTP {status}")
    status, _ = http("PATCH",
                     f"{FIRESTORE}/users/{uid}/scoreHistory/2020-01-01",
                     forged, id_token)
    check("the owner's client CANNOT invent a new day", status == 403,
          f"HTTP {status}")
    status, _ = http("GET", f"{FIRESTORE}/users/{uid}/scoreHistory/{today}")
    check("an anonymous read is denied", status == 403, f"HTTP {status}")

    # Outcome-based, not status-based: a 403 that still let the value through
    # would be worse than an honest 200. Read back before the delete attempt
    # below, which is the one attack that can remove the row entirely.
    survived = score_profile.history_collection(db, uid).document(
        today).get().to_dict()
    check("today's row still holds the pipeline's own value after the attack",
          survived is not None
          and survived["overall_score"] == gap["overall_score"]
          and survived["computed_at"] == gap["computed_at"],
          "row was altered or destroyed" if survived is None
          else f"score={survived['overall_score']}")

    status, _ = http("DELETE", f"{FIRESTORE}/users/{uid}/scoreHistory/{today}",
                     token=id_token)
    check("the owner's client CANNOT delete a snapshot", status == 403,
          f"HTTP {status}")
    still_there = score_profile.history_collection(db, uid).document(
        today).get().exists
    check("the row is still present after the delete attempt", still_there)
    if not still_there:
        # Restore it so B7 counts the history it expects. A rewrite through the
        # real writer, not a hand-built document.
        score_profile.refresh_score_profile(db, uid, notify=False)
        print("  (row restored via the real writer so B7 can proceed)")

    print()
    print("=" * 78)
    print("B7. Account deletion clears the collection")
    print("=" * 78)

    before_delete = len(list(score_profile.history_collection(
        db, uid).list_documents()))
    check("there is a non-trivial history to delete", before_delete >= 5,
          f"{before_delete} docs")

    result = account_deletion.delete_account(db, uid, remove_plaid_items=False)
    expect("the cascade counted the scoreHistory documents it removed",
           result["firestore_documents_deleted"].get(
               score_profile.HISTORY_COLLECTION), before_delete)
    check("scoreHistory is empty afterwards",
          list(score_profile.history_collection(db, uid).list_documents()) == [],
          "documents survived")
    check("no document survives anywhere under users/{uid}",
          account_deletion.surviving_paths(db, uid) == [])
    check("the cascade reports Firestore verified empty",
          result["firestore_verified_empty"] is True)

finally:
    try:
        for doc in score_profile.history_collection(db, uid).list_documents():
            doc.delete()
        for coll in db.collection("users").document(uid).collections():
            for doc in coll.list_documents():
                doc.delete()
        db.collection("users").document(uid).delete()
    except Exception as exc:
        print(f"cleanup (firestore): {exc}")
    try:
        admin_auth.delete_user(uid)
        print(f"cleaned up test user {uid}")
    except Exception as exc:
        print(f"cleanup (auth): {exc}")

print()
print("=" * 78)
print(f"{len(passed)} passed, {len(failed)} failed")
if failed:
    for name in failed:
        print(f"  FAILED: {name}")
print("=" * 78)
sys.exit(1 if failed else 0)
