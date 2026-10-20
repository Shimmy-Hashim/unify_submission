"""financialProfile derived fields — Home Phase 1 (brief Sections 5 & 8).

Section 8 documents users/{uid}/financialProfile as a single document.
Firestore documents must live in a collection, so it is implemented as a
one-document subcollection with the fixed ID "profile":

    users/{uid}/financialProfile/profile

Field ownership is split by writer, so there is exactly one write path per
field:
  - the Flutter client writes monthly_income_sgd / income_source once, at
    onboarding (the only client write path — accounts predating this
    collection are test/dev-only and get deleted, not retrofitted), and
    never writes the derived fields beyond their initial 0.0
  - this module owns the derived fields (monthly_expenses_sgd,
    net_worth_sgd, total_assets_sgd, total_debts_sgd). Timing: recomputed on every Plaid sync trigger
    (exchange-token, sync-transactions, sync-investments — see main.py).
    POST /profile/refresh exposes the same recompute on demand (unused by
    the app today; kept for manual testing and as the likely Phase 4
    Home-screen refresh hook)
  - overall_score / checklist_flags / checklist_last_updated / ml_profile
    are Phase 3 outputs — created null by the client and never touched here

refresh_financial_profile() deliberately no-ops when the profile document
does not exist yet: document existence is the signal that the user has
provided their income, and a derived-fields-only document would confuse the
client's "needs income setup" routing check.
"""
from datetime import datetime, timedelta, timezone

from google.cloud.firestore_v1.base_query import FieldFilter

PROFILE_DOC_ID = "profile"

# Both constants mirror the Savings screen's Emergency Fund estimator
# (lib/models/plaid_account.dart, SavingsSummary.avgMonthlySpent): outflows
# over the last 30 days, computed from at most the 100 most recent
# transactions per account — the same window the client fetches.
EXPENSE_WINDOW_DAYS = 30
_TXN_FETCH_LIMIT = 100

# Net-worth bucketing by plaidAccounts.account_type: depository balances are
# assets; credit and loan balances are debt. (Plaid reports a credit card's
# balance_current as the positive amount owed, so summing "all balances"
# would count debt as an asset.) The plaidAccounts.liability map itself
# carries payment metadata, not balances — the balance of a liability
# account is its balance_current, and unlike the debt-to-income feature
# this includes loan accounts Plaid's Liabilities product doesn't cover,
# since their balance is real debt regardless of payment-data coverage.
_ASSET_ACCOUNT_TYPES = ("depository",)
_LIABILITY_ACCOUNT_TYPES = ("credit", "loan")


def profile_ref(db, uid: str):
    return (
        db.collection("users")
        .document(uid)
        .collection("financialProfile")
        .document(PROFILE_DOC_ID)
    )


def compute_monthly_expenses(db, uid: str, now: datetime | None = None) -> float:
    """Replicates SavingsSummary.avgMonthlySpent (the Savings screen's
    Emergency Fund estimator) in Python: sum of outflows (amount > 0, Plaid
    sign convention) from depository accounts over the last 30 days,
    reading at most the 100 most recent transactions per account exactly
    like the client does. Credit-card transactions are excluded — card
    repayments already leave the depository accounts, so counting both
    would double-count.

    Replicated rather than shared because the original is client-side Dart
    and this is the Python backend; the constants above keep them aligned.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=EXPENSE_WINDOW_DAYS)
    user_ref = db.collection("users").document(uid)
    total = 0.0
    for account in (
        user_ref.collection("plaidAccounts")
        .where(filter=FieldFilter("account_type", "in", list(_ASSET_ACCOUNT_TYPES)))
        .stream()
    ):
        for txn_snap in (
            user_ref.collection("plaidAccounts")
            .document(account.id)
            .collection("transactions")
            .order_by("date", direction="DESCENDING")
            .limit(_TXN_FETCH_LIMIT)
            .stream()
        ):
            txn = txn_snap.to_dict()
            txn_date = txn.get("date")
            amount = txn.get("amount")
            if txn_date is None or amount is None:
                continue
            if txn_date >= cutoff and amount > 0:
                total += float(amount)
    return round(total, 2)


def compute_balance_totals(db, uid: str) -> dict:
    """The two halves of net worth, and net worth itself, from ONE read of
    the accounts — so the three can never disagree.

        total_assets_sgd   depository balance_current + investmentAccounts
                           total_value_sgd
        total_debts_sgd    credit/loan balance_current
        net_worth_sgd      total_assets_sgd - total_debts_sgd

    Uses balance_current, matching the Savings screen's Total Savings figure.

    Net worth is taken from the ROUNDED totals, so the client can subtract the
    two stored figures and land on the stored net worth to the cent. Balances
    are cent-exact, but investment total_value_sgd is not (Sandbox carries
    values like 25125.63318); rounding each total first is what makes the
    identity hold by construction rather than by luck.
    """
    user_ref = db.collection("users").document(uid)
    assets = 0.0
    liabilities = 0.0
    for snap in user_ref.collection("plaidAccounts").stream():
        account = snap.to_dict()
        balance = account.get("balance_current") or 0.0
        if account.get("account_type") in _ASSET_ACCOUNT_TYPES:
            assets += float(balance)
        elif account.get("account_type") in _LIABILITY_ACCOUNT_TYPES:
            liabilities += float(balance)
    for snap in user_ref.collection("investmentAccounts").stream():
        assets += float(snap.to_dict().get("total_value_sgd") or 0.0)
    total_assets = round(assets, 2)
    total_debts = round(liabilities, 2)
    return {
        "total_assets_sgd": total_assets,
        "total_debts_sgd": total_debts,
        "net_worth_sgd": round(total_assets - total_debts, 2),
    }


def compute_net_worth(db, uid: str) -> float:
    """(depository balance_current + investmentAccounts total_value_sgd)
    minus (credit/loan balance_current) — see compute_balance_totals()."""
    return compute_balance_totals(db, uid)["net_worth_sgd"]


def refresh_financial_profile(db, uid: str) -> dict | None:
    """Recomputes and stores the derived fields. Returns the stored values,
    or None when the user has no financialProfile document yet (income not
    provided — nothing is written, see module docstring)."""
    ref = profile_ref(db, uid)
    if not ref.get().exists:
        return None
    derived = {
        "monthly_expenses_sgd": compute_monthly_expenses(db, uid),
        # net_worth_sgd, total_assets_sgd, total_debts_sgd — one read, so the
        # stored assets minus the stored debts is the stored net worth.
        **compute_balance_totals(db, uid),
    }
    ref.update(derived)
    return derived
