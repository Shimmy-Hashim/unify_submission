"""Plaid Sandbox service layer — Phases 1 (banking) and 3 (investments).

Owns the Plaid API client plus Firestore persistence for
users/{uid}/plaidItems, users/{uid}/plaidAccounts,
users/{uid}/investmentAccounts and their transactions subcollections
(project brief Section 8). Investment-type accounts returned by Plaid are
never stored in plaidAccounts — they belong in investmentAccounts, written
by the Phase 3 sync functions at the bottom of this module.

The Plaid access_token lives only on the plaidItems document. No function
in this module that feeds an API response includes it in its return value.
"""
import logging
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import plaid
from dotenv import dotenv_values
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from plaid.api import plaid_api
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.country_code import CountryCode
from plaid.model.institutions_get_by_id_request import (
    InstitutionsGetByIdRequest,
)
from plaid.model.item_get_request import ItemGetRequest
from plaid.model.item_public_token_exchange_request import (
    ItemPublicTokenExchangeRequest,
)
from plaid.model.investments_holdings_get_request import (
    InvestmentsHoldingsGetRequest,
)
from plaid.model.investments_transactions_get_request import (
    InvestmentsTransactionsGetRequest,
)
from plaid.model.investments_transactions_get_request_options import (
    InvestmentsTransactionsGetRequestOptions,
)
from plaid.model.liabilities_get_request import LiabilitiesGetRequest
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import (
    LinkTokenCreateRequestUser,
)
from plaid.model.products import Products
from plaid.model.transactions_sync_request import TransactionsSyncRequest

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[1]

# Section 8: plaidAccounts stores banking only — investment-type accounts
# belong in investmentAccounts (Phase 3) and are skipped here.
BANKING_ACCOUNT_TYPES = ("depository", "credit", "loan")

# Firestore batches cap at 500 ops; stay under it with headroom.
_BATCH_LIMIT = 450

_client: plaid_api.PlaidApi | None = None


# ---------------------------------------------------------------------------
# Phase 1 — Banking (brief Section 6/8)
# ---------------------------------------------------------------------------
def _plaid_config() -> tuple[str, str, str]:
    """Returns (client_id, secret, host). Reads backend/.env via
    dotenv_values, where the LAST occurrence of a duplicated key wins —
    backend/.env currently carries a stale duplicate PLAID_CLIENT_ID on an
    earlier line, so file values deliberately take precedence over
    os.environ (which load_dotenv populated first-occurrence-wins).
    """
    file_values = {
        k: v for k, v in dotenv_values(_BACKEND_DIR / ".env").items() if v
    }
    cfg = {**os.environ, **file_values}
    missing = [
        k for k in ("PLAID_CLIENT_ID", "PLAID_SECRET") if not cfg.get(k)
    ]
    if missing:
        raise RuntimeError(
            f"Missing in backend/.env: {', '.join(missing)} — fill in the "
            "Plaid Sandbox keys from the Plaid dashboard"
        )
    env = (cfg.get("PLAID_ENV") or "sandbox").strip().lower()
    hosts = {
        "sandbox": plaid.Environment.Sandbox,
        "production": plaid.Environment.Production,
    }
    if env not in hosts:
        raise RuntimeError(f"Unsupported PLAID_ENV {env!r}")
    return cfg["PLAID_CLIENT_ID"], cfg["PLAID_SECRET"], hosts[env]


def get_plaid_client() -> plaid_api.PlaidApi:
    """Lazy singleton, same pattern as main.get_parser — the API can start
    without Plaid keys; endpoints fail with a clear error until configured.
    """
    global _client
    if _client is None:
        client_id, secret, host = _plaid_config()
        configuration = plaid.Configuration(
            host=host,
            api_key={"clientId": client_id, "secret": secret},
        )
        _client = plaid_api.PlaidApi(plaid.ApiClient(configuration))
    return _client


def create_link_token(uid: str) -> dict:
    """Creates a Link token for the signed-in user. country_codes is US-only
    by design: Sandbox only offers US test institutions (First Platypus
    Bank), regardless of target market — brief Section 6.
    """
    request = LinkTokenCreateRequest(
        user=LinkTokenCreateRequestUser(client_user_id=uid),
        client_name="Unify",
        # investments sits in products (a hard requirement, unlike
        # liabilities below): Phase 3's holdings/transactions data is core
        # to the Investments screen and the investment-rate feature, not a
        # nice-to-have enrichment.
        products=[Products("transactions"), Products("investments")],
        # required_if_supported (not products) so institutions without
        # Liabilities support are not excluded from Link in a real
        # deployment; Sandbox institutions support it regardless.
        required_if_supported_products=[Products("liabilities")],
        country_codes=[CountryCode("US")],
        language="en",
    )
    response = get_plaid_client().link_token_create(request).to_dict()
    return {
        "link_token": response["link_token"],
        "expiration": response["expiration"].isoformat(),
    }


def _institution_for_item(access_token: str) -> tuple[str | None, str | None]:
    client = get_plaid_client()
    item = client.item_get(ItemGetRequest(access_token=access_token))
    institution_id = item.to_dict()["item"].get("institution_id")
    if not institution_id:
        return None, None
    inst = client.institutions_get_by_id(
        InstitutionsGetByIdRequest(
            institution_id=institution_id,
            country_codes=[CountryCode("US")],
        )
    )
    return institution_id, inst.to_dict()["institution"]["name"]


def _liability_map(kind: str, entry: dict) -> dict:
    """Normalizes one /liabilities/get entry (kind: credit | mortgage |
    student) into the plaidAccounts.liability map.

    monthly_payment_amount aliases different concepts for a uniform schema:
    the actual installment (next_monthly_payment) for mortgages, but the
    minimum payment (minimum_payment_amount) for credit cards and student
    loans. For credit cards interest_rate_percentage is the purchase APR
    (cards carry several APRs; purchase is the canonical one). Mortgages
    have no is_overdue in Plaid's response — stored as null there.
    """
    if kind == "mortgage":
        monthly = entry.get("next_monthly_payment")
        rate = (entry.get("interest_rate") or {}).get("percentage")
    elif kind == "student":
        monthly = entry.get("minimum_payment_amount")
        rate = entry.get("interest_rate_percentage")
    else:
        monthly = entry.get("minimum_payment_amount")
        rate = next(
            (
                apr.get("apr_percentage")
                for apr in entry.get("aprs") or []
                if apr.get("apr_type") == "purchase_apr"
            ),
            None,
        )
    due = entry.get("next_payment_due_date")
    return {
        "monthly_payment_amount": monthly,
        "next_payment_due_date": (
            datetime.combine(due, time.min, tzinfo=timezone.utc)
            if due
            else None
        ),
        "interest_rate_percentage": rate,
        "is_overdue": entry.get("is_overdue"),
    }


def fetch_liabilities(access_token: str) -> dict[str, dict]:
    """Returns {account_id: liability map} for the Item. Plaid only covers
    credit cards, mortgages and student loans — other loan types (personal,
    auto, lines of credit) never appear here. Returns {} instead of raising
    when the Item/institution has no Liabilities support, so connect and
    sync degrade gracefully rather than fail.
    """
    try:
        response = (
            get_plaid_client()
            .liabilities_get(LiabilitiesGetRequest(access_token=access_token))
            .to_dict()
        )
    except plaid.ApiException as exc:
        logger.warning("liabilities/get unavailable for item: %s", exc.status)
        return {}
    liabilities = response.get("liabilities") or {}
    out = {}
    for kind in ("credit", "mortgage", "student"):
        for entry in liabilities.get(kind) or []:
            account_id = entry.get("account_id")
            if account_id:
                out[account_id] = _liability_map(kind, entry)
    return out


def connect_bank_item(db, uid: str, public_token: str) -> dict:
    """Exchanges a public_token, then persists the Item and its banking
    accounts per Section 8. Returns a client-safe summary (no access_token).
    """
    client = get_plaid_client()
    exchange = client.item_public_token_exchange(
        ItemPublicTokenExchangeRequest(public_token=public_token)
    ).to_dict()
    access_token = exchange["access_token"]
    item_id = exchange["item_id"]

    institution_id, institution_name = _institution_for_item(access_token)
    accounts = client.accounts_get(
        AccountsGetRequest(access_token=access_token)
    ).to_dict()["accounts"]
    liabilities = fetch_liabilities(access_token)

    user_ref = db.collection("users").document(uid)
    user_ref.collection("plaidItems").document(item_id).set(
        {
            "item_id": item_id,
            "uid": uid,
            # Backend-only secret — see the firestore.rules note about
            # carving plaidItems out of client access.
            "access_token": access_token,
            "institution_id": institution_id,
            "institution_name": institution_name,
            "plaid_cursor": None,
            "last_synced": None,
        }
    )

    stored, skipped = [], []
    for acct in accounts:
        account_type = str(acct["type"])
        summary = {
            "account_id": acct["account_id"],
            "account_name": acct.get("name"),
            "account_type": account_type,
            "account_subtype": (
                str(acct["subtype"]) if acct.get("subtype") else None
            ),
            # Denormalized for the client: plaidItems (where
            # institution_name canonically lives) is not client-readable
            # by design, and Plaid's mask feeds the "•••• 1234" display.
            "mask": acct.get("mask"),
            "institution_name": institution_name,
        }
        if account_type not in BANKING_ACCOUNT_TYPES:
            skipped.append(summary)
            continue
        balances = acct.get("balances") or {}
        user_ref.collection("plaidAccounts").document(
            acct["account_id"]
        ).set(
            {
                **summary,
                "item_id": item_id,
                "uid": uid,
                "balance_current": balances.get("current"),
                "balance_available": balances.get("available"),
                "iso_currency_code": balances.get("iso_currency_code"),
                # Explicit null (not empty map) on the accounts Plaid's
                # Liabilities product doesn't cover.
                "liability": liabilities.get(acct["account_id"]),
                "last_synced": firestore.SERVER_TIMESTAMP,
            }
        )
        stored.append(summary)

    return {
        "item_id": item_id,
        "institution_id": institution_id,
        "institution_name": institution_name,
        "accounts": stored,
        "skipped_non_banking_accounts": skipped,
    }


def _transaction_doc(txn: dict) -> dict:
    """Maps one Plaid transaction to the Section 8 transactions schema."""
    category = txn.get("category")
    if not category:
        # Plaid is phasing out the legacy category array; fall back to the
        # personal_finance_category pair so the field stays populated.
        pfc = txn.get("personal_finance_category") or {}
        category = [v for v in (pfc.get("primary"), pfc.get("detailed")) if v]
    txn_date = txn.get("date")
    return {
        "transaction_id": txn["transaction_id"],
        "account_id": txn["account_id"],
        "amount": txn.get("amount"),
        "iso_currency_code": txn.get("iso_currency_code"),
        "date": (
            datetime.combine(txn_date, time.min, tzinfo=timezone.utc)
            if txn_date
            else None
        ),
        "name": txn.get("name"),
        "category": list(category or []),
        "pending": bool(txn.get("pending")),
    }


def sync_item_transactions(db, uid: str, item_id: str) -> dict:
    """Runs /transactions/sync for one Item using the cursor stored on its
    plaidItems document, upserts into
    plaidAccounts/{accountId}/transactions/{transactionId}, and advances the
    cursor only after all writes commit.

    Idempotent: documents are keyed by transaction_id (set() upserts), and a
    re-run with an up-to-date cursor returns zero changes from Plaid. If a
    run dies mid-write, the cursor is not advanced, so the next run replays
    the same page onto the same document IDs.
    """
    user_ref = db.collection("users").document(uid)
    item_ref = user_ref.collection("plaidItems").document(item_id)
    snap = item_ref.get()
    if not snap.exists:
        raise LookupError(f"No Plaid item {item_id} for this user")
    item = snap.to_dict()

    cursor = item.get("plaid_cursor") or ""
    added: list[dict] = []
    modified: list[dict] = []
    removed: list[dict] = []
    while True:
        response = (
            get_plaid_client()
            .transactions_sync(
                TransactionsSyncRequest(
                    access_token=item["access_token"], cursor=cursor
                )
            )
            .to_dict()
        )
        added.extend(response["added"])
        modified.extend(response["modified"])
        removed.extend(response["removed"])
        cursor = response["next_cursor"]
        if not response["has_more"]:
            break

    banking_ids = {
        doc.id
        for doc in user_ref.collection("plaidAccounts")
        .where(filter=FieldFilter("item_id", "==", item_id))
        .stream()
    }

    def txn_ref(account_id: str, transaction_id: str):
        return (
            user_ref.collection("plaidAccounts")
            .document(account_id)
            .collection("transactions")
            .document(transaction_id)
        )

    batch = db.batch()
    ops = written = deleted = skipped_non_banking = 0

    def flush(force: bool = False) -> None:
        nonlocal batch, ops
        if ops and (force or ops >= _BATCH_LIMIT):
            batch.commit()
            batch = db.batch()
            ops = 0

    for txn in added + modified:
        if txn["account_id"] not in banking_ids:
            skipped_non_banking += 1
            continue
        batch.set(txn_ref(txn["account_id"], txn["transaction_id"]),
                  _transaction_doc(txn))
        ops += 1
        written += 1
        flush()
    for txn in removed:
        account_id = txn.get("account_id")
        if account_id not in banking_ids:
            continue
        batch.delete(txn_ref(account_id, txn["transaction_id"]))
        ops += 1
        deleted += 1
        flush()

    # Refresh liability data (payment amounts, due dates, is_overdue) on
    # every sync — this is the recurring path, so the snapshot stays
    # current without a dedicated endpoint. One extra Plaid call per Item.
    liabilities = fetch_liabilities(item["access_token"])
    liabilities_updated = 0
    for account_id, liability in liabilities.items():
        if account_id not in banking_ids:
            continue
        batch.update(
            user_ref.collection("plaidAccounts").document(account_id),
            {"liability": liability},
        )
        ops += 1
        liabilities_updated += 1
        flush()
    flush(force=True)

    item_ref.update(
        {"plaid_cursor": cursor, "last_synced": firestore.SERVER_TIMESTAMP}
    )
    return {
        "item_id": item_id,
        "added": len(added),
        "modified": len(modified),
        "removed": len(removed),
        "written": written,
        "deleted": deleted,
        "skipped_non_banking": skipped_non_banking,
        "liabilities_updated": liabilities_updated,
    }


def sync_all_items(db, uid: str) -> list[dict]:
    """Syncs transactions for every Plaid Item the user has connected."""
    item_ids = [
        doc.id
        for doc in db.collection("users")
        .document(uid)
        .collection("plaidItems")
        .stream()
    ]
    return [
        sync_item_transactions(db, uid, item_id) for item_id in item_ids
    ]


def refresh_item_balances(db, uid: str, item_id: str) -> dict:
    """Re-reads one Item's account balances with /accounts/get and writes
    balance_current, balance_available and last_synced onto its existing
    plaidAccounts documents — nothing else.

    connect_bank_item is otherwise the only writer of balances, so without
    this they freeze at link time. /accounts/get (not the paid, slower
    /accounts/balance/get) returns Plaid's cached balances, which for a
    Transactions-enabled Item are at least as recent as its latest
    transaction update.

    Values are copied straight across, None included. On credit and loan
    accounts balance_current is the amount owed and balance_available is
    remaining credit, so neither falls back to the other. Only accounts this
    Item already has in plaidAccounts are touched: an account Plaid newly
    reports is not created here, and investment accounts are never written.
    """
    user_ref = db.collection("users").document(uid)
    snap = user_ref.collection("plaidItems").document(item_id).get()
    if not snap.exists:
        raise LookupError(f"No Plaid item {item_id} for this user")
    item = snap.to_dict()

    accounts = (
        get_plaid_client()
        .accounts_get(AccountsGetRequest(access_token=item["access_token"]))
        .to_dict()["accounts"]
    )
    banking_ids = {
        doc.id
        for doc in user_ref.collection("plaidAccounts")
        .where(filter=FieldFilter("item_id", "==", item_id))
        .stream()
    }

    batch = db.batch()
    updated = not_stored = 0
    for acct in accounts:
        if str(acct["type"]) not in BANKING_ACCOUNT_TYPES:
            continue
        if acct["account_id"] not in banking_ids:
            not_stored += 1
            continue
        balances = acct.get("balances") or {}
        batch.update(
            user_ref.collection("plaidAccounts").document(acct["account_id"]),
            {
                "balance_current": balances.get("current"),
                "balance_available": balances.get("available"),
                "last_synced": firestore.SERVER_TIMESTAMP,
            },
        )
        updated += 1
    # One Item carries a handful of accounts, far below the batch cap.
    if updated:
        batch.commit()
    return {
        "item_id": item_id,
        "accounts_updated": updated,
        "accounts_not_stored": not_stored,
    }


# ---------------------------------------------------------------------------
# Phase 3 — Investments (brief Section 6/8)
# ---------------------------------------------------------------------------

# Plaid reports investment accounts as type "investment" (with "brokerage"
# as its legacy alias); these are the accounts Phase 1 skips out of
# plaidAccounts and Phase 3 stores in investmentAccounts instead.
INVESTMENT_ACCOUNT_TYPES = ("investment", "brokerage")

# /investments/transactions/get is date-ranged (unlike the cursor-based
# banking /transactions/sync). 90 days matches the two-year-max product
# window comfortably and covers the monthly investment-rate feature's
# "recent months" need; a wider backfill can raise this later.
INVESTMENT_TXN_WINDOW_DAYS = 90

# Section 8's account_type enum is SG-flavoured ("brokerage" | "srs" |
# "cpfis" | "other") while Plaid's subtypes are US/UK/CA-centric. Mapping
# decisions (flagged, not silent):
#   - plain investment/brokerage wrappers -> "brokerage"
#   - tax-advantaged retirement wrappers (401k, IRA, pension, RRSP...) ->
#     "srs", the closest SG analogue (Supplementary Retirement Scheme);
#     these are US/CA constructs, so the fit is approximate by design
#   - "cpfis" is unreachable from Plaid data — no US subtype maps to a CPF
#     Investment Scheme account; it exists for manual entries only
#   - everything else (529, HSA, annuities, life insurance, ISAs, TFSA,
#     custodial/trust...) -> "other" rather than a wrong guess
_BROKERAGE_SUBTYPES = {
    "brokerage",
    "mutual fund",
    "stock plan",
    "crypto exchange",
}
_RETIREMENT_SUBTYPES = {
    "401a",
    "401k",
    "457b",
    "ira",
    "keogh",
    "lif",
    "lira",
    "lrif",
    "lrsp",
    "pension",
    "prif",
    "profit sharing plan",
    "retirement",
    "rlif",
    "roth",
    "roth 401k",
    "roth 457b",
    "roth pension",
    "roth profit sharing plan",
    "roth thrift savings plan",
    "rrif",
    "rrsp",
    "sarsep",
    "sep ira",
    "simple ira",
    "sipp",
    "thrift savings plan",
}

# Security type -> asset_allocation bucket. ETFs and mutual funds are
# treated as equities — Plaid gives no fund look-through, so a bond ETF
# would be miscounted; documented simplification, not an oversight.
# Unmapped types (cryptocurrency, derivative, loan, other) land in an
# explicit "other" bucket rather than being forced into equities/bonds/cash.
_ASSET_CLASS_BY_SECURITY_TYPE = {
    "equity": "equities",
    "etf": "equities",
    "mutual fund": "equities",
    "fixed income": "bonds",
    "cash": "cash",
}

# Investment transaction mapping into Section 8's strict 5-value enum
# ("buy" | "sell" | "dividend" | "deposit" | "withdrawal"). Subtype is
# checked first (it is the more specific signal — e.g. subtype "deposit"
# under type "cash"), then the coarse type for plain buys/sells. Plaid
# subtypes with no honest home in the enum (fees, taxes, transfers, splits,
# mergers, interest...) are skipped and counted per raw type/subtype in the
# sync summary — reported, never silently miscategorized.
_TXN_TYPE_BY_SUBTYPE = {
    "dividend": "dividend",
    "qualified dividend": "dividend",
    "non-qualified dividend": "dividend",
    "deposit": "deposit",
    "contribution": "deposit",
    "withdrawal": "withdrawal",
    "distribution": "withdrawal",
}
_TXN_TYPE_BY_TYPE = {"buy": "buy", "sell": "sell"}


def map_investment_account_type(subtype: str | None) -> str:
    """Maps a Plaid investment account subtype onto Section 8's
    ("brokerage" | "srs" | "cpfis" | "other") enum. See the mapping notes
    on the constants above; "cpfis" is never produced from Plaid data."""
    if subtype in _BROKERAGE_SUBTYPES:
        return "brokerage"
    if subtype in _RETIREMENT_SUBTYPES:
        return "srs"
    return "other"


def map_investment_txn_type(txn: dict) -> str | None:
    """Maps one Plaid investment transaction onto Section 8's 5-value type
    enum, or None when it has no honest mapping (caller skips + counts)."""
    subtype = str(txn.get("subtype") or "")
    if subtype in _TXN_TYPE_BY_SUBTYPE:
        return _TXN_TYPE_BY_SUBTYPE[subtype]
    return _TXN_TYPE_BY_TYPE.get(str(txn.get("type") or ""))


def _allocation_and_concentration(
    holdings: list[dict], total_value: float
) -> tuple[dict, float | None]:
    """Computes the two cached analytics fields Section 8 stores on each
    investmentAccounts doc, recomputed on every holdings sync (Flutter never
    derives them). Both are fractions of total_value in [0, 1] — the same
    unit the schema's asset_allocation example uses ({equities: 0.6, ...});
    concentration_risk is largest single holding / total. None/empty when
    the account holds nothing (never fabricated zeros)."""
    if not holdings or not total_value:
        return {}, None
    buckets: dict[str, float] = {}
    for h in holdings:
        bucket = _ASSET_CLASS_BY_SECURITY_TYPE.get(
            str(h.get("asset_type") or ""), "other"
        )
        buckets[bucket] = buckets.get(bucket, 0.0) + (h["market_value"] or 0)
    allocation = {
        k: round(v / total_value, 4) for k, v in buckets.items() if v
    }
    largest = max(h["market_value"] or 0 for h in holdings)
    return allocation, round(largest / total_value, 4)


def sync_investment_holdings(db, uid: str, item_id: str) -> dict:
    """Runs /investments/holdings/get for one Item and upserts one
    users/{uid}/investmentAccounts/{accountId} document per investment
    account, per Section 8. Idempotent: docs are keyed by account_id and
    set() replaces them wholesale, so a re-run cannot duplicate holdings.

    total_value_sgd carries the value in the account's own currency (USD in
    Sandbox), with iso_currency_code stored alongside — same
    no-fabricated-FX convention as Phase 1's balance fields.
    """
    user_ref = db.collection("users").document(uid)
    snap = user_ref.collection("plaidItems").document(item_id).get()
    if not snap.exists:
        raise LookupError(f"No Plaid item {item_id} for this user")
    item = snap.to_dict()

    response = (
        get_plaid_client()
        .investments_holdings_get(
            InvestmentsHoldingsGetRequest(access_token=item["access_token"])
        )
        .to_dict()
    )
    securities = {s["security_id"]: s for s in response["securities"]}

    holdings_by_account: dict[str, list[dict]] = {}
    for holding in response["holdings"]:
        sec = securities.get(holding.get("security_id"), {})
        holdings_by_account.setdefault(holding["account_id"], []).append(
            {
                "ticker": sec.get("ticker_symbol"),
                "name": sec.get("name"),
                "quantity": holding.get("quantity"),
                "market_value": holding.get("institution_value"),
                # Raw Plaid security type (equity/etf/fixed income/...);
                # the equities/bonds/cash grouping happens in
                # asset_allocation, keeping the raw signal auditable.
                "asset_type": sec.get("type"),
                # Nullable — passed through when Plaid provides it. Sandbox
                # fixture values are semantically inconsistent (per-share vs
                # total cost); the Flutter side sanity-guards derived stats.
                "cost_basis": holding.get("cost_basis"),
            }
        )

    stored = []
    for acct in response["accounts"]:
        if str(acct["type"]) not in INVESTMENT_ACCOUNT_TYPES:
            continue
        account_id = acct["account_id"]
        holdings = holdings_by_account.get(account_id, [])
        total_value = sum(h["market_value"] or 0 for h in holdings)
        allocation, concentration = _allocation_and_concentration(
            holdings, total_value
        )
        subtype = str(acct["subtype"]) if acct.get("subtype") else None
        balances = acct.get("balances") or {}
        user_ref.collection("investmentAccounts").document(account_id).set(
            {
                "account_id": account_id,
                "uid": uid,
                "item_id": item_id,
                "source": "plaid",
                "account_name": acct.get("name"),
                "account_type": map_investment_account_type(subtype),
                # Raw Plaid subtype (e.g. "401k") kept for audit/display,
                # same rationale as insurance's policy_type_raw — the
                # SG-enum mapping above is deliberately lossy.
                "account_subtype_raw": subtype,
                "total_value_sgd": total_value,
                "holdings": holdings,
                "asset_allocation": allocation,
                "concentration_risk": concentration,
                "iso_currency_code": balances.get("iso_currency_code"),
                "last_synced": firestore.SERVER_TIMESTAMP,
            }
        )
        stored.append(
            {
                "account_id": account_id,
                "account_name": acct.get("name"),
                "account_type": map_investment_account_type(subtype),
                "account_subtype_raw": subtype,
                "holdings_count": len(holdings),
                "total_value": total_value,
            }
        )
    return {"item_id": item_id, "accounts": stored}


def sync_investment_transactions(
    db, uid: str, item_id: str, days: int = INVESTMENT_TXN_WINDOW_DAYS
) -> dict:
    """Runs /investments/transactions/get for one Item over the last `days`
    days and upserts into investmentAccounts/{accountId}/transactions/
    {transactionId} per Section 8. Idempotent: docs are keyed by Plaid's
    investment_transaction_id, so re-syncing an overlapping window rewrites
    the same documents instead of duplicating them.

    Amounts keep Plaid's sign convention (positive = cash debited, e.g. a
    buy; negative = cash credited, e.g. a deposit into the account or sale
    proceeds) — the investment-rate feature sums abs(amount) of
    type == "deposit" rows per month.
    """
    user_ref = db.collection("users").document(uid)
    snap = user_ref.collection("plaidItems").document(item_id).get()
    if not snap.exists:
        raise LookupError(f"No Plaid item {item_id} for this user")
    item = snap.to_dict()

    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    client = get_plaid_client()

    transactions: list[dict] = []
    securities: dict[str, dict] = {}
    offset = 0
    while True:
        response = client.investments_transactions_get(
            InvestmentsTransactionsGetRequest(
                access_token=item["access_token"],
                start_date=start_date,
                end_date=end_date,
                options=InvestmentsTransactionsGetRequestOptions(
                    count=500, offset=offset
                ),
            )
        ).to_dict()
        transactions.extend(response["investment_transactions"])
        securities.update(
            {s["security_id"]: s for s in response["securities"]}
        )
        offset = len(transactions)
        if offset >= response["total_investment_transactions"]:
            break

    investment_ids = {
        doc.id
        for doc in user_ref.collection("investmentAccounts")
        .where(filter=FieldFilter("item_id", "==", item_id))
        .stream()
    }

    batch = db.batch()
    ops = written = 0
    skipped_unmapped: dict[str, int] = {}

    def flush(force: bool = False) -> None:
        nonlocal batch, ops
        if ops and (force or ops >= _BATCH_LIMIT):
            batch.commit()
            batch = db.batch()
            ops = 0

    for txn in transactions:
        mapped_type = map_investment_txn_type(txn)
        if mapped_type is None:
            # No honest home in the 5-value enum (fees, transfers, taxes,
            # corporate actions...) — count per raw kind, never guess.
            raw = f"{txn.get('type')}/{txn.get('subtype')}"
            skipped_unmapped[raw] = skipped_unmapped.get(raw, 0) + 1
            continue
        if txn["account_id"] not in investment_ids:
            # Transactions only land under accounts the holdings sync has
            # written; run sync_investment_holdings first (the combined
            # sync below always does).
            continue
        sec = securities.get(txn.get("security_id") or "", {})
        txn_date = txn.get("date")
        doc_ref = (
            user_ref.collection("investmentAccounts")
            .document(txn["account_id"])
            .collection("transactions")
            .document(txn["investment_transaction_id"])
        )
        batch.set(
            doc_ref,
            {
                "transaction_id": txn["investment_transaction_id"],
                "account_id": txn["account_id"],
                "type": mapped_type,
                "amount": txn.get("amount"),
                "quantity": txn.get("quantity"),
                "price": txn.get("price"),
                "ticker": sec.get("ticker_symbol"),
                "iso_currency_code": txn.get("iso_currency_code"),
                "date": (
                    datetime.combine(txn_date, time.min, tzinfo=timezone.utc)
                    if txn_date
                    else None
                ),
            },
        )
        ops += 1
        written += 1
        flush()
    flush(force=True)

    return {
        "item_id": item_id,
        "fetched": len(transactions),
        "written": written,
        "skipped_unmapped": skipped_unmapped,
        "window_days": days,
    }


def sync_item_investments(db, uid: str, item_id: str) -> dict:
    """Full investments refresh for one Item: holdings (which creates the
    investmentAccounts docs and their cached asset_allocation /
    concentration_risk fields) first, then the transactions window. Also
    stamps last_synced on the plaidItems doc, mirroring the banking sync."""
    holdings = sync_investment_holdings(db, uid, item_id)
    transactions = sync_investment_transactions(db, uid, item_id)
    db.collection("users").document(uid).collection("plaidItems").document(
        item_id
    ).update({"last_synced": firestore.SERVER_TIMESTAMP})
    return {
        "item_id": item_id,
        "holdings": holdings["accounts"],
        "transactions": {
            k: v for k, v in transactions.items() if k != "item_id"
        },
    }


def sync_all_items_investments(db, uid: str) -> list[dict]:
    """Investments refresh for every Plaid Item the user has connected."""
    item_ids = [
        doc.id
        for doc in db.collection("users")
        .document(uid)
        .collection("plaidItems")
        .stream()
    ]
    return [
        sync_item_investments(db, uid, item_id) for item_id in item_ids
    ]
