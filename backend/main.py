import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import plaid
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from pydantic import BaseModel

from data import (
    account_deletion,
    financial_profile,
    item_disconnect,
    plaid_service,
    policy_deletion,
    score_profile,
)
from firebase import get_current_uid, get_db, init_firebase
from insurance_parser import GeminiInsuranceParser
from ml import score_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="Unify API")

# Real policy documents in the gold set top out well under this; the limit
# exists to stop accidental huge uploads from being streamed to Gemini.
MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB

_parser: GeminiInsuranceParser | None = None


def get_parser() -> GeminiInsuranceParser:
    """Lazy singleton so the API can start (and /health work) without a
    Gemini key; the parser endpoint returns 503 until one is configured."""
    global _parser
    if _parser is None:
        try:
            _parser = GeminiInsuranceParser(
                api_key=os.environ.get("GEMINI_API_KEY", "")
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail="Insurance parser is not configured on the server "
                "(GEMINI_API_KEY missing)",
            ) from exc
    return _parser


@app.on_event("startup")
def startup() -> None:
    init_firebase()


# ---------------------------------------------------------------------------
# Health + identity
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Unify Backend"}


@app.get("/me")
def get_me(uid: str = Depends(get_current_uid)):
    """Returns the caller's users/{uid} document. Proves the full chain:
    Flutter signs in -> sends ID token -> backend verifies -> reads Firestore.
    """
    snap = get_db().collection("users").document(uid).get()
    if not snap.exists:
        raise HTTPException(
            status_code=404,
            detail="User profile not found — complete onboarding first",
        )
    data = snap.to_dict()
    return {"uid": uid, **data}


# ---------------------------------------------------------------------------
# Insurance policy upload (extraction-only, see brief Section 8)
# ---------------------------------------------------------------------------


@app.post("/insurance/parse")
def parse_insurance_pdf(
    file: UploadFile = File(...),
    uid: str = Depends(get_current_uid),
):
    """Parses an uploaded insurance policy PDF with Gemini and persists the
    extracted fields to users/{uid}/insurancePolicies/{policyId}.

    The raw PDF is never persisted — it exists only in memory for the
    duration of this request (only the bytes sent to Gemini leave the
    process). Returns the full written document (including the generated
    policy_id) so the client needs no follow-up Firestore read.
    """
    filename = Path(file.filename or "").name
    content_type = (file.content_type or "").lower()
    looks_like_pdf = (filename.lower().endswith(".pdf")
                      or content_type == "application/pdf")
    if not looks_like_pdf:
        raise HTTPException(
            status_code=400, detail="Only PDF files are accepted"
        )
    if not filename:
        filename = "policy.pdf"  # display-only fallback for original_filename

    pdf_bytes = file.file.read(MAX_PDF_BYTES + 1)
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MB limit",
        )
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400, detail="File is not a valid PDF"
        )

    parser = get_parser()

    # extract_bytes() owns all retry/validation logic; it returns a validated
    # InsurancePolicyExtraction or None once the retry budget is spent. The
    # PDF bytes stay in memory — nothing is written to disk or Storage.
    try:
        result = parser.extract_bytes(pdf_bytes, filename=filename)
    except Exception as exc:
        logger.exception("Insurance parser raised for %s", filename)
        raise HTTPException(
            status_code=502,
            detail="Insurance parser failed unexpectedly — the document "
            "was not saved. Please try again.",
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=502,
            detail="Could not extract policy details from this PDF after "
            "multiple attempts — the document was not saved. Please try a "
            "clearer copy of the policy.",
        )

    extracted = result.model_dump()

    # Section 8 stores renewal_date as a Firestore timestamp; the parser emits
    # an ISO YYYY-MM-DD string (or null).
    renewal_ts = None
    if extracted["renewal_date"]:
        try:
            renewal_ts = datetime.fromisoformat(
                extracted["renewal_date"]
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning(
                "Non-ISO renewal_date %r from parser — storing null",
                extracted["renewal_date"],
            )

    db = get_db()
    doc_ref = (
        db.collection("users")
        .document(uid)
        .collection("insurancePolicies")
        .document()
    )
    doc = {
        "policy_id": doc_ref.id,
        "uid": uid,
        **extracted,
        "renewal_date": renewal_ts,
        "original_filename": filename,
        "uploaded_at": firestore.SERVER_TIMESTAMP,
    }
    write_result = doc_ref.set(doc)

    # A new policy changes the Insurance features, and can be the last missing
    # piece that makes a score computable at all — recompute, non-fatally. The
    # policy is already written, so a score failure must not fail this upload.
    score = _refresh_score_nonfatal(db, uid)

    # renewal_ts (a datetime) serializes to the same ISO 8601 timestamp that
    # Firestore stores, so response and document carry one consistent type.
    return {
        **doc,
        "uploaded_at": write_result.update_time.isoformat(),
        "score": score,
    }


@app.delete("/insurance/{policy_id}")
def delete_insurance_policy(
    policy_id: str,
    uid: str = Depends(get_current_uid),
):
    """Deletes one of the caller's parsed policies, then recomputes the score.

    AUTHORISATION. Same model as /plaid/disconnect-item: policy_id only selects
    a document beneath users/{uid}/insurancePolicies/, where uid comes from the
    verified ID token. Someone else's policy_id resolves to nothing at that
    path and 404s — not 403, which would confirm that another user holds that
    policy. See data/policy_deletion.py.

    A document still present after the delete is a 500 with the policy still
    listed, so the user can simply try again.
    """
    db = get_db()
    try:
        summary = policy_deletion.delete_policy(db, uid, policy_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404, detail="No policy with that ID for this user"
        ) from exc
    except RuntimeError as exc:
        logger.error("policy delete incomplete for %s: %s", uid, exc)
        raise HTTPException(
            status_code=500,
            detail="The policy could not be removed. It is still listed — "
            "please try deleting it again.",
        ) from exc

    # The score pipeline reads every policy, so removing one moves the
    # Insurance features and can move the tier and the overall score. Score
    # only — unlike disconnect there is no financialProfile refresh, for the
    # same reason upload has none: policies feed no balance or net-worth field.
    return {
        "status": "deleted",
        **summary,
        "score": _refresh_score_nonfatal(db, uid),
    }


# ---------------------------------------------------------------------------
# financialProfile (Home Phase 1, see brief Sections 5 & 8)
# ---------------------------------------------------------------------------


def _refresh_profile_nonfatal(db, uid: str) -> dict | None:
    """Derived-field refresh attached to the Plaid sync endpoints. A refresh
    failure must never fail the sync that triggered it — the fields catch up
    on the next trigger."""
    try:
        return financial_profile.refresh_financial_profile(db, uid)
    except Exception:
        logger.exception("financialProfile refresh failed for %s", uid)
        return None


def _refresh_score_nonfatal(db, uid: str) -> dict | None:
    """Phase 3 score recompute attached to every endpoint that can change a
    score input. Same non-fatal contract as _refresh_profile_nonfatal above —
    the score is derived data hanging off somebody else's operation, and a
    failure here must never fail the sync or the policy upload that triggered
    it.

    ORDER MATTERS on the Plaid endpoints: Banking's savings_rate and
    emergency_fund_months are computed from monthly_expenses_sgd /
    net_worth_sgd, so this must run AFTER _refresh_profile_nonfatal, never
    beside it.
    """
    return score_profile.refresh_score_nonfatal(db, uid)


@app.post("/profile/refresh")
def refresh_financial_profile(uid: str = Depends(get_current_uid)):
    """Recomputes the financialProfile derived fields (monthly_expenses_sgd,
    net_worth_sgd) from the caller's synced Plaid data on demand. The Plaid
    sync endpoints below refresh implicitly on every sync, which covers the
    app's needs today — no client code calls this endpoint currently; it is
    kept for manual testing and as the likely Phase 4 Home-screen refresh
    hook."""
    db = get_db()
    derived = financial_profile.refresh_financial_profile(db, uid)
    if derived is None:
        raise HTTPException(
            status_code=404,
            detail="No financialProfile document — provide monthly income "
            "first (onboarding or the income prompt)",
        )
    # The derived fields feed Banking's features, so a refresh can change the
    # score — recompute it here too, non-fatally.
    return {"status": "refreshed", **derived,
            "score": _refresh_score_nonfatal(db, uid)}


# ---------------------------------------------------------------------------
# Phase 3 score (Layer 1 + Layer 2 tiers + Layer 3, see brief Section 5)
# ---------------------------------------------------------------------------


@app.post("/score/refresh")
def refresh_score(uid: str = Depends(get_current_uid)):
    """Runs the full score pipeline for the caller and caches the result to
    financialProfile (overall_score, checklist_flags, checklist_last_updated,
    ml_profile), returning the whole result object.

    Every sync endpoint below already recomputes implicitly, so the client
    normally never needs this. It exists for the one trigger the backend cannot
    see: onboarding writes monthly_income_sgd straight to Firestore from the
    client, and income is what makes a score computable at all — so the app
    should call this once after onboarding completes.

    A user-data problem is NOT an error here: no income, an age outside the
    models' 20-70 range, or only one connected domain all return 200 with an
    explicit reason code (see ml/score_pipeline.py). The only 404 is a user with
    no financialProfile document at all, i.e. onboarding never finished.
    """
    db = get_db()
    result = score_profile.refresh_score_profile(db, uid)
    if result["unavailable_reason"] == score_pipeline.REASON_NO_FINANCIAL_PROFILE:
        raise HTTPException(
            status_code=404,
            detail="No financialProfile document — complete onboarding first",
        )
    return result


@app.get("/score")
def get_score(uid: str = Depends(get_current_uid)):
    """Returns the CACHED score without recomputing — the fast path for a normal
    app open. Use POST /score/refresh to recompute.

    ``computed: false`` (with everything else null) is the explicit
    never-computed-yet state, returned as 200: a brand-new account that has not
    connected anything yet is a normal state, not an error. 404 means no
    financialProfile document at all.
    """
    cached = score_profile.read_cached_score(get_db(), uid)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="No financialProfile document — complete onboarding first",
        )
    return cached


# The default window the Home trend chart asks for, and the hard ceiling on
# what one request can return.
#
# 90 days because that is roughly what a trend chart can render legibly on a
# phone at one point per day, and because a snapshot is a few hundred bytes of
# nested maps — 90 of them is a small response, 365 is not. The ceiling exists
# so the window stays a client choice without becoming an unbounded read of a
# collection that grows by one document a day forever.
#
# A shorter window than the data available is not a gap: snapshots are written
# once a day, so `limit` days back is `limit` calendar days back at most.
HISTORY_DEFAULT_LIMIT = 90
HISTORY_MAX_LIMIT = 365


@app.get("/score/history")
def get_score_history(
    uid: str = Depends(get_current_uid),
    limit: int = Query(HISTORY_DEFAULT_LIMIT, ge=1, le=HISTORY_MAX_LIMIT),
):
    """The caller's daily score snapshots, newest first.

    One document per UTC day, written by the score pipeline's cache hook — see
    data/score_profile.py's scoreHistory block. This endpoint only reads them;
    nothing here recomputes or backfills.

    An empty list is a normal 200, not a 404: a user who has never been scored,
    and one whose account predates this collection, both legitimately have no
    history yet, and neither is an error the client should route on. That is
    the same reasoning GET /score uses for ``computed: false``.

    A snapshot with ``overall_score: null`` is a real day, not a hole to filter
    out — it is a day the score could not be computed, and
    ``overall_score_reason`` says why. The chart should show it as a gap.
    """
    snapshots = score_profile.read_score_history(get_db(), uid, limit)
    return {"count": len(snapshots), "limit": limit, "snapshots": snapshots}


# ---------------------------------------------------------------------------
# Delete Account (Settings Pass 2)
# ---------------------------------------------------------------------------


@app.delete("/account")
def delete_account(uid: str = Depends(get_current_uid)):
    """Irreversibly deletes the caller's account: every Firestore document under
    users/{uid}, then the Firebase Auth record.

    Authenticated by the same get_current_uid dependency as every other
    endpoint, which is also the authorisation model — a caller can only ever
    delete the uid inside their own verified ID token. There is deliberately no
    uid parameter; an endpoint that took one would be an account-deletion
    primitive aimed at other people's accounts.

    The client is expected to have reauthenticated the user immediately before
    calling this (Settings > Delete Account does, reusing the same Firebase Auth
    reauthentication Password & Security established in Pass 1). That is a
    client-side friction requirement rather than something this endpoint can
    enforce: an ID token does not carry how recently the password was entered.

    Deletion ORDER is the safety property — see data/account_deletion.py. A
    Firestore failure leaves the Auth user intact and the account fully usable,
    which is why that case is a 500 with the account still working rather than a
    half-deleted account.
    """
    try:
        return account_deletion.delete_account(get_db(), uid)
    except RuntimeError as exc:
        logger.error("account deletion incomplete for %s: %s", uid, exc)
        raise HTTPException(
            status_code=500,
            detail="Your account was NOT deleted — some data could not be "
            "removed, so nothing was finalised and your account still works. "
            "Please try again.",
        ) from exc


# ---------------------------------------------------------------------------
# Plaid (Phases 1 banking + 3 investments, see brief Section 6)
# ---------------------------------------------------------------------------


class PublicTokenExchangeBody(BaseModel):
    public_token: str


class SyncTransactionsBody(BaseModel):
    # Omit item_id to sync every Item the user has connected.
    item_id: str | None = None


def _plaid_error_code(exc: plaid.ApiException) -> str:
    """Plaid's error_code from an ApiException, or UNKNOWN."""
    try:
        return json.loads(exc.body).get("error_code", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def _plaid_http_error(exc: plaid.ApiException) -> HTTPException:
    """Surfaces only Plaid's error_code — never the raw response body, which
    is not client-facing material."""
    code = _plaid_error_code(exc)
    logger.error("Plaid API error: %s", code)
    return HTTPException(status_code=502, detail=f"Plaid API error: {code}")


@app.post("/plaid/link-token")
def create_plaid_link_token(uid: str = Depends(get_current_uid)):
    """Creates a Plaid Link token the Flutter app uses to launch Plaid Link
    (Phase 2). Sandbox is US-institutions-only by design."""
    try:
        return plaid_service.create_link_token(uid)
    except plaid.ApiException as exc:
        raise _plaid_http_error(exc) from exc


@app.post("/plaid/exchange-token")
def exchange_plaid_public_token(
    body: PublicTokenExchangeBody,
    uid: str = Depends(get_current_uid),
):
    """Exchanges the public_token from Plaid Link, stores the Item and its
    banking accounts (Section 8 schema), then runs the initial transactions
    sync in the same request so Phase 2 needs no follow-up call.

    The access_token obtained here is written to plaidItems only — it is
    never included in this (or any) response body.
    """
    db = get_db()
    try:
        summary = plaid_service.connect_bank_item(db, uid, body.public_token)
    except plaid.ApiException as exc:
        raise _plaid_http_error(exc) from exc

    # The Item and accounts are already stored, so a sync hiccup (e.g.
    # Plaid still preparing transactions for a brand-new Item) should not
    # fail the whole exchange — report it and let the client re-sync.
    try:
        initial_sync = plaid_service.sync_item_transactions(
            db, uid, summary["item_id"]
        )
    except plaid.ApiException as exc:
        code = _plaid_http_error(exc).detail
        initial_sync = {"error": code}

    # Phase 3: the investment-type accounts Phase 1 skips out of
    # plaidAccounts land in investmentAccounts here, at connect time, so
    # the Investments screen has data without a separate first call.
    # Equally non-fatal — /plaid/sync-investments re-runs it.
    try:
        initial_investments = plaid_service.sync_item_investments(
            db, uid, summary["item_id"]
        )
    except plaid.ApiException as exc:
        code = _plaid_http_error(exc).detail
        initial_investments = {"error": code}

    # Home Phase 1: newly synced balances/transactions change the
    # financialProfile derived fields — refresh them on this sync trigger.
    financial_profile_derived = _refresh_profile_nonfatal(db, uid)

    return {
        "status": "connected",
        **summary,
        "initial_sync": initial_sync,
        "initial_investments_sync": initial_investments,
        "financial_profile": financial_profile_derived,
        # Phase 3: a first bank connection is usually what makes Banking and
        # Investment assessable at all. After the derived-field refresh, never
        # before — the score reads those fields.
        "score": _refresh_score_nonfatal(db, uid),
    }


@app.post("/plaid/sync-transactions")
def sync_plaid_transactions(
    body: SyncTransactionsBody,
    uid: str = Depends(get_current_uid),
):
    """Incremental /transactions/sync using the cursor on each plaidItems
    document. Pass item_id to sync one Item, omit it to sync all. Safe to
    call repeatedly — writes are keyed by transaction_id (upserts)."""
    db = get_db()
    try:
        if body.item_id:
            results = [
                plaid_service.sync_item_transactions(db, uid, body.item_id)
            ]
        else:
            results = plaid_service.sync_all_items(db, uid)
    except LookupError as exc:
        raise HTTPException(
            status_code=404, detail="No Plaid item with that ID for this user"
        ) from exc
    except plaid.ApiException as exc:
        raise _plaid_http_error(exc) from exc
    financial_profile_derived = _refresh_profile_nonfatal(db, uid)
    return {
        "items_synced": len(results),
        "results": results,
        "financial_profile": financial_profile_derived,
        "score": _refresh_score_nonfatal(db, uid),
    }


@app.post("/plaid/sync-investments")
def sync_plaid_investments(
    body: SyncTransactionsBody,
    uid: str = Depends(get_current_uid),
):
    """Phase 3: refreshes investment holdings (investmentAccounts docs with
    their cached asset_allocation/concentration_risk) and the investment
    transactions window. A separate endpoint from /plaid/sync-transactions
    — banking sync is cursor-based/incremental while this is a date-ranged
    full refresh, and keeping them apart leaves the verified Phase 1 path
    untouched. Pass item_id for one Item, omit it for all. Safe to call
    repeatedly — holdings docs and transactions are keyed by their Plaid
    IDs (upserts)."""
    db = get_db()
    try:
        if body.item_id:
            results = [
                plaid_service.sync_item_investments(db, uid, body.item_id)
            ]
        else:
            results = plaid_service.sync_all_items_investments(db, uid)
    except LookupError as exc:
        raise HTTPException(
            status_code=404, detail="No Plaid item with that ID for this user"
        ) from exc
    except plaid.ApiException as exc:
        raise _plaid_http_error(exc) from exc
    financial_profile_derived = _refresh_profile_nonfatal(db, uid)
    return {
        "items_synced": len(results),
        "results": results,
        "financial_profile": financial_profile_derived,
        "score": _refresh_score_nonfatal(db, uid),
    }


# How recently a targeted Item may have synced before POST /plaid/refresh
# declines to call Plaid again. Long enough to absorb a repeated pull-to-refresh
# or a double tap; short enough that a deliberate retry soon works. There is
# deliberately no request field that bypasses it.
REFRESH_COOLDOWN_SECONDS = 30


def _item_has_investment_accounts(db, uid: str, item_id: str) -> bool:
    return any(
        db.collection("users").document(uid)
        .collection("investmentAccounts")
        .where(filter=FieldFilter("item_id", "==", item_id))
        .limit(1)
        .stream()
    )


def _refresh_one_item(db, uid: str, item_id: str, institution_name) -> dict:
    """Transactions, then balances, then investments (only when the Item has
    investment accounts), for one Item. Never raises: a failure is recorded
    on the returned entry so the caller can carry on with the other Items.
    Steps that completed before a failure stay written — the sync functions
    are idempotent, so the next refresh simply repeats them."""
    entry = {
        "item_id": item_id,
        "institution_name": institution_name,
        "status": "refreshed",
        "error_code": None,
        "transactions": None,
        "accounts_updated": 0,
        "investments_synced": False,
    }
    try:
        txn = plaid_service.sync_item_transactions(db, uid, item_id)
        entry["transactions"] = {
            k: txn[k] for k in ("added", "modified", "removed")
        }
        balances = plaid_service.refresh_item_balances(db, uid, item_id)
        entry["accounts_updated"] = balances["accounts_updated"]
        if _item_has_investment_accounts(db, uid, item_id):
            plaid_service.sync_item_investments(db, uid, item_id)
            entry["investments_synced"] = True
    except plaid.ApiException as exc:
        entry["status"] = "failed"
        entry["error_code"] = _plaid_error_code(exc)
        logger.error("refresh failed for item %s: Plaid %s",
                     item_id, entry["error_code"])
    except Exception:
        entry["status"] = "failed"
        entry["error_code"] = "INTERNAL_ERROR"
        logger.exception("refresh failed for item %s", item_id)
    return entry


@app.post("/plaid/refresh")
def refresh_plaid_data(
    body: SyncTransactionsBody | None = None,
    uid: str = Depends(get_current_uid),
):
    """Brings the caller's connected data up to date without reconnecting:
    new transactions, current balances and (where present) holdings, then
    the derived financialProfile fields and the score.

    Pass item_id to refresh one Item; omit it (or send no body) for all.
    Like every other route, item_id only selects a document beneath the
    caller's own users/{uid} subtree, so a foreign or unknown one is a 404.

    Statuses: "no_connections" (nothing to refresh, score recomputed anyway),
    "skipped_recent" (a targeted Item synced within REFRESH_COOLDOWN_SECONDS;
    no Plaid call made), and otherwise "refreshed" / "partial" / "failed"
    with one entry per Item. A per-Item Plaid error such as
    ITEM_LOGIN_REQUIRED is reported on that Item's entry and does not stop
    the others. All of these are 200s.
    """
    db = get_db()
    items_ref = db.collection("users").document(uid).collection("plaidItems")
    item_id = body.item_id if body else None
    if item_id:
        snap = items_ref.document(item_id).get()
        if not snap.exists:
            raise HTTPException(
                status_code=404, detail="No Plaid item with that ID for this user"
            )
        targets = [snap]
    else:
        targets = list(items_ref.stream())

    if not targets:
        return {
            "status": "no_connections",
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
            "score": _refresh_score_nonfatal(db, uid),
        }

    synced_times = [
        t for t in (snap.to_dict().get("last_synced") for snap in targets) if t
    ]
    newest = max(synced_times) if synced_times else None
    now = datetime.now(timezone.utc)
    if newest is not None and (
        (now - newest).total_seconds() < REFRESH_COOLDOWN_SECONDS
    ):
        return {
            "status": "skipped_recent",
            "last_synced": newest.isoformat(),
            "cooldown_seconds": REFRESH_COOLDOWN_SECONDS,
        }

    entries = [
        _refresh_one_item(
            db, uid, snap.id, snap.to_dict().get("institution_name")
        )
        for snap in targets
    ]
    succeeded = sum(e["status"] == "refreshed" for e in entries)
    if succeeded == len(entries):
        status = "refreshed"
    elif succeeded:
        status = "partial"
    else:
        status = "failed"

    financial_profile_derived = score = None
    if succeeded:
        # Same order as the sync and disconnect routes: derived fields first,
        # score second — Banking's features are computed from them.
        financial_profile_derived = _refresh_profile_nonfatal(db, uid)
        score = _refresh_score_nonfatal(db, uid)

    return {
        "status": status,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "items": entries,
        "financial_profile": financial_profile_derived,
        "score": score,
    }


class DisconnectItemBody(BaseModel):
    item_id: str


@app.post("/plaid/disconnect-item")
def disconnect_plaid_item(
    body: DisconnectItemBody,
    uid: str = Depends(get_current_uid),
):
    """Revokes one Plaid Item at Plaid and deletes everything stored under it
    for the caller: its plaidAccounts and investmentAccounts documents, each of
    their transactions subcollections, and the plaidItems document itself.

    AUTHORISATION. The item_id in the body is not trusted as an identifier of
    whose data to touch — it only selects a document underneath
    users/{uid}/..., where uid comes from the verified ID token. An item_id
    belonging to another user resolves to nothing at that path and 404s. See
    data/item_disconnect.py's docstring; there is deliberately no uid parameter,
    for the same reason DELETE /account has none.

    404 rather than 403 on a foreign item_id, matching the existing
    /plaid/sync-* routes: the two cases are genuinely indistinguishable from
    inside the caller's own subtree, and a 403 would confirm that some other
    user holds that Item — an existence oracle this endpoint should not offer.

    A Plaid revocation failure is a 502 with nothing deleted, so the user can
    retry (item_disconnect's docstring covers why this is fatal here while the
    same call is best-effort during full account deletion). An incomplete
    Firestore cascade is a 500 with the connection still listed and still
    retryable — the same degrade-to-"nothing happened" contract DELETE /account
    uses.
    """
    db = get_db()
    try:
        summary = item_disconnect.disconnect_item(db, uid, body.item_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404, detail="No Plaid item with that ID for this user"
        ) from exc
    except plaid.ApiException as exc:
        # Nothing local was touched — the cascade runs only after revocation.
        raise _plaid_http_error(exc) from exc
    except RuntimeError as exc:
        logger.error("disconnect incomplete for %s: %s", uid, exc)
        raise HTTPException(
            status_code=500,
            detail="The connection was revoked but some of its data could not "
            "be removed. It is still listed — please try disconnecting again.",
        ) from exc

    # Removing accounts changes balances, net worth and the investment totals,
    # so the derived fields move and the score moves with them. Same order as
    # the sync endpoints: derived fields first, score second — Banking's
    # features are computed from them.
    financial_profile_derived = _refresh_profile_nonfatal(db, uid)
    return {
        "status": "disconnected",
        **summary,
        "financial_profile": financial_profile_derived,
        "score": _refresh_score_nonfatal(db, uid),
    }
