"""Disconnect one bank connection — the cascade behind POST /plaid/disconnect-item.

The single-Item sibling of account_deletion.py. Same shape, same batched-delete
helpers (imported from there rather than re-implemented, so a fix to the batching
or the "missing document" handling lands in both), but scoped to one Plaid Item
instead of an entire account — and with one deliberate difference in failure
policy, described below.

OWNERSHIP IS STRUCTURAL, NOT A COMPARISON
There is no `if item["uid"] == uid` check here, and that is the point. Every
lookup starts at users/{uid}/plaidItems/{item_id}, where the uid comes from the
caller's verified Firebase ID token and nothing else. An item_id belonging to
another user is not "rejected" — it is simply not present at that path, so it
raises LookupError exactly like a typo would. A stolen or guessed item_id
therefore cannot reach another user's data even in principle; there is no code
path that takes a uid from the request body. This mirrors sync_item_transactions
and is the same authorisation model DELETE /account uses (see its docstring: an
endpoint that accepted a uid would be a primitive aimed at other people).

Every Firestore query below is likewise rooted at users/{uid}/..., so even a
malformed item_id can only ever scope a delete to the caller's own subtree.

DISCONNECT ORDER, AND WHY

  1. Plaid /item/remove FIRST, while the access_token on plaidItems is still
     readable. Same forced ordering as account_deletion: once plaidItems is
     gone the token is gone with it, and the Item would stay live at Plaid
     forever with nothing left to revoke it with.

  2. Firestore, deepest first: each account's transactions subcollection, then
     the account documents, for BOTH plaidAccounts and investmentAccounts —
     then the plaidItems document LAST. Deepest-first for the reason
     account_deletion documents (a deleted parent does not take its
     subcollections with it). plaidItems last so that a failure partway through
     leaves the connection still listed in Settings and still retryable, rather
     than stranding account documents whose owning Item has vanished.

  3. Scoped emptiness is VERIFIED, not assumed — re-read by item_id, and if
     anything survived, raise.

WHY REVOCATION IS FATAL HERE BUT BEST-EFFORT IN account_deletion
account_deletion treats a Plaid failure as non-fatal because a user must always
be able to delete their account; blocking that on a third-party outage would be
worse than leaving a Sandbox Item alive. Disconnect has the opposite balance:
revoking Plaid's access IS the user's intent, and the local rows are the only
handle left on that access_token. Deleting them after a failed revocation would
destroy the only means of ever completing it — a silent, permanent leak of a
live connection, presented to the user as "disconnected". So a revocation
failure aborts before anything local is touched, and the user can retry.

An Item already absent at Plaid (ITEM_NOT_FOUND) counts as revoked, so a retry
after a partial failure can still finish — the same reasoning that lets
account_deletion treat an already-absent Auth user as deleted.
"""
from __future__ import annotations

import json
import logging

import plaid
from google.cloud.firestore_v1.base_query import FieldFilter

# Reused rather than re-implemented: these carry the 400-op batching and the
# list_documents() (not stream()) behaviour that catches fieldless "missing"
# parent documents holding subcollections. Duplicating them would mean a future
# fix to either landing in only one of the two cascades.
from .account_deletion import _delete_collection, _delete_refs

logger = logging.getLogger(__name__)

# The account-scoped collections that can carry an item_id and own a
# transactions subcollection of their own. manualAccounts is deliberately
# absent: it has no item_id (Section 8 — user-entered, not Plaid-sourced), so
# no Plaid disconnect can ever be responsible for removing it.
ITEM_SCOPED_COLLECTIONS = ("plaidAccounts", "investmentAccounts")


def disconnect_item(db, uid: str, item_id: str) -> dict:
    """Revoke one Plaid Item and delete everything under it for this user.

    Raises LookupError if the Item is not this user's (see the module docstring
    — that covers both "does not exist" and "belongs to someone else", which are
    indistinguishable by design). Raises plaid.ApiException if Plaid refuses the
    revocation, having touched nothing. Raises RuntimeError if Firestore is not
    verifiably clean afterwards, in which case the connection stays listed and
    the operation can be retried.
    """
    user_ref = db.collection("users").document(uid)
    item_ref = user_ref.collection("plaidItems").document(item_id)
    snap = item_ref.get()
    if not snap.exists:
        # Not 'forbidden' — from this uid's perspective the Item genuinely does
        # not exist. See the module docstring on structural ownership.
        raise LookupError(f"No Plaid item {item_id} for this user")
    item = snap.to_dict() or {}

    institution_name = item.get("institution_name")

    # --- 1. revoke at Plaid, before the token becomes unreachable
    revocation = _remove_at_plaid(item.get("access_token"), uid, item_id)

    # --- 2. local cascade, deepest first
    counts = _delete_item_data(db, uid, item_id)

    # --- plaidItems LAST: while it exists, the row is still listed and the
    # disconnect is still retryable.
    item_ref.delete()
    counts["plaidItems"] = 1

    # --- 3. verify, do not assume
    leftovers = surviving_item_paths(db, uid, item_id)
    if leftovers:
        raise RuntimeError(
            "Disconnect incomplete for %s/%s — surviving: %s"
            % (uid, item_id, ", ".join(leftovers))
        )

    return {
        "item_id": item_id,
        "institution_name": institution_name,
        "plaid": revocation,
        "firestore_documents_deleted": counts,
        "firestore_verified_empty": True,
    }


def _delete_item_data(db, uid: str, item_id: str) -> dict:
    """Every account document carrying this item_id, plus their transactions.

    Split out from disconnect_item so a test can make THIS step fail and assert
    that the connection survives it intact — the same seam delete_user_data
    provides for the account cascade.
    """
    user_ref = db.collection("users").document(uid)
    counts: dict[str, int] = {}

    for collection in ITEM_SCOPED_COLLECTIONS:
        # A query, not list_documents(), because scoping to one Item requires
        # reading the item_id field. Safe here: every document in both
        # collections is written by plaid_service with item_id set, so a
        # fieldless document cannot be an Item's account. (investmentAccounts
        # allows a null item_id for manual entries — those correctly never
        # match, and must not be deleted by a Plaid disconnect.)
        account_refs = [
            doc.reference
            for doc in user_ref.collection(collection)
            .where(filter=FieldFilter("item_id", "==", item_id))
            .stream()
        ]
        transactions = 0
        for account_ref in account_refs:
            for sub in account_ref.collections():
                transactions += _delete_collection(db, sub)
        parents = _delete_refs(db, account_refs)
        if transactions:
            counts[f"{collection}/transactions"] = transactions
        if parents:
            counts[collection] = parents

    return counts


def surviving_item_paths(db, uid: str, item_id: str) -> list[str]:
    """What still references this item_id. Empty list means a clean disconnect.

    Used as the post-cascade gate and by the verification script. Checks the
    account documents AND their subcollections separately, because a deleted
    parent says nothing about its children.
    """
    user_ref = db.collection("users").document(uid)
    surviving: list[str] = []

    if user_ref.collection("plaidItems").document(item_id).get().exists:
        surviving.append(f"users/{uid}/plaidItems/{item_id}")

    for collection in ITEM_SCOPED_COLLECTIONS:
        for doc in (
            user_ref.collection(collection)
            .where(filter=FieldFilter("item_id", "==", item_id))
            .stream()
        ):
            surviving.append(f"users/{uid}/{collection}/{doc.id}")
            for sub in doc.reference.collections():
                for sub_doc in sub.list_documents():
                    surviving.append(
                        f"users/{uid}/{collection}/{doc.id}/{sub.id}/{sub_doc.id}"
                    )
    return surviving


def _remove_at_plaid(access_token: str | None, uid: str, item_id: str) -> dict:
    """Revoke the access_token at Plaid. Fatal on failure — see the module
    docstring for why this differs from account_deletion's best-effort stance.
    """
    from plaid.model.item_remove_request import ItemRemoveRequest

    from . import plaid_service

    if not access_token:
        # Nothing to revoke. Reachable only for a malformed Item document; the
        # local rows should still be removable rather than permanently stuck.
        logger.warning("plaidItems/%s for %s has no access_token — nothing to "
                       "revoke at Plaid", item_id, uid)
        return {"revoked": False, "reason": "no_access_token"}

    try:
        plaid_service.get_plaid_client().item_remove(
            ItemRemoveRequest(access_token=access_token))
        return {"revoked": True}
    except plaid.ApiException as exc:
        if _error_code(exc) == "ITEM_NOT_FOUND":
            # Already gone at Plaid — a retry after a partial failure, or a
            # connection the user removed at the institution. The goal state is
            # satisfied, so the local cascade should proceed.
            logger.info("Plaid Item %s already absent — treating as revoked",
                        item_id)
            return {"revoked": True, "already_absent": True}
        raise


def _error_code(exc: plaid.ApiException) -> str:
    try:
        return json.loads(exc.body).get("error_code", "UNKNOWN")
    except Exception:
        return "UNKNOWN"
