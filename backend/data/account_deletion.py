"""Delete Account — the full cascade behind DELETE /account.

Firestore has no multi-collection transaction at this scale (a transaction caps
at 500 writes and cannot span an unbounded document tree), so the cascade is a
sequence of batched deletes whose ORDER is the safety mechanism. The order is
deliberate and is the main thing to understand in this file.

DELETION ORDER, AND WHY

  1. Plaid Items are removed AT PLAID first, while their access_tokens are
     still readable. Best-effort: a Plaid outage must not make the account
     undeletable. Doing this before the local delete is the only possible
     order — once plaidItems is gone the tokens are gone with it, and the
     Items would stay live at Plaid forever with no way left to reach them.

  2. Firestore, deepest first: every transactions subcollection, then the
     account documents that own them, then the flat subcollections, then the
     users/{uid} document itself. Deepest-first because Firestore documents do
     not own their subcollections — deleting users/{uid} would leave every
     subcollection beneath it intact but orphaned and invisible in the console,
     which is the classic way to "delete" an account and silently keep all of
     its data.

  3. Firestore emptiness is VERIFIED. Not assumed — re-read, and if anything
     survived, stop here and raise.

  4. The Firebase Auth user is deleted LAST, and only if step 3 passed.

WHY AUTH LAST — the failure mode this order exists to prevent
If Auth were deleted first and a Firestore delete then failed, the leftover
data would be unreachable and unrecoverable by any normal path: the uid can
never authenticate again, so the user cannot retry deletion, the client can
never read it back, and DELETE /account itself is unreachable because it is
authenticated by that same uid. Cleanup would need manual Admin SDK
intervention by a developer with the service account key, per orphan.

With Auth last, the same Firestore failure leaves the account fully intact and
fully usable: the user is still signed in, still owns their data, and can
simply press Delete Account again. A partial failure degrades to "nothing
happened", which is the only safe way for an irreversible operation to fail.
verify_account_deletion.py exercises exactly this scenario.
"""
from __future__ import annotations

import logging

from firebase_admin import auth as admin_auth

logger = logging.getLogger(__name__)

# Firestore caps a batch at 500 operations; 400 leaves headroom and keeps the
# batches small enough that a failure retries cheaply.
_BATCH_SIZE = 400

# Section 8's account-scoped subcollections that own a `transactions`
# subcollection of their own. Deleted deepest-first.
NESTED_SUBCOLLECTIONS = ("plaidAccounts", "manualAccounts", "investmentAccounts")

# Everything else under users/{uid}. Listed explicitly rather than discovered,
# so that adding a collection to the schema without adding it here shows up as
# a failed emptiness check in step 3 instead of as data quietly left behind.
# (list_collections() below is still the authority for the check — this tuple
# only drives the deletion pass.)
FLAT_SUBCOLLECTIONS = (
    "financialProfile",
    "plaidItems",
    "insurancePolicies",
    # Settings Pass 2:
    "notificationSettings",
    "notificationTokens",
    "notificationEvents",
    # Score History:
    "scoreHistory",
)


def delete_account(db, uid: str, *, remove_plaid_items: bool = True) -> dict:
    """The whole cascade, in the order documented above.

    Returns a summary of what was deleted. Raises RuntimeError if Firestore is
    not verifiably empty afterwards — in which case the Auth user is deliberately
    left alive and the account remains usable.
    """
    plaid_summary = (_remove_plaid_items(db, uid) if remove_plaid_items
                     else {"skipped": True})

    deleted = delete_user_data(db, uid)

    leftovers = surviving_paths(db, uid)
    if leftovers:
        # Auth is NOT touched. See the module docstring: this is the whole point
        # of the ordering.
        raise RuntimeError(
            "Firestore cleanup incomplete for %s — Auth user left intact so the "
            "account stays usable and deletion can be retried. Surviving: %s"
            % (uid, ", ".join(leftovers))
        )

    auth_deleted = _delete_auth_user(uid)

    return {
        "uid": uid,
        "firestore_documents_deleted": deleted,
        "firestore_verified_empty": True,
        "auth_user_deleted": auth_deleted,
        "plaid": plaid_summary,
    }


def delete_user_data(db, uid: str) -> dict:
    """Every Firestore document under users/{uid}, plus the document itself.

    Split out from delete_account so a test can make THIS step fail and assert
    that the Auth user survives it.
    """
    user_ref = db.collection("users").document(uid)
    counts: dict[str, int] = {}

    # --- deepest level first: accounts' transactions
    for collection in NESTED_SUBCOLLECTIONS:
        transactions = 0
        accounts = list(user_ref.collection(collection).list_documents())
        for account_ref in accounts:
            for sub in account_ref.collections():
                transactions += _delete_collection(db, sub)
        # ...then the account documents that owned them
        parents = _delete_refs(db, accounts)
        if transactions:
            counts[f"{collection}/transactions"] = transactions
        if parents:
            counts[collection] = parents

    # --- flat subcollections
    for collection in FLAT_SUBCOLLECTIONS:
        removed = _delete_collection(db, user_ref.collection(collection))
        if removed:
            counts[collection] = removed

    # --- anything not in the lists above (schema drift, an old experiment).
    # Discovered rather than assumed, so nothing can be left behind just
    # because this file was not updated.
    for extra in user_ref.collections():
        removed = _delete_collection(db, extra, recursive=True)
        if removed:
            counts[f"{extra.id} (undeclared)"] = removed

    # --- finally the profile document itself
    user_ref.delete()
    counts["users/{uid}"] = 1
    return counts


def surviving_paths(db, uid: str) -> list[str]:
    """What is still there under users/{uid}. Empty list means a clean delete.

    Used as the pre-Auth-deletion gate and by the verification script. Checks
    the document AND its subcollections separately, because in Firestore a
    deleted parent document says nothing about its children.
    """
    user_ref = db.collection("users").document(uid)
    surviving: list[str] = []
    if user_ref.get().exists:
        surviving.append("users/%s" % uid)
    for collection in user_ref.collections():
        for doc_ref in collection.list_documents():
            surviving.append("users/%s/%s/%s" % (uid, collection.id, doc_ref.id))
            for sub in doc_ref.collections():
                for sub_doc in sub.list_documents():
                    surviving.append("users/%s/%s/%s/%s/%s" % (
                        uid, collection.id, doc_ref.id, sub.id, sub_doc.id))
    return surviving


# ---------------------------------------------------------------------------
# Cascade steps
# ---------------------------------------------------------------------------


def _remove_plaid_items(db, uid: str) -> dict:
    """Tell Plaid to forget the user's Items before their tokens are deleted.

    Best-effort and non-fatal by design: a Plaid failure must not block a user
    from deleting their account. Anything that fails here leaves a Sandbox Item
    alive at Plaid with a token nobody holds — reported, not raised.
    """
    from plaid.model.item_remove_request import ItemRemoveRequest

    from . import plaid_service

    removed, failed = 0, 0
    try:
        item_snaps = list(db.collection("users").document(uid)
                          .collection("plaidItems").stream())
    except Exception:
        logger.exception("could not list plaidItems for %s", uid)
        return {"removed": 0, "failed": 0, "listed": False}

    for snap in item_snaps:
        access_token = (snap.to_dict() or {}).get("access_token")
        if not access_token:
            continue
        try:
            plaid_service.get_plaid_client().item_remove(
                ItemRemoveRequest(access_token=access_token))
            removed += 1
        except Exception:
            logger.warning("Plaid item_remove failed for %s/%s — the Item stays "
                           "live at Plaid", uid, snap.id, exc_info=True)
            failed += 1
    return {"removed": removed, "failed": failed, "listed": True}


def _delete_auth_user(uid: str) -> bool:
    """Delete the Auth record. An already-absent user counts as success —
    a retry after a partial failure must be able to finish."""
    try:
        admin_auth.delete_user(uid)
        return True
    except admin_auth.UserNotFoundError:
        logger.info("Auth user %s already absent — treating as deleted", uid)
        return True


# ---------------------------------------------------------------------------
# Batched deletes
# ---------------------------------------------------------------------------


def _delete_collection(db, collection_ref, recursive: bool = False) -> int:
    """Delete every document in one collection. Returns how many.

    list_documents() (not stream()) because it also returns "missing"
    documents — path nodes that hold subcollections without carrying fields of
    their own. stream() skips those, which would leave their children behind.
    """
    refs = list(collection_ref.list_documents())
    nested = 0
    if recursive:
        for ref in refs:
            for sub in ref.collections():
                nested += _delete_collection(db, sub, recursive=True)
    return _delete_refs(db, refs) + nested


def _delete_refs(db, refs) -> int:
    deleted = 0
    batch = db.batch()
    pending = 0
    for ref in refs:
        batch.delete(ref)
        pending += 1
        deleted += 1
        if pending >= _BATCH_SIZE:
            batch.commit()
            batch = db.batch()
            pending = 0
    if pending:
        batch.commit()
    return deleted
