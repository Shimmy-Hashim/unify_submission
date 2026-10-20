"""Delete one insurance policy — the logic behind DELETE /insurance/{policy_id}.

The smallest sibling of item_disconnect.py and account_deletion.py. Same
ownership model, same verify-don't-assume finish, but none of their cascade
machinery, because a policy has nothing underneath it: no subcollections, no
third-party credential to revoke, and no PDF in Storage (brief Section 8's
no-persistence decision). One document goes, and that is the whole operation.

OWNERSHIP IS STRUCTURAL, NOT A COMPARISON
Identical in spirit to item_disconnect: the only lookup is
users/{uid}/insurancePolicies/{policy_id}, and the uid comes from the caller's
verified Firebase ID token and nothing else. Another user's policy_id is not
"rejected" — it simply does not exist at that path, so it raises LookupError
exactly as a typo would, and the endpoint answers 404. There is no `if
policy["uid"] == uid` check to get wrong, and no code path that takes a uid from
the request.

VERIFIED, NOT ASSUMED
A Firestore delete() of a document succeeds whether or not anything was
removed, so its return says nothing. The document is read back afterwards and a
survivor raises RuntimeError — the caller turns that into a 500 with the policy
still listed and the delete still retryable, the same "nothing happened"
contract disconnect and Delete Account use.

WHAT IS DELIBERATELY LEFT BEHIND
Any notificationEvents premium-reminder marker for this policy. Its document ID
is `premium_due_{policy_id}_{renewal_date}` (notifications/rules.py), and
policy_id is a Firestore auto-ID minted fresh by every /insurance/parse call,
never reused — so a later upload, even of the very same PDF, gets a new
policy_id and a new key, and the stale marker can never suppress it. The marker
is only ever read by service.deliver()'s create() collision check; nothing lists
or displays these documents. It is inert audit history, and it is still removed
by DELETE /account's cascade like every other notificationEvents document.
"""
from __future__ import annotations

POLICIES_COLLECTION = "insurancePolicies"


def delete_policy(db, uid: str, policy_id: str) -> dict:
    """Delete one of this user's policies.

    Raises LookupError if the policy is not this user's (see the module
    docstring — "does not exist" and "belongs to someone else" are
    indistinguishable by design). Raises RuntimeError if the document is still
    there after the delete, in which case it can be retried.
    """
    # Names Firestore refuses as document IDs would otherwise surface as a 500
    # from the client library. No policy can live at one — every real
    # policy_id is an auto-ID — so they are simply "not found".
    if (not policy_id or policy_id in (".", "..")
            or (policy_id.startswith("__") and policy_id.endswith("__"))):
        raise LookupError(f"No insurance policy {policy_id!r} for this user")

    policy_ref = (
        db.collection("users").document(uid)
        .collection(POLICIES_COLLECTION).document(policy_id)
    )
    snap = policy_ref.get()
    if not snap.exists:
        raise LookupError(f"No insurance policy {policy_id} for this user")
    policy = snap.to_dict() or {}

    policy_ref.delete()

    if policy_ref.get().exists:
        raise RuntimeError(
            f"Policy delete incomplete — users/{uid}/{POLICIES_COLLECTION}/"
            f"{policy_id} still exists"
        )

    return {
        "policy_id": policy_id,
        "policy_name": policy.get("policy_name"),
        "firestore_verified_deleted": True,
    }
