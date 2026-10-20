"""scoreHistory security-rule verification, against a LOCAL Firestore emulator.

    # terminal 1
    npx firebase-tools emulators:start --only firestore --project YOUR_FIREBASE_PROJECT_ID
    # terminal 2
    cd backend && venv\\Scripts\\activate && python scripts/verify_score_history_rules.py

WHY THIS EXISTS SEPARATELY FROM verify_score_history.py
That script's section B6 is the real enforcement test: it attacks the live
project with a real ID token, and so it can only pass once the ruleset has
actually been deployed. This script answers the narrower question that comes
first — "is the rule TEXT correct?" — by loading this repo's firestore.rules
into a local emulator and attacking that. It needs no credentials, touches no
production data, and can be run before deciding to deploy anything.

Neither script replaces the other. This one proves the rule says what we meant;
B6 proves the project is actually running it.

WHAT IS BEING PROVED
The read-only posture, in both directions:
  - the owner CAN read their own snapshots     (the Home chart depends on it)
  - the owner CANNOT create, update or delete  (only the pipeline writes here)
  - another signed-in user CANNOT read them    (ordinary per-user isolation)
  - an anonymous caller CANNOT read them
Plus the regression that motivated the rule's shape: the catch-all at the
bottom of firestore.rules grants read+write to any collection NOT named in its
exclusion list, so a match block alone would not have been enough.
"""
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

PROJECT = "YOUR_FIREBASE_PROJECT_ID"
HOST = "127.0.0.1:8085"
BASE = f"http://{HOST}/v1/projects/{PROJECT}/databases/(default)/documents"

OWNER = "owner-uid-1"
OTHER = "other-uid-2"
TODAY = "2026-09-09"

passed, failed = [], []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def fake_id_token(uid: str) -> str:
    """An unsigned JWT. The emulator accepts alg=none deliberately, so a rules
    test needs no Auth emulator and no real credentials."""
    def part(obj):
        return base64.urlsafe_b64encode(
            json.dumps(obj, separators=(",", ":")).encode()).rstrip(b"=").decode()
    now = int(time.time())
    return ".".join([
        part({"alg": "none", "typ": "JWT"}),
        part({"iss": f"https://securetoken.google.com/{PROJECT}",
              "aud": PROJECT, "sub": uid, "user_id": uid,
              "auth_time": now, "iat": now, "exp": now + 3600,
              "email": f"{uid}@example.com", "email_verified": True,
              "firebase": {"identities": {}, "sign_in_provider": "password"}}),
        "",
    ])


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
    except urllib.error.URLError as e:
        sys.exit(f"\nCannot reach the Firestore emulator at {HOST}: {e}\n"
                 "Start it first:\n  npx firebase-tools emulators:start "
                 f"--only firestore --project {PROJECT}")


SNAPSHOT = {"fields": {
    "date": {"stringValue": TODAY},
    "overall_score": {"doubleValue": 69.0},
    "schema_version": {"stringValue": "1.1.0"},
    "age_band": {"integerValue": "2"},
}}
FORGED = {"fields": {"date": {"stringValue": TODAY},
                     "overall_score": {"doubleValue": 100.0}}}

owner_token = fake_id_token(OWNER)
other_token = fake_id_token(OTHER)

print("=" * 78)
print("Seeding as the backend (Admin bypasses rules, exactly as in production)")
print("=" * 78)

# "owner" is the emulator's admin bearer token — the Admin SDK equivalent.
status, _ = http("PATCH", f"{BASE}/users/{OWNER}/scoreHistory/{TODAY}",
                 SNAPSHOT, "owner")
check("the backend can write a snapshot", status == 200, f"HTTP {status}")
status, _ = http("PATCH", f"{BASE}/users/{OWNER}/scoreHistory/2026-09-08",
                 SNAPSHOT, "owner")
check("the backend can write a second day", status == 200, f"HTTP {status}")

print()
print("=" * 78)
print("The owner's own client — may read, may not write")
print("=" * 78)

status, body = http("GET", f"{BASE}/users/{OWNER}/scoreHistory/{TODAY}",
                    token=owner_token)
check("owner CAN read a snapshot (the Home chart depends on this)",
      status == 200, f"HTTP {status}")
check("  and gets the real stored value",
      body.get("fields", {}).get("overall_score", {}).get("doubleValue") == 69.0,
      str(body.get("fields", {}).get("overall_score")))

status, _ = http("GET", f"{BASE}/users/{OWNER}/scoreHistory", token=owner_token)
check("owner CAN list the collection (a chart reads a range, not one doc)",
      status == 200, f"HTTP {status}")

status, _ = http("PATCH", f"{BASE}/users/{OWNER}/scoreHistory/{TODAY}",
                 FORGED, owner_token)
check("owner CANNOT overwrite an existing snapshot", status == 403,
      f"HTTP {status}")

status, _ = http("PATCH", f"{BASE}/users/{OWNER}/scoreHistory/2020-01-01",
                 FORGED, owner_token)
check("owner CANNOT invent a day that never happened", status == 403,
      f"HTTP {status}")

status, _ = http("DELETE", f"{BASE}/users/{OWNER}/scoreHistory/{TODAY}",
                 token=owner_token)
check("owner CANNOT delete a snapshot", status == 403, f"HTTP {status}")

# Outcome, not status code: a 403 that still let the value through would be
# worse than an honest 200.
status, body = http("GET", f"{BASE}/users/{OWNER}/scoreHistory/{TODAY}",
                    token="owner")
check("the row still holds the pipeline's value after all three attacks",
      status == 200
      and body["fields"]["overall_score"]["doubleValue"] == 69.0,
      f"HTTP {status}, "
      f"{body.get('fields', {}).get('overall_score')}")

print()
print("=" * 78)
print("Other callers — no read at all")
print("=" * 78)

status, _ = http("GET", f"{BASE}/users/{OWNER}/scoreHistory/{TODAY}",
                 token=other_token)
check("another signed-in user CANNOT read someone else's history",
      status == 403, f"HTTP {status}")
status, _ = http("PATCH", f"{BASE}/users/{OWNER}/scoreHistory/{TODAY}",
                 FORGED, other_token)
check("another signed-in user CANNOT write it either", status == 403,
      f"HTTP {status}")
status, _ = http("GET", f"{BASE}/users/{OWNER}/scoreHistory/{TODAY}")
check("an anonymous caller CANNOT read it", status == 403, f"HTTP {status}")

print()
print("=" * 78)
print("The catch-all regression — why the exclusion list is load-bearing")
print("=" * 78)
print("  firestore.rules' bottom match grants read+write to any collection NOT")
print("  named in its exclusion list. A collection outside that list is")
print("  client-writable, which is exactly what scoreHistory must not be.")

status, _ = http("PATCH", f"{BASE}/users/{OWNER}/someUnlistedCollection/doc-1",
                 FORGED, owner_token)
check("control: an UNLISTED collection IS client-writable via the catch-all",
      status == 200,
      f"HTTP {status} — if this fails the control itself is broken, and the "
      "scoreHistory denials above prove less than they appear to")

print()
print("=" * 78)
print(f"{len(passed)} passed, {len(failed)} failed")
if failed:
    for name in failed:
        print(f"  FAILED: {name}")
print("=" * 78)
sys.exit(1 if failed else 0)
