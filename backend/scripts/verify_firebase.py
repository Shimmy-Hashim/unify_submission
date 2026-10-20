"""End-to-end Firebase verification. Run after enabling the Email/Password
provider in the Firebase console (Authentication -> Get started -> enable
Email/Password):

    cd backend && venv\\Scripts\\activate && python scripts/verify_firebase.py

Exercises the real production chain with two throwaway users:
  1. sign up user A and user B (Firebase Auth REST, as the Flutter app would)
  2. as A, create users/{uidA} through the client API   -> rules must allow
  3. as B, read users/{uidA} through the client API     -> rules must deny
  4. call GET /me in-process with A's real ID token     -> backend verifies
     the token with firebase-admin and reads Firestore
  5. clean up both users and the test document via the Admin SDK
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from firebase import get_db, init_firebase  # noqa: E402
from firebase_admin import auth as admin_auth  # noqa: E402

PROJECT = "YOUR_FIREBASE_PROJECT_ID"
GSERVICES = BACKEND_DIR.parent / "android" / "app" / "google-services.json"
API_KEY = json.loads(GSERVICES.read_text())["client"][0]["api_key"][0]["current_key"]
FIRESTORE = f"https://firestore.googleapis.com/v1/projects/{PROJECT}/databases/(default)/documents"

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
        msg = body.get("error", {}).get("message", "")
        if msg.startswith("CONFIGURATION_NOT_FOUND"):
            sys.exit(
                "\nFirebase Auth is not provisioned yet.\n"
                "Open https://console.firebase.google.com/project/"
                f"{PROJECT}/authentication, click 'Get started', enable the\n"
                "Email/Password provider (and Google), then rerun this script."
            )
        sys.exit(f"\nSign-up failed unexpectedly ({status}): {msg}")
    return body["localId"], body["idToken"]


def main():
    init_firebase()
    stamp = int(time.time())
    uid_a, token_a = sign_up(f"e2e.a.{stamp}@unify-test.example", "E2eTestA-12345")
    uid_b, token_b = sign_up(f"e2e.b.{stamp}@unify-test.example", "E2eTestB-12345")
    check("sign up user A and user B via Auth REST", True, f"uidA={uid_a}")
    try:
        doc = {"fields": {
            "name": {"stringValue": "E2E User A"},
            "email": {"stringValue": f"e2e.a.{stamp}@unify-test.example"},
            "date_of_birth": {"timestampValue": "2000-01-15T00:00:00Z"},
            "dependants": {"integerValue": "0"},
            "created_at": {"timestampValue": "2026-07-04T00:00:00Z"},
            "last_active": {"timestampValue": "2026-07-04T00:00:00Z"},
        }}
        status, _ = http("PATCH", f"{FIRESTORE}/users/{uid_a}", doc, token_a)
        check("user A creates own users/{uid} (rules allow)", status == 200, f"HTTP {status}")

        status, _ = http("GET", f"{FIRESTORE}/users/{uid_a}", token=token_b)
        check("user B blocked from reading A's doc (rules deny)", status == 403, f"HTTP {status}")

        status, _ = http("GET", f"{FIRESTORE}/users/{uid_a}")
        check("anonymous blocked from reading A's doc", status == 403, f"HTTP {status}")

        from fastapi.testclient import TestClient
        import main as backend_main
        client = TestClient(backend_main.app)
        r = client.get("/me", headers={"Authorization": f"Bearer {token_a}"})
        ok = r.status_code == 200 and r.json().get("uid") == uid_a \
            and r.json().get("name") == "E2E User A"
        check("GET /me verifies real ID token and returns A's doc", ok,
              f"HTTP {r.status_code}")

        r = client.get("/me", headers={"Authorization": f"Bearer {token_b}"})
        check("GET /me for B (no profile yet) returns 404", r.status_code == 404,
              f"HTTP {r.status_code}")
    finally:
        get_db().collection("users").document(uid_a).delete()
        admin_auth.delete_user(uid_a)
        admin_auth.delete_user(uid_b)
        print("cleanup: removed both test users and the test document")

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
