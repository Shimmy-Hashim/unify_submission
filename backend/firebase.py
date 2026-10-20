"""Firebase Admin SDK initialization and the auth dependency shared by every
authenticated endpoint in main.py.

The backend authenticates to Firebase with a service account key file
(gitignored). GOOGLE_APPLICATION_CREDENTIALS in backend/.env points to it;
relative paths are resolved against this directory so it works regardless of
the uvicorn working directory.
"""
import os
from pathlib import Path

import firebase_admin
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth, credentials, firestore

_BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(_BACKEND_DIR / ".env")


def init_firebase() -> None:
    """Idempotent Firebase Admin initialization."""
    if firebase_admin._apps:
        return
    key_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    if not key_path.is_absolute():
        key_path = _BACKEND_DIR / key_path
    firebase_admin.initialize_app(credentials.Certificate(str(key_path)))


def get_db() -> firestore.firestore.Client:
    init_firebase()
    return firestore.client()


_bearer = HTTPBearer(auto_error=False)


def get_current_uid(
    token: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Verifies the Firebase ID token from `Authorization: Bearer <token>`
    and returns the caller's uid. Use as a dependency on any endpoint that
    acts on behalf of a signed-in user.
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization: Bearer <Firebase ID token>",
        )
    init_firebase()
    try:
        decoded = auth.verify_id_token(token.credentials)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Firebase ID token",
        ) from exc
    return decoded["uid"]
