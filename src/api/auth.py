"""JWT authentication and role-based access for AyurYukti API."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "ayuryukti-dev-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))

security = HTTPBearer(auto_error=False)

# Demo users for development/judging
DEMO_USERS = {
    "doctor": {"password": "doctor123", "role": "doctor", "name": "Dr. Demo Physician"},
    "official": {"password": "official123", "role": "official", "name": "Public Health Official"},
    "admin": {"password": "admin123", "role": "admin", "name": "System Administrator"},
}


class TokenPayload(BaseModel):
    sub: str
    role: str
    name: str
    exp: float


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str
    name: str


def _create_token(username: str, role: str, name: str) -> str:
    """Create a JWT token."""
    try:
        import jwt
    except ImportError:
        # Fallback: simple base64 token if PyJWT not installed
        import base64
        import json

        payload = {"sub": username, "role": role, "name": name, "exp": (datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES)).timestamp()}
        return base64.b64encode(json.dumps(payload).encode()).decode()

    payload = {
        "sub": username,
        "role": role,
        "name": name,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token."""
    try:
        import jwt

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except ImportError:
        import base64
        import json

        try:
            payload = json.loads(base64.b64decode(token).decode())
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if payload.get("exp", 0) < datetime.now(timezone.utc).timestamp():
        raise HTTPException(status_code=401, detail="Token has expired")

    return TokenPayload(**payload)


def authenticate_user(username: str, password: str) -> Optional[TokenResponse]:
    """Validate credentials and return token."""
    user = DEMO_USERS.get(username)
    if not user or user["password"] != password:
        return None
    token = _create_token(username, user["role"], user["name"])
    return TokenResponse(
        access_token=token,
        expires_in=JWT_EXPIRY_MINUTES * 60,
        role=user["role"],
        name=user["name"],
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[TokenPayload]:
    """Extract current user from JWT token. Returns None if no token (open access mode)."""
    if credentials is None:
        return None
    return _decode_token(credentials.credentials)


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> TokenPayload:
    """Require authentication — raises 401 if no valid token."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return _decode_token(credentials.credentials)


def require_role(*roles: str):
    """Dependency that requires specific role(s)."""

    async def _check(user: TokenPayload = Depends(require_auth)) -> TokenPayload:
        if user.role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user.role}' not authorized. Required: {', '.join(roles)}",
            )
        return user

    return _check
