"""JWT authentication and role-based access control tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.auth import DEMO_USERS, _create_token, _decode_token, authenticate_user
from src.api.main import app

client = TestClient(app)


# --- Token creation and validation ---

def test_create_token_returns_string():
    token = _create_token("doctor", "doctor", "Dr. Demo")
    assert isinstance(token, str)
    assert len(token) > 10


def test_decode_valid_token():
    token = _create_token("doctor", "doctor", "Dr. Demo")
    payload = _decode_token(token)
    assert payload.sub == "doctor"
    assert payload.role == "doctor"
    assert payload.name == "Dr. Demo"


def test_decode_invalid_token_raises():
    with pytest.raises(Exception):
        _decode_token("completely-invalid-token")


def test_authenticate_valid_user():
    result = authenticate_user("doctor", "doctor123")
    assert result is not None
    assert result.role == "doctor"
    assert result.access_token


def test_authenticate_invalid_password():
    result = authenticate_user("doctor", "wrong")
    assert result is None


def test_authenticate_unknown_user():
    result = authenticate_user("hacker", "password")
    assert result is None


# --- All demo roles can log in ---

def test_all_demo_roles_can_login():
    for username, info in DEMO_USERS.items():
        resp = client.post("/auth/login", json={"username": username, "password": info["password"]})
        assert resp.status_code == 200, f"Login failed for {username}"
        data = resp.json()
        assert data["role"] == info["role"]
        assert data["name"] == info["name"]
        assert "access_token" in data


# --- Protected endpoint access with token ---

def test_api_with_bearer_token():
    """Endpoints should accept requests with a valid Bearer token."""
    login = client.post("/auth/login", json={"username": "doctor", "password": "doctor123"})
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        "/api/v1/vaksetu/generate-ehr",
        json={"transcript": "35 saal ki mahila, Vata Prakriti. kabz ki shikayat. Triphala Churna 5g.", "language": "hi"},
        headers=headers,
    )
    assert resp.status_code == 200


def test_api_works_without_token():
    """API should also work without token (open access mode for demo)."""
    resp = client.post(
        "/api/v1/vaksetu/generate-ehr",
        json={"transcript": "35 saal ki mahila, Vata Prakriti. kabz. Triphala Churna 5g.", "language": "hi"},
    )
    assert resp.status_code == 200
