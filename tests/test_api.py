"""FastAPI endpoint tests for AyurYukti REST API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AyurYukti API"
    assert "version" in data


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_demo_credentials_endpoint() -> None:
    response = client.get("/auth/demo-credentials")
    assert response.status_code == 200
    data = response.json()
    assert "doctor" in data
    assert "official" in data


def test_login_valid_credentials() -> None:
    response = client.post("/auth/login", json={"username": "doctor", "password": "doctor123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "doctor"


def test_login_invalid_credentials() -> None:
    response = client.post("/auth/login", json={"username": "doctor", "password": "wrong"})
    assert response.status_code == 401


def test_vaksetu_generate_ehr() -> None:
    response = client.post(
        "/api/v1/vaksetu/generate-ehr",
        json={
            "transcript": "35 saal ki mahila hai, Vata Prakriti. 2 hafte se kabz ki shikayat hai. Triphala Churna 5 gram subah shaam garam paani ke saath. Abhayarishta 15ml khana ke baad.",
            "language": "hi",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ayush_diagnosis"]["name"] == "Vibandha"
    assert len(data["prescriptions"]) >= 2


def test_vaksetu_empty_transcript_rejected() -> None:
    response = client.post("/api/v1/vaksetu/generate-ehr", json={"transcript": "", "language": "hi"})
    assert response.status_code == 400


def test_vaksetu_samples() -> None:
    response = client.get("/api/v1/vaksetu/samples")
    assert response.status_code == 200
    assert len(response.json()["samples"]) == 3


def test_prakritimitra_recommend() -> None:
    response = client.post(
        "/api/v1/prakritimitra/recommend",
        json={
            "patient_prakriti": "Vata",
            "condition": "Vibandha",
            "patient_age": 35,
            "patient_sex": "Female",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["patient_prakriti"] == "Vata"
    assert data["condition"] == "Vibandha"
    assert len(data["recommended_formulations"]) > 0


def test_prakritimitra_lifestyle() -> None:
    response = client.get("/api/v1/prakritimitra/lifestyle/Vata?condition=Vibandha")
    assert response.status_code == 200
    data = response.json()
    assert data["prakriti_type"] == "Vata"
    assert len(data["dietary"]) > 0
    assert len(data["yoga"]) > 0


def test_rogaradar_alerts() -> None:
    response = client.get("/api/v1/rogaradar/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_rogaradar_dashboard() -> None:
    response = client.get("/api/v1/rogaradar/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "total_districts" in data
    assert "active_alerts" in data


def test_rogaradar_districts() -> None:
    response = client.get("/api/v1/rogaradar/districts")
    assert response.status_code == 200
    assert response.json()["total"] > 0


def test_yuktishaala_feedback() -> None:
    response = client.post(
        "/api/v1/yuktishaala/feedback",
        json={
            "encounter_id": "test-encounter-001",
            "patient_prakriti": "Vata",
            "condition_code": "Vibandha",
            "formulations": ["Triphala Churna"],
            "outcome": "Improved",
            "follow_up_days": 14,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_yuktishaala_effectiveness() -> None:
    response = client.get("/api/v1/yuktishaala/effectiveness/Vibandha")
    assert response.status_code == 200
    data = response.json()
    assert data["condition"] == "Vibandha"


def test_yuktishaala_summary() -> None:
    response = client.get("/api/v1/yuktishaala/outcomes/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_outcomes" in data
    assert "improvement_rate" in data
