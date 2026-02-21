# AyurYukti REST API Reference

**Base URL:** `http://localhost:8000`
**API Version:** v1
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Health & Auth

### GET /health
Health check endpoint.
```json
{"status": "healthy"}
```

### POST /auth/login
Authenticate and receive JWT token.
```json
// Request
{"username": "doctor", "password": "doctor123"}

// Response
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600,
  "role": "doctor",
  "name": "Dr. Demo Physician"
}
```

### GET /auth/demo-credentials
Returns demo credentials for testing.

---

## VakSetu (Voice-to-EHR)

### POST /api/v1/vaksetu/generate-ehr
Generate structured EHR from a consultation transcript.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| transcript | string | Yes | Doctor-patient consultation text |
| language | string | No | Language code (default: "hi") |
| centre_id | string | No | Centre identifier |
| doctor_id | string | No | Doctor identifier |

**Example:**
```json
{
  "transcript": "35 saal ki mahila hai, Vata Prakriti. kabz ki shikayat...",
  "language": "hi",
  "centre_id": "C001",
  "doctor_id": "D001"
}
```

**Response:** Structured EHR with patient demographics, diagnosis, prescriptions, lifestyle advice, and confidence scores.

### GET /api/v1/vaksetu/samples
Returns 3 curated demo transcripts.

---

## PrakritiMitra (Recommendations)

### POST /api/v1/prakritimitra/recommend
Generate Prakriti-aware treatment recommendations.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| patient_prakriti | string | Yes | Prakriti type (Vata, Pitta, etc.) |
| condition | string | Yes | AYUSH diagnosis name |
| patient_age | int | No | Patient age (default: 35) |
| patient_sex | string | No | Patient sex |
| existing_prescriptions | list | No | Already prescribed formulations |

### POST /api/v1/prakritimitra/classify
Classify patient prakriti from questionnaire responses.

**Request Body:**
```json
{
  "responses": {"q1": 4, "q2": 2, "q3": 5, ...}
}
```

### GET /api/v1/prakritimitra/lifestyle/{prakriti_type}
Get lifestyle, dietary, and yoga advice for a prakriti type.

---

## RogaRadar (Surveillance)

### GET /api/v1/rogaradar/alerts
Get active surveillance alerts. Optional query parameters: `district`, `severity`.

### GET /api/v1/rogaradar/dashboard
Full dashboard summary with metrics and all alerts.

### GET /api/v1/rogaradar/districts
List monitored districts with coordinates.

---

## YuktiShaala (Learning)

### POST /api/v1/yuktishaala/feedback
Record treatment outcome for continuous learning.

**Request Body:**
```json
{
  "encounter_id": "enc-001",
  "patient_prakriti": "Vata",
  "condition_code": "Vibandha",
  "formulations": ["Triphala Churna"],
  "outcome": "Improved",
  "follow_up_days": 14
}
```

### GET /api/v1/yuktishaala/effectiveness/{condition}
Treatment effectiveness analysis for a condition.

### GET /api/v1/yuktishaala/outcomes/summary
Overall outcome tracking summary statistics.

---

## Authentication

Protected endpoints require a Bearer token in the Authorization header:
```
Authorization: Bearer <token>
```

**Roles:**
- `doctor` — Access to VakSetu, PrakritiMitra, YuktiShaala
- `official` — Access to RogaRadar, analytics
- `admin` — Full access

**Demo credentials:**
- doctor / doctor123
- official / official123
- admin / admin123
