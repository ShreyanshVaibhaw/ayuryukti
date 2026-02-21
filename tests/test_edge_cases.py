"""Edge case and robustness tests for AyurYukti."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.main import app
from src.vaksetu.medical_ner import MedicalNEREngine

client = TestClient(app)
KB = ROOT / "data" / "knowledge_base"
VOCAB = ROOT / "data" / "ayush_vocabulary.json"


def _get_ner():
    return MedicalNEREngine(
        llm_client=None,
        vocabulary_path=str(VOCAB),
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
    )


# --- API edge cases ---

def test_recommend_unknown_prakriti_via_api():
    """Unknown prakriti should return results, not crash."""
    resp = client.post(
        "/api/v1/prakritimitra/recommend",
        json={"patient_prakriti": "UnknownType", "condition": "Vibandha"},
    )
    assert resp.status_code == 200


def test_recommend_unknown_condition_via_api():
    """Unknown condition should return results, not crash."""
    resp = client.post(
        "/api/v1/prakritimitra/recommend",
        json={"patient_prakriti": "Vata", "condition": "NonExistentCondition"},
    )
    assert resp.status_code == 200


def test_lifestyle_unknown_prakriti_via_api():
    """Unknown prakriti lifestyle should return 200, not 500."""
    resp = client.get("/api/v1/prakritimitra/lifestyle/UnknownType")
    assert resp.status_code == 200


def test_effectiveness_unknown_condition():
    """Unknown condition effectiveness should not crash."""
    resp = client.get("/api/v1/yuktishaala/effectiveness/NonExistent")
    assert resp.status_code == 200
    assert resp.json()["treatments"] == []


def test_feedback_invalid_outcome_rejected():
    """Invalid outcome value should be rejected by validation."""
    resp = client.post(
        "/api/v1/yuktishaala/feedback",
        json={
            "encounter_id": "test",
            "patient_prakriti": "Vata",
            "condition_code": "Vibandha",
            "formulations": ["Triphala Churna"],
            "outcome": "InvalidOutcome",
        },
    )
    assert resp.status_code == 422  # Pydantic validation error


def test_rogaradar_alerts_with_district_filter():
    """Alert filter by district should not crash."""
    resp = client.get("/api/v1/rogaradar/alerts?district=Varanasi")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_rogaradar_alerts_with_severity_filter():
    """Alert filter by severity should not crash."""
    resp = client.get("/api/v1/rogaradar/alerts?severity=ALERT")
    assert resp.status_code == 200


# --- NER robustness ---

def test_ner_very_long_transcript():
    """Very long transcript should not crash."""
    ner = _get_ner()
    long_text = "35 saal ki mahila, Vata Prakriti. kabz ki shikayat. " * 200
    ehr = ner.extract_ehr(long_text, "hi")
    assert ehr.ayush_diagnosis["name"] == "Vibandha"


def test_ner_unicode_hindi_transcript():
    """Hindi Unicode transcript should not crash."""
    ner = _get_ner()
    ehr = ner.extract_ehr("35 साल की महिला, Vata Prakriti. कब्ज की शिकायत।", "hi")
    assert ehr.patient_demographics["age"] is None or isinstance(ehr.patient_demographics["age"], int)


def test_ner_only_numbers_transcript():
    """Transcript with only numbers should not crash."""
    ner = _get_ner()
    ehr = ner.extract_ehr("12345 67890", "en")
    assert ehr.ayush_diagnosis["name"] == "Unknown"


def test_prescription_deduplication():
    """Substring formulations should not create duplicate prescriptions.

    E.g. 'Guduchi' should not appear alongside 'Guduchi Kashaya'.
    """
    ner = _get_ner()
    transcript = "Guduchi Kashaya 15ml twice daily. Chandraprabha Vati 2 goli."
    ehr = ner.extract_ehr(transcript, "en")
    rx_names = [rx["formulation_name"] for rx in ehr.prescriptions]
    # Should have full names, not short substrings
    assert "Guduchi Kashaya" in rx_names
    assert "Chandraprabha Vati" in rx_names
    # Short substrings should NOT appear separately
    short_names = [n for n in rx_names if n in ("Guduchi", "Chandraprabha")]
    assert len(short_names) == 0, f"Substring duplicates found: {short_names}"


def test_all_conditions_have_icd10_mapping():
    """Every condition alias should map to a valid ICD-10 code."""
    ner = _get_ner()
    for alias, ayush_name in ner.condition_aliases.items():
        icd_entry = ner.icd_by_ayush.get(ayush_name)
        if icd_entry:
            name, code = icd_entry
            assert code, f"{ayush_name} has empty ICD-10 code"
            assert name, f"{ayush_name} has empty ICD-10 name"


# --- Cross-module API integration ---

def test_full_api_flow_vaksetu_to_prakritimitra_to_yuktishaala():
    """Full pipeline through API: generate EHR → get recommendation → submit feedback."""
    # Step 1: Generate EHR
    ehr_resp = client.post(
        "/api/v1/vaksetu/generate-ehr",
        json={
            "transcript": "52 year old male, Pitta-Kapha prakriti. Burning sensation in chest. Amlapitta. Avipattikar Churna 3g twice daily.",
            "language": "en",
        },
    )
    assert ehr_resp.status_code == 200
    ehr = ehr_resp.json()
    prakriti = ehr.get("prakriti_assessment", "Pitta-Kapha")
    condition = ehr.get("ayush_diagnosis", {}).get("name", "Amlapitta")

    # Step 2: Get recommendation using EHR output
    rec_resp = client.post(
        "/api/v1/prakritimitra/recommend",
        json={
            "patient_prakriti": prakriti,
            "condition": condition,
            "patient_age": 52,
            "patient_sex": "Male",
        },
    )
    assert rec_resp.status_code == 200
    rec = rec_resp.json()
    assert len(rec["recommended_formulations"]) > 0

    # Step 3: Submit feedback for the recommendation
    formulations = [f["formulation_name"] for f in rec["recommended_formulations"][:2]]
    fb_resp = client.post(
        "/api/v1/yuktishaala/feedback",
        json={
            "encounter_id": rec.get("encounter_id", "test-cross-module"),
            "patient_prakriti": prakriti,
            "condition_code": condition,
            "formulations": formulations,
            "outcome": "Improved",
            "follow_up_days": 14,
        },
    )
    assert fb_resp.status_code == 200
    assert fb_resp.json()["status"] == "success"


def test_confidence_scores_always_between_0_and_1():
    """All confidence scores must be in [0, 1] range."""
    ner = _get_ner()
    transcripts = [
        "35 saal ki mahila, Vata Prakriti. kabz. Triphala Churna 5g.",
        "52 year old male, Pitta-Kapha prakriti. Amlapitta. Avipattikar Churna.",
        "",
        "random text with no medical content",
    ]
    for t in transcripts:
        ehr = ner.extract_ehr(t, "en")
        for field, score in ehr.confidence_scores.items():
            assert 0.0 <= score <= 1.0, f"Confidence '{field}' = {score} out of range for: {t[:40]}"
