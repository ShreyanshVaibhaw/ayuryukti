"""Shared Pydantic models used across AyurYukti modules."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PatientRecord(BaseModel):
    """Core patient profile used across capture and recommendation workflows."""

    patient_id: str
    abha_id: Optional[str] = None
    name: Optional[str] = None
    age: int
    sex: Literal["Male", "Female", "Other"]
    prakriti_type: Optional[str] = None
    prakriti_score: Optional[Dict[str, float]] = None


class Prescription(BaseModel):
    """Normalized prescription entity for AYUSH formulations."""

    formulation_name: str
    formulation_id: Optional[str] = None
    dosage: str
    frequency: str
    duration: str
    route: str
    classical_reference: Optional[str] = None
    special_instructions: Optional[str] = None


class ClinicalEncounter(BaseModel):
    """Full encounter payload from voice capture through clinical confirmation."""

    encounter_id: str
    patient_id: str
    doctor_id: str
    centre_id: str
    timestamp: datetime
    raw_transcript: Optional[str] = None
    language: str
    chief_complaints: List[str] = Field(default_factory=list)
    duration_of_illness: Optional[str] = None
    prakriti_assessment: Optional[str] = None
    diagnosis_ayush: str
    diagnosis_ayush_code: str
    diagnosis_icd10: str
    diagnosis_icd10_code: str
    prescriptions: List[Prescription] = Field(default_factory=list)
    lifestyle_recommendations: List[str] = Field(default_factory=list)
    yoga_recommendations: List[str] = Field(default_factory=list)
    dietary_recommendations: List[str] = Field(default_factory=list)
    follow_up_date: Optional[str] = None
    outcome: Optional[Literal["Improved", "No Change", "Worsened"]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AyushFormulation(BaseModel):
    """Structured formulation definition derived from classical and official sources."""

    formulation_id: str
    name_sanskrit: str
    name_english: str
    formulation_type: str
    system: str
    ingredients: List[Dict[str, str]] = Field(default_factory=list)
    indicated_conditions: List[str] = Field(default_factory=list)
    contraindicated_prakriti: List[str] = Field(default_factory=list)
    indicated_prakriti: List[str] = Field(default_factory=list)
    dosha_action: Dict[str, str] = Field(default_factory=dict)
    dosage_range: str
    route: str
    classical_reference: str
    chapter_reference: str
    pharmacopoeia_reference: Optional[str] = None
    safety_notes: Optional[str] = None


class PrakritiAssessment(BaseModel):
    """Questionnaire and score output for Prakriti classification."""

    assessment_id: str
    patient_id: str
    responses: Dict[str, int] = Field(default_factory=dict)
    vata_score: float
    pitta_score: float
    kapha_score: float
    dominant_prakriti: str
    secondary_prakriti: Optional[str] = None
    prakriti_type: str
    confidence: float
    timestamp: datetime


class TreatmentRecommendation(BaseModel):
    """Recommendation package for clinicians with evidence and rationale."""

    recommendation_id: str
    encounter_id: str
    patient_prakriti: str
    condition: str
    recommended_formulations: List[Dict[str, Any]] = Field(default_factory=list)
    lifestyle_suggestions: List[str] = Field(default_factory=list)
    yoga_suggestions: List[str] = Field(default_factory=list)
    dietary_suggestions: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    confidence: float
    reasoning: str
    classical_references: List[str] = Field(default_factory=list)
    generated_at: datetime


class OutbreakAlert(BaseModel):
    """Public-health alert payload generated from RogaRadar anomalies."""

    alert_id: str
    alert_level: Literal["WATCH", "WARNING", "ALERT"]
    condition_ayush: str
    condition_icd10: str
    district: str
    state: str
    current_cases: int
    baseline_cases: float
    ratio: float
    trend: str
    affected_centres: List[str] = Field(default_factory=list)
    neighboring_districts_affected: List[str] = Field(default_factory=list)
    recommended_action: str
    generated_at: datetime


class AyushMorbidityCode(BaseModel):
    """Mapping model for AYUSH morbidity taxonomy and ICD-10 interoperability."""

    code_id: str
    ayush_name: str
    english_name: str
    system: str
    category: str
    icd10_codes: List[str] = Field(default_factory=list)
    common_symptoms: List[str] = Field(default_factory=list)
    dosha_involvement: List[str] = Field(default_factory=list)


class EHROutput(BaseModel):
    """Final structured EHR format emitted by VakSetu after clinician review."""

    patient_demographics: Dict[str, Any] = Field(default_factory=dict)
    prakriti_assessment: Optional[str] = None
    chief_complaints: List[Dict[str, str]] = Field(default_factory=list)
    examination_findings: Optional[str] = None
    ayush_diagnosis: Dict[str, str] = Field(default_factory=dict)
    icd10_diagnosis: Dict[str, str] = Field(default_factory=dict)
    prescriptions: List[Dict[str, str]] = Field(default_factory=list)
    lifestyle_advice: List[str] = Field(default_factory=list)
    dietary_advice: List[str] = Field(default_factory=list)
    yoga_advice: List[str] = Field(default_factory=list)
    follow_up: Optional[str] = None
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    encounter_metadata: Dict[str, Any] = Field(default_factory=dict)
