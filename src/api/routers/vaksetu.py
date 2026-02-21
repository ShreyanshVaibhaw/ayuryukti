"""VakSetu API router — Voice-to-EHR endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.common.models import EHROutput
from src.llm.ollama_client import LLMClient
from src.vaksetu.code_mapper import CodeMapper
from src.vaksetu.ehr_generator import EHRGenerator
from src.vaksetu.medical_ner import MedicalNEREngine
from src.vaksetu.speech_engine import SAMPLE_TRANSCRIPTS

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
KB = ROOT / "data" / "knowledge_base"
VOCAB = ROOT / "data" / "ayush_vocabulary.json"


class TranscriptRequest(BaseModel):
    transcript: str = Field(..., description="Doctor-patient consultation transcript")
    language: str = Field(default="hi", description="Language code (hi, en, ta, etc.)")
    centre_id: str = Field(default="C001", description="Centre identifier")
    doctor_id: str = Field(default="D001", description="Doctor identifier")


class EHRResponse(BaseModel):
    patient_demographics: Dict[str, Any] = {}
    prakriti_assessment: Optional[str] = None
    chief_complaints: List[Dict[str, str]] = []
    examination_findings: Optional[str] = None
    ayush_diagnosis: Dict[str, str] = {}
    icd10_diagnosis: Dict[str, str] = {}
    prescriptions: List[Dict[str, str]] = []
    lifestyle_advice: List[str] = []
    dietary_advice: List[str] = []
    yoga_advice: List[str] = []
    follow_up: Optional[str] = None
    confidence_scores: Dict[str, float] = {}
    encounter_metadata: Dict[str, Any] = {}


def _get_generator() -> EHRGenerator:
    llm = None
    try:
        client = LLMClient()
        health = client.health_check()
        if health.get("ok") and health.get("model_available"):
            llm = client
    except Exception:
        pass
    mapper = CodeMapper(
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
        icd10_mapping_path=str(KB / "icd10_mapping.json"),
    )
    ner = MedicalNEREngine(
        llm_client=llm,
        vocabulary_path=str(VOCAB),
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
    )
    return EHRGenerator(ner_engine=ner, code_mapper=mapper)


@router.post("/generate-ehr", response_model=EHRResponse)
async def generate_ehr(request: TranscriptRequest) -> EHRResponse:
    """Generate structured EHR from a consultation transcript."""
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")
    generator = _get_generator()
    ehr = generator.generate_from_transcript(
        transcript=request.transcript,
        language=request.language,
        centre_id=request.centre_id,
        doctor_id=request.doctor_id,
    )
    return EHRResponse(
        patient_demographics=ehr.patient_demographics,
        prakriti_assessment=ehr.prakriti_assessment,
        chief_complaints=ehr.chief_complaints,
        examination_findings=ehr.examination_findings,
        ayush_diagnosis=ehr.ayush_diagnosis,
        icd10_diagnosis=ehr.icd10_diagnosis,
        prescriptions=ehr.prescriptions,
        lifestyle_advice=ehr.lifestyle_advice,
        dietary_advice=ehr.dietary_advice,
        yoga_advice=ehr.yoga_advice,
        follow_up=ehr.follow_up,
        confidence_scores=ehr.encounter_metadata.get("confidence_scores", {}),
        encounter_metadata=ehr.encounter_metadata,
    )


@router.get("/samples")
async def list_samples() -> Dict[str, Any]:
    """Return curated demo transcripts."""
    return {
        "samples": [
            {"id": i + 1, "transcript": t}
            for i, t in enumerate(SAMPLE_TRANSCRIPTS)
        ]
    }
