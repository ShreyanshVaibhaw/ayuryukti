"""EHR generation pipeline for VakSetu."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from src.common.models import EHROutput
from src.vaksetu.code_mapper import CodeMapper
from src.vaksetu.medical_ner import MedicalNEREngine
from src.vaksetu.speech_engine import SpeechEngine, TranscriptionResult


class EHRGenerator:
    """Build structured AHMIS-compatible EHR from transcript/audio input."""

    def __init__(self, ner_engine: MedicalNEREngine, code_mapper: CodeMapper) -> None:
        self.ner_engine = ner_engine
        self.code_mapper = code_mapper

    def generate_from_transcript(
        self,
        transcript: str,
        language: str,
        centre_id: str,
        doctor_id: str,
    ) -> EHROutput:
        """Generate structured EHR from a transcript."""
        corrected = self.ner_engine.correct_transcript(raw_transcript=transcript, language=language)
        ehr = self.ner_engine.extract_ehr(transcript=corrected, language=language)

        ayush_name = ehr.ayush_diagnosis.get("name", "")
        mapped = self.code_mapper.map_condition(ayush_name)
        if mapped.get("code_id"):
            ehr.ayush_diagnosis["code"] = mapped["code_id"]
        if mapped.get("icd10_codes"):
            ehr.icd10_diagnosis["code"] = mapped["icd10_codes"][0]

        ehr.encounter_metadata.update(
            {
                "centre_id": centre_id,
                "doctor_id": doctor_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "transcript",
            }
        )
        return ehr

    def generate_from_audio(
        self,
        audio_bytes: bytes,
        language: str,
        centre_id: str,
        doctor_id: str,
        speech_engine: SpeechEngine,
    ) -> Tuple[EHROutput, TranscriptionResult]:
        """Transcribe audio and generate EHR."""
        transcription = speech_engine.transcribe(audio_bytes=audio_bytes, language=language)
        ehr = self.generate_from_transcript(
            transcript=transcription.text,
            language=language,
            centre_id=centre_id,
            doctor_id=doctor_id,
        )
        ehr.encounter_metadata["transcription_method"] = transcription.method
        ehr.encounter_metadata["transcription_confidence"] = transcription.confidence
        return ehr, transcription

    def to_ahmis_json(self, ehr: EHROutput) -> Dict[str, Any]:
        """Convert EHR model into AHMIS-aligned JSON payload."""
        return {
            "patientDemographics": ehr.patient_demographics,
            "prakritiAssessment": ehr.prakriti_assessment,
            "chiefComplaints": ehr.chief_complaints,
            "examinationFindings": ehr.examination_findings,
            "diagnosis": {
                "ayush": ehr.ayush_diagnosis,
                "icd10": ehr.icd10_diagnosis,
            },
            "prescriptions": ehr.prescriptions,
            "lifestyleAdvice": ehr.lifestyle_advice,
            "dietaryAdvice": ehr.dietary_advice,
            "yogaAdvice": ehr.yoga_advice,
            "followUp": ehr.follow_up,
            "metadata": ehr.encounter_metadata,
        }


def generate_ehr_payload(entities: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible helper used by legacy tests."""
    return {
        "patient_demographics": {},
        "chief_complaints": entities.get("symptoms", []),
        "ayush_diagnosis": {},
        "icd10_diagnosis": {},
        "prescriptions": [],
        "follow_up": None,
    }
