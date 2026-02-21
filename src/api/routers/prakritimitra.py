"""PrakritiMitra API router — Prakriti classification and recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from src.prakritimitra.knowledge_graph import AyushKnowledgeGraph
from src.prakritimitra.lifestyle_advisor import LifestyleAdvisor
from src.prakritimitra.prakriti_classifier import PrakritiClassifier
from src.prakritimitra.recommendation_engine import RecommendationEngine

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
KB = ROOT / "data" / "knowledge_base"
Q_PATH = ROOT / "data" / "prakriti" / "questionnaire.json"
TRAIN_PATH = ROOT / "data" / "prakriti" / "training_data.csv"


class RecommendRequest(BaseModel):
    patient_prakriti: str = Field(..., description="Patient prakriti type (Vata, Pitta, Kapha, etc.)")
    condition: str = Field(..., description="AYUSH diagnosis name (e.g. Vibandha)")
    patient_age: int = Field(default=35, ge=1, le=120)
    patient_sex: str = Field(default="Male")
    existing_prescriptions: List[str] = Field(default_factory=list)


class ClassifyRequest(BaseModel):
    responses: Dict[str, int] = Field(..., description="Questionnaire responses (q1-q30, values 1-5)")


class RecommendResponse(BaseModel):
    patient_prakriti: str
    condition: str
    recommended_formulations: List[Dict[str, Any]]
    lifestyle_suggestions: List[str]
    yoga_suggestions: List[str]
    dietary_suggestions: List[str]
    contraindications: List[str]
    confidence: float
    reasoning: str
    classical_references: List[str]


class ClassifyResponse(BaseModel):
    prakriti_type: str
    vata_score: float
    pitta_score: float
    kapha_score: float
    confidence: float
    description: str


def _get_kg():
    kg = AyushKnowledgeGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    try:
        kg.setup_schema()
        kg.seed_from_json(
            formulations_path=str(KB / "formulations.json"),
            morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
            prakriti_rules_path=str(KB / "prakriti_rules.json"),
        )
    except Exception:
        pass
    return kg


PRAKRITI_DESCRIPTIONS = {
    "Vata": "Light, quick, and creative. Benefits from grounding routines and warm nourishment.",
    "Pitta": "Sharp, focused, and driven. Benefits from cooling balance and moderate intensity.",
    "Kapha": "Steady, strong, and nurturing. Benefits from stimulation and active movement.",
    "Vata-Pitta": "Combines mobility with intensity. Needs regular meals and calm environment.",
    "Pitta-Kapha": "Strong metabolism with stable endurance. Needs moderation in heat and heaviness.",
    "Vata-Kapha": "Alternates between lightness and heaviness. Needs warmth and consistent routine.",
    "Sama": "Balanced tri-dosha profile. Maintain seasonal and daily rhythm.",
}


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """Generate Prakriti-aware treatment recommendations."""
    kg = _get_kg()
    engine = RecommendationEngine(knowledge_graph=kg, llm_client=None)
    rec = engine.recommend(
        patient_prakriti=request.patient_prakriti,
        condition_ayush_code=request.condition,
        patient_age=request.patient_age,
        patient_sex=request.patient_sex,
        existing_prescriptions=request.existing_prescriptions,
    )
    return RecommendResponse(
        patient_prakriti=rec.patient_prakriti,
        condition=rec.condition,
        recommended_formulations=rec.recommended_formulations,
        lifestyle_suggestions=rec.lifestyle_suggestions,
        yoga_suggestions=rec.yoga_suggestions,
        dietary_suggestions=rec.dietary_suggestions,
        contraindications=rec.contraindications,
        confidence=rec.confidence,
        reasoning=rec.reasoning,
        classical_references=rec.classical_references,
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify_prakriti(request: ClassifyRequest) -> ClassifyResponse:
    """Classify patient prakriti from questionnaire responses."""
    if len(request.responses) < 10:
        raise HTTPException(status_code=400, detail="At least 10 questionnaire responses required.")
    classifier = PrakritiClassifier(
        questionnaire_path=str(Q_PATH),
        training_data_path=str(TRAIN_PATH),
    )
    result = classifier.classify(request.responses)
    return ClassifyResponse(
        prakriti_type=result.prakriti_type,
        vata_score=result.vata_score,
        pitta_score=result.pitta_score,
        kapha_score=result.kapha_score,
        confidence=result.confidence,
        description=PRAKRITI_DESCRIPTIONS.get(result.prakriti_type, "Prakriti profile available."),
    )


@router.get("/lifestyle/{prakriti_type}")
async def get_lifestyle(prakriti_type: str, condition: str = "general") -> Dict[str, Any]:
    """Get lifestyle, dietary, and yoga advice for a prakriti type."""
    advisor = LifestyleAdvisor(prakriti_rules_path=KB / "prakriti_rules.json")
    return {
        "prakriti_type": prakriti_type,
        "condition": condition,
        "dietary": advisor.get_dietary_advice(prakriti_type, condition),
        "lifestyle": advisor.get_lifestyle_advice(prakriti_type, condition),
        "yoga": advisor.get_yoga_advice(prakriti_type, condition),
    }
