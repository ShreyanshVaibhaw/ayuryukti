"""YuktiShaala API router — Learning and feedback endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.yuktishaala.analytics import TreatmentAnalytics
from src.yuktishaala.contextual_bandit import ThompsonSamplingBandit
from src.yuktishaala.outcome_tracker import OutcomeTracker

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
VISITS_PATH = ROOT / "data" / "synthetic" / "patient_visits.csv"


class FeedbackRequest(BaseModel):
    encounter_id: str = Field(..., description="Encounter identifier")
    patient_prakriti: str = Field(..., description="Patient prakriti type")
    condition_code: str = Field(..., description="AYUSH condition name")
    formulations: List[str] = Field(..., description="Formulations prescribed")
    outcome: Literal["Improved", "No Change", "Worsened"] = Field(..., description="Follow-up outcome")
    follow_up_days: int = Field(default=14, ge=1)


class FeedbackResponse(BaseModel):
    status: str
    message: str


class EffectivenessResponse(BaseModel):
    condition: str
    treatments: List[Dict[str, Any]]


def _get_components():
    tracker = OutcomeTracker()
    bandit = ThompsonSamplingBandit(exploration_rate=0.15)
    if VISITS_PATH.exists():
        tracker.seed_from_synthetic(str(VISITS_PATH))
        for row in tracker.list_all():
            reward = 1.0 if row["outcome"] == "Improved" else 0.5 if row["outcome"] == "No Change" else 0.0
            bandit.update(
                prakriti=row["patient_prakriti"],
                condition=row["condition_code"],
                formulation=row["formulation_name"],
                reward=reward,
            )
    analytics = TreatmentAnalytics(tracker, bandit)
    return tracker, bandit, analytics


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record treatment outcome for continuous learning."""
    tracker, bandit, _ = _get_components()
    tracker.record_outcome(
        encounter_id=request.encounter_id,
        patient_prakriti=request.patient_prakriti,
        condition_code=request.condition_code,
        formulations=request.formulations,
        outcome=request.outcome,
        follow_up_days=request.follow_up_days,
    )
    reward = 1.0 if request.outcome == "Improved" else 0.5 if request.outcome == "No Change" else 0.0
    for formulation in request.formulations:
        bandit.update(request.patient_prakriti, request.condition_code, formulation, reward)
    return FeedbackResponse(
        status="success",
        message=f"Outcome '{request.outcome}' recorded for {len(request.formulations)} formulation(s).",
    )


@router.get("/effectiveness/{condition}")
async def get_effectiveness(condition: str) -> EffectivenessResponse:
    """Get treatment effectiveness analysis for a condition."""
    _, _, analytics = _get_components()
    df = analytics.get_treatment_effectiveness(condition)
    treatments = df.to_dict(orient="records") if not df.empty else []
    return EffectivenessResponse(condition=condition, treatments=treatments)


@router.get("/outcomes/summary")
async def outcomes_summary() -> Dict[str, Any]:
    """Get overall outcome tracking summary."""
    tracker, _, _ = _get_components()
    all_outcomes = tracker.list_all()
    total = len(all_outcomes)
    improved = sum(1 for o in all_outcomes if o.get("outcome") == "Improved")
    no_change = sum(1 for o in all_outcomes if o.get("outcome") == "No Change")
    worsened = sum(1 for o in all_outcomes if o.get("outcome") == "Worsened")
    return {
        "total_outcomes": total,
        "improved": improved,
        "no_change": no_change,
        "worsened": worsened,
        "improvement_rate": round(improved / total, 3) if total else 0.0,
    }
