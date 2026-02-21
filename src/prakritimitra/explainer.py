"""Human-readable explanation helpers for treatment recommendations."""

from __future__ import annotations

from typing import Dict


def explain(recommendation: Dict) -> str:
    """Generate concise clinical reasoning text."""
    prakriti = recommendation.get("prakriti", "patient")
    condition = recommendation.get("condition", "condition")
    items = recommendation.get("items", [])
    top_name = items[0]["formulation_name"] if items else "selected formulation"
    return (
        f"{top_name} is prioritized because it aligns with {prakriti} traits and "
        f"the presenting condition ({condition}). The recommendation balances dosha-specific "
        "needs and references classical AYUSH treatment logic. Final prescription should be "
        "validated by the treating physician based on full clinical context."
    )


def generate_patient_summary(recommendation: Dict, language: str) -> str:
    """Generate patient-friendly summary of recommendation plan."""
    _ = language
    items = recommendation.get("recommended_formulations", [])
    if not items:
        return "No medicine recommendation available right now. Please consult your doctor."
    first = items[0]
    return (
        f"Suggested medicine: {first.get('formulation_name')} ({first.get('dosage')}). "
        "Please follow doctor advice on food, yoga, and follow-up."
    )


# Backward-compatible helper name used by early scaffold.
def explain_recommendation(recommendation: Dict[str, str]) -> str:
    formulation = recommendation.get("name", "Unknown formulation")
    return f"Recommended based on condition-Prakriti match: {formulation}."

