"""Treatment recommendation engine for PrakritiMitra."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.common.models import TreatmentRecommendation
from src.llm.prompt_templates import RECOMMENDATION_EXPLANATION_TEMPLATE
from src.prakritimitra.explainer import explain
from src.prakritimitra.safety_checker import SafetyChecker

logger = logging.getLogger("AyurYukti.RecommendationEngine")


class RecommendationEngine:
    """Produce ranked AYUSH recommendations for patient context."""

    def __init__(self, knowledge_graph, llm_client, outcome_tracker=None, bandit=None, safety_checker=None):
        self.kg = knowledge_graph
        self.llm = llm_client
        self.outcomes = outcome_tracker
        self.bandit = bandit
        self.safety = safety_checker or SafetyChecker()
        self._encounter_context: Dict[str, Dict] = {}

    def _outcome_score(self, formulation_name: str) -> float:
        """Pull simple historical outcome score if tracker is available."""
        if not self.outcomes or not hasattr(self.outcomes, "list_all"):
            return 0.5
        history = self.outcomes.list_all()
        hits = [x for x in history if x.get("formulation_name") == formulation_name]
        if not hits:
            return 0.5
        improved = sum(1 for x in hits if x.get("outcome") == "Improved")
        return improved / len(hits)

    def _prakriti_bonus(self, formulation: Dict, patient_prakriti: str) -> float:
        indicated = [x.lower() for x in formulation.get("raw", {}).get("indicated_prakriti", [])]
        contraindicated = [x.lower() for x in formulation.get("raw", {}).get("contraindicated_prakriti", [])]
        p = patient_prakriti.lower()
        if p in contraindicated:
            return -0.5
        if p in indicated:
            return 1.0
        return 0.4

    def _age_adjust(self, formulation: Dict, age: int) -> str:
        """Adjust dosage hint for children and elderly."""
        dose = formulation.get("dosage_range", "As prescribed")
        if age < 12:
            return f"{dose} (50% pediatric adjustment)"
        if age > 65:
            return f"{dose} (75% geriatric adjustment)"
        return dose

    def recommend(
        self,
        patient_prakriti: str,
        condition_ayush_code: str,
        patient_age: int,
        patient_sex: str,
        existing_prescriptions: Optional[List[str]] = None,
    ) -> TreatmentRecommendation:
        """Run the recommendation pipeline with weighted ranking."""
        existing = {x.lower() for x in (existing_prescriptions or [])}
        candidates = self.kg.query_treatments(condition_ayush_code, patient_prakriti, top_k=20)

        bandit_scores = {}
        if self.bandit:
            actions = [c["name"] for c in candidates if c["name"].lower() not in existing]
            sampled = self.bandit.select_action(patient_prakriti, condition_ayush_code, actions)
            bandit_scores = {name: score for name, score in sampled}
            has_outcome_data = any(s.get("n_trials", 0) > 0 for s in self.bandit.get_arm_stats(patient_prakriti, condition_ayush_code))
        else:
            has_outcome_data = False

        ranked = []
        for candidate in candidates:
            if candidate["name"].lower() in existing:
                continue
            base_relevance = float(candidate.get("score", 0.5))
            prakriti_bonus = self._prakriti_bonus(candidate, patient_prakriti)
            if has_outcome_data and self.bandit:
                bandit_score = float(bandit_scores.get(candidate["name"], 0.5))
                final = 0.4 * base_relevance + 0.3 * prakriti_bonus + 0.3 * bandit_score
            else:
                # Cold start from knowledge graph only.
                final = base_relevance
            if final <= 0:
                continue
            ranked.append((final, candidate))

        ranked.sort(key=lambda x: x[0], reverse=True)

        # Safety check: filter out high-severity contraindications
        safe_ranked = []
        safety_warnings = []
        for final, item in ranked:
            raw = item.get("raw", item)
            warnings = self.safety.check_contraindications(raw, patient_prakriti, patient_age, patient_sex)
            high = [w for w in warnings if w["severity"] == "HIGH"]
            if not high:
                item["safety_warnings"] = [w for w in warnings if w["severity"] != "HIGH"]
                safe_ranked.append((final, item))
            else:
                safety_warnings.extend(high)

        top = safe_ranked[:5]

        # Check pairwise interactions among top formulations
        top_names = [item.get("name", "") for _, item in top]
        interaction_warnings = self.safety.check_interactions(top_names)
        if interaction_warnings:
            safety_warnings.extend(interaction_warnings)

        formatted = []
        refs = []
        for final, item in top:
            adjusted_dose = self._age_adjust(item, patient_age)
            refs.append(item.get("classical_reference", "Classical source"))
            formatted.append(
                {
                    "formulation_id": item.get("formulation_id"),
                    "formulation_name": item.get("name"),
                    "dosage": adjusted_dose,
                    "score": round(final, 3),
                    "classical_reference": item.get("classical_reference"),
                    "chapter_reference": item.get("chapter_reference"),
                }
            )

        lifestyle = self.kg.query_lifestyle(patient_prakriti, condition_ayush_code)

        if self.llm:
            prompt = RECOMMENDATION_EXPLANATION_TEMPLATE.format(
                prakriti_type=patient_prakriti,
                age=patient_age,
                sex=patient_sex,
                condition_ayush=condition_ayush_code,
                condition_english=condition_ayush_code,
                dosha_involvement="Context-specific",
                treatment_details=formatted,
            )
            explanation = self.llm.generate(prompt, temperature=0.1)
        else:
            explanation = explain({"prakriti": patient_prakriti, "condition": condition_ayush_code, "items": formatted})

        # Merge safety warnings into contraindications
        contraindications = list(lifestyle.get("avoid", []))
        for sw in safety_warnings:
            msg = sw.get("message", sw.get("description", ""))
            if msg:
                contraindications.append(f"[{sw.get('severity', 'WARN')}] {msg}")

        recommendation = TreatmentRecommendation(
            recommendation_id=str(uuid.uuid4()),
            encounter_id=str(uuid.uuid4()),
            patient_prakriti=patient_prakriti,
            condition=condition_ayush_code,
            recommended_formulations=formatted,
            lifestyle_suggestions=lifestyle.get("lifestyle", []),
            yoga_suggestions=lifestyle.get("yoga", []),
            dietary_suggestions=lifestyle.get("dietary", []),
            contraindications=contraindications,
            confidence=float(top[0][0]) if top else 0.0,
            reasoning=explanation,
            classical_references=list(dict.fromkeys(refs)),
            generated_at=datetime.now(timezone.utc),
        )
        self._encounter_context[recommendation.encounter_id] = {
            "patient_prakriti": patient_prakriti,
            "condition": condition_ayush_code,
            "formulations": [x["formulation_name"] for x in formatted],
        }
        return recommendation

    def record_feedback(self, encounter_id: str, outcome: str) -> bool:
        """Update OutcomeTracker and Bandit after follow-up feedback."""
        if encounter_id not in self._encounter_context:
            return False
        context = self._encounter_context[encounter_id]
        formulations = context["formulations"][:1] if context["formulations"] else []
        if not formulations:
            return False

        if self.outcomes:
            self.outcomes.record_outcome(
                encounter_id=encounter_id,
                patient_prakriti=context["patient_prakriti"],
                condition_code=context["condition"],
                formulations=formulations,
                outcome=outcome,
                follow_up_days=14,
            )

        if self.bandit:
            reward = 1.0 if outcome == "Improved" else 0.5 if outcome == "No Change" else 0.0
            for formulation in formulations:
                self.bandit.update(context["patient_prakriti"], context["condition"], formulation, reward)
        return True


def rank_recommendations(candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Backward-compatible helper from scaffold."""
    return sorted(candidates, key=lambda c: c.get("confidence", ""), reverse=True)
