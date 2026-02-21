"""YuktiShaala tests: outcomes, bandit, and recommendation integration."""

from __future__ import annotations

from pathlib import Path

from src.prakritimitra.knowledge_graph import InMemoryKnowledgeGraph
from src.prakritimitra.recommendation_engine import RecommendationEngine
from src.yuktishaala.contextual_bandit import ThompsonSamplingBandit
from src.yuktishaala.outcome_tracker import OutcomeTracker


ROOT = Path(__file__).resolve().parents[1]
FORM_PATH = ROOT / "data" / "knowledge_base" / "formulations.json"
CODE_PATH = ROOT / "data" / "knowledge_base" / "ayush_morbidity_codes.json"
RULES_PATH = ROOT / "data" / "knowledge_base" / "prakriti_rules.json"
VISITS_PATH = ROOT / "data" / "synthetic" / "patient_visits.csv"


def test_outcome_recording_and_retrieval() -> None:
    tracker = OutcomeTracker()
    tracker.record_outcome("E1", "Vata", "Vibandha", ["Triphala Churna"], "Improved", 7)
    stats = tracker.get_outcomes_for_treatment("Vata", "Vibandha", "Triphala Churna")
    assert stats["total"] == 1
    assert stats["success_rate"] == 1.0


def test_thompson_sampling_converges_after_100_positive_updates() -> None:
    bandit = ThompsonSamplingBandit(exploration_rate=0.0)
    for _ in range(100):
        bandit.update("Vata", "Vibandha", "Triphala Churna", 1.0)
    stats = bandit.get_arm_stats("Vata", "Vibandha")
    top = next(x for x in stats if x["formulation"] == "Triphala Churna")
    assert top["mean"] > 0.8


def test_exploration_recommends_low_data_formulations_sometimes() -> None:
    bandit = ThompsonSamplingBandit(exploration_rate=1.0)
    available = ["Triphala Churna", "Abhayarishta", "Avipattikar Churna"]
    seen = set()
    for _ in range(15):
        ranked = bandit.select_action("Vata", "Vibandha", available)
        seen.add(ranked[0][0])
    assert len(seen) >= 2


def test_cold_start_bandit_uniform_scores() -> None:
    bandit = ThompsonSamplingBandit(exploration_rate=0.0)
    available = ["A", "B", "C"]
    ranked = bandit.select_action("Vata", "Vibandha", available)
    means = [x["mean"] for x in bandit.get_arm_stats("Vata", "Vibandha")]
    assert len(ranked) == 3
    assert all(abs(m - 0.5) < 1e-6 for m in means)


def test_recommendation_engine_uses_outcome_feedback_when_available() -> None:
    kg = InMemoryKnowledgeGraph()
    kg.seed_from_json(str(FORM_PATH), str(CODE_PATH), str(RULES_PATH))
    tracker = OutcomeTracker()
    bandit = ThompsonSamplingBandit(exploration_rate=0.0)
    engine = RecommendationEngine(knowledge_graph=kg, llm_client=None, outcome_tracker=tracker, bandit=bandit)

    rec = engine.recommend("Vata", "Vibandha", 35, "Female")
    assert rec.recommended_formulations
    encounter_id = rec.encounter_id
    ok = engine.record_feedback(encounter_id=encounter_id, outcome="Improved")
    assert ok

    stats = bandit.get_arm_stats("Vata", "Vibandha")
    assert any(x["n_trials"] > 0 for x in stats)

    # Seed from synthetic outcomes should load without crashing.
    if VISITS_PATH.exists():
        tracker.seed_from_synthetic(str(VISITS_PATH))
        assert len(tracker.list_all()) > 0
