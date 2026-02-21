"""PrakritiMitra tests for classifier, knowledge graph, recommendations, and safety."""

from __future__ import annotations

from pathlib import Path

from src.prakritimitra.knowledge_graph import AyushKnowledgeGraph
from src.prakritimitra.lifestyle_advisor import LifestyleAdvisor
from src.prakritimitra.prakriti_classifier import PrakritiClassifier
from src.prakritimitra.recommendation_engine import RecommendationEngine
from src.prakritimitra.safety_checker import SafetyChecker


ROOT = Path(__file__).resolve().parents[1]
Q_PATH = ROOT / "data" / "prakriti" / "questionnaire.json"
TRAIN_PATH = ROOT / "data" / "prakriti" / "training_data.csv"
FORM_PATH = ROOT / "data" / "knowledge_base" / "formulations.json"
CODE_PATH = ROOT / "data" / "knowledge_base" / "ayush_morbidity_codes.json"
RULES_PATH = ROOT / "data" / "knowledge_base" / "prakriti_rules.json"
INTERACTION_PATH = ROOT / "data" / "knowledge_base" / "interaction_matrix.json"


def _build_graph():
    kg = AyushKnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password="neo4j")
    kg.setup_schema()
    kg.seed_from_json(str(FORM_PATH), str(CODE_PATH), str(RULES_PATH))
    return kg


def test_classifier_accuracy_over_90_on_synthetic() -> None:
    classifier = PrakritiClassifier(questionnaire_path=str(Q_PATH), training_data_path=str(TRAIN_PATH))
    acc = classifier.evaluate_accuracy()
    assert acc > 0.9


def test_knowledge_graph_seeding_and_query() -> None:
    kg = _build_graph()
    rows = kg.query_treatments("Vibandha", "Vata", top_k=5)
    assert rows
    assert all("name" in r for r in rows)


def test_vata_vibandha_has_triphala_in_top3() -> None:
    kg = _build_graph()
    rows = kg.query_treatments("Vibandha", "Vata", top_k=3)
    names = [r["name"].lower() for r in rows]
    assert any("triphala churna" in x for x in names)


def test_pitta_amlapitta_has_avipattikar_in_top3() -> None:
    kg = _build_graph()
    rows = kg.query_treatments("Amlapitta", "Pitta", top_k=3)
    names = [r["name"].lower() for r in rows]
    assert any("avipattikar churna" in x for x in names)


def test_contraindicated_formulations_not_present() -> None:
    kg = _build_graph()
    rows = kg.query_treatments("Vibandha", "Pitta-Kapha", top_k=10)
    for row in rows:
        raw = row.get("raw", {})
        contraindicated = [x.lower() for x in raw.get("contraindicated_prakriti", [])]
        assert "pitta-kapha" not in contraindicated


def test_lifestyle_advisor_returns_for_all_prakriti() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    for prakriti in ["Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Sama"]:
        assert advisor.get_dietary_advice(prakriti, "Vibandha")
        assert advisor.get_yoga_advice(prakriti, "Vibandha")
        assert advisor.get_lifestyle_advice(prakriti, "Vibandha")


def test_recommendation_engine_pipeline() -> None:
    kg = _build_graph()
    engine = RecommendationEngine(knowledge_graph=kg, llm_client=None, outcome_tracker=None)
    rec = engine.recommend(
        patient_prakriti="Vata",
        condition_ayush_code="Vibandha",
        patient_age=35,
        patient_sex="Female",
        existing_prescriptions=[],
    )
    assert rec.recommended_formulations
    assert rec.condition == "Vibandha"


# --- Phase 4 additions ---

def test_lifestyle_advice_is_differentiated_per_prakriti() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    vata_diet = advisor.get_dietary_advice("Vata", "general")
    pitta_diet = advisor.get_dietary_advice("Pitta", "general")
    kapha_diet = advisor.get_dietary_advice("Kapha", "general")
    # All three should be different
    assert vata_diet != pitta_diet
    assert pitta_diet != kapha_diet
    assert vata_diet != kapha_diet


def test_vata_specific_dietary_content() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    vata_diet = advisor.get_dietary_advice("Vata", "general")
    diet_text = " ".join(vata_diet).lower()
    assert "warm" in diet_text  # Vata should get warming foods
    assert "ghee" in diet_text or "sesame" in diet_text


def test_pitta_specific_dietary_content() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    pitta_diet = advisor.get_dietary_advice("Pitta", "general")
    diet_text = " ".join(pitta_diet).lower()
    assert "cool" in diet_text  # Pitta should get cooling foods
    assert "spicy" in diet_text  # Should warn about spicy


def test_kapha_specific_dietary_content() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    kapha_diet = advisor.get_dietary_advice("Kapha", "general")
    diet_text = " ".join(kapha_diet).lower()
    assert "light" in diet_text  # Kapha should get light foods


def test_yoga_advice_differentiated() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    vata_yoga = advisor.get_yoga_advice("Vata", "general")
    pitta_yoga = advisor.get_yoga_advice("Pitta", "general")
    kapha_yoga = advisor.get_yoga_advice("Kapha", "general")
    assert vata_yoga != pitta_yoga
    assert pitta_yoga != kapha_yoga


def test_condition_specific_advice() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    vata_vibandha = advisor.get_dietary_advice("Vata", "Vibandha")
    vata_general = advisor.get_dietary_advice("Vata", "general")
    # Vibandha-specific advice should include extra items
    assert len(vata_vibandha) >= len(vata_general)


def test_full_profile_returns_all_sections() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    profile = advisor.get_full_profile("Vata")
    assert "general_principles" in profile
    assert "aggravating_factors" in profile
    assert "pacifying_factors" in profile
    assert "preferred_dosage_forms" in profile


def test_safety_checker_interaction_detection() -> None:
    checker = SafetyChecker(str(INTERACTION_PATH))
    warnings = checker.check_interactions(["Yogaraja Guggulu", "Simhanada Guggulu"])
    assert len(warnings) > 0
    assert warnings[0]["severity"] in {"LOW", "MODERATE", "HIGH"}


def test_safety_checker_no_interaction() -> None:
    checker = SafetyChecker(str(INTERACTION_PATH))
    warnings = checker.check_interactions(["Triphala Churna", "Ashwagandha Churna"])
    assert len(warnings) == 0


def test_safety_checker_pediatric_warning() -> None:
    checker = SafetyChecker(str(INTERACTION_PATH))
    warnings = checker.check_contraindications(
        {"name_sanskrit": "Some Medicine", "contraindicated_prakriti": []},
        "Vata", 3, "Male",
    )
    pediatric = [w for w in warnings if w["type"] == "PEDIATRIC_CAUTION"]
    assert len(pediatric) > 0


def test_safety_checker_geriatric_warning() -> None:
    checker = SafetyChecker(str(INTERACTION_PATH))
    warnings = checker.check_contraindications(
        {"name_sanskrit": "Some Medicine", "contraindicated_prakriti": []},
        "Vata", 80, "Male",
    )
    geriatric = [w for w in warnings if w["type"] == "GERIATRIC_CAUTION"]
    assert len(geriatric) > 0


def test_safety_checker_prakriti_conflict() -> None:
    checker = SafetyChecker(str(INTERACTION_PATH))
    warnings = checker.check_contraindications(
        {"name_sanskrit": "Test Med", "contraindicated_prakriti": ["Pitta"]},
        "Pitta", 35, "Male",
    )
    conflicts = [w for w in warnings if w["type"] == "PRAKRITI_CONFLICT"]
    assert len(conflicts) > 0


def test_unknown_prakriti_does_not_crash() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    result = advisor.get_dietary_advice("UnknownType", "Vibandha")
    assert isinstance(result, list)


def test_unknown_condition_does_not_crash() -> None:
    advisor = LifestyleAdvisor(prakriti_rules_path=RULES_PATH)
    result = advisor.get_dietary_advice("Vata", "NonexistentCondition")
    assert isinstance(result, list)
    assert len(result) > 0  # Should still return base advice
