"""Data quality tests for curated Phase 4 assets."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KB = DATA / "knowledge_base"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_formulations_have_no_placeholder_names() -> None:
    formulations = _load(KB / "formulations.json")
    assert len(formulations) >= 200
    assert not any("Classical Support Formula" in row["name_sanskrit"] for row in formulations)


def test_morbidity_and_vocabulary_size() -> None:
    morbidity = _load(KB / "ayush_morbidity_codes.json")
    vocabulary = _load(DATA / "ayush_vocabulary.json")
    assert len(morbidity) >= 100
    assert len(vocabulary) >= 500


# --- Phase 4 additions ---

def test_formulations_have_required_fields() -> None:
    formulations = _load(KB / "formulations.json")
    required_fields = ["formulation_id", "name_sanskrit", "formulation_type", "system"]
    for f in formulations[:50]:
        for field in required_fields:
            assert field in f, f"Missing {field} in formulation {f.get('formulation_id', 'unknown')}"


def test_formulations_have_ingredients() -> None:
    formulations = _load(KB / "formulations.json")
    with_ingredients = sum(1 for f in formulations if len(f.get("ingredients", [])) > 0)
    assert with_ingredients > 100, "At least 100 formulations should have ingredients"


def test_formulations_have_indicated_conditions() -> None:
    formulations = _load(KB / "formulations.json")
    with_conditions = sum(1 for f in formulations if len(f.get("indicated_conditions", [])) > 0)
    assert with_conditions > 100, "At least 100 formulations should have indicated conditions"


def test_morbidity_codes_have_icd10_mapping() -> None:
    morbidity = _load(KB / "ayush_morbidity_codes.json")
    with_icd = sum(1 for c in morbidity if len(c.get("icd10_codes", [])) > 0)
    assert with_icd > 50, "At least 50 morbidity codes should have ICD-10 mappings"


def test_morbidity_codes_have_symptoms() -> None:
    morbidity = _load(KB / "ayush_morbidity_codes.json")
    with_symptoms = sum(1 for c in morbidity if len(c.get("common_symptoms", [])) > 0)
    assert with_symptoms > 50, "At least 50 morbidity codes should have common symptoms"


def test_prakriti_rules_all_seven_types() -> None:
    rules = _load(KB / "prakriti_rules.json")
    expected = {"Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Sama"}
    assert set(rules.keys()) == expected


def test_prakriti_rules_differentiated() -> None:
    rules = _load(KB / "prakriti_rules.json")
    vata_diet = rules["Vata"]["dietary_guidelines"]["favor"]
    pitta_diet = rules["Pitta"]["dietary_guidelines"]["favor"]
    kapha_diet = rules["Kapha"]["dietary_guidelines"]["favor"]
    assert vata_diet != pitta_diet
    assert pitta_diet != kapha_diet


def test_prakriti_rules_have_yoga() -> None:
    rules = _load(KB / "prakriti_rules.json")
    for ptype, rule in rules.items():
        yoga = rule.get("yoga_recommendations", [])
        assert len(yoga) >= 3, f"{ptype} should have at least 3 yoga recommendations"


def test_prakriti_rules_have_seasonal_advice() -> None:
    rules = _load(KB / "prakriti_rules.json")
    for ptype, rule in rules.items():
        season = rule.get("lifestyle_guidelines", {}).get("season_advice", {})
        assert len(season) >= 1, f"{ptype} should have seasonal advice"


def test_icd10_mapping_file_exists_and_valid() -> None:
    mapping = _load(KB / "icd10_mapping.json")
    assert len(mapping) > 20


def test_interaction_matrix_exists() -> None:
    interaction = _load(KB / "interaction_matrix.json")
    assert "interactions" in interaction
    assert len(interaction["interactions"]) >= 5


def test_interaction_matrix_has_severity() -> None:
    interaction = _load(KB / "interaction_matrix.json")
    for entry in interaction["interactions"]:
        assert "severity" in entry
        assert entry["severity"] in {"LOW", "MODERATE", "HIGH"}


def test_evaluation_transcripts_exist() -> None:
    eval_data = _load(DATA / "evaluation" / "labeled_transcripts.json")
    assert len(eval_data) >= 20
    for case in eval_data:
        assert "transcript" in case
        assert "expected" in case
        assert "age" in case["expected"]


def test_vocabulary_has_formulation_types() -> None:
    vocab = _load(DATA / "ayush_vocabulary.json")
    types = {entry.get("type") for entry in vocab}
    assert "formulation" in types
    assert "condition" in types


def test_vocabulary_entries_have_terms() -> None:
    vocab = _load(DATA / "ayush_vocabulary.json")
    for entry in vocab[:100]:
        assert "term" in entry
        assert len(entry["term"]) > 0
