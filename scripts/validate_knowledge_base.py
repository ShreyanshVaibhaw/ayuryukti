"""Validate AyurYukti knowledge-base quality and consistency checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KB = DATA / "knowledge_base"

ICD_RE = re.compile(r"^[A-Z][0-9]{1,2}(\.[0-9A-Z]{1,2})?(-[A-Z]?[0-9]{1,2}(\.[0-9A-Z]{1,2})?)?$")
ALLOWED_DOSHA_ACTIONS = {"pacifies", "aggravates", "neutral"}


def load_json(path: Path):
    """Load and return JSON data."""
    return json.loads(path.read_text(encoding="utf-8"))


def assert_true(condition: bool, message: str) -> None:
    """Raise ValueError when condition fails."""
    if not condition:
        raise ValueError(message)


def _valid_icd(code: str) -> bool:
    return bool(ICD_RE.match(code.strip()))


def main() -> None:
    """Execute validation on all critical knowledge assets."""
    formulations = load_json(KB / "formulations.json")
    morbidity = load_json(KB / "ayush_morbidity_codes.json")
    rules = load_json(KB / "prakriti_rules.json")
    questionnaire = load_json(DATA / "prakriti" / "questionnaire.json")
    vocabulary = load_json(DATA / "ayush_vocabulary.json")

    assert_true(len(formulations) >= 200, "Need at least 200 formulations")
    assert_true(len(morbidity) >= 100, "Need at least 100 morbidity codes")
    assert_true(len(rules) == 7, "Need exactly 7 prakriti rule sets")
    assert_true(len(questionnaire) == 30, "Need exactly 30 questionnaire items")
    assert_true(len(vocabulary) >= 500, "Need at least 500 vocabulary terms")

    bad_formulation_names = [
        x["name_sanskrit"]
        for x in formulations
        if "Classical Support Formula" in x.get("name_sanskrit", "") or "Variant" in x.get("name_sanskrit", "")
    ]
    assert_true(not bad_formulation_names, "Found synthetic placeholder formulations")

    assert_true(
        all(x.get("classical_reference", "").strip() for x in formulations),
        "Every formulation must have a classical_reference",
    )
    assert_true(
        all("classical ayurvedic compendia" not in x.get("classical_reference", "").lower() for x in formulations),
        "Generic classical_reference placeholder detected",
    )
    assert_true(
        all(x.get("chapter_reference", "").strip() for x in formulations),
        "Every formulation must have a chapter_reference",
    )
    assert_true(
        all("pending precise chapter mapping" not in x.get("chapter_reference", "").lower() for x in formulations),
        "chapter_reference placeholder detected",
    )
    assert_true(
        all("qualified AYUSH physician" in x.get("safety_notes", "") for x in formulations),
        "Every formulation must include clinical safety supervision note",
    )

    for item in formulations:
        dosha_action = item.get("dosha_action", {})
        invalid = [k for k, v in dosha_action.items() if str(v).lower() not in ALLOWED_DOSHA_ACTIONS]
        assert_true(not invalid, f"Invalid dosha_action values in {item.get('formulation_id')}: {invalid}")

    bad_codes = []
    for row in morbidity:
        assert_true(
            not any("specialist-curated" in str(s).lower() for s in row.get("common_symptoms", [])),
            f"Placeholder symptoms found in {row.get('code_id')}",
        )
        for icd in row.get("icd10_codes", []):
            if not _valid_icd(icd):
                bad_codes.append((row.get("code_id"), icd))
    assert_true(not bad_codes, f"Invalid ICD-10 mappings: {bad_codes[:8]}")

    expected_prakritis = {"Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Sama"}
    assert_true(set(rules.keys()) == expected_prakritis, "Prakriti rule keys are inconsistent")

    for prakriti, payload in rules.items():
        assert_true(payload.get("general_principles"), f"Missing general_principles for {prakriti}")
        assert_true("dietary_guidelines" in payload, f"Missing dietary_guidelines for {prakriti}")
        assert_true("lifestyle_guidelines" in payload, f"Missing lifestyle_guidelines for {prakriti}")
        assert_true("yoga_recommendations" in payload, f"Missing yoga_recommendations for {prakriti}")

    print("knowledge-base-validation=ok")
    print(
        f"formulations={len(formulations)} morbidity={len(morbidity)} questionnaire={len(questionnaire)} "
        f"vocab={len(vocabulary)}"
    )


if __name__ == "__main__":
    main()
