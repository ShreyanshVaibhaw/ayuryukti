"""Benchmark NER accuracy on labeled transcripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vaksetu.medical_ner import MedicalNEREngine

VOCAB = ROOT / "data" / "ayush_vocabulary.json"
MORBIDITY = ROOT / "data" / "knowledge_base" / "ayush_morbidity_codes.json"
EVAL_DATA = ROOT / "data" / "evaluation" / "labeled_transcripts.json"


def run_benchmark() -> dict:
    """Run NER benchmark and return metrics."""
    ner = MedicalNEREngine(llm_client=None, vocabulary_path=str(VOCAB), morbidity_codes_path=str(MORBIDITY))
    cases = json.loads(EVAL_DATA.read_text(encoding="utf-8"))

    total = len(cases)
    age_correct = 0
    sex_correct = 0
    prakriti_correct = 0
    condition_correct = 0
    icd_correct = 0
    rx_precision_sum = 0.0
    rx_recall_sum = 0.0
    complaint_recall_sum = 0.0
    confidence_sum = 0.0

    results = []
    for case in cases:
        ehr = ner.extract_ehr(case["transcript"], case["language"])
        expected = case["expected"]

        age_ok = ehr.patient_demographics.get("age") == expected["age"]
        sex_ok = ehr.patient_demographics.get("sex") == expected.get("sex")
        prakriti_ok = ehr.prakriti_assessment == expected.get("prakriti")
        condition_ok = ehr.ayush_diagnosis.get("name") == expected.get("condition")
        icd_ok = ehr.icd10_diagnosis.get("code") == expected.get("icd10_code")

        if age_ok:
            age_correct += 1
        if sex_ok:
            sex_correct += 1
        if prakriti_ok:
            prakriti_correct += 1
        if condition_ok:
            condition_correct += 1
        if icd_ok:
            icd_correct += 1

        # Prescription accuracy
        extracted_rx = {rx.get("formulation_name", "").lower() for rx in ehr.prescriptions}
        expected_rx = {rx.lower() for rx in expected.get("prescriptions", [])}
        if expected_rx:
            hits = extracted_rx & expected_rx
            precision = len(hits) / len(extracted_rx) if extracted_rx else 0
            recall = len(hits) / len(expected_rx) if expected_rx else 0
            rx_precision_sum += precision
            rx_recall_sum += recall

        # Complaint recall
        expected_complaints_lower = {c.lower() for c in expected.get("complaints", [])}
        extracted_complaints_lower = set()
        for comp in ehr.chief_complaints:
            extracted_complaints_lower.add(comp.get("complaint", "").lower())
        if expected_complaints_lower:
            comp_hits = sum(
                1 for ec in expected_complaints_lower
                if any(ec in ext for ext in extracted_complaints_lower)
            )
            complaint_recall_sum += comp_hits / len(expected_complaints_lower)

        overall_conf = ehr.confidence_scores.get("overall", 0)
        confidence_sum += overall_conf

        results.append({
            "id": case["id"],
            "age_ok": age_ok,
            "sex_ok": sex_ok,
            "prakriti_ok": prakriti_ok,
            "condition_ok": condition_ok,
            "icd_ok": icd_ok,
            "rx_extracted": list(extracted_rx),
            "rx_expected": list(expected_rx),
            "confidence": overall_conf,
        })

    metrics = {
        "total_cases": total,
        "age_accuracy": round(age_correct / total, 3),
        "sex_accuracy": round(sex_correct / total, 3),
        "prakriti_accuracy": round(prakriti_correct / total, 3),
        "condition_accuracy": round(condition_correct / total, 3),
        "icd10_accuracy": round(icd_correct / total, 3),
        "rx_precision": round(rx_precision_sum / total, 3),
        "rx_recall": round(rx_recall_sum / total, 3),
        "complaint_recall": round(complaint_recall_sum / total, 3),
        "avg_confidence": round(confidence_sum / total, 3),
    }

    # Composite accuracy (weighted)
    metrics["composite_accuracy"] = round(
        0.2 * metrics["age_accuracy"]
        + 0.1 * metrics["sex_accuracy"]
        + 0.15 * metrics["prakriti_accuracy"]
        + 0.25 * metrics["condition_accuracy"]
        + 0.15 * metrics["icd10_accuracy"]
        + 0.15 * metrics["rx_recall"],
        3,
    )

    return {"metrics": metrics, "results": results}


if __name__ == "__main__":
    output = run_benchmark()
    print("\n=== NER Benchmark Results ===")
    for key, value in output["metrics"].items():
        print(f"  {key}: {value}")

    passed = output["metrics"]["composite_accuracy"] >= 0.85
    print(f"\nTarget: composite_accuracy >= 0.85")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
