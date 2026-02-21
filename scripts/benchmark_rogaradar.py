"""Benchmark RogaRadar outbreak detection precision, recall, and F1."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALERT_LEVELS
from scripts.generate_synthetic_data import generate_outbreak_scenarios, generate_patient_visits
from src.rogaradar.alert_generator import AlertGenerator
from src.rogaradar.anomaly_detector import AnomalyDetector
from src.rogaradar.baseline_model import BaselineModel
from src.rogaradar.data_ingestion import DataIngestion
from src.rogaradar.geo_cluster import GeoCluster

VISITS_PATH = ROOT / "data" / "synthetic" / "patient_visits.csv"

# Known injected outbreaks (ground truth)
EXPECTED_OUTBREAKS = [
    {"district": "Varanasi", "condition": "Jwara"},
    {"district": "Chennai", "condition": "Kushtha"},
    {"district": "Jaipur", "condition": "Prameha"},
]


def run_benchmark() -> dict:
    """Run outbreak detection benchmark."""
    # Ensure fresh data
    visits = generate_patient_visits(n_records=10000, n_centres=50, n_districts=25, n_states=5)
    VISITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    visits.to_csv(VISITS_PATH, index=False)

    # Run full detection pipeline
    ingestion = DataIngestion(str(VISITS_PATH))
    df = ingestion.load_visit_data()
    agg = ingestion.aggregate_by_district_condition_week(df)
    baseline = BaselineModel()
    baseline.fit_all(agg)
    detector = AnomalyDetector(baseline_model=baseline)
    anomalies = detector.run_all_detectors(agg)
    geo = GeoCluster(ingestion.get_district_metadata())
    clusters = geo.cluster_anomalies(anomalies)
    alerts = AlertGenerator(ALERT_LEVELS).generate_alerts(anomalies, clusters)

    # Convert to DataFrame for analysis
    alert_pairs = {(a.district, a.condition_ayush) for a in alerts}
    expected_pairs = {(e["district"], e["condition"]) for e in EXPECTED_OUTBREAKS}

    # True positives: detected outbreaks that are real
    true_positives = alert_pairs & expected_pairs
    # False positives: detected outbreaks that are not real
    false_positives = alert_pairs - expected_pairs
    # False negatives: real outbreaks that were missed
    false_negatives = expected_pairs - alert_pairs

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Severity accuracy check
    severity_correct = 0
    for alert in alerts:
        pair = (alert.district, alert.condition_ayush)
        if pair in expected_pairs:
            if alert.alert_level in {"WARNING", "ALERT"}:
                severity_correct += 1

    metrics = {
        "total_alerts": len(alerts),
        "expected_outbreaks": len(EXPECTED_OUTBREAKS),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "severity_accuracy": round(severity_correct / len(EXPECTED_OUTBREAKS), 3) if EXPECTED_OUTBREAKS else 0,
        "detected_outbreaks": list(true_positives),
        "missed_outbreaks": list(false_negatives),
    }

    return metrics


if __name__ == "__main__":
    output = run_benchmark()
    print("\n=== RogaRadar Outbreak Detection Benchmark ===")
    print(f"  Total Alerts Generated: {output['total_alerts']}")
    print(f"  Expected Outbreaks: {output['expected_outbreaks']}")
    print(f"  True Positives: {output['true_positives']}")
    print(f"  False Positives: {output['false_positives']}")
    print(f"  False Negatives: {output['false_negatives']}")
    print(f"  Precision: {output['precision']}")
    print(f"  Recall: {output['recall']}")
    print(f"  F1 Score: {output['f1_score']}")
    print(f"  Severity Accuracy: {output['severity_accuracy']}")

    if output["detected_outbreaks"]:
        print(f"\n  Detected: {output['detected_outbreaks']}")
    if output["missed_outbreaks"]:
        print(f"  Missed: {output['missed_outbreaks']}")

    passed = output["recall"] >= 0.66  # Detect at least 2/3 injected outbreaks
    print(f"\nTarget: recall >= 0.66")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
