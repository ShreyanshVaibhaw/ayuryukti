"""Benchmark Prakriti classifier accuracy with cross-validation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prakritimitra.prakriti_classifier import PrakritiClassifier

Q_PATH = ROOT / "data" / "prakriti" / "questionnaire.json"
TRAIN_PATH = ROOT / "data" / "prakriti" / "training_data.csv"


def run_benchmark() -> dict:
    """Run classifier benchmark with cross-validation."""
    classifier = PrakritiClassifier(
        questionnaire_path=str(Q_PATH),
        training_data_path=str(TRAIN_PATH),
    )

    # Overall accuracy
    accuracy = classifier.evaluate_accuracy()

    # Per-class metrics using cross-validation predictions
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix

    df = pd.read_csv(TRAIN_PATH)
    feature_cols = [c for c in df.columns if c.startswith("q")]
    X = df[feature_cols].values
    y = df["prakriti"].values

    y_pred = cross_val_predict(classifier.model, X, y, cv=5)

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=sorted(set(y)))

    per_class = {}
    for cls in sorted(set(y)):
        cls_report = report.get(cls, {})
        per_class[cls] = {
            "precision": round(cls_report.get("precision", 0), 3),
            "recall": round(cls_report.get("recall", 0), 3),
            "f1_score": round(cls_report.get("f1-score", 0), 3),
            "support": int(cls_report.get("support", 0)),
        }

    metrics = {
        "overall_accuracy": round(accuracy, 3),
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 3),
        "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0), 3),
        "n_classes": len(set(y)),
        "n_samples": len(y),
        "per_class": per_class,
    }

    return metrics


if __name__ == "__main__":
    output = run_benchmark()
    print("\n=== Prakriti Classifier Benchmark ===")
    print(f"  Overall Accuracy: {output['overall_accuracy']}")
    print(f"  Macro F1: {output['macro_f1']}")
    print(f"  Weighted F1: {output['weighted_f1']}")
    print(f"  Classes: {output['n_classes']}")
    print(f"  Samples: {output['n_samples']}")
    print("\n  Per-class:")
    for cls, m in output["per_class"].items():
        print(f"    {cls}: P={m['precision']} R={m['recall']} F1={m['f1_score']} (n={m['support']})")

    passed = output["overall_accuracy"] >= 0.90
    print(f"\nTarget: overall_accuracy >= 0.90")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
