"""Benchmark regression tests — ensure accuracy stays above thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_ner_composite_accuracy_above_85():
    """NER composite accuracy must stay >= 85%."""
    from scripts.benchmark_ner import run_benchmark

    output = run_benchmark()
    assert output["metrics"]["composite_accuracy"] >= 0.85


def test_ner_all_field_accuracies_above_90():
    """Every NER field accuracy must be >= 90%."""
    from scripts.benchmark_ner import run_benchmark

    m = run_benchmark()["metrics"]
    for key in ["age_accuracy", "sex_accuracy", "prakriti_accuracy", "condition_accuracy",
                "icd10_accuracy", "rx_precision", "rx_recall", "complaint_recall", "avg_confidence"]:
        assert m[key] >= 0.90, f"{key} = {m[key]} is below 90%"


def test_prakriti_overall_accuracy_above_90():
    """Prakriti classifier overall accuracy must stay >= 90%."""
    from scripts.benchmark_prakriti import run_benchmark

    output = run_benchmark()
    assert output["overall_accuracy"] >= 0.90


def test_prakriti_per_class_f1_above_90():
    """Every Prakriti class F1 must be >= 90%."""
    from scripts.benchmark_prakriti import run_benchmark

    output = run_benchmark()
    for cls, metrics in output["per_class"].items():
        assert metrics["f1_score"] >= 0.90, f"{cls} F1 = {metrics['f1_score']} is below 90%"


def test_rogaradar_recall_above_66():
    """RogaRadar outbreak recall must stay >= 66%."""
    from scripts.benchmark_rogaradar import run_benchmark

    output = run_benchmark()
    assert output["recall"] >= 0.66


def test_rogaradar_zero_false_negatives():
    """RogaRadar should not miss any known outbreak."""
    from scripts.benchmark_rogaradar import run_benchmark

    output = run_benchmark()
    assert output["false_negatives"] == 0, f"Missed: {output.get('missed_outbreaks')}"
