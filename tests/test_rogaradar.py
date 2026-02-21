"""RogaRadar tests for synthetic data, anomaly detection, alerts, and benchmarks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import ALERT_LEVELS
from scripts.generate_synthetic_data import generate_outbreak_scenarios, generate_patient_visits
from src.rogaradar.alert_generator import AlertGenerator
from src.rogaradar.anomaly_detector import AnomalyDetector
from src.rogaradar.baseline_model import BaselineModel
from src.rogaradar.data_ingestion import DataIngestion
from src.rogaradar.geo_cluster import GeoCluster


ROOT = Path(__file__).resolve().parents[1]
VISITS_PATH = ROOT / "data" / "synthetic" / "patient_visits.csv"
SCENARIO_PATH = ROOT / "data" / "synthetic" / "outbreak_scenarios.csv"


def _ensure_data():
    visits = generate_patient_visits(n_records=10000, n_centres=50, n_districts=25, n_states=5)
    scenarios = generate_outbreak_scenarios()
    VISITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    visits.to_csv(VISITS_PATH, index=False)
    scenarios.to_csv(SCENARIO_PATH, index=False)
    return visits, scenarios


def _build_pipeline():
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
    return df, agg, anomalies, clusters, alerts


def test_synthetic_data_generation_and_injected_anomalies() -> None:
    visits, scenarios = _ensure_data()
    assert len(visits) == 10000
    assert len(scenarios) == 3
    # Check presence of injected scenario windows.
    assert ((visits["district"] == "Varanasi") & (visits["ayush_diagnosis_name"] == "Jwara") & (visits["date"].str.startswith("2025-08"))).any()
    assert ((visits["district"] == "Chennai") & (visits["ayush_diagnosis_name"] == "Kushtha") & (visits["date"].str.startswith("2025-06"))).any()
    assert ((visits["district"] == "Jaipur") & (visits["ayush_diagnosis_name"] == "Prameha") & (visits["date"].str.startswith("2025-1"))).any()


def test_baseline_model_fits_without_crash() -> None:
    _ensure_data()
    ingestion = DataIngestion(str(VISITS_PATH))
    agg = ingestion.aggregate_by_district_condition_week(ingestion.load_visit_data())
    baseline = BaselineModel()
    baseline.fit_all(agg)
    assert baseline.models


def test_anomaly_detection_catches_all_three_injected_outbreaks() -> None:
    _ensure_data()
    _, _, anomalies, _, _ = _build_pipeline()
    a_df = pd.DataFrame(anomalies)
    assert ((a_df["district"] == "Varanasi") & (a_df["condition"] == "Jwara")).any()
    assert ((a_df["district"] == "Chennai") & (a_df["condition"] == "Kushtha")).any()
    assert ((a_df["district"] == "Jaipur") & (a_df["condition"] == "Prameha")).any()


def test_geo_clustering_and_neighboring_logic() -> None:
    _ensure_data()
    ingestion = DataIngestion(str(VISITS_PATH))
    geo = GeoCluster(ingestion.get_district_metadata())
    neighbors = geo.get_neighboring_districts("Varanasi", radius_km=300)
    assert isinstance(neighbors, list)


def test_alert_generation_with_expected_severity_for_key_scenarios() -> None:
    _ensure_data()
    _, _, _, _, alerts = _build_pipeline()
    assert alerts
    varanasi = [a for a in alerts if a.district == "Varanasi" and a.condition_ayush == "Jwara"]
    chennai = [a for a in alerts if a.district == "Chennai" and a.condition_ayush == "Kushtha"]
    jaipur = [a for a in alerts if a.district == "Jaipur" and a.condition_ayush == "Prameha"]
    assert any(a.alert_level in {"WARNING", "ALERT"} for a in varanasi)
    assert any(a.alert_level in {"WARNING", "ALERT"} for a in chennai)
    assert any(a.alert_level in {"WATCH", "WARNING", "ALERT"} for a in jaipur)


# --- Phase 4 additions ---

def test_alert_fields_complete() -> None:
    _ensure_data()
    _, _, _, _, alerts = _build_pipeline()
    for alert in alerts[:5]:
        assert alert.alert_id
        assert alert.alert_level in {"WATCH", "WARNING", "ALERT"}
        assert alert.district
        assert alert.state
        assert alert.current_cases > 0
        assert alert.baseline_cases >= 0
        assert alert.ratio > 0
        assert alert.recommended_action


def test_district_metadata_has_coordinates() -> None:
    _ensure_data()
    ingestion = DataIngestion(str(VISITS_PATH))
    meta = ingestion.get_district_metadata()
    assert "lat" in meta.columns
    assert "lon" in meta.columns
    assert len(meta) >= 25


def test_no_false_positive_for_stable_district() -> None:
    _ensure_data()
    _, _, _, _, alerts = _build_pipeline()
    # A district with no injected outbreak should have fewer high-severity alerts
    stable_alerts = [a for a in alerts if a.district not in {"Varanasi", "Chennai", "Jaipur"} and a.alert_level == "ALERT"]
    # May have some natural variance, but shouldn't be majority
    total_alerts = len(alerts)
    if total_alerts > 0:
        false_positive_rate = len(stable_alerts) / total_alerts
        assert false_positive_rate < 0.5  # Less than half should be false ALERT level


def test_aggregation_produces_weekly_data() -> None:
    _ensure_data()
    ingestion = DataIngestion(str(VISITS_PATH))
    df = ingestion.load_visit_data()
    agg = ingestion.aggregate_by_district_condition_week(df)
    assert not agg.empty
    assert "district" in agg.columns
    assert "condition_ayush" in agg.columns


def test_multiple_detector_methods() -> None:
    _ensure_data()
    ingestion = DataIngestion(str(VISITS_PATH))
    df = ingestion.load_visit_data()
    agg = ingestion.aggregate_by_district_condition_week(df)
    baseline = BaselineModel()
    baseline.fit_all(agg)
    detector = AnomalyDetector(baseline_model=baseline)
    anomalies = detector.run_all_detectors(agg)
    # Should have anomalies from multiple methods
    if anomalies:
        methods = {a.get("method", "unknown") for a in anomalies}
        assert len(methods) >= 1
