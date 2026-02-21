"""End-to-end integration tests for AyurYukti pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generate_synthetic_data import main as synth_main
from scripts.seed_knowledge_base import seed_all
from src.prakritimitra.knowledge_graph import AyushKnowledgeGraph
from src.prakritimitra.recommendation_engine import RecommendationEngine
from src.rogaradar.alert_generator import AlertGenerator
from src.rogaradar.anomaly_detector import AnomalyDetector
from src.rogaradar.baseline_model import BaselineModel
from src.rogaradar.data_ingestion import DataIngestion
from src.rogaradar.geo_cluster import GeoCluster
from src.vaksetu.code_mapper import CodeMapper
from src.vaksetu.ehr_generator import EHRGenerator
from src.vaksetu.medical_ner import MedicalNEREngine
from src.vaksetu.speech_engine import SAMPLE_TRANSCRIPTS
from src.yuktishaala.contextual_bandit import ThompsonSamplingBandit
from src.yuktishaala.outcome_tracker import OutcomeTracker


ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "knowledge_base"
SYN = ROOT / "data" / "synthetic"
VISITS = SYN / "patient_visits.csv"
SCENARIOS = SYN / "outbreak_scenarios.csv"
VOCAB = ROOT / "data" / "ayush_vocabulary.json"


def _ensure_data() -> None:
    synth_main()


def _build_vaksetu() -> EHRGenerator:
    mapper = CodeMapper(
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
        icd10_mapping_path=str(KB / "icd10_mapping.json"),
    )
    ner = MedicalNEREngine(
        llm_client=None,
        vocabulary_path=str(VOCAB),
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
    )
    return EHRGenerator(ner_engine=ner, code_mapper=mapper)


def _build_recommender(tracker: OutcomeTracker | None = None, bandit: ThompsonSamplingBandit | None = None):
    kg = AyushKnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password="ayuryukti2026")
    kg.setup_schema()
    kg.seed_from_json(
        formulations_path=str(KB / "formulations.json"),
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
        prakriti_rules_path=str(KB / "prakriti_rules.json"),
    )
    return RecommendationEngine(knowledge_graph=kg, llm_client=None, outcome_tracker=tracker, bandit=bandit), kg


def _run_surveillance_pipeline():
    ingestion = DataIngestion(str(VISITS))
    visits = ingestion.load_visit_data()
    agg = ingestion.aggregate_by_district_condition_week(visits)

    baseline = BaselineModel()
    baseline.fit_all(agg)
    detector = AnomalyDetector(baseline)
    anomalies = detector.run_all_detectors(agg)

    geo = GeoCluster(ingestion.get_district_metadata())
    clusters = geo.cluster_anomalies(anomalies)
    alerts = AlertGenerator({"WATCH": 1.5, "WARNING": 2.0, "ALERT": 3.0}).generate_alerts(anomalies, clusters)

    return visits, agg, anomalies, alerts


def test_full_pipeline() -> None:
    """Complete integration flow from data generation to learning update."""
    _ensure_data()
    seed_all()

    visits = pd.read_csv(VISITS)
    assert len(visits) == 10000

    generator = _build_vaksetu()
    ehr = generator.generate_from_transcript(SAMPLE_TRANSCRIPTS[0], "hi", "C001", "D001")
    payload = generator.to_ahmis_json(ehr)

    tracker = OutcomeTracker()
    tracker.seed_from_synthetic(str(VISITS))
    bandit = ThompsonSamplingBandit(exploration_rate=0.0)
    for row in tracker.list_all():
        reward = 1.0 if row["outcome"] == "Improved" else 0.5 if row["outcome"] == "No Change" else 0.0
        bandit.update(row["patient_prakriti"], row["condition_code"], row["formulation_name"], reward)

    engine, kg = _build_recommender(tracker=tracker, bandit=bandit)
    rec = engine.recommend(
        patient_prakriti=ehr.prakriti_assessment or "Vata",
        condition_ayush_code=ehr.ayush_diagnosis["name"],
        patient_age=int(ehr.patient_demographics.get("age") or 35),
        patient_sex=ehr.patient_demographics.get("sex") or "Female",
    )

    visits_df, agg, anomalies, alerts = _run_surveillance_pipeline()
    assert not visits_df.empty
    assert not agg.empty
    assert anomalies
    assert alerts

    pre_trials = sum(x["n_trials"] for x in bandit.get_arm_stats(rec.patient_prakriti, rec.condition))
    ok = engine.record_feedback(rec.encounter_id, "Improved")
    post_trials = sum(x["n_trials"] for x in bandit.get_arm_stats(rec.patient_prakriti, rec.condition))

    assert ok
    assert post_trials > pre_trials

    # Schema checks.
    assert {"patientDemographics", "diagnosis", "prescriptions", "metadata"}.issubset(payload.keys())
    assert rec.recommended_formulations
    assert all("ratio" in a for a in anomalies)
    kg.close()


def test_vaksetu_to_prakritimitra() -> None:
    """VakSetu output should directly feed PrakritiMitra input."""
    _ensure_data()
    generator = _build_vaksetu()
    ehr = generator.generate_from_transcript(SAMPLE_TRANSCRIPTS[1], "en", "C002", "D002")

    engine, kg = _build_recommender()
    rec = engine.recommend(
        patient_prakriti=ehr.prakriti_assessment or "Pitta-Kapha",
        condition_ayush_code=ehr.ayush_diagnosis["name"],
        patient_age=int(ehr.patient_demographics.get("age") or 52),
        patient_sex=ehr.patient_demographics.get("sex") or "Male",
        existing_prescriptions=[p.get("formulation_name") for p in ehr.prescriptions],
    )

    assert rec.condition == ehr.ayush_diagnosis["name"]
    assert rec.patient_prakriti in {ehr.prakriti_assessment, "Pitta-Kapha"}
    assert len(rec.recommended_formulations) >= 1
    kg.close()


def test_outbreak_detection_accuracy() -> None:
    """All three injected anomalies should be detected with expected severities."""
    _ensure_data()
    _ = pd.read_csv(SCENARIOS)

    _, _, anomalies, alerts = _run_surveillance_pipeline()
    a_df = pd.DataFrame(anomalies)
    assert ((a_df["district"] == "Varanasi") & (a_df["condition"] == "Jwara")).any()
    assert ((a_df["district"] == "Chennai") & (a_df["condition"] == "Kushtha")).any()
    assert ((a_df["district"] == "Jaipur") & (a_df["condition"] == "Prameha")).any()

    varanasi = [a for a in alerts if a.district == "Varanasi" and a.condition_ayush == "Jwara"]
    chennai = [a for a in alerts if a.district == "Chennai" and a.condition_ayush == "Kushtha"]
    jaipur = [a for a in alerts if a.district == "Jaipur" and a.condition_ayush == "Prameha"]

    assert any(a.alert_level in {"WARNING", "ALERT"} for a in varanasi)
    assert any(a.alert_level in {"WARNING", "ALERT"} for a in chennai)
    assert any(a.alert_level in {"WATCH", "WARNING", "ALERT"} for a in jaipur)


def test_learning_engine_improvement() -> None:
    """After feedback accumulation, ranking should shift from cold KG ordering."""
    tracker = OutcomeTracker()
    bandit = ThompsonSamplingBandit(exploration_rate=0.0)
    engine, kg = _build_recommender(tracker=tracker, bandit=bandit)

    baseline = engine.recommend(
        patient_prakriti="Vata",
        condition_ayush_code="Prameha",
        patient_age=45,
        patient_sex="Male",
    )
    before_names = [x["formulation_name"] for x in baseline.recommended_formulations[:3]]

    candidate_names = [x["formulation_name"] for x in baseline.recommended_formulations]
    target = candidate_names[-1]
    top = candidate_names[0]

    for i in range(100):
        tracker.record_outcome(f"E_POS_{i}", "Vata", "Prameha", [target], "Improved", 14)
        bandit.update("Vata", "Prameha", target, 1.0)
        tracker.record_outcome(f"E_NEG_{i}", "Vata", "Prameha", [top], "Worsened", 14)
        bandit.update("Vata", "Prameha", top, 0.0)

    after = engine.recommend(
        patient_prakriti="Vata",
        condition_ayush_code="Prameha",
        patient_age=45,
        patient_sex="Male",
    )
    after_names = [x["formulation_name"] for x in after.recommended_formulations[:3]]

    assert before_names != after_names
    assert target in after_names
    kg.close()
