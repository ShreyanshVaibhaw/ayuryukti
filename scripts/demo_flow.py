"""End-to-end demo flow runner for AyurYukti video-ready walkthrough."""

from __future__ import annotations

import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ALERT_LEVELS, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from scripts.generate_synthetic_data import main as generate_synthetic_data
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
from src.yuktishaala.analytics import TreatmentAnalytics
from src.yuktishaala.contextual_bandit import ThompsonSamplingBandit
from src.yuktishaala.outcome_tracker import OutcomeTracker


KB = ROOT / "data" / "knowledge_base"
VISITS = ROOT / "data" / "synthetic" / "patient_visits.csv"


def _ensure_data() -> None:
    if not VISITS.exists():
        generate_synthetic_data()


def _header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _build_vaksetu():
    mapper = CodeMapper(
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
        icd10_mapping_path=str(KB / "icd10_mapping.json"),
    )
    ner = MedicalNEREngine(
        llm_client=None,
        vocabulary_path=str(ROOT / "data" / "ayush_vocabulary.json"),
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
    )
    return ner, EHRGenerator(ner_engine=ner, code_mapper=mapper)


def _build_recommender() -> tuple[RecommendationEngine, OutcomeTracker, ThompsonSamplingBandit, TreatmentAnalytics]:
    kg = AyushKnowledgeGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    kg.setup_schema()
    kg.seed_from_json(
        formulations_path=str(KB / "formulations.json"),
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
        prakriti_rules_path=str(KB / "prakriti_rules.json"),
    )

    tracker = OutcomeTracker()
    bandit = ThompsonSamplingBandit(exploration_rate=0.15)
    tracker.seed_from_synthetic(str(VISITS))

    for row in tracker.list_all():
        reward = 1.0 if row["outcome"] == "Improved" else 0.5 if row["outcome"] == "No Change" else 0.0
        bandit.update(row["patient_prakriti"], row["condition_code"], row["formulation_name"], reward)

    analytics = TreatmentAnalytics(tracker, bandit)
    engine = RecommendationEngine(knowledge_graph=kg, llm_client=None, outcome_tracker=tracker, bandit=bandit)
    return engine, tracker, bandit, analytics


def run_demo() -> None:
    """Complete demo flow that runs in <3 minutes for recording."""
    start_all = time.perf_counter()
    _ensure_data()

    # Scene 1: VakSetu
    _header("Scene 1: VakSetu Demo (Voice-to-EHR)")
    ner, generator = _build_vaksetu()
    scene_start = time.perf_counter()

    raw = SAMPLE_TRANSCRIPTS[0]
    corrected = ner.correct_transcript(raw, "hi")
    ehr = generator.generate_from_transcript(corrected, "hi", "C001", "D001")

    print("Raw transcript:")
    print(raw)
    print("\nCorrected transcript:")
    print(corrected)
    print("\nExtracted EHR:")
    print(f"- Prakriti: {ehr.prakriti_assessment}")
    print(f"- AYUSH Diagnosis: {ehr.ayush_diagnosis.get('code')} | {ehr.ayush_diagnosis.get('name')}")
    print(f"- ICD-10: {ehr.icd10_diagnosis.get('code')} | {ehr.icd10_diagnosis.get('name')}")
    print(f"- Prescriptions: {len(ehr.prescriptions)}")

    elapsed = time.perf_counter() - scene_start
    print(f"Generated AHMIS-compatible EHR in {elapsed:.1f} seconds")

    # Scene 2: PrakritiMitra
    _header("Scene 2: PrakritiMitra Demo (Personalized Recommendations)")
    scene_start = time.perf_counter()

    engine, tracker, bandit, analytics = _build_recommender()
    rec = engine.recommend(
        patient_prakriti=ehr.prakriti_assessment or "Vata",
        condition_ayush_code=ehr.ayush_diagnosis.get("name", "Vibandha"),
        patient_age=int(ehr.patient_demographics.get("age") or 35),
        patient_sex=ehr.patient_demographics.get("sex") or "Female",
        existing_prescriptions=[x.get("formulation_name") for x in ehr.prescriptions],
    )

    print(f"Context: Prakriti={rec.patient_prakriti}, Condition={rec.condition}")
    print("Top 3 formulations:")
    for idx, item in enumerate(rec.recommended_formulations[:3], start=1):
        stats = tracker.get_outcomes_for_treatment(rec.patient_prakriti, rec.condition, item["formulation_name"])
        print(
            f"{idx}. {item['formulation_name']} | score={item['score']:.3f} | "
            f"ref={item.get('classical_reference', '-') or '-'} | "
            f"outcomes={stats['success_rate'] * 100:.0f}% improved (n={stats['total']})"
        )

    print("\nLifestyle:")
    print(", ".join(rec.lifestyle_suggestions[:4]) if rec.lifestyle_suggestions else "No lifestyle suggestions")
    print("Diet:")
    print(", ".join(rec.dietary_suggestions[:4]) if rec.dietary_suggestions else "No diet suggestions")
    print("Yoga:")
    print(", ".join(rec.yoga_suggestions[:4]) if rec.yoga_suggestions else "No yoga suggestions")

    short_reasoning = rec.reasoning.strip().replace("\n", " ")
    print("\nReasoning:")
    print(short_reasoning[:380] + ("..." if len(short_reasoning) > 380 else ""))
    print(f"Scene runtime: {time.perf_counter() - scene_start:.1f}s")

    # Scene 3: RogaRadar
    _header("Scene 3: RogaRadar Demo (Outbreak Detection)")
    scene_start = time.perf_counter()

    ingestion = DataIngestion(str(VISITS))
    visits = ingestion.load_visit_data()
    agg = ingestion.aggregate_by_district_condition_week(visits)

    baseline = BaselineModel()
    baseline.fit_all(agg)

    detector = AnomalyDetector(baseline)
    anomalies = detector.run_all_detectors(agg)

    geo = GeoCluster(ingestion.get_district_metadata())
    clusters = geo.cluster_anomalies(anomalies)
    alerts = AlertGenerator(ALERT_LEVELS).generate_alerts(anomalies, clusters)

    print(f"Detected {len(anomalies)} outbreak anomalies")
    key_alerts = [a for a in alerts if a.district == "Varanasi" and a.condition_ayush == "Jwara"]
    if key_alerts:
        top = sorted(key_alerts, key=lambda x: x.ratio, reverse=True)[0]
        print(
            "Varanasi Jwara spike: "
            f"level={top.alert_level}, cases={top.current_cases}, baseline={top.baseline_cases:.1f}, ratio={top.ratio:.2f}"
        )
    else:
        print("Varanasi Jwara spike alert not found in this run.")

    neighbors = geo.get_neighboring_districts("Varanasi", radius_km=350)
    print(f"Neighboring districts monitored: {', '.join(neighbors[:6]) if neighbors else 'None'}")
    print(f"Scene runtime: {time.perf_counter() - scene_start:.1f}s")

    # Scene 4: YuktiShaala
    _header("Scene 4: YuktiShaala Demo (Learning Engine)")
    scene_start = time.perf_counter()

    stats = tracker.get_outcomes_for_treatment("Vata", "Vibandha", "Triphala Churna")
    curve = analytics.get_learning_curve("Vata", "Vibandha", "Triphala Churna")
    total_outcomes = len(tracker.list_all())

    print("Treatment outcome stats: Triphala Churna | Vata | Vibandha")
    print(
        f"- n={stats['total']} | Improved={stats['improved']} | "
        f"No Change={stats['no_change']} | Worsened={stats['worsened']} | Success={stats['success_rate'] * 100:.1f}%"
    )

    if not curve.empty:
        first = curve.iloc[0]
        last = curve.iloc[-1]
        print(
            f"Learning curve confidence: mean {first['mean']:.3f} -> {last['mean']:.3f}; "
            f"CI width {(first['ci_high'] - first['ci_low']):.3f} -> {(last['ci_high'] - last['ci_low']):.3f}"
        )

    print(f"System has learned from {total_outcomes:,} patient outcomes")
    print("Recommendations improve with every patient")
    print(f"Scene runtime: {time.perf_counter() - scene_start:.1f}s")

    total = time.perf_counter() - start_all
    _header("Demo Complete")
    print(f"Total runtime: {total:.1f}s")


if __name__ == "__main__":
    run_demo()
