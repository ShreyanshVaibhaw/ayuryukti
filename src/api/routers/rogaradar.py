"""RogaRadar API router — Disease surveillance endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from config import ALERT_LEVELS
from src.rogaradar.alert_generator import AlertGenerator
from src.rogaradar.anomaly_detector import AnomalyDetector
from src.rogaradar.baseline_model import BaselineModel
from src.rogaradar.data_ingestion import DataIngestion
from src.rogaradar.geo_cluster import GeoCluster

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
VISITS_PATH = ROOT / "data" / "synthetic" / "patient_visits.csv"


class AlertResponse(BaseModel):
    alert_id: str
    alert_level: str
    condition_ayush: str
    condition_icd10: str
    district: str
    state: str
    current_cases: int
    baseline_cases: float
    ratio: float
    trend: str
    recommended_action: str


class DashboardResponse(BaseModel):
    total_districts: int
    active_alerts: int
    conditions_under_watch: int
    data_points: int
    alerts: List[AlertResponse]


def _run_pipeline():
    ingestion = DataIngestion(str(VISITS_PATH))
    visits = ingestion.load_visit_data()
    agg = ingestion.aggregate_by_district_condition_week(visits)
    baseline = BaselineModel()
    if not agg.empty:
        baseline.fit_all(agg)
    detector = AnomalyDetector(baseline_model=baseline)
    anomalies = detector.run_all_detectors(agg) if not agg.empty else []
    district_meta = ingestion.get_district_metadata()
    geo = GeoCluster(district_metadata=district_meta)
    clusters = geo.cluster_anomalies(anomalies)
    alerts = AlertGenerator(ALERT_LEVELS).generate_alerts(anomalies, clusters)
    return visits, agg, anomalies, alerts, district_meta


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    district: Optional[str] = Query(None, description="Filter by district"),
    severity: Optional[str] = Query(None, description="Filter by severity (WATCH, WARNING, ALERT)"),
) -> List[AlertResponse]:
    """Get active surveillance alerts with optional filters."""
    _, _, _, alerts, _ = _run_pipeline()
    filtered = alerts
    if district:
        filtered = [a for a in filtered if a.district.lower() == district.lower()]
    if severity:
        filtered = [a for a in filtered if a.alert_level == severity.upper()]
    return [
        AlertResponse(
            alert_id=a.alert_id,
            alert_level=a.alert_level,
            condition_ayush=a.condition_ayush,
            condition_icd10=a.condition_icd10,
            district=a.district,
            state=a.state,
            current_cases=a.current_cases,
            baseline_cases=a.baseline_cases,
            ratio=a.ratio,
            trend=a.trend,
            recommended_action=a.recommended_action,
        )
        for a in filtered
    ]


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard() -> DashboardResponse:
    """Get full surveillance dashboard summary."""
    visits, agg, _, alerts, district_meta = _run_pipeline()
    return DashboardResponse(
        total_districts=int(district_meta["district"].nunique()),
        active_alerts=len(alerts),
        conditions_under_watch=len({a.condition_ayush for a in alerts}),
        data_points=len(agg),
        alerts=[
            AlertResponse(
                alert_id=a.alert_id,
                alert_level=a.alert_level,
                condition_ayush=a.condition_ayush,
                condition_icd10=a.condition_icd10,
                district=a.district,
                state=a.state,
                current_cases=a.current_cases,
                baseline_cases=a.baseline_cases,
                ratio=a.ratio,
                trend=a.trend,
                recommended_action=a.recommended_action,
            )
            for a in alerts
        ],
    )


@router.get("/districts")
async def list_districts() -> Dict[str, Any]:
    """List monitored districts with metadata."""
    ingestion = DataIngestion(str(VISITS_PATH))
    meta = ingestion.get_district_metadata()
    return {
        "total": len(meta),
        "districts": meta.to_dict(orient="records"),
    }
