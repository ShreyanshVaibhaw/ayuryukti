"""Reporting export tests for Phase 4."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.common.models import EHROutput, OutbreakAlert, TreatmentRecommendation
from src.reporting.analytics_report import export_analytics_excel, generate_analytics_pdf
from src.reporting.ehr_report import export_ehr_json, generate_ehr_pdf
from src.reporting.recommendation_report import generate_recommendation_pdf
from src.reporting.surveillance_report import generate_surveillance_pdf


def test_ehr_report_exports(tmp_path) -> None:
    ehr = EHROutput(
        patient_demographics={"age": 35, "sex": "Female"},
        prakriti_assessment="Vata",
        chief_complaints=[{"complaint": "Constipation", "duration": "2 weeks", "severity": "Moderate"}],
        ayush_diagnosis={"code": "AYM_001", "name": "Vibandha"},
        icd10_diagnosis={"code": "K59.0", "name": "Constipation"},
        prescriptions=[{"formulation_name": "Triphala Churna", "dosage": "5g", "frequency": "BID", "duration": "14 days", "route": "Oral"}],
        lifestyle_advice=["Regular sleep"],
        dietary_advice=["Warm water"],
        yoga_advice=["Vajrasana"],
        follow_up="2 weeks",
        encounter_metadata={"encounter_id": "E001"},
    )

    pdf_path = tmp_path / "ehr.pdf"
    json_path = tmp_path / "ehr.json"
    generate_ehr_pdf(ehr, str(pdf_path))
    export_ehr_json(ehr, str(json_path))

    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert json_path.exists() and json_path.stat().st_size > 0


def test_recommendation_and_analytics_reports(tmp_path) -> None:
    rec = TreatmentRecommendation(
        recommendation_id="R001",
        encounter_id="E001",
        patient_prakriti="Vata",
        condition="Vibandha",
        recommended_formulations=[
            {
                "formulation_name": "Triphala Churna",
                "dosage": "5g",
                "score": 0.82,
                "classical_reference": "Charaka",
            }
        ],
        lifestyle_suggestions=["Daily routine"],
        yoga_suggestions=["Pavanamuktasana"],
        dietary_suggestions=["Warm meals"],
        contraindications=[],
        confidence=0.82,
        reasoning="Test reasoning",
        classical_references=["Charaka"],
        generated_at=datetime.now(timezone.utc),
    )

    rec_pdf = tmp_path / "recommendation.pdf"
    generate_recommendation_pdf(rec, str(rec_pdf))

    analytics_data = {
        "effectiveness": pd.DataFrame(
            [
                {
                    "formulation": "Triphala Churna",
                    "prakriti": "Vata",
                    "n_patients": 20,
                    "success_rate": 0.75,
                    "ci_low": 0.62,
                    "ci_high": 0.88,
                }
            ]
        ),
        "prakriti_response": {"Vata": {"Triphala Churna": 0.75}},
        "raw_outcomes": pd.DataFrame([{"outcome": "Improved"}]),
    }

    analytics_pdf = tmp_path / "analytics.pdf"
    analytics_xlsx = tmp_path / "analytics.xlsx"
    generate_analytics_pdf(analytics_data, str(analytics_pdf))
    export_analytics_excel(analytics_data, str(analytics_xlsx))

    assert rec_pdf.exists() and rec_pdf.stat().st_size > 0
    assert analytics_pdf.exists() and analytics_pdf.stat().st_size > 0
    assert analytics_xlsx.exists() and analytics_xlsx.stat().st_size > 0


def test_surveillance_report_exports(tmp_path) -> None:
    alerts = [
        OutbreakAlert(
            alert_id="A1",
            alert_level="ALERT",
            condition_ayush="Jwara",
            condition_icd10="R50",
            district="Varanasi",
            state="Uttar Pradesh",
            current_cases=50,
            baseline_cases=12.0,
            ratio=4.2,
            trend="Increasing",
            affected_centres=[],
            neighboring_districts_affected=["Prayagraj"],
            recommended_action="Investigate",
            generated_at=datetime.now(timezone.utc),
        )
    ]
    agg = pd.DataFrame(
        [
            {
                "district": "Varanasi",
                "state": "Uttar Pradesh",
                "condition_ayush": "Jwara",
                "condition_icd10": "R50",
                "week_start": pd.Timestamp("2025-08-04"),
                "case_count": 50,
            }
        ]
    )

    pdf = tmp_path / "surveillance.pdf"
    generate_surveillance_pdf(alerts, agg, str(pdf))
    assert pdf.exists() and pdf.stat().st_size > 0
