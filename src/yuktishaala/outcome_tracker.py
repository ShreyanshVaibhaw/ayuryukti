"""Outcome tracking for YuktiShaala learning engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from src.common.database import DatabaseManager


class OutcomeTracker:
    """Track treatment outcomes with database and in-memory fallback."""

    def __init__(self, db_uri: str = None):
        """Initialize tracker and optional DB connection."""
        self.outcomes: List[Dict] = []
        self.db_uri = db_uri
        self.db = DatabaseManager(db_uri=db_uri) if db_uri else None

    def record_outcome(
        self,
        encounter_id: str,
        patient_prakriti: str,
        condition_code: str,
        formulations: List[str],
        outcome: str,
        follow_up_days: int,
    ):
        """Record treatment outcome row(s) for formulations."""
        timestamp = datetime.now(timezone.utc).isoformat()
        for formulation in formulations:
            row = {
                "encounter_id": encounter_id,
                "patient_prakriti": patient_prakriti,
                "condition_code": condition_code,
                "formulation_name": formulation,
                "outcome": outcome,
                "timestamp": timestamp,
                "follow_up_days": follow_up_days,
            }
            self.outcomes.append(row)
            if self.db:
                self.db.insert_outcome(row)

    def get_outcomes_for_treatment(self, prakriti_type: str, condition_code: str, formulation_id: str) -> Dict:
        """Return aggregate outcome stats for a treatment arm."""
        rows = [
            x
            for x in self.outcomes
            if x["patient_prakriti"] == prakriti_type
            and x["condition_code"] == condition_code
            and x["formulation_name"] == formulation_id
        ]
        total = len(rows)
        improved = sum(1 for x in rows if x["outcome"] == "Improved")
        no_change = sum(1 for x in rows if x["outcome"] == "No Change")
        worsened = sum(1 for x in rows if x["outcome"] == "Worsened")
        success_rate = (improved / total) if total else 0.0
        return {
            "total": total,
            "improved": improved,
            "no_change": no_change,
            "worsened": worsened,
            "success_rate": success_rate,
        }

    def get_all_outcomes_for_condition(self, condition_code: str, prakriti_type: str = None) -> pd.DataFrame:
        """Get all outcomes filtered by condition and optional prakriti."""
        rows = [x for x in self.outcomes if x["condition_code"] == condition_code]
        if prakriti_type is not None:
            rows = [x for x in rows if x["patient_prakriti"] == prakriti_type]
        return pd.DataFrame(rows)

    def seed_from_synthetic(self, patient_visits_path: str):
        """Seed outcomes from synthetic visits where follow-up outcome is available."""
        df = pd.read_csv(patient_visits_path)
        out = df[df["outcome"].notna() & (df["outcome"].astype(str).str.len() > 0)].copy()
        for _, row in out.iterrows():
            formulations = [x.strip() for x in str(row.get("formulations_prescribed", "")).split(";") if x.strip()]
            if not formulations:
                continue
            self.record_outcome(
                encounter_id=str(row.get("visit_id")),
                patient_prakriti=str(row.get("prakriti_type")),
                condition_code=str(row.get("ayush_diagnosis_code")),
                formulations=formulations,
                outcome=str(row.get("outcome")),
                follow_up_days=14,
            )

    def list_all(self) -> List[Dict]:
        """Return all outcome rows (backward-compatible helper)."""
        return list(self.outcomes)

