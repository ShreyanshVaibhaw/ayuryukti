"""Data ingestion and preprocessing for RogaRadar."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


class DataIngestion:
    """Load and aggregate AYUSH visit data for outbreak monitoring."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_visit_data(self) -> pd.DataFrame:
        """Load patient visit data from CSV and normalize fields."""
        path = Path(self.data_path)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "district", "state", "ayush_diagnosis_name", "icd10_code"])
        return df

    def aggregate_by_district_condition_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate visits per district-condition-week."""
        if df.empty:
            return pd.DataFrame(
                columns=["district", "state", "condition_ayush", "condition_icd10", "week_start", "case_count"]
            )
        tmp = df.copy()
        tmp["week_start"] = tmp["date"].dt.to_period("W-MON").dt.start_time
        grouped = (
            tmp.groupby(["district", "state", "ayush_diagnosis_name", "icd10_code", "week_start"], as_index=False)
            .size()
            .rename(columns={"ayush_diagnosis_name": "condition_ayush", "icd10_code": "condition_icd10", "size": "case_count"})
        )
        return grouped

    def get_district_metadata(self) -> pd.DataFrame:
        """Return fixed district-level coordinates for mapping and clustering."""
        rows: List[Dict] = [
            {"state": "Uttar Pradesh", "district": "Varanasi", "lat": 25.3176, "lon": 82.9739},
            {"state": "Uttar Pradesh", "district": "Lucknow", "lat": 26.8467, "lon": 80.9462},
            {"state": "Uttar Pradesh", "district": "Prayagraj", "lat": 25.4358, "lon": 81.8463},
            {"state": "Uttar Pradesh", "district": "Kanpur", "lat": 26.4499, "lon": 80.3319},
            {"state": "Uttar Pradesh", "district": "Agra", "lat": 27.1767, "lon": 78.0081},
            {"state": "Rajasthan", "district": "Jaipur", "lat": 26.9124, "lon": 75.7873},
            {"state": "Rajasthan", "district": "Jodhpur", "lat": 26.2389, "lon": 73.0243},
            {"state": "Rajasthan", "district": "Udaipur", "lat": 24.5854, "lon": 73.7125},
            {"state": "Rajasthan", "district": "Ajmer", "lat": 26.4499, "lon": 74.6399},
            {"state": "Rajasthan", "district": "Kota", "lat": 25.2138, "lon": 75.8648},
            {"state": "Tamil Nadu", "district": "Chennai", "lat": 13.0827, "lon": 80.2707},
            {"state": "Tamil Nadu", "district": "Coimbatore", "lat": 11.0168, "lon": 76.9558},
            {"state": "Tamil Nadu", "district": "Madurai", "lat": 9.9252, "lon": 78.1198},
            {"state": "Tamil Nadu", "district": "Salem", "lat": 11.6643, "lon": 78.1460},
            {"state": "Tamil Nadu", "district": "Tiruchirappalli", "lat": 10.7905, "lon": 78.7047},
            {"state": "Maharashtra", "district": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"state": "Maharashtra", "district": "Pune", "lat": 18.5204, "lon": 73.8567},
            {"state": "Maharashtra", "district": "Nagpur", "lat": 21.1458, "lon": 79.0882},
            {"state": "Maharashtra", "district": "Nashik", "lat": 19.9975, "lon": 73.7898},
            {"state": "Maharashtra", "district": "Aurangabad", "lat": 19.8762, "lon": 75.3433},
            {"state": "Karnataka", "district": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
            {"state": "Karnataka", "district": "Mysuru", "lat": 12.2958, "lon": 76.6394},
            {"state": "Karnataka", "district": "Hubli", "lat": 15.3647, "lon": 75.1240},
            {"state": "Karnataka", "district": "Mangalore", "lat": 12.9141, "lon": 74.8560},
            {"state": "Karnataka", "district": "Belagavi", "lat": 15.8497, "lon": 74.4977},
        ]
        return pd.DataFrame(rows)


def load_visit_data(path: str) -> pd.DataFrame:
    """Backward-compatible functional API."""
    return DataIngestion(path).load_visit_data()


def summarize_visits(df: pd.DataFrame) -> Dict[str, int]:
    """Backward-compatible summary helper."""
    return {"rows": int(len(df))}

