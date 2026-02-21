"""Synthetic clinical data generator for RogaRadar Phase 2."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_VISITS = ROOT / "data" / "synthetic" / "patient_visits.csv"
OUT_SCENARIOS = ROOT / "data" / "synthetic" / "outbreak_scenarios.csv"
OUT_PRAKRITI_TRAIN = ROOT / "data" / "prakriti" / "training_data.csv"


DISTRICT_BY_STATE = {
    "Uttar Pradesh": ["Varanasi", "Lucknow", "Prayagraj", "Kanpur", "Agra"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Ajmer", "Kota"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubli", "Mangalore", "Belagavi"],
}


CONDITIONS: List[Tuple[str, str, float]] = [
    ("Amlapitta", "K21", 0.18),
    ("Sandhivata", "M15-M19", 0.15),
    ("Vibandha", "K59.0", 0.12),
    ("Jwara", "R50", 0.10),
    ("Prameha", "E11", 0.08),
    ("Sthaulya", "E66", 0.06),
    ("Raktachapa", "I10", 0.05),
    ("Kushtha", "L30", 0.04),
    ("Kasa", "R05", 0.04),
    ("Ashmari", "N20", 0.06),
    ("Mutrakricchra", "R30", 0.06),
    ("Pratishyaya", "J31", 0.06),
    ("Aruchi", "R63.0", 0.06),
]


PRAKRITI_TYPES = ["Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Sama"]
FORMULATION_POOL = [
    "Triphala Churna",
    "Abhayarishta",
    "Avipattikar Churna",
    "Yogaraja Guggulu",
    "Maharasnadi Kashaya",
    "Chandraprabha Vati",
    "Gokshuradi Guggulu",
    "Sudarshana Churna",
    "Khadirarishta",
]

FORMULATION_BY_CONDITION = {
    "Amlapitta": ["Avipattikar Churna", "Kamdudha Rasa", "Sutashekhar Rasa"],
    "Sandhivata": ["Yogaraja Guggulu", "Maharasnadi Kashaya", "Kottamchukkadi Taila"],
    "Vibandha": ["Triphala Churna", "Abhayarishta", "Panchsakar Churna"],
    "Jwara": ["Sudarshana Churna", "Samshamani Vati", "Guduchi Satva"],
    "Prameha": ["Chandraprabha Vati", "Nishamalaki", "Triphala Guggulu"],
    "Sthaulya": ["Medohar Guggulu", "Triphala Guggulu", "Arogyavardhini Vati"],
    "Raktachapa": ["Sarpagandha Vati", "Arjunarishta", "Punarnava Mandoor"],
    "Kushtha": ["Khadirarishta", "Gandhaka Rasayana", "Neem Churna"],
    "Kasa": ["Sitopaladi Churna", "Talisadi Churna", "Vasavaleha"],
    "Ashmari": ["Gokshuradi Guggulu", "Pashanabheda", "Varunadi Kwath"],
    "Mutrakricchra": ["Punarnavadi Kashaya", "Chandanasava", "Yavakshar"],
    "Pratishyaya": ["Talisadi Churna", "Haridra Khanda", "Vyoshadi Vati"],
    "Aruchi": ["Chitrakadi Vati", "Lavanbhaskar Churna", "Jeerakarishta"],
}


def _weighted_condition() -> Tuple[str, str]:
    items = [(name, icd) for name, icd, _ in CONDITIONS]
    weights = [w for _, _, w in CONDITIONS]
    return random.choices(items, weights=weights, k=1)[0]


def _seasonal_adjustment(condition: str, dt: datetime) -> float:
    month = dt.month
    if condition == "Jwara" and month in [7, 8, 9]:
        return 2.2
    if condition == "Pratishyaya" and month in [11, 12, 1]:
        return 2.0
    return 1.0


def _geo_adjustment(condition: str, district: str, state: str) -> float:
    northern = {"Varanasi", "Lucknow", "Prayagraj", "Kanpur", "Agra", "Jaipur", "Jodhpur", "Udaipur", "Ajmer", "Kota"}
    if condition == "Sandhivata" and district in northern:
        return 1.5
    if condition == "Kushtha" and state == "Tamil Nadu":
        return 1.2
    return 1.0


def _inject_anomaly_multiplier(condition: str, district: str, dt: datetime) -> float:
    # 1) Jwara spike in Varanasi, August 2025 — 4x.
    if condition == "Jwara" and district == "Varanasi" and dt.year == 2025 and dt.month == 8:
        return 4.0
    # 2) Kushtha spike in Chennai, June 2025 — 3x.
    if condition == "Kushtha" and district == "Chennai" and dt.year == 2025 and dt.month == 6:
        return 3.0
    # 3) Gradual Prameha increase in Jaipur Oct-Dec 2025 — 2.5x by December.
    if condition == "Prameha" and district == "Jaipur" and dt.year == 2025 and dt.month in [10, 11, 12]:
        return {10: 1.6, 11: 2.0, 12: 2.5}[dt.month]
    return 1.0


def generate_patient_visits(n_records: int = 10000, n_centres: int = 50, n_districts: int = 25, n_states: int = 5):
    """Generate realistic synthetic AYUSH patient visit data."""
    random.seed(26)
    start = datetime(2025, 1, 1)
    end = datetime(2025, 12, 31)
    date_range_days = (end - start).days

    states = list(DISTRICT_BY_STATE.keys())[:n_states]
    district_state_pairs = []
    for state in states:
        for district in DISTRICT_BY_STATE[state]:
            district_state_pairs.append((district, state))
    district_state_pairs = district_state_pairs[:n_districts]

    centres = [f"C{idx:03d}" for idx in range(1, n_centres + 1)]
    age_groups = ["0-14", "15-29", "30-44", "45-59", "60+"]
    sexes = ["M", "F"]
    outcomes = ["Improved", "No Change", "Worsened"]

    records: List[Dict] = []
    i = 1
    while len(records) < n_records:
        district, state = random.choice(district_state_pairs)
        centre_id = random.choice(centres)
        dt = start + timedelta(days=random.randint(0, date_range_days))
        condition, icd10 = _weighted_condition()

        multiplier = _seasonal_adjustment(condition, dt) * _geo_adjustment(condition, district, state) * _inject_anomaly_multiplier(condition, district, dt)
        # Stochastic accept/reject to implement multipliers.
        if random.random() > min(1.0, 0.28 * multiplier):
            continue

        candidates = FORMULATION_BY_CONDITION.get(condition, FORMULATION_POOL)
        prescribed = random.sample(candidates, k=min(2, len(candidates)))
        has_followup = random.random() < 0.4

        records.append(
            {
                "visit_id": f"V{i:06d}",
                "patient_id": f"P{uuid.uuid4().hex[:10]}",
                "centre_id": centre_id,
                "district": district,
                "state": state,
                "date": dt.strftime("%Y-%m-%d"),
                "age_group": random.choices(age_groups, weights=[0.1, 0.22, 0.28, 0.24, 0.16], k=1)[0],
                "sex": random.choice(sexes),
                "ayush_diagnosis_code": condition,
                "ayush_diagnosis_name": condition,
                "icd10_code": icd10,
                "prakriti_type": random.choices(
                    PRAKRITI_TYPES,
                    weights=[0.16, 0.16, 0.16, 0.18, 0.14, 0.11, 0.09],
                    k=1,
                )[0],
                "formulations_prescribed": "; ".join(prescribed),
                "outcome": random.choices(outcomes, weights=[0.60, 0.25, 0.15], k=1)[0] if has_followup else "",
            }
        )
        i += 1

    # Force clear anomaly signatures while preserving total record count.
    def _rewrite_rows(target_count: int, district: str, state: str, condition: str, icd10: str, month: int):
        picks = random.sample(range(len(records)), k=min(target_count, len(records)))
        for idx in picks:
            day = random.randint(1, 28)
            dt = datetime(2025, month, day)
            records[idx]["district"] = district
            records[idx]["state"] = state
            records[idx]["date"] = dt.strftime("%Y-%m-%d")
            records[idx]["ayush_diagnosis_code"] = condition
            records[idx]["ayush_diagnosis_name"] = condition
            records[idx]["icd10_code"] = icd10

    # 1) Varanasi Jwara August strong spike.
    _rewrite_rows(target_count=240, district="Varanasi", state="Uttar Pradesh", condition="Jwara", icd10="R50", month=8)
    # 2) Chennai Kushtha June strong spike.
    _rewrite_rows(target_count=180, district="Chennai", state="Tamil Nadu", condition="Kushtha", icd10="L30", month=6)
    # 3) Jaipur Prameha gradual rise Oct-Dec.
    _rewrite_rows(target_count=70, district="Jaipur", state="Rajasthan", condition="Prameha", icd10="E11", month=10)
    _rewrite_rows(target_count=110, district="Jaipur", state="Rajasthan", condition="Prameha", icd10="E11", month=11)
    _rewrite_rows(target_count=150, district="Jaipur", state="Rajasthan", condition="Prameha", icd10="E11", month=12)

    return pd.DataFrame(records)


def generate_outbreak_scenarios():
    """Create expected outbreak detection ground truth."""
    rows = [
        {
            "scenario_id": "OUTBREAK_001",
            "district": "Varanasi",
            "state": "Uttar Pradesh",
            "condition_ayush": "Jwara",
            "start_date": "2025-08-01",
            "end_date": "2025-08-31",
            "expected_multiplier": 4.0,
            "type": "sudden_spike",
        },
        {
            "scenario_id": "OUTBREAK_002",
            "district": "Chennai",
            "state": "Tamil Nadu",
            "condition_ayush": "Kushtha",
            "start_date": "2025-06-01",
            "end_date": "2025-06-30",
            "expected_multiplier": 3.0,
            "type": "sudden_spike",
        },
        {
            "scenario_id": "OUTBREAK_003",
            "district": "Jaipur",
            "state": "Rajasthan",
            "condition_ayush": "Prameha",
            "start_date": "2025-10-01",
            "end_date": "2025-12-31",
            "expected_multiplier": 2.5,
            "type": "gradual_rise",
        },
    ]
    return pd.DataFrame(rows)


def generate_prakriti_training_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate 5,000 synthetic Prakriti training samples (q1..q30 + label)."""
    rng = np.random.default_rng(26)
    dist = [
        ("Vata", 0.15),
        ("Pitta", 0.15),
        ("Kapha", 0.15),
        ("Vata-Pitta", 0.18),
        ("Pitta-Kapha", 0.15),
        ("Vata-Kapha", 0.12),
        ("Sama", 0.10),
    ]

    rows = []

    def clip(v: float) -> int:
        return int(max(1, min(5, round(v))))

    for label, frac in dist:
        count = int(n_samples * frac)
        for _ in range(count):
            base = np.full(30, 3.0)
            noise = rng.normal(0, 0.5, size=30)
            if label == "Vata":
                base[[0, 3, 5, 8, 15, 19, 27]] = [2, 2, 2, 4.8, 4.5, 4.2, 4.0]
            elif label == "Pitta":
                base[[1, 4, 7, 10, 13, 16, 25]] = [4.2, 4.5, 4.0, 4.6, 4.5, 4.3, 4.0]
            elif label == "Kapha":
                base[[2, 5, 8, 11, 14, 17, 26]] = [4.8, 4.8, 2.0, 4.8, 2.0, 2.2, 2.5]
            elif label == "Vata-Pitta":
                base[[0, 1, 3, 4, 8, 13, 16]] = [2.5, 3.7, 2.7, 3.8, 4.0, 4.0, 3.8]
            elif label == "Pitta-Kapha":
                base[[1, 2, 4, 5, 10, 14, 20]] = [3.8, 4.0, 4.0, 4.2, 3.8, 3.2, 3.2]
            elif label == "Vata-Kapha":
                base[[0, 2, 3, 5, 8, 15, 26]] = [3.8, 3.8, 3.6, 3.8, 3.1, 3.7, 3.0]
            values = [clip(v) for v in (base + noise)]
            row = {f"q{i+1}": values[i] for i in range(30)}
            row["prakriti"] = label
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=26).reset_index(drop=True)
    return df


def main() -> None:
    """Generate and persist synthetic datasets."""
    visits = generate_patient_visits()
    scenarios = generate_outbreak_scenarios()
    prakriti_train = generate_prakriti_training_data(5000)
    OUT_VISITS.parent.mkdir(parents=True, exist_ok=True)
    OUT_PRAKRITI_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    visits.to_csv(OUT_VISITS, index=False)
    scenarios.to_csv(OUT_SCENARIOS, index=False)
    prakriti_train.to_csv(OUT_PRAKRITI_TRAIN, index=False)
    print(f"patient_visits={len(visits)}")
    print(f"outbreak_scenarios={len(scenarios)}")
    print(f"prakriti_training_rows={len(prakriti_train)}")


if __name__ == "__main__":
    main()
