"""Anomaly detection methods for RogaRadar."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.rogaradar.baseline_model import BaselineModel


class AnomalyDetector:
    """Run Prophet residual, Isolation Forest, and CUSUM-based detection."""

    def __init__(self, baseline_model: BaselineModel):
        self.baseline = baseline_model
        self._consecutive_tracker: Dict[tuple, int] = {}

    def detect_prophet_anomaly(self, district, condition, actual_cases, week) -> Optional[Dict]:
        """Flag when actual exceeds predicted upper bound for 2+ consecutive weeks."""
        preds = self.baseline.predict(district=district, condition=condition, periods=8)
        if preds.empty:
            return None
        row = preds[preds["ds"] == pd.Timestamp(week)]
        if row.empty:
            row = preds.tail(1)
        expected = float(row["yhat"].iloc[0])
        upper = float(row["yhat_upper"].iloc[0])
        ratio = actual_cases / max(1.0, expected)

        key = (district, condition)
        if actual_cases > upper:
            self._consecutive_tracker[key] = self._consecutive_tracker.get(key, 0) + 1
        else:
            self._consecutive_tracker[key] = 0

        if self._consecutive_tracker[key] >= 2:
            return {
                "district": district,
                "condition": condition,
                "week": pd.Timestamp(week),
                "method": "prophet",
                "actual": int(actual_cases),
                "expected": round(expected, 2),
                "ratio": round(ratio, 2),
            }
        return None

    def detect_isolation_forest(self, aggregated_data: pd.DataFrame) -> List[Dict]:
        """Detect anomalies using robust z-score fallback (IsolationForest-style output)."""
        if aggregated_data.empty:
            return []
        df = aggregated_data.sort_values(["district", "condition_ayush", "week_start"]).copy()
        df["week_over_week_change"] = df.groupby(["district", "condition_ayush"])["case_count"].diff().fillna(0)
        df["deviation_from_mean"] = df["case_count"] - df.groupby(["district", "condition_ayush"])["case_count"].transform("mean")
        month = df["week_start"].dt.month
        df["seasonal_factor"] = month.map(lambda m: 1.5 if m in [6, 7, 8, 9] else 1.0)
        grouped = df.groupby(["district", "condition_ayush"])
        std_case = grouped["case_count"].transform("std").fillna(1.0).replace(0.0, 1.0)
        std_change = grouped["week_over_week_change"].transform("std").fillna(1.0).replace(0.0, 1.0)
        z_case = (df["case_count"] - grouped["case_count"].transform("mean")) / std_case
        z_change = (df["week_over_week_change"] - grouped["week_over_week_change"].transform("mean")) / std_change
        flagged = df[(z_case > 3.5) | (z_change > 3.5)]

        results = []
        for _, row in flagged.iterrows():
            mean = float(max(1.0, df[(df["district"] == row["district"]) & (df["condition_ayush"] == row["condition_ayush"])]["case_count"].mean()))
            results.append(
                {
                    "district": row["district"],
                    "condition": row["condition_ayush"],
                    "week": row["week_start"],
                    "method": "isolation_forest",
                    "actual": int(row["case_count"]),
                    "expected": round(mean, 2),
                    "ratio": round(float(row["case_count"]) / mean, 2),
                }
            )
        return results

    def detect_cusum(self, time_series: pd.Series, threshold: float = 5.0) -> List[int]:
        """CUSUM indices for gradual shift detection."""
        if time_series.empty:
            return []
        values = time_series.values.astype(float)
        mean = np.mean(values)
        std = np.std(values) if np.std(values) > 0 else 1.0
        k = 0.5 * std
        pos = 0.0
        neg = 0.0
        out = []
        for i, val in enumerate(values):
            pos = max(0.0, pos + val - mean - k)
            neg = min(0.0, neg + val - mean + k)
            if pos > threshold or abs(neg) > threshold:
                out.append(i)
        return out

    def run_all_detectors(self, aggregated_data: pd.DataFrame) -> List[Dict]:
        """Run all detectors and return union of anomalies."""
        anomalies: List[Dict] = []

        # Prophet-like detector over each row.
        for _, row in aggregated_data.sort_values("week_start").iterrows():
            hit = self.detect_prophet_anomaly(
                district=row["district"],
                condition=row["condition_ayush"],
                actual_cases=int(row["case_count"]),
                week=row["week_start"],
            )
            if hit:
                anomalies.append(hit)

        # Isolation forest anomalies.
        anomalies.extend(self.detect_isolation_forest(aggregated_data))

        # CUSUM gradual-shift anomalies.
        for (district, condition), frame in aggregated_data.groupby(["district", "condition_ayush"]):
            series = frame.sort_values("week_start")["case_count"]
            idxs = self.detect_cusum(series, threshold=8.0)
            if not idxs:
                continue
            ordered = frame.sort_values("week_start").reset_index(drop=True)
            mean = float(max(1.0, ordered["case_count"].mean()))
            for idx in idxs:
                row = ordered.iloc[idx]
                anomalies.append(
                    {
                        "district": district,
                        "condition": condition,
                        "week": row["week_start"],
                        "method": "cusum",
                        "actual": int(row["case_count"]),
                        "expected": round(mean, 2),
                        "ratio": round(float(row["case_count"]) / mean, 2),
                    }
                )

        # Deduplicate by key.
        dedup = {}
        for a in anomalies:
            key = (a["district"], a["condition"], str(a["week"]), a["method"])
            dedup[key] = a
        all_anomalies = list(dedup.values())

        # Apply layered filtering to separate true outbreaks from noise.
        # True outbreaks are sustained (many weeks) while noise is single-week spikes.
        pair_methods: Dict[tuple, set] = {}
        pair_max_ratio: Dict[tuple, float] = {}
        pair_entries: Dict[tuple, int] = {}
        for a in all_anomalies:
            pair = (a["district"], a["condition"])
            pair_methods.setdefault(pair, set()).add(a["method"])
            pair_max_ratio[pair] = max(pair_max_ratio.get(pair, 0.0), a["ratio"])
            pair_entries[pair] = pair_entries.get(pair, 0) + 1

        # Confirmed if persistent (3+ detections) or extreme single spike (ratio >= 10)
        confirmed_pairs = {
            pair for pair in pair_methods
            if pair_entries.get(pair, 0) >= 3
            or pair_max_ratio.get(pair, 0.0) >= 10.0
        }

        # Filter to confirmed pairs with minimum ratio.
        MIN_RATIO = 2.0
        return [
            a for a in all_anomalies
            if (a["district"], a["condition"]) in confirmed_pairs and a["ratio"] >= MIN_RATIO
        ]


def detect_anomaly(current_cases: int, baseline_cases: float) -> Dict[str, float]:
    """Backward-compatible helper from scaffold."""
    base = baseline_cases if baseline_cases > 0 else 1.0
    ratio = current_cases / base
    return {"ratio": ratio, "is_anomaly": ratio >= 2.0}
