"""Alert generation for outbreak anomalies."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List

from src.common.models import OutbreakAlert


class AlertGenerator:
    """Convert anomaly detections into structured alert objects."""

    def __init__(self, alert_levels: Dict[str, float]):
        self.levels = alert_levels

    @staticmethod
    def _raise_level(level: str) -> str:
        order = ["WATCH", "WARNING", "ALERT"]
        if level not in order:
            return level
        idx = min(len(order) - 1, order.index(level) + 1)
        return order[idx]

    def _level_from_ratio(self, ratio: float) -> str | None:
        if ratio >= self.levels["ALERT"]:
            return "ALERT"
        if ratio >= self.levels["WARNING"]:
            return "WARNING"
        if ratio >= self.levels["WATCH"]:
            return "WATCH"
        return None

    def _recommended_action(self, level: str, district_text: str) -> str:
        if level == "WATCH":
            return f"Increase surveillance in {district_text}. Monitor for 1 more week."
        if level == "WARNING":
            return "Investigate potential outbreak. Cross-reference with IDSP data."
        return f"Immediate investigation required. Possible epidemic in {district_text}."

    def generate_alerts(self, anomalies: List[Dict], clusters: List[Dict]) -> List[OutbreakAlert]:
        """Generate structured outbreak alerts."""
        alerts: List[OutbreakAlert] = []
        cluster_by_condition = {c["condition"]: c for c in clusters}

        for a in anomalies:
            ratio = float(a["ratio"])
            level = self._level_from_ratio(ratio)
            if level is None:
                continue  # Below minimum threshold — skip
            cluster = cluster_by_condition.get(a["condition"])
            if cluster and cluster.get("cluster_type") == "regional_spread":
                level = self._raise_level(level)
                neighboring = [d for d in cluster.get("districts", []) if d != a["district"]]
                district_text = ", ".join(cluster.get("districts", []))
            else:
                neighboring = []
                district_text = a["district"]

            condition_to_icd = {
                "Jwara": "R50",
                "Kushtha": "L30",
                "Prameha": "E11",
                "Vibandha": "K59.0",
            }

            alert = OutbreakAlert(
                alert_id=str(uuid.uuid4()),
                alert_level=level,
                condition_ayush=a["condition"],
                condition_icd10=condition_to_icd.get(a["condition"], "R69"),
                district=a["district"],
                state=a.get("state", "Unknown"),
                current_cases=int(a["actual"]),
                baseline_cases=float(a["expected"]),
                ratio=float(ratio),
                trend="Increasing",
                affected_centres=[],
                neighboring_districts_affected=neighboring,
                recommended_action=self._recommended_action(level, district_text),
                generated_at=datetime.now(timezone.utc),
            )
            alerts.append(alert)
        return alerts


def generate_alert(condition: str, ratio: float) -> Dict[str, str]:
    """Backward-compatible helper retained for existing usage."""
    levels = {"WATCH": 1.5, "WARNING": 2.0, "ALERT": 3.0}
    if ratio >= levels["ALERT"]:
        level = "ALERT"
    elif ratio >= levels["WARNING"]:
        level = "WARNING"
    elif ratio >= levels["WATCH"]:
        level = "WATCH"
    else:
        level = "NORMAL"
    return {"condition": condition, "level": level, "ratio": f"{ratio:.2f}"}
