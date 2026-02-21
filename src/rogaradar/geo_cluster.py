"""Geospatial clustering for outbreak anomalies."""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Compute great-circle distance between two lat/lon points."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class GeoCluster:
    """Cluster anomalies in space-time to identify spread patterns."""

    def __init__(self, district_metadata: pd.DataFrame):
        self.districts = district_metadata.copy()
        self.lookup = {row["district"]: row for _, row in self.districts.iterrows()}

    def get_neighboring_districts(self, district: str, radius_km: float = 150) -> List[str]:
        """Find districts within radius using haversine distance."""
        if district not in self.lookup:
            return []
        src = self.lookup[district]
        out = []
        for name, row in self.lookup.items():
            if name == district:
                continue
            d = haversine_km(src["lat"], src["lon"], row["lat"], row["lon"])
            if d <= radius_km:
                out.append(name)
        return sorted(out)

    def cluster_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """Cluster anomalies by geography+time and summarize spread severity."""
        if not anomalies:
            return []
        rows = []
        for a in anomalies:
            if a["district"] not in self.lookup:
                continue
            meta = self.lookup[a["district"]]
            week = pd.Timestamp(a["week"])
            rows.append(
                {
                    **a,
                    "lat": float(meta["lat"]),
                    "lon": float(meta["lon"]),
                    "week_ordinal": int(week.toordinal()),
                }
            )
        if not rows:
            return []

        df = pd.DataFrame(rows)
        output = []
        for condition, frame in df.groupby("condition"):
            frame = frame.sort_values("week").reset_index(drop=True).copy()

            # Build simple spatiotemporal graph and extract connected components.
            # This avoids heavy sklearn backends and keeps clustering deterministic.
            n = len(frame)
            neighbors: List[List[int]] = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    dist = haversine_km(
                        float(frame.loc[i, "lat"]),
                        float(frame.loc[i, "lon"]),
                        float(frame.loc[j, "lat"]),
                        float(frame.loc[j, "lon"]),
                    )
                    week_delta = abs(int(frame.loc[i, "week_ordinal"]) - int(frame.loc[j, "week_ordinal"]))
                    if dist <= 250.0 and week_delta <= 21:
                        neighbors[i].append(j)
                        neighbors[j].append(i)

            labels = [-1] * n
            current = 0
            for i in range(n):
                if labels[i] != -1:
                    continue
                stack = [i]
                labels[i] = current
                while stack:
                    node = stack.pop()
                    for nb in neighbors[node]:
                        if labels[nb] == -1:
                            labels[nb] = current
                            stack.append(nb)
                current += 1

            frame["cluster_id"] = labels
            for cid, group in frame.groupby("cluster_id"):
                districts = sorted(group["district"].unique().tolist())
                span = (str(group["week"].min().date()), str(group["week"].max().date()))
                severity = float(group["ratio"].max())
                output.append(
                    {
                        "cluster_id": f"{condition}_{cid}",
                        "condition": condition,
                        "districts": districts,
                        "temporal_span": span,
                        "severity": severity,
                        "cluster_type": "regional_spread" if len(districts) >= 2 else "local_cluster",
                    }
                )
        return output


def cluster_districts(districts: List[str]) -> List[List[str]]:
    """Backward-compatible helper retained from scaffold."""
    if not districts:
        return []
    return [districts]
