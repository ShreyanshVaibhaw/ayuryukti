"""Visualization layer for outbreak surveillance outputs."""

from __future__ import annotations

from typing import List

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class SurveillanceDashboard:
    """Build geospatial and chart visualizations for surveillance results."""

    COLORS = {"WATCH": "yellow", "WARNING": "orange", "ALERT": "red"}

    def create_district_map(self, alerts, district_metadata: pd.DataFrame) -> folium.Map:
        """Create map with district-level alert markers."""
        fmap = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles="cartodbpositron")
        alert_lookup = {a.district: a for a in alerts}
        for _, row in district_metadata.iterrows():
            district = row["district"]
            if district in alert_lookup:
                alert = alert_lookup[district]
                color = self.COLORS.get(alert.alert_level, "green")
                popup = (
                    f"{district}<br>{alert.condition_ayush}<br>"
                    f"cases={alert.current_cases}, baseline={alert.baseline_cases:.1f}, ratio={alert.ratio:.2f}"
                )
            else:
                color = "green"
                popup = f"{district}<br>Normal"
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.85,
                popup=popup,
            ).add_to(fmap)
        return fmap

    def create_time_series_chart(self, aggregated_data, district, condition) -> go.Figure:
        """Plot actual weekly cases and highlight anomaly candidates."""
        frame = aggregated_data[
            (aggregated_data["district"] == district) & (aggregated_data["condition_ayush"] == condition)
        ].sort_values("week_start")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame["week_start"], y=frame["case_count"], mode="lines+markers", name="Actual cases"))
        if not frame.empty:
            mean = frame["case_count"].mean()
            fig.add_trace(
                go.Scatter(
                    x=frame["week_start"],
                    y=[mean] * len(frame),
                    mode="lines",
                    line=dict(dash="dash"),
                    name="Baseline mean",
                )
            )
            outliers = frame[frame["case_count"] > 1.8 * mean]
            if not outliers.empty:
                fig.add_trace(
                    go.Scatter(
                        x=outliers["week_start"],
                        y=outliers["case_count"],
                        mode="markers",
                        marker=dict(color="red", size=10),
                        name="Anomaly",
                    )
                )
        fig.update_layout(title=f"{district} - {condition}: weekly cases vs baseline")
        return fig

    def create_alert_summary_table(self, alerts) -> pd.DataFrame:
        """Build tabular summary from alert objects."""
        rows = []
        for a in alerts:
            rows.append(
                {
                    "Alert Level": a.alert_level,
                    "District": a.district,
                    "State": a.state,
                    "Condition": a.condition_ayush,
                    "Cases": a.current_cases,
                    "Baseline": round(a.baseline_cases, 2),
                    "Ratio": round(a.ratio, 2),
                    "Trend": a.trend,
                }
            )
        return pd.DataFrame(rows)

    def create_condition_heatmap(self, aggregated_data) -> go.Figure:
        """Heatmap for districts x conditions by normalized weekly burden."""
        if aggregated_data.empty:
            return go.Figure()
        pivot = aggregated_data.pivot_table(
            index="district",
            columns="condition_ayush",
            values="case_count",
            aggfunc="mean",
            fill_value=0,
        )
        fig = px.imshow(
            pivot,
            labels=dict(x="Condition", y="District", color="Avg Cases"),
            title="District-condition burden heatmap",
            aspect="auto",
        )
        return fig


def build_cases_chart(df: pd.DataFrame):
    """Backward-compatible helper from scaffold."""
    if df.empty:
        df = pd.DataFrame({"date": [], "cases": []})
    return px.line(df, x="date", y="cases", title="AYUSH Condition Cases Over Time")

