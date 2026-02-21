"""Baseline time-series modeling for outbreak surveillance."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


class BaselineModel:
    """Store district-condition baseline models (Prophet or fallback moving average)."""

    def __init__(self):
        self.models: Dict[Tuple[str, str], Dict] = {}
        enable_prophet = os.getenv("AYURYUKTI_ENABLE_PROPHET", "0") == "1"
        if enable_prophet:
            try:
                from prophet import Prophet

                self.Prophet = Prophet
            except Exception:  # pragma: no cover - optional dependency
                self.Prophet = None
        else:
            self.Prophet = None

    def fit(self, aggregated_data: pd.DataFrame, district: str, condition: str):
        """Fit baseline model for one district-condition pair."""
        subset = aggregated_data[
            (aggregated_data["district"] == district) & (aggregated_data["condition_ayush"] == condition)
        ].sort_values("week_start")

        key = (district, condition)
        if len(subset) < 8 or self.Prophet is None:
            self.models[key] = {
                "type": "moving_average",
                "mean": float(subset["case_count"].rolling(4, min_periods=1).mean().iloc[-1]) if not subset.empty else 0.0,
                "std": float(subset["case_count"].std(ddof=0)) if len(subset) > 1 else 1.0,
                "history": subset.copy(),
            }
            return

        model = self.Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
        fit_df = subset.rename(columns={"week_start": "ds", "case_count": "y"})[["ds", "y"]]
        model.fit(fit_df)
        self.models[key] = {"type": "prophet", "model": model, "history": subset.copy()}

    def predict(self, district: str, condition: str, periods: int = 4) -> pd.DataFrame:
        """Predict baseline for next periods."""
        key = (district, condition)
        if key not in self.models:
            return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

        payload = self.models[key]
        if payload["type"] == "moving_average":
            history = payload.get("history", pd.DataFrame())
            start = history["week_start"].max() if not history.empty else pd.Timestamp("2025-01-01")
            rows = []
            for i in range(1, periods + 1):
                ds = pd.Timestamp(start) + pd.Timedelta(days=7 * i)
                mean = float(payload["mean"])
                std = float(max(1.0, payload["std"]))
                rows.append({"ds": ds, "yhat": mean, "yhat_lower": max(0.0, mean - 2.5 * std), "yhat_upper": mean + 2.5 * std})
            return pd.DataFrame(rows)

        model = payload["model"]
        future = model.make_future_dataframe(periods=periods, freq="W-MON")
        forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return forecast.tail(periods).reset_index(drop=True)

    def fit_all(self, aggregated_data: pd.DataFrame):
        """Fit for every district-condition pair with at least 4 points."""
        groups = aggregated_data.groupby(["district", "condition_ayush"])
        for (district, condition), frame in groups:
            if len(frame) < 4:
                continue
            self.fit(aggregated_data=aggregated_data, district=district, condition=condition)


def build_baseline(df: pd.DataFrame) -> Dict[str, float]:
    """Backward-compatible helper retained for scaffold compatibility."""
    if df.empty:
        return {"baseline_cases": 0.0}
    col = "cases" if "cases" in df.columns else "case_count"
    return {"baseline_cases": float(np.mean(df[col]))}
