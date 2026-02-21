"""Analytics and visualization for treatment outcome learning."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.yuktishaala.contextual_bandit import ThompsonSamplingBandit
from src.yuktishaala.outcome_tracker import OutcomeTracker


class TreatmentAnalytics:
    """Build effectiveness reports and learning-curve visualizations."""

    def __init__(self, outcome_tracker: OutcomeTracker, bandit: ThompsonSamplingBandit):
        self.tracker = outcome_tracker
        self.bandit = bandit

    def get_treatment_effectiveness(self, condition_code: str) -> pd.DataFrame:
        """Return treatment effectiveness by formulation and prakriti."""
        df = self.tracker.get_all_outcomes_for_condition(condition_code)
        if df.empty:
            return pd.DataFrame(
                columns=["formulation", "prakriti", "n_patients", "success_rate", "ci_low", "ci_high"]
            )

        rows = []
        grouped = df.groupby(["formulation_name", "patient_prakriti"])
        for (formulation, prakriti), frame in grouped:
            n = len(frame)
            improved = (frame["outcome"] == "Improved").sum()
            p = improved / n if n else 0.0
            std = (p * (1 - p) / n) ** 0.5 if n else 0.0
            rows.append(
                {
                    "formulation": formulation,
                    "prakriti": prakriti,
                    "n_patients": n,
                    "success_rate": p,
                    "ci_low": max(0.0, p - 1.96 * std),
                    "ci_high": min(1.0, p + 1.96 * std),
                }
            )
        return pd.DataFrame(rows).sort_values(["success_rate", "n_patients"], ascending=[False, False])

    def get_prakriti_response_analysis(self, condition_code: str) -> Dict:
        """Return response matrix: prakriti -> formulation -> success rate."""
        report = self.get_treatment_effectiveness(condition_code)
        out: Dict = {}
        if report.empty:
            return out
        for _, row in report.iterrows():
            out.setdefault(row["prakriti"], {})[row["formulation"]] = float(row["success_rate"])
        return out

    def get_learning_curve(self, prakriti: str, condition: str, formulation: str) -> pd.DataFrame:
        """Return pseudo-learning curve from arm posterior evolution snapshot."""
        stats = self.bandit.get_arm_stats(prakriti, condition)
        for row in stats:
            if row["formulation"] != formulation:
                continue
            n = max(1, int(row["n_trials"]))
            # Create a monotonic synthetic curve ending at current posterior mean.
            points = []
            for i in range(1, n + 1):
                frac = i / n
                mean = 0.5 + (row["mean"] - 0.5) * frac
                ci = (row["95%_ci"][1] - row["95%_ci"][0]) * (1 - 0.7 * frac)
                points.append(
                    {"trial": i, "mean": mean, "ci_low": max(0.0, mean - ci / 2), "ci_high": min(1.0, mean + ci / 2)}
                )
            return pd.DataFrame(points)
        return pd.DataFrame(columns=["trial", "mean", "ci_low", "ci_high"])

    def create_effectiveness_chart(self, condition_code: str):
        """Grouped bar chart for success rates by prakriti/formulation."""
        df = self.get_treatment_effectiveness(condition_code)
        if df.empty:
            return go.Figure()
        fig = px.bar(
            df,
            x="formulation",
            y="success_rate",
            color="prakriti",
            barmode="group",
            title=f"Treatment effectiveness for {condition_code}",
        )
        return fig

    def create_learning_curve_chart(self, prakriti: str, condition: str):
        """Create line chart showing posterior means for all arms."""
        stats = self.bandit.get_arm_stats(prakriti, condition)
        fig = go.Figure()
        for row in stats:
            curve = self.get_learning_curve(prakriti, condition, row["formulation"])
            if curve.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=curve["trial"],
                    y=curve["mean"],
                    mode="lines",
                    name=row["formulation"],
                )
            )
        fig.update_layout(title=f"Learning curves: {prakriti} / {condition}", xaxis_title="Trial", yaxis_title="Posterior mean")
        return fig


def summarize_outcomes(outcomes: List[Dict[str, str]]) -> Dict[str, int]:
    """Backward-compatible helper retained from scaffold."""
    summary = {"Improved": 0, "No Change": 0, "Worsened": 0}
    for item in outcomes:
        key = item.get("outcome")
        if key in summary:
            summary[key] += 1
    return summary

