"""Surveillance report generation utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.common.models import OutbreakAlert


def generate_surveillance_pdf(alerts: List[OutbreakAlert], aggregated_data: pd.DataFrame, output_path: str):
    """Generate outbreak surveillance PDF report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(out), pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AyurYukti RogaRadar — Disease Surveillance Report</b>", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()} UTC", styles["BodyText"]))
    story.append(Spacer(1, 10))

    districts = len(set(a.district for a in alerts))
    level_counts = {"WATCH": 0, "WARNING": 0, "ALERT": 0}
    for alert in alerts:
        level_counts[alert.alert_level] = level_counts.get(alert.alert_level, 0) + 1

    summary = Table(
        [
            ["Total Districts", str(districts)],
            ["Active Alerts", str(len(alerts))],
            ["WATCH / WARNING / ALERT", f"{level_counts['WATCH']} / {level_counts['WARNING']} / {level_counts['ALERT']}"],
            ["Rows Analyzed", str(len(aggregated_data))],
        ],
        colWidths=[180, 300],
    )
    summary.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(summary)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Active Alert Details</b>", styles["Heading3"]))
    rows = [["Severity", "District", "State", "Condition", "Cases", "Baseline", "Ratio", "Action"]]
    for alert in alerts[:25]:
        rows.append(
            [
                alert.alert_level,
                alert.district,
                alert.state,
                alert.condition_ayush,
                str(alert.current_cases),
                f"{alert.baseline_cases:.1f}",
                f"{alert.ratio:.2f}",
                alert.recommended_action[:70],
            ]
        )

    detail = Table(rows, colWidths=[52, 60, 60, 65, 42, 48, 40, 160])
    detail.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ]
        )
    )
    story.append(detail)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Methodology</b>", styles["Heading3"]))
    story.append(
        Paragraph(
            "Anomaly signals are generated using moving-baseline residual checks, robust multivariate outlier detection, and CUSUM-style shift detection.",
            styles["BodyText"],
        )
    )

    doc.build(story)
