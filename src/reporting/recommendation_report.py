"""Treatment recommendation report generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.common.models import TreatmentRecommendation


def generate_recommendation_pdf(recommendation: TreatmentRecommendation, output_path: str):
    """Generate treatment recommendation PDF."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4)
    story = []

    story.append(Paragraph("<b>AyurYukti — Treatment Recommendation Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Patient Prakriti: <b>{recommendation.patient_prakriti}</b>", styles["BodyText"]))
    story.append(Paragraph(f"Condition: <b>{recommendation.condition}</b>", styles["BodyText"]))
    story.append(Paragraph(f"Recommendation ID: {recommendation.recommendation_id}", styles["BodyText"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Top Formulations</b>", styles["Heading3"]))
    rows = [["Rank", "Formulation", "Dosage", "Reference", "Confidence"]]
    for idx, item in enumerate(recommendation.recommended_formulations[:3], start=1):
        score = float(item.get("score", 0.0))
        badge = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.5 else "LOW"
        rows.append(
            [
                str(idx),
                str(item.get("formulation_name", "")),
                str(item.get("dosage", "")),
                str(item.get("classical_reference", "")),
                f"{badge} ({score:.2f})",
            ]
        )

    table = Table(rows, colWidths=[40, 150, 120, 130, 70])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"<b>Lifestyle:</b> {', '.join(recommendation.lifestyle_suggestions or []) or 'NA'}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Diet:</b> {', '.join(recommendation.dietary_suggestions or []) or 'NA'}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Yoga:</b> {', '.join(recommendation.yoga_suggestions or []) or 'NA'}", styles["BodyText"]))
    story.append(Spacer(1, 8))

    reasoning = recommendation.reasoning.strip() if recommendation.reasoning else "No reasoning available."
    story.append(Paragraph("<b>Clinical Reasoning</b>", styles["Heading3"]))
    story.append(Paragraph(reasoning[:1800], styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI-generated recommendation. Doctor makes final decision.", styles["Italic"]))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).isoformat()} UTC", styles["BodyText"]))

    doc.build(story)
