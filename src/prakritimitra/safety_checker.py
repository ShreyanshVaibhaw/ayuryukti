"""Drug interaction and safety checks for AYUSH formulations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AyurYukti.SafetyChecker")

ROOT = Path(__file__).resolve().parents[2]
INTERACTION_PATH = ROOT / "data" / "knowledge_base" / "interaction_matrix.json"


class SafetyChecker:
    """Check drug interactions, contraindications, age limits, and prakriti conflicts."""

    def __init__(self, interaction_matrix_path: Optional[str] = None) -> None:
        path = Path(interaction_matrix_path) if interaction_matrix_path else INTERACTION_PATH
        if path.exists():
            self.interactions = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.interactions = {"interactions": [], "age_restrictions": {}, "pregnancy_warnings": []}
        self._interaction_lookup: Dict[str, Dict] = {}
        for entry in self.interactions.get("interactions", []):
            key = self._pair_key(entry["formulation_a"], entry["formulation_b"])
            self._interaction_lookup[key] = entry

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        return "|".join(sorted([a.lower().strip(), b.lower().strip()]))

    def check_interactions(self, formulations: List[str]) -> List[Dict[str, Any]]:
        """Check pairwise interactions between prescribed formulations."""
        warnings = []
        for i in range(len(formulations)):
            for j in range(i + 1, len(formulations)):
                key = self._pair_key(formulations[i], formulations[j])
                if key in self._interaction_lookup:
                    entry = self._interaction_lookup[key]
                    warnings.append({
                        "formulation_a": formulations[i],
                        "formulation_b": formulations[j],
                        "severity": entry.get("severity", "MODERATE"),
                        "description": entry.get("description", "Potential interaction detected."),
                        "recommendation": entry.get("recommendation", "Monitor closely."),
                    })
        return warnings

    def check_contraindications(
        self,
        formulation: Dict[str, Any],
        patient_prakriti: str,
        patient_age: int,
        patient_sex: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Check if a formulation is contraindicated for this patient."""
        warnings = []

        # Prakriti conflict check
        contraindicated = [p.lower() for p in formulation.get("contraindicated_prakriti", [])]
        if patient_prakriti.lower() in contraindicated:
            warnings.append({
                "type": "PRAKRITI_CONFLICT",
                "severity": "HIGH",
                "message": f"{formulation.get('name_sanskrit', 'Formulation')} is contraindicated for {patient_prakriti} prakriti.",
                "recommendation": "Consider alternative formulation suitable for patient's prakriti.",
            })

        # Age restriction checks
        age_limits = self.interactions.get("age_restrictions", {})
        form_name = formulation.get("name_sanskrit", "").lower()
        if form_name in age_limits:
            limits = age_limits[form_name]
            min_age = limits.get("min_age", 0)
            max_age = limits.get("max_age", 120)
            if patient_age < min_age:
                warnings.append({
                    "type": "AGE_RESTRICTION",
                    "severity": "HIGH",
                    "message": f"{formulation.get('name_sanskrit', '')} not recommended for patients under {min_age} years.",
                    "recommendation": limits.get("pediatric_alternative", "Consult physician for age-appropriate alternative."),
                })
            if patient_age > max_age:
                warnings.append({
                    "type": "AGE_RESTRICTION",
                    "severity": "MEDIUM",
                    "message": f"{formulation.get('name_sanskrit', '')} — use caution in patients over {max_age} years.",
                    "recommendation": "Reduce dosage or consult physician.",
                })

        # Pediatric general warning
        if patient_age < 5:
            warnings.append({
                "type": "PEDIATRIC_CAUTION",
                "severity": "MEDIUM",
                "message": "Patient is under 5 years. Dosage must be adjusted by physician.",
                "recommendation": "Apply Sharangadhara's pediatric dosage formula.",
            })

        # Geriatric general warning
        if patient_age > 75:
            warnings.append({
                "type": "GERIATRIC_CAUTION",
                "severity": "LOW",
                "message": "Patient is over 75 years. Monitor for tolerance.",
                "recommendation": "Start with 50-75% of standard dose.",
            })

        # Pregnancy warning
        pregnancy_warned = self.interactions.get("pregnancy_warnings", [])
        if form_name in [p.lower() for p in pregnancy_warned] and patient_sex == "Female":
            warnings.append({
                "type": "PREGNANCY_CAUTION",
                "severity": "HIGH",
                "message": f"{formulation.get('name_sanskrit', '')} may be contraindicated in pregnancy.",
                "recommendation": "Confirm pregnancy status before prescribing.",
            })

        return warnings

    def filter_safe_formulations(
        self,
        formulations: List[Dict[str, Any]],
        patient_prakriti: str,
        patient_age: int,
        patient_sex: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter formulations, returning (safe, warnings) tuple."""
        safe = []
        all_warnings = []
        for f in formulations:
            raw = f.get("raw", f)
            warnings = self.check_contraindications(raw, patient_prakriti, patient_age, patient_sex)
            high_severity = [w for w in warnings if w["severity"] == "HIGH"]
            if not high_severity:
                f["safety_warnings"] = warnings
                safe.append(f)
            else:
                for w in high_severity:
                    w["formulation"] = raw.get("name_sanskrit", f.get("name", "Unknown"))
                all_warnings.extend(high_severity)
            all_warnings.extend([w for w in warnings if w["severity"] != "HIGH"])
        return safe, all_warnings
