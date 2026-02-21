"""Lifestyle, yoga, and dietary advisory helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class LifestyleAdvisor:
    """Fetch lifestyle advice from prakriti rule definitions with condition awareness."""

    def __init__(self, prakriti_rules_path: str | Path):
        self.rules = json.loads(Path(prakriti_rules_path).read_text(encoding="utf-8"))

    def get_dietary_advice(self, prakriti_type: str, condition: str) -> List[str]:
        """Return dietary do's and don'ts for prakriti-condition context."""
        rule = self.rules.get(prakriti_type, {})
        diet = rule.get("dietary_guidelines", {})
        favor = diet.get("favor", [])
        avoid = [f"Avoid: {x}" for x in diet.get("avoid", [])]
        result = list(favor) + avoid

        # Add condition-specific dietary advice
        condition_rules = rule.get("condition_specific", {}).get(condition, {})
        extra = condition_rules.get("extra_dietary", [])
        if extra:
            result = [f"[{condition}] {x}" for x in extra] + result

        return result

    def get_yoga_advice(self, prakriti_type: str, condition: str) -> List[str]:
        """Return yoga suggestions for prakriti type."""
        items = self.rules.get(prakriti_type, {}).get("yoga_recommendations", [])
        result = []
        for x in items:
            if isinstance(x, dict):
                result.append(f"{x.get('asana', '')} — {x.get('purpose', '')}")
            else:
                result.append(str(x))
        return result

    def get_lifestyle_advice(self, prakriti_type: str, condition: str, season: Optional[str] = None) -> List[str]:
        """Return lifestyle suggestions with condition-aware and seasonal guidance."""
        rule = self.rules.get(prakriti_type, {})
        daily = list(rule.get("lifestyle_guidelines", {}).get("daily_routine", []))

        # Add exercise recommendation
        exercise = rule.get("lifestyle_guidelines", {}).get("exercise", "")
        if exercise:
            daily.insert(0, exercise)

        # Add sleep recommendation
        sleep = rule.get("lifestyle_guidelines", {}).get("sleep", "")
        if sleep:
            daily.insert(1, f"Sleep: {sleep}")

        if season:
            season_info = rule.get("lifestyle_guidelines", {}).get("season_advice", {})
            if season in season_info:
                daily.append(f"[Seasonal] {season_info[season]}")

        # Add condition-specific lifestyle advice
        condition_rules = rule.get("condition_specific", {}).get(condition, {})
        extra = condition_rules.get("extra_lifestyle", [])
        if extra:
            daily = [f"[{condition}] {x}" for x in extra] + daily

        return daily

    def get_full_profile(self, prakriti_type: str) -> Dict[str, any]:
        """Return complete prakriti profile including aggravating/pacifying factors."""
        rule = self.rules.get(prakriti_type, {})
        return {
            "general_principles": rule.get("general_principles", ""),
            "aggravating_factors": rule.get("aggravating_factors", {}),
            "pacifying_factors": rule.get("pacifying_factors", {}),
            "preferred_dosage_forms": rule.get("preferred_dosage_forms", []),
        }


def build_lifestyle_plan(prakriti: str) -> Dict[str, List[str]]:
    """Backward-compatible scaffold helper."""
    return {
        "prakriti": prakriti,
        "diet": ["Prefer warm freshly cooked meals"],
        "lifestyle": ["Follow regular sleep-wake cycle"],
        "yoga": ["Pavanamuktasana"],
    }
