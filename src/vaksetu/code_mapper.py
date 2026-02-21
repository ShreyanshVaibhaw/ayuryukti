"""AYUSH and ICD-10 code mapping utilities."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List

from src.common.models import AyushMorbidityCode


class CodeMapper:
    """Maps condition text to AYUSH morbidity + ICD references."""

    def __init__(self, morbidity_codes_path: str, icd10_mapping_path: str) -> None:
        raw_codes = json.loads(Path(morbidity_codes_path).read_text(encoding="utf-8"))
        self.codes = [AyushMorbidityCode(**row) for row in raw_codes]
        self.icd_mapping = json.loads(Path(icd10_mapping_path).read_text(encoding="utf-8"))
        self._by_code = {code.code_id: code for code in self.codes}

    def map_condition(self, condition_text: str) -> Dict:
        """Map condition using exact, fuzzy, and keyword lookup."""
        query = condition_text.strip().lower()
        if not query:
            return {"ayush_name": "", "code_id": "", "icd10_codes": [], "confidence": 0.0, "match_type": "none"}

        # Exact name match.
        for code in self.codes:
            if query == code.ayush_name.lower() or query == code.english_name.lower():
                return {
                    "ayush_name": code.ayush_name,
                    "code_id": code.code_id,
                    "icd10_codes": code.icd10_codes,
                    "confidence": 1.0,
                    "match_type": "exact",
                }

        # Fuzzy match.
        best = None
        best_score = 0.0
        for code in self.codes:
            score = max(
                SequenceMatcher(None, query, code.ayush_name.lower()).ratio(),
                SequenceMatcher(None, query, code.english_name.lower()).ratio(),
            )
            if score > best_score:
                best_score = score
                best = code
        if best and best_score >= 0.72:
            return {
                "ayush_name": best.ayush_name,
                "code_id": best.code_id,
                "icd10_codes": best.icd10_codes,
                "confidence": round(best_score, 3),
                "match_type": "fuzzy",
            }

        # Keyword match.
        keyword_hits = []
        for code in self.codes:
            text = f"{code.ayush_name} {code.english_name} {' '.join(code.common_symptoms)}".lower()
            hit = sum(1 for token in query.split() if token in text)
            if hit > 0:
                keyword_hits.append((hit, code))
        keyword_hits.sort(key=lambda x: x[0], reverse=True)
        if keyword_hits:
            top = keyword_hits[0][1]
            return {
                "ayush_name": top.ayush_name,
                "code_id": top.code_id,
                "icd10_codes": top.icd10_codes,
                "confidence": 0.55,
                "match_type": "keyword",
            }

        return {"ayush_name": "", "code_id": "", "icd10_codes": [], "confidence": 0.0, "match_type": "none"}

    def search_codes(self, query: str, top_k: int = 5) -> List[AyushMorbidityCode]:
        """Keyword search over AYUSH morbidity taxonomy."""
        q = query.lower().strip()
        scored = []
        for code in self.codes:
            text = f"{code.ayush_name} {code.english_name} {' '.join(code.common_symptoms)}".lower()
            score = sum(1 for token in q.split() if token in text)
            if score > 0:
                scored.append((score, code))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [code for _, code in scored[:top_k]]

    def get_icd10_for_ayush(self, ayush_code_id: str) -> List[str]:
        """Lookup ICD-10 codes by AYUSH code id."""
        code = self._by_code.get(ayush_code_id)
        if code:
            return code.icd10_codes
        return []


def map_diagnosis_to_codes(ayush_diagnosis: str) -> Dict[str, str]:
    """Backward-compatible helper retained for legacy imports."""
    root = Path(__file__).resolve().parents[2]
    mapper = CodeMapper(
        morbidity_codes_path=str(root / "data" / "knowledge_base" / "ayush_morbidity_codes.json"),
        icd10_mapping_path=str(root / "data" / "knowledge_base" / "icd10_mapping.json"),
    )
    mapped = mapper.map_condition(ayush_diagnosis)
    return {
        "ayush_code": mapped.get("code_id", ""),
        "icd10_code": (mapped.get("icd10_codes") or [""])[0],
    }

