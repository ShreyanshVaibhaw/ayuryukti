"""AYUSH vocabulary correction and discovery utilities."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class AyushVocabulary:
    """Vocabulary helper for variant correction and term detection."""

    def __init__(self, vocabulary_path: str | Path) -> None:
        """Load vocabulary JSON and build fast lookup indexes."""
        path = Path(vocabulary_path)
        self.entries: List[Dict] = json.loads(path.read_text(encoding="utf-8"))
        self.variant_to_term: Dict[str, str] = {}
        self.term_types: Dict[str, str] = {}

        for entry in self.entries:
            term = str(entry.get("term", "")).strip()
            if not term:
                continue
            self.term_types[term] = str(entry.get("type", "unknown"))
            self.variant_to_term[term.lower()] = term
            for variant in entry.get("variants", []):
                v = str(variant).strip().lower()
                if v:
                    self.variant_to_term[v] = term

        self.known_terms = sorted(set(self.variant_to_term.values()))

    def correct_text(self, text: str) -> str:
        """Replace known AYUSH variants using safe word-boundary substitutions."""
        corrected = text
        # Longer variants first to avoid partial overwrite.
        variants = sorted(self.variant_to_term.keys(), key=len, reverse=True)
        for variant in variants:
            canonical = self.variant_to_term[variant]
            pattern = re.compile(rf"\b{re.escape(variant)}\b", flags=re.IGNORECASE)
            corrected = pattern.sub(canonical, corrected)
        return corrected

    def find_terms(self, text: str) -> List[Dict]:
        """Return detected AYUSH terms and their offsets in text."""
        found: List[Dict] = []
        lowered = text.lower()
        for term in self.known_terms:
            start = lowered.find(term.lower())
            if start == -1:
                continue
            found.append(
                {
                    "term": term,
                    "type": self.term_types.get(term, "unknown"),
                    "start": start,
                    "end": start + len(term),
                }
            )
        return sorted(found, key=lambda x: x["start"])

    def suggest_correction(self, word: str) -> Optional[str]:
        """Suggest fuzzy correction for a single token."""
        token = word.strip().lower()
        token_compact = re.sub(r"[\s_\-]+", "", token)
        if not token:
            return None
        if token in self.variant_to_term:
            return self.variant_to_term[token]

        best_term = None
        best_score = 0.0
        for variant, canonical in self.variant_to_term.items():
            variant_compact = re.sub(r"[\s_\-]+", "", variant)
            score = max(
                SequenceMatcher(None, token, variant).ratio(),
                SequenceMatcher(None, token_compact, variant_compact).ratio(),
            )
            if score > best_score:
                best_score = score
                best_term = canonical
        return best_term if best_score >= 0.62 else None


def correct_ayush_terms(text: str, vocabulary: Iterable[str]) -> str:
    """Backward-compatible helper for simple term replacement."""
    corrected = text
    for term in vocabulary:
        corrected = corrected.replace(term.lower(), term)
    return corrected
