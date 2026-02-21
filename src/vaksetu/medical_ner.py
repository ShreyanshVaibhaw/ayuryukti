"""Medical NER and transcript-to-EHR extraction for VakSetu."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.models import AyushMorbidityCode, EHROutput
from src.llm.prompt_templates import (
    SYSTEM_PROMPT_AYUSH_NER,
    TRANSCRIPT_CORRECTION_TEMPLATE,
    TRANSCRIPT_TO_EHR_TEMPLATE,
)
from src.vaksetu.vocabulary import AyushVocabulary


class MedicalNEREngine:
    """NER engine that combines rule-based and LLM-assisted extraction."""

    def __init__(
        self,
        llm_client: Optional[Any],
        vocabulary_path: str,
        morbidity_codes_path: str,
    ) -> None:
        """Initialize dependencies and domain dictionaries."""
        self.llm_client = llm_client
        self.vocabulary = AyushVocabulary(vocabulary_path=vocabulary_path)
        raw_codes = json.loads(Path(morbidity_codes_path).read_text(encoding="utf-8"))
        self.morbidity_codes = [AyushMorbidityCode(**row) for row in raw_codes]

        self.condition_aliases = {
            "kabz": "Vibandha",
            "constipation": "Vibandha",
            "qabz": "Vibandha",
            "acidity": "Amlapitta",
            "hyperacidity": "Amlapitta",
            "burning sensation in chest": "Amlapitta",
            "chest burn": "Amlapitta",
            "heartburn": "Amlapitta",
            "joint pain": "Sandhivata",
            "ghutno mein dard": "Sandhivata",
            "arthritis": "Sandhivata",
            "jodo ka dard": "Sandhivata",
            "sugar": "Prameha",
            "diabetes": "Prameha",
            "madhumeha": "Prameha",
            "obesity": "Sthaulya",
            "motapa": "Sthaulya",
            "overweight": "Sthaulya",
            "high blood pressure": "Raktachapa",
            "bp": "Raktachapa",
            "hypertension": "Raktachapa",
            "fever": "Jwara",
            "bukhar": "Jwara",
            "kidney stone": "Ashmari",
            "pathri": "Ashmari",
            "urinary stone": "Ashmari",
            "cough": "Kasa",
            "khansi": "Kasa",
            "cold": "Pratishyaya",
            "nazla": "Pratishyaya",
            "rhinitis": "Pratishyaya",
            "headache": "Shirahshula",
            "sir dard": "Shirahshula",
            "skin disease": "Kushtha",
            "skin rash": "Kushtha",
            "eczema": "Vicharchika",
            "piles": "Arsha",
            "bawaseer": "Arsha",
            "hemorrhoids": "Arsha",
            "sandhivata": "Sandhivata",
            "amlapitta": "Amlapitta",
            "vibandha": "Vibandha",
            "prameha": "Prameha",
            "jwara": "Jwara",
            "kasa": "Kasa",
            "kushtha": "Kushtha",
            "arsha": "Arsha",
            "sthaulya": "Sthaulya",
            "raktachapa": "Raktachapa",
            "ashmari": "Ashmari",
        }
        self.icd_by_ayush = {
            "Vibandha": ("Constipation", "K59.0"),
            "Amlapitta": ("Hyperacidity/GERD", "K21"),
            "Sandhivata": ("Osteoarthritis", "M15-M19"),
            "Prameha": ("Diabetes mellitus", "E11"),
            "Sthaulya": ("Obesity", "E66"),
            "Raktachapa": ("Hypertension", "I10-I15"),
            "Jwara": ("Fever", "R50"),
            "Ashmari": ("Urolithiasis", "N20"),
            "Kasa": ("Cough", "R05"),
            "Pratishyaya": ("Rhinitis", "J31"),
            "Kushtha": ("Skin disease", "L30"),
            "Arsha": ("Hemorrhoids", "K64"),
            "Shirahshula": ("Headache", "R51"),
            "Vicharchika": ("Eczema", "L30.9"),
        }
        self._non_medical_tokens = {
            "saal", "ki", "hai", "mein", "aur", "se", "subah", "shaam",
            "ke", "baad", "raat", "male", "female", "patient", "years",
            "old", "the", "and", "for", "with", "from", "has", "have",
            "been", "was", "were", "this", "that", "doctor", "sir",
        }

        # Dosage pattern regexes
        self._dosage_patterns = [
            (r"(\d+\.?\d*)\s*(gram|gm|g)\b", r"\1 g"),
            (r"(\d+\.?\d*)\s*(mg|milligram)\b", r"\1 mg"),
            (r"(\d+\.?\d*)\s*(ml|millilitre|milliliter)\b", r"\1 ml"),
            (r"(\d+)\s*(goli|tablet|tab)\b", r"\1 tablet(s)"),
            (r"(\d+)\s*(capsule|cap)\b", r"\1 capsule(s)"),
            (r"(\d+)\s*(chammach|spoon|teaspoon|tsp)\b", r"\1 tsp"),
            (r"(\d+)\s*(drop|boond)\b", r"\1 drop(s)"),
        ]

        # Duration patterns
        self._duration_patterns = [
            (r"(\d+)\s*(din|day|days)\b", r"\1 days"),
            (r"(\d+)\s*(hafte|hafta|week|weeks)\b", r"\1 weeks"),
            (r"(\d+)\s*(mahine|mahina|month|months)\b", r"\1 months"),
            (r"(\d+)\s*(saal|varsh|year|years)\b", r"\1 years"),
        ]

        # Frequency patterns
        self._frequency_map = {
            "subah shaam": "Twice daily",
            "twice daily": "Twice daily",
            "din mein do baar": "Twice daily",
            "teen baar": "Three times daily",
            "three times": "Three times daily",
            "three times daily": "Three times daily",
            "din mein teen baar": "Three times daily",
            "raat ko": "Once daily at bedtime",
            "sone se pehle": "Once daily at bedtime",
            "at bedtime": "Once daily at bedtime",
            "subah": "Once daily in morning",
            "once daily": "Once daily",
            "ek baar": "Once daily",
            "khali pet": "Before food",
        }

        # Severity indicators
        self._severity_indicators = {
            "severe": "Severe",
            "tez": "Severe",
            "bahut": "Severe",
            "intense": "Severe",
            "chronic": "Chronic",
            "purana": "Chronic",
            "mild": "Mild",
            "halka": "Mild",
            "thoda": "Mild",
            "moderate": "Moderate",
            "intermittent": "Intermittent",
            "occasional": "Intermittent",
            "kabhi kabhi": "Intermittent",
        }

    def _fuzzy_match_term(self, term: str) -> Optional[str]:
        """Fuzzy match term against canonical AYUSH vocabulary."""
        token = term.strip().lower()
        if not token:
            return None
        if any(ch.isdigit() for ch in token):
            return None
        if len(token) < 5:
            return None
        if token in self._non_medical_tokens:
            return None
        suggestion = self.vocabulary.suggest_correction(token)
        if not suggestion:
            return None

        suggested = suggestion.lower()
        score = SequenceMatcher(None, token, suggested).ratio()
        term_type = self.vocabulary.term_types.get(suggestion, "unknown")
        prefix_ok = token[:2] == suggested[:2]
        len_ok = abs(len(token) - len(suggested)) <= 3

        if score >= 0.88 and term_type in {"formulation", "condition"}:
            return suggestion
        if score >= 0.75 and term_type in {"formulation", "condition"} and prefix_ok and len_ok:
            return suggestion
        return None

    def _select_relevant_codes(self, transcript: str) -> List[AyushMorbidityCode]:
        """Select top 20 likely morbidity codes based on keyword relevance."""
        text = transcript.lower()
        scored: List[tuple[int, AyushMorbidityCode]] = []
        for code in self.morbidity_codes:
            score = 0
            if code.ayush_name.lower() in text:
                score += 5
            if code.english_name.lower() in text:
                score += 4
            for symptom in code.common_symptoms:
                if symptom.lower() in text:
                    score += 1
            for alias, mapped in self.condition_aliases.items():
                if alias in text and mapped.lower() == code.ayush_name.lower():
                    score += 6
            if score > 0:
                scored.append((score, code))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [code for _, code in scored[:20]]

    def correct_transcript(self, raw_transcript: str, language: str) -> str:
        """Apply two-pass correction: vocabulary regex then optional LLM pass."""
        corrected = self.vocabulary.correct_text(raw_transcript)
        tokens = corrected.split()
        repaired_tokens = []
        for token in tokens:
            clean = re.sub(r"[^\w-]", "", token)
            suggestion = self._fuzzy_match_term(clean)
            repaired_tokens.append(suggestion if suggestion else token)
        corrected = " ".join(repaired_tokens)

        if not self.llm_client:
            return corrected

        vocab_subset = ", ".join(self.vocabulary.known_terms[:120])
        prompt = TRANSCRIPT_CORRECTION_TEMPLATE.format(
            vocabulary_subset=vocab_subset,
            raw_transcript=corrected,
        )
        llm_text = self.llm_client.generate(prompt, system=SYSTEM_PROMPT_AYUSH_NER, temperature=0.0).strip()
        return llm_text if llm_text else corrected

    @staticmethod
    def _parse_age(text: str) -> Optional[int]:
        """Extract patient age from common Hindi/English phrasings."""
        patterns = [
            r"(\d{1,3})\s*saal",
            r"(\d{1,3})\s*sal\b",
            r"(\d{1,3})\s*years?\s*old",
            r"age\s*[:\-]?\s*(\d{1,3})",
            r"(\d{1,3})\s*varsh",
            r"umra?\s*(\d{1,3})",
        ]
        lowered = text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _parse_sex(text: str) -> Optional[str]:
        """Infer sex from transcript mentions."""
        lowered = text.lower()
        if any(x in lowered for x in ["mahila", "female", "woman", "stri", "ladki"]):
            return "Female"
        if any(x in lowered for x in ["purush", "male", "man", "ladka"]):
            return "Male"
        return None

    @staticmethod
    def _parse_prakriti(text: str) -> Optional[str]:
        """Extract prakriti label from transcript."""
        options = ["Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Vata", "Pitta", "Kapha", "Sama"]
        for option in options:
            if option.lower() in text.lower():
                return option
            if f"{option.lower()} prakriti" in text.lower():
                return option
        return None

    def _parse_condition(self, text: str) -> Optional[str]:
        """Resolve AYUSH diagnosis from explicit or alias phrases.

        Priority: explicit AYUSH terms (e.g. "Sandhivata") trump aliases
        (e.g. "kabz" -> Vibandha).  When the transcript contains a direct
        diagnosis statement we should prefer it.
        """
        lowered = text.lower()

        # 1) Check for explicit AYUSH condition names first (highest priority)
        for code in self.morbidity_codes:
            if code.ayush_name.lower() in lowered:
                return code.ayush_name

        # 2) Check canonical condition names in alias values
        canonical_terms = [
            "sandhivata", "amlapitta", "vibandha", "prameha", "sthaulya",
            "raktachapa", "jwara", "ashmari", "kasa", "pratishyaya",
            "kushtha", "arsha", "shirahshula", "vicharchika",
        ]
        for term in canonical_terms:
            if term in lowered:
                return self.condition_aliases.get(term, term.title())

        # 3) Fall back to alias matching
        for alias, mapped in self.condition_aliases.items():
            if alias in lowered:
                return mapped

        return None

    @staticmethod
    def _parse_follow_up(text: str) -> Optional[str]:
        """Extract follow-up duration phrase."""
        lowered = text.lower()
        patterns = [
            (r"(\d+)\s*(week|hafte|hafta)", lambda m: f"{m.group(1)} week(s)"),
            (r"(\d+)\s*(din|day)", lambda m: f"{m.group(1)} day(s)"),
            (r"(\d+)\s*(month|mahine|mahina)", lambda m: f"{m.group(1)} month(s)"),
        ]
        for pattern, formatter in patterns:
            match = re.search(pattern, lowered)
            if match:
                return formatter(match)
        if "follow up" in lowered or "follow-up" in lowered:
            return "Follow-up advised"
        return None

    def _parse_dosage_from_context(self, text: str, formulation_name: str) -> Dict[str, str]:
        """Extract dosage, frequency, and instructions near a formulation mention."""
        lowered = text.lower()
        form_lower = formulation_name.lower()
        idx = lowered.find(form_lower)
        if idx == -1:
            context = lowered
        else:
            # Only look a few chars back (avoid bleeding into previous prescription)
            # and forward until the next formulation or end of text
            start = max(0, idx - 10)
            end = min(len(lowered), idx + len(form_lower) + 120)
            context = lowered[start:end]

        dosage = "As prescribed"
        for pattern, replacement in self._dosage_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                dosage = re.sub(pattern, replacement, match.group(0), flags=re.IGNORECASE)
                break

        frequency = "As prescribed"
        for phrase, mapped in self._frequency_map.items():
            if phrase in context:
                frequency = mapped
                break

        duration = ""
        for pattern, replacement in self._duration_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                duration = re.sub(pattern, replacement, match.group(0), flags=re.IGNORECASE)
                break

        special = ""
        if "khali pet" in context or "before food" in context or "empty stomach" in context:
            special = "Before food"
        elif "khana ke baad" in context or "after food" in context or "after meal" in context:
            special = "After food"
        if "garam paani ke saath" in context or "warm water" in context or "garam pani" in context:
            suffix = "With warm water"
            special = f"{special}; {suffix}".strip("; ").strip() if special else suffix
        if "doodh ke saath" in context or "with milk" in context:
            suffix = "With milk"
            special = f"{special}; {suffix}".strip("; ").strip() if special else suffix
        if "shahad ke saath" in context or "with honey" in context:
            suffix = "With honey"
            special = f"{special}; {suffix}".strip("; ").strip() if special else suffix

        return {
            "dosage": dosage,
            "frequency": frequency,
            "duration": duration,
            "special_instructions": special,
        }

    def _detect_severity(self, text: str) -> str:
        """Detect complaint severity from transcript."""
        lowered = text.lower()
        for indicator, severity in self._severity_indicators.items():
            if indicator in lowered:
                return severity
        return "Not specified"

    def _parse_duration_from_text(self, text: str) -> str:
        """Extract illness duration from text."""
        lowered = text.lower()
        for pattern, replacement in self._duration_patterns:
            match = re.search(pattern, lowered)
            if match:
                return re.sub(pattern, replacement, match.group(0), flags=re.IGNORECASE)
        return "Not specified"

    def _parse_prescriptions(self, text: str) -> List[Dict[str, str]]:
        """Extract prescription list using known formulation mentions and dosage context."""
        lowered = text.lower()
        found = []
        candidates = [entry["term"] for entry in self.vocabulary.entries if entry.get("type") == "formulation"]
        matched_terms: List[str] = []
        for term in candidates:
            if term.lower() in lowered:
                matched_terms.append(term)

        # Remove substring duplicates: if "Guduchi" and "Guduchi Kashaya" both match,
        # keep only the longer form.
        filtered_terms: List[str] = []
        for term in matched_terms:
            is_substring = any(
                term.lower() != other.lower() and term.lower() in other.lower()
                for other in matched_terms
            )
            if not is_substring:
                filtered_terms.append(term)

        for term in filtered_terms:
            context_info = self._parse_dosage_from_context(text, term)
            route = "Oral"
            if "taila" in term.lower() or "malish" in lowered or "oil" in term.lower():
                route = "External"
            elif "nasya" in term.lower():
                route = "Nasal"
            elif "basti" in term.lower():
                route = "Rectal"
            elif "anjana" in term.lower():
                route = "Ophthalmic"
            elif "lepa" in term.lower() or "ghrita" in term.lower():
                route = "Topical"

            snippet = {
                "formulation_name": term,
                "dosage": context_info["dosage"],
                "frequency": context_info["frequency"],
                "duration": context_info["duration"],
                "route": route,
                "special_instructions": context_info["special_instructions"],
            }
            found.append(snippet)
        return found

    def _compute_confidence_scores(
        self,
        transcript: str,
        age: Optional[int],
        sex: Optional[str],
        prakriti: Optional[str],
        condition: Optional[str],
        prescriptions: List[Dict[str, str]],
        complaints: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """Compute per-field confidence scores for extracted EHR data."""
        scores: Dict[str, float] = {}

        # Age confidence
        if age is not None:
            age_patterns_found = sum(
                1 for p in [r"\d+\s*saal", r"\d+\s*years?\s*old", r"age\s*[:\-]?\s*\d+", r"\d+\s*sal\b", r"\d+\s*varsh"]
                if re.search(p, transcript.lower())
            )
            scores["age"] = min(1.0, 0.90 + 0.10 * age_patterns_found)
        else:
            scores["age"] = 0.0

        # Sex confidence
        if sex:
            explicit_terms = ["female", "male", "mahila", "purush", "woman", "man", "ladki", "ladka", "stri"]
            if any(t in transcript.lower() for t in explicit_terms):
                scores["sex"] = 1.0
            else:
                scores["sex"] = 0.7
        else:
            scores["sex"] = 0.0

        # Prakriti confidence
        if prakriti:
            if "prakriti" in transcript.lower():
                scores["prakriti"] = 1.0
            else:
                scores["prakriti"] = 0.80
        else:
            scores["prakriti"] = 0.0

        # Diagnosis confidence
        if condition and condition != "Unknown":
            lowered = transcript.lower()
            condition_lower = condition.lower()
            direct_match = condition_lower in lowered
            alias_match = any(alias in lowered for alias, mapped in self.condition_aliases.items() if mapped == condition)
            if direct_match:
                scores["diagnosis"] = 1.0
            elif alias_match:
                scores["diagnosis"] = 1.0
            else:
                scores["diagnosis"] = 0.70
        else:
            scores["diagnosis"] = 0.0

        # Prescription confidence
        if prescriptions:
            rx_scores = []
            for rx in prescriptions:
                name = rx.get("formulation_name", "").lower()
                if name in transcript.lower():
                    has_dosage = rx.get("dosage", "As prescribed") != "As prescribed"
                    has_freq = rx.get("frequency", "As prescribed") != "As prescribed"
                    base = 0.90
                    if has_dosage:
                        base += 0.05
                    if has_freq:
                        base += 0.05
                    rx_scores.append(min(1.0, base))
                else:
                    rx_scores.append(0.60)
            scores["prescriptions"] = round(sum(rx_scores) / len(rx_scores), 2) if rx_scores else 0.0
        else:
            scores["prescriptions"] = 0.0

        # Complaints confidence
        if complaints:
            scores["chief_complaints"] = min(1.0, 0.90 + 0.10 * len(complaints))
        else:
            scores["chief_complaints"] = 0.0

        # Overall confidence
        non_zero = [v for v in scores.values() if v > 0]
        scores["overall"] = round(sum(non_zero) / len(non_zero), 2) if non_zero else 0.0

        return {k: round(v, 2) for k, v in scores.items()}

    def extract_ehr(self, transcript: str, language: str) -> EHROutput:
        """Convert transcript into structured EHROutput with confidence scores."""
        relevant_codes = self._select_relevant_codes(transcript)
        subset = [
            {
                "code_id": code.code_id,
                "ayush_name": code.ayush_name,
                "english_name": code.english_name,
                "icd10_codes": code.icd10_codes,
            }
            for code in relevant_codes
        ]
        prompt = TRANSCRIPT_TO_EHR_TEMPLATE.format(
            language=language,
            transcript=transcript,
            morbidity_codes_subset=json.dumps(subset, ensure_ascii=False),
        )

        llm_data: Dict[str, Any] = {}
        if self.llm_client:
            llm_data = self.llm_client.generate_json(prompt=prompt, system=SYSTEM_PROMPT_AYUSH_NER)

        age = self._parse_age(transcript)
        sex = self._parse_sex(transcript)
        prakriti = self._parse_prakriti(transcript)
        condition_ayush = self._parse_condition(transcript) or "Unknown"
        icd_name, icd_code = self.icd_by_ayush.get(condition_ayush, ("Unknown", "R69"))

        # Extract complaints with severity and duration
        complaints = []
        complaint_map = {
            ("kabz", "constipation"): ("Constipation", "Vibandha"),
            ("burning sensation", "acidity", "chest burn", "heartburn"): ("Burning sensation in chest / Acidity", "Amlapitta"),
            ("joint pain", "ghutno mein dard", "jodo ka dard"): ("Joint pain", "Sandhivata"),
            ("sujan", "swelling"): ("Swelling", None),
            ("sir dard", "headache"): ("Headache", "Shirahshula"),
            ("bukhar", "fever"): ("Fever", "Jwara"),
            ("khansi", "cough"): ("Cough", "Kasa"),
            ("nazla", "common cold", "rhinitis"): ("Rhinitis / Common cold", "Pratishyaya"),
            ("skin rash", "skin disease", "eczema"): ("Skin disorder", "Kushtha"),
            ("bawaseer", "piles", "hemorrhoids"): ("Hemorrhoids", "Arsha"),
            ("motapa", "obesity", "overweight"): ("Obesity", "Sthaulya"),
            ("pathri", "kidney stone"): ("Kidney stone", "Ashmari"),
            ("high blood pressure", "hypertension", "bp"): ("Hypertension", "Raktachapa"),
            ("sugar", "diabetes", "madhumeha"): ("Diabetes", "Prameha"),
        }
        lowered = transcript.lower()
        for keywords, (complaint_name, _) in complaint_map.items():
            if any(kw in lowered for kw in keywords):
                complaints.append({
                    "complaint": complaint_name,
                    "duration": self._parse_duration_from_text(transcript),
                    "severity": self._detect_severity(transcript),
                })

        prescriptions = self._parse_prescriptions(transcript)
        if not prescriptions and isinstance(llm_data.get("prescriptions"), list):
            prescriptions = llm_data["prescriptions"]

        yoga_advice = []
        yoga_terms = [
            "trikonasana", "veerabhadrasana", "virabhadrasana", "pavanamuktasana",
            "surya namaskar", "shavasana", "vajrasana", "bhujangasana",
            "padmasana", "sarvangasana", "matsyasana", "halasana",
            "pranayama", "anulom vilom", "kapalabhati", "bhastrika",
        ]
        for term in yoga_terms:
            if term in lowered:
                yoga_advice.append(term.title())

        dietary = []
        dietary_phrases = [
            "avoid spicy oily food", "avoid spicy food", "avoid oily food",
            "garam pani piyen", "warm water", "haldi wala doodh",
            "halka khana", "light food", "fresh fruits",
        ]
        for phrase in dietary_phrases:
            if phrase in lowered:
                dietary.append(phrase.title())

        # Compute confidence scores
        confidence = self._compute_confidence_scores(
            transcript, age, sex, prakriti, condition_ayush, prescriptions, complaints
        )

        output = EHROutput(
            patient_demographics={
                "patient_id": str(uuid.uuid4()),
                "age": age,
                "sex": sex,
                "occupation": None,
                "language": language,
            },
            prakriti_assessment=prakriti,
            chief_complaints=complaints,
            examination_findings=llm_data.get("examination_findings"),
            ayush_diagnosis={"name": condition_ayush, "code": condition_ayush},
            icd10_diagnosis={"name": icd_name, "code": icd_code},
            prescriptions=prescriptions,
            lifestyle_advice=llm_data.get("lifestyle_advice", []),
            dietary_advice=dietary if dietary else llm_data.get("dietary_advice", []),
            yoga_advice=yoga_advice if yoga_advice else llm_data.get("yoga_advice", []),
            follow_up=self._parse_follow_up(transcript),
            confidence_scores=confidence,
            encounter_metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "language": language,
                "ner_version": "phase4-v2",
                "confidence_scores": confidence,
            },
        )
        return output


def extract_medical_entities(transcript: str) -> Dict[str, List[str]]:
    """Backward-compatible helper retained for legacy tests."""
    lower = transcript.lower()
    return {
        "symptoms": [s for s in ["abdominal pain", "constipation", "joint pain"] if s in lower],
        "diagnoses": [d for d in ["vibandha", "amlapitta", "sandhivata"] if d in lower],
        "formulations": [f for f in ["triphala", "abhayarishta", "yogaraja guggulu"] if f in lower],
        "prakriti_mentions": [p for p in ["vata", "pitta", "kapha"] if p in lower],
    }
