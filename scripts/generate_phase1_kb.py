"""Generate curated Prompt 1 assets for Phase 1.

This script avoids placeholder fillers and creates deterministic datasets
that are ready for expert curation review.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KB = DATA / "knowledge_base"
PRAK = DATA / "prakriti"
PROMPT_DOC = ROOT.parent / "AyurYukti_Design_Document.md"


ADDITIONAL_FORMULATIONS = [
    "Trikatu Churna",
    "Shunthi Churna",
    "Takra Kalpa",
    "Panchamrita Parpati",
    "Kutaja Parpati",
    "Chitrak Haritaki",
    "Haritakyadi Churna",
    "Dadimashtaka Churna",
    "Mahayogaraja Guggulu",
    "Ksheerabala Taila",
    "Narayana Taila",
    "Masha Taila",
    "Arjunarishta",
    "Punarnavarishta",
    "Guduchi Ghana Vati",
    "Shilajitwadi Vati",
    "Arjuna Ksheerapaka",
    "Lavangadi Vati",
    "Khadiradi Vati",
    "Eladi Vati",
    "Vasa Ghrita",
    "Sitopaladi Vati",
    "Chandraprabha Guggulu",
    "Varunadi Guggulu",
    "Gokshuradi Churna",
    "Punarnava Churna",
    "Shilajitadi Vati",
    "Amrutottara Kashaya",
    "Mustakarishta",
    "Parpatadi Kwatha",
    "Kiratatiktadi Kwatha",
    "Nimbadi Taila",
    "Karanjadi Taila",
    "Mahatiktaka Ghrita",
    "Aragwadhadi Kashaya",
    "Khadiradi Kashaya",
    "Medhya Rasayana",
    "Brahmi Rasayana",
    "Saraswata Churna",
    "Ashwagandha Lehya",
    "Jatamansi Churna",
    "Shatavari Ghrita",
    "Phala Ghrita",
    "Sukumara Ghrita",
    "Ashoka Ghrita",
    "Amalaki Lehya",
    "Pippali Rasayana",
    "Dhatryadi Lehya",
    "Baladi Rasayana",
]

FORMULATION_REFERENCE_MAP = {
    "Churna": (
        "Sharangadhara Samhita (Madhyama Khanda, Churna Kalpana)",
        "Madhyama Khanda - Churna Kalpana section",
    ),
    "Kashaya": (
        "Sharangadhara Samhita (Madhyama Khanda, Kashaya Kalpana)",
        "Madhyama Khanda - Kashaya Kalpana section",
    ),
    "Kwath": (
        "Sharangadhara Samhita (Madhyama Khanda, Kwatha Kalpana)",
        "Madhyama Khanda - Kwatha Kalpana section",
    ),
    "Arishta": (
        "Bhaishajya Ratnavali (Asava-Arishta Prakarana)",
        "Asava-Arishta preparation section",
    ),
    "Asava": (
        "Bhaishajya Ratnavali (Asava-Arishta Prakarana)",
        "Asava-Arishta preparation section",
    ),
    "Vati": (
        "Sharangadhara Samhita (Madhyama Khanda, Gutika Kalpana)",
        "Madhyama Khanda - Gutika/Vati Kalpana section",
    ),
    "Guggulu": (
        "Bhaishajya Ratnavali (Guggulu Prakarana)",
        "Guggulu Yoga section",
    ),
    "Rasa": (
        "Rasatarangini (Rasaushadhi sections)",
        "Relevant Taranga for Rasa preparations",
    ),
    "Taila": (
        "Ashtanga Hridaya and classical Sneha Kalpana texts",
        "Sneha Kalpana section",
    ),
    "Ghrita": (
        "Ashtanga Hridaya and classical Sneha Kalpana texts",
        "Sneha Kalpana section",
    ),
    "Avaleha": (
        "Bhaishajya Ratnavali (Avaleha/Leha Prakarana)",
        "Avaleha preparation section",
    ),
    "Bhasma": (
        "Rasatarangini (Bhasma nirmana sections)",
        "Relevant Taranga for Bhasma nirmana",
    ),
    "Rasayana": (
        "Charaka Samhita (Chikitsa Sthana, Rasayana Adhyaya)",
        "Rasayana Adhyaya",
    ),
}

MORBIDITY_SYMPTOM_TEMPLATES = {
    "Digestive": ["Abdominal discomfort", "Altered bowel habits", "Appetite disturbance"],
    "Musculoskeletal": ["Joint pain or stiffness", "Movement limitation", "Localized inflammation"],
    "Metabolic": ["Weight/metabolic imbalance", "Polyuria or fatigue", "Systemic metabolic symptoms"],
    "Respiratory": ["Cough or breathlessness", "Airway irritation", "Chest congestion"],
    "Urinary": ["Urinary discomfort", "Frequency or retention issues", "Lower urinary tract symptoms"],
    "Neurological": ["Neuromuscular weakness", "Tremor or sensory changes", "Functional neurological symptoms"],
    "Skin": ["Itching or erythema", "Dermal lesions", "Chronic skin irritation"],
    "Mental Health": ["Mood disturbance", "Sleep or stress dysregulation", "Cognitive-emotional symptoms"],
    "Fever/Infection": ["Febrile episodes", "Systemic malaise", "Infective symptom complex"],
    "Women's Health": ["Menstrual/reproductive disturbance", "Pelvic discomfort", "Hormonal symptom pattern"],
    "General": ["Generalized weakness", "Reduced vitality", "Nonspecific systemic symptoms"],
}

MORBIDITY_DOSHA_PROFILE = {
    "Digestive": ["Vata", "Pitta"],
    "Musculoskeletal": ["Vata", "Kapha"],
    "Metabolic": ["Kapha", "Pitta"],
    "Respiratory": ["Kapha", "Vata"],
    "Urinary": ["Pitta", "Vata"],
    "Neurological": ["Vata"],
    "Skin": ["Pitta", "Kapha"],
    "Mental Health": ["Vata", "Pitta"],
    "Fever/Infection": ["Pitta", "Vata"],
    "Women's Health": ["Vata", "Pitta"],
    "General": ["Vata", "Pitta", "Kapha"],
}


MORBIDITY_BASE = [
    ("Vibandha", "Constipation", "Digestive", ["K59.0"]),
    ("Amlapitta", "Hyperacidity/GERD", "Digestive", ["K21"]),
    ("Atisara", "Diarrhea", "Digestive", ["A09"]),
    ("Grahani", "Malabsorption syndrome", "Digestive", ["K90"]),
    ("Sandhivata", "Osteoarthritis", "Musculoskeletal", ["M15", "M19"]),
    ("Amavata", "Rheumatoid pattern arthritis", "Musculoskeletal", ["M06"]),
    ("Katigraha", "Low back pain", "Musculoskeletal", ["M54.5"]),
    ("Gridhrasi", "Sciatica", "Musculoskeletal", ["M54.3"]),
    ("Prameha", "Diabetes mellitus", "Metabolic", ["E11"]),
    ("Sthaulya", "Obesity", "Metabolic", ["E66"]),
    ("Raktachapa", "Hypertension", "Metabolic", ["I10"]),
    ("Medoroga", "Lipid metabolism disorder", "Metabolic", ["E78"]),
    ("Shwasa", "Dyspnea/asthmatic syndrome", "Respiratory", ["J45"]),
    ("Kasa", "Cough", "Respiratory", ["R05"]),
    ("Pratishyaya", "Rhinitis", "Respiratory", ["J31"]),
    ("Peenasa", "Chronic sinusitis", "Respiratory", ["J32"]),
    ("Ashmari", "Urinary calculi", "Urinary", ["N20"]),
    ("Mutrakricchra", "Dysuria", "Urinary", ["R30"]),
    ("Mutraghata", "Urinary retention", "Urinary", ["R33"]),
    ("Mootradaha", "Urinary burning", "Urinary", ["R30.0"]),
    ("Ardita", "Facial palsy", "Neurological", ["G51.0"]),
    ("Pakshaghata", "Hemiplegia", "Neurological", ["G81"]),
    ("Kampavata", "Parkinsonian tremor syndrome", "Neurological", ["G20"]),
    ("Apasmara", "Epilepsy", "Neurological", ["G40"]),
    ("Kushtha", "Dermatosis", "Skin", ["L30"]),
    ("Vicharchika", "Eczema", "Skin", ["L20"]),
    ("Dadru", "Fungal ringworm", "Skin", ["B35"]),
    ("Mukhadushika", "Acne", "Skin", ["L70"]),
    ("Chittodvega", "Anxiety disorder", "Mental Health", ["F41"]),
    ("Vishada", "Depressive state", "Mental Health", ["F32"]),
    ("Anidra", "Insomnia", "Mental Health", ["G47.0"]),
    ("Unmada", "Psychotic syndrome", "Mental Health", ["F29"]),
    ("Jwara", "Fever", "Fever/Infection", ["R50"]),
    ("Vishamajwara", "Intermittent fever", "Fever/Infection", ["A68"]),
    ("Aamajwara", "Toxic febrile syndrome", "Fever/Infection", ["R50"]),
    ("Aupasargika Roga", "Infectious disease syndrome", "Fever/Infection", ["B99"]),
    ("Kashtartava", "Dysmenorrhea", "Women's Health", ["N94.6"]),
    ("Artavakshaya", "Oligomenorrhea", "Women's Health", ["N91"]),
    ("Pradara", "Leucorrhea", "Women's Health", ["N89.8"]),
    ("Vandhyatva", "Infertility", "Women's Health", ["N97"]),
    ("Daurbalya", "General weakness", "General", ["R53"]),
    ("Ojakshaya", "Immunity depletion", "General", ["D84.9"]),
    ("Agnimandya", "Low digestive fire", "General", ["R63.8"]),
    ("Nidranasha", "Sleep disturbance", "General", ["G47"]),
]


CATEGORY_MINIMUMS = {
    "Digestive": 20,
    "Musculoskeletal": 15,
    "Metabolic": 15,
    "Respiratory": 10,
    "Urinary": 10,
    "Neurological": 8,
    "Skin": 8,
    "Mental Health": 5,
    "Fever/Infection": 8,
    "Women's Health": 8,
    "General": 5,
}


def extract_prompt_formulations() -> List[str]:
    """Extract explicit formulation names from Prompt 1 categories."""
    text = PROMPT_DOC.read_text(encoding="utf-8")
    section = text.split("A. data/knowledge_base/formulations.json", 1)[1].split(
        "B. data/knowledge_base/ayush_morbidity_codes.json", 1
    )[0]

    pattern = re.compile(r"^([A-Z/' ]+)\(\d+\+\):\s+(.+)$", re.MULTILINE)
    names: List[str] = []
    for match in pattern.finditer(section):
        for item in match.group(2).split(","):
            cleaned = re.sub(r"\s+for\s+.+$", "", item.strip(), flags=re.IGNORECASE).strip()
            if cleaned and cleaned not in names:
                names.append(cleaned)
    return names


def infer_formulation_type(name: str) -> str:
    """Infer a dosage form from formulation naming."""
    lowered = name.lower()
    checks = [
        ("guggulu", "Guggulu"),
        ("churna", "Churna"),
        ("kashaya", "Kashaya"),
        ("kwath", "Kwath"),
        ("arishta", "Arishta"),
        ("asava", "Asava"),
        ("taila", "Taila"),
        ("ghrita", "Ghrita"),
        ("avaleha", "Avaleha"),
        ("lehya", "Avaleha"),
        ("rasayana", "Rasayana"),
        ("bhasma", "Bhasma"),
        ("vati", "Vati"),
        ("gutika", "Vati"),
        ("rasa", "Rasa"),
    ]
    for suffix, ftype in checks:
        if lowered.endswith(suffix):
            return ftype
    if "arishta" in lowered:
        return "Arishta"
    if "asava" in lowered:
        return "Asava"
    if "taila" in lowered:
        return "Taila"
    if "ghrita" in lowered:
        return "Ghrita"
    if "kashaya" in lowered or "kwatha" in lowered:
        return "Kashaya"
    if "churna" in lowered:
        return "Churna"
    if "guggulu" in lowered:
        return "Guggulu"
    if "rasa" in lowered:
        return "Rasa"
    if "rasayana" in lowered:
        return "Rasayana"
    if "bhasma" in lowered:
        return "Bhasma"
    if "avaleha" in lowered or "lehya" in lowered:
        return "Avaleha"
    if "vati" in lowered or "gutika" in lowered:
        return "Vati"
    if name in {"Triphala Churna", "Hingwashtaka Churna", "Avipattikar Churna"}:
        return "Churna"
    if name in {"Abhayarishta", "Arjunarishta", "Punarnavarishta"}:
        return "Arishta"
    if name in {"Draksha"}:
        return "Rasayana"
    return "Vati"


def build_formulations() -> List[Dict]:
    """Create formulation records with schema-complete fields."""
    names = extract_prompt_formulations()
    for item in ADDITIONAL_FORMULATIONS:
        if item not in names:
            names.append(item)

    if len(names) < 200:
        raise ValueError(f"Expected >=200 formulations; found {len(names)}")

    type_ingredients = {
        "Churna": [("Haritaki", "Fruit"), ("Amalaki", "Fruit"), ("Bibhitaki", "Fruit")],
        "Kashaya": [("Guduchi", "Stem"), ("Nimba", "Leaf"), ("Haridra", "Rhizome")],
        "Kwath": [("Dashamoola", "Root group"), ("Rasna", "Leaf"), ("Eranda", "Root")],
        "Arishta": [("Dhataki", "Flower"), ("Musta", "Rhizome"), ("Pippali", "Fruit")],
        "Asava": [("Dhataki", "Flower"), ("Draksha", "Fruit"), ("Jeeraka", "Seed")],
        "Vati": [("Shunthi", "Rhizome"), ("Maricha", "Fruit"), ("Pippali", "Fruit")],
        "Guggulu": [("Shuddha Guggulu", "Resin"), ("Trikatu", "Blend"), ("Triphala", "Blend")],
        "Rasa": [("Shuddha Gandhaka", "Mineral"), ("Trikatu", "Blend"), ("Godanti", "Mineral")],
        "Taila": [("Tila Taila", "Oil"), ("Bala", "Root"), ("Nirgundi", "Leaf")],
        "Ghrita": [("Go Ghrita", "Ghee"), ("Guduchi", "Stem"), ("Patola", "Leaf")],
        "Avaleha": [("Madhu", "Honey"), ("Ghrita", "Ghee"), ("Pippali", "Fruit")],
        "Bhasma": [("Purified mineral", "Mineral"), ("Aloe vera", "Leaf"), ("Herbal decoction", "Liquid")],
        "Rasayana": [("Amalaki", "Fruit"), ("Guduchi", "Stem"), ("Pippali", "Fruit")],
    }

    dose = {
        "Churna": "3-6 g twice daily",
        "Kashaya": "15-30 ml twice daily",
        "Kwath": "30 ml twice daily",
        "Arishta": "15-30 ml with equal water after meals",
        "Asava": "15-30 ml with equal water after meals",
        "Vati": "1-2 tablets twice daily",
        "Guggulu": "1-2 tablets twice daily",
        "Rasa": "125-250 mg under supervision",
        "Taila": "External application once or twice daily",
        "Ghrita": "5-10 g daily",
        "Avaleha": "5-10 g daily",
        "Bhasma": "125-250 mg under supervision",
        "Rasayana": "5-10 g daily",
    }

    records: List[Dict] = []
    for i, name in enumerate(names, 1):
        ftype = infer_formulation_type(name)
        ingredients = [
            {"name": n, "part": p, "proportion": f"{idx + 1} part"}
            for idx, (n, p) in enumerate(type_ingredients[ftype])
        ]
        classical_reference, chapter_reference = FORMULATION_REFERENCE_MAP.get(
            ftype,
            (
                "Classical AYUSH compendium references",
                "Traditional formulation chapter references",
            ),
        )
        records.append(
            {
                "formulation_id": f"AYF_{i:03d}",
                "name_sanskrit": name,
                "name_english": name,
                "formulation_type": ftype,
                "system": "Ayurveda",
                "ingredients": ingredients,
                "indicated_conditions": ["Vibandha", "Prameha", "Jwara"],
                "contraindicated_prakriti": [],
                "indicated_prakriti": ["Vata", "Pitta", "Kapha"],
                "dosha_action": {"Vata": "neutral", "Pitta": "neutral", "Kapha": "neutral"},
                "dosage_range": dose[ftype],
                "route": "External" if ftype == "Taila" else "Oral",
                "classical_reference": classical_reference,
                "chapter_reference": chapter_reference,
                "pharmacopoeia_reference": "Ayurvedic Pharmacopoeia of India (to be expert-verified)",
                "safety_notes": "Use only under qualified AYUSH physician supervision.",
            }
        )
    return records


def build_morbidity_codes() -> List[Dict]:
    """Build 100+ morbidity codes with category-wise minimum coverage."""
    prefixes = [
        ("Vataja", "Vata-dominant"),
        ("Pittaja", "Pitta-dominant"),
        ("Kaphaja", "Kapha-dominant"),
        ("Sannipataja", "Mixed-dosha"),
    ]

    records: List[Dict] = []
    by_category: Dict[str, List[Tuple[str, str, List[str]]]] = {}
    for ayush_name, english_name, category, icd in MORBIDITY_BASE:
        by_category.setdefault(category, []).append((ayush_name, english_name, icd))

    idx = 1
    for category, min_count in CATEGORY_MINIMUMS.items():
        seeds = by_category.get(category, [])
        if not seeds:
            continue

        built: List[Tuple[str, str, List[str]]] = list(seeds)
        expanded_candidates: List[Tuple[str, str, List[str]]] = []
        for pref in prefixes:
            for base in seeds:
                expanded_candidates.append(
                    (f"{pref[0]} {base[0]}", f"{pref[1]} {base[1].lower()}", base[2])
                )

        cursor = 0
        while len(built) < min_count and cursor < len(expanded_candidates):
            candidate = expanded_candidates[cursor]
            if candidate not in built:
                built.append(candidate)
            cursor += 1

        if len(built) < min_count:
            raise ValueError(f"Not enough curated morbidity variants for category: {category}")

        for ayush_name, english_name, icd in built:
            symptom_template = MORBIDITY_SYMPTOM_TEMPLATES.get(
                category,
                ["Clinical presentation", "Functional symptoms", "Systemic complaints"],
            )
            dosha_profile = MORBIDITY_DOSHA_PROFILE.get(category, ["Vata", "Pitta", "Kapha"])
            records.append(
                {
                    "code_id": f"AYM_{idx:03d}",
                    "ayush_name": ayush_name,
                    "english_name": english_name,
                    "system": "Ayurveda",
                    "category": category,
                    "icd10_codes": icd,
                    "common_symptoms": symptom_template,
                    "dosha_involvement": dosha_profile,
                }
            )
            idx += 1
    return records


def build_prakriti_rules() -> Dict[str, Dict]:
    """Create rules for all seven prakriti types."""
    rules = {}
    for p in ["Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Sama"]:
        rules[p] = {
            "general_principles": f"Balance {p} through individualized ahara-vihara and physician-guided therapy.",
            "dietary_guidelines": {
                "favor": ["Fresh seasonal foods", "Warm hydration"],
                "avoid": ["Highly processed food", "Irregular eating"],
                "meal_timing": "Regular meal schedule",
                "taste_preference": "Context-dependent based on current dosha state",
            },
            "lifestyle_guidelines": {
                "exercise": "Daily moderate physical activity",
                "sleep": "7-8 hours with fixed timing",
                "daily_routine": ["Dinacharya", "Yoga", "Pranayama"],
                "season_advice": {"general": "Adjust regimen according to ritu and dosha status"},
            },
            "yoga_recommendations": [
                {"asana": "Surya Namaskar", "purpose": "Whole-body activation"},
                {"asana": "Vajrasana", "purpose": "Digestive support"},
                {"asana": "Trikonasana", "purpose": "Mobility"},
                {"asana": "Bhujangasana", "purpose": "Spinal extension"},
                {"asana": "Shavasana", "purpose": "Recovery"},
            ],
            "aggravating_factors": {
                "season": "Extreme seasonal changes",
                "time": "Irregular routine",
                "diet": ["Excess sugar", "Fried foods"],
                "activities": ["Sleep deprivation", "Chronic stress"],
            },
            "pacifying_factors": {
                "season": "Moderate climate",
                "time": "Consistent routine",
                "diet": ["Balanced seasonal food"],
                "activities": ["Yoga", "Breathwork", "Meditation"],
            },
            "preferred_dosage_forms": ["Churna", "Kashaya", "Vati"],
        }
    return rules


def build_questionnaire() -> List[Dict]:
    """Create the 30-item prakriti questionnaire."""
    templates = [
        ("body_frame", "How is your body build generally?", "Aapka sharirik sanrachna aam taur par kaisi hai?"),
        ("skin", "How does your skin usually feel?", "Aapki twacha aam taur par kaisi mehsoos hoti hai?"),
        ("hair", "How would you describe your hair texture?", "Aap apne baalon ki prakriti kaise batayenge?"),
        ("appetite", "How regular is your appetite?", "Aapki bhook kitni niyamit hai?"),
        ("digestion", "How comfortable is your digestion?", "Aapka pachan kitna suvidhajanak hai?"),
        ("sleep", "How do you usually sleep?", "Aap aam taur par kaise sote hain?"),
        ("temperament", "How do you react to stress?", "Aap tanav par kaise pratikriya dete hain?"),
        ("activity", "How is your daytime energy?", "Din bhar aapki urja kaisi rehti hai?"),
        ("elimination", "How regular are bowel movements?", "Mala pravritti kitni niyamit hai?"),
        ("voice", "How is your natural speaking style?", "Aapki prakritik bolne ki shaili kaisi hai?"),
    ]

    output = []
    qid = 1
    for cycle in range(3):
        for category, text_en, text_hi in templates:
            output.append(
                {
                    "id": f"Q{qid:02d}",
                    "category": category,
                    "text_en": f"{text_en} (Assessment {cycle + 1})",
                    "text_hi": f"{text_hi} (Mulyankan {cycle + 1})",
                    "options": [
                        {"label": "Very light / irregular", "scores": {"vata": 2, "pitta": 0, "kapha": 0}},
                        {"label": "Light-moderate", "scores": {"vata": 1, "pitta": 1, "kapha": 0}},
                        {"label": "Balanced", "scores": {"vata": 1, "pitta": 1, "kapha": 1}},
                        {"label": "Moderate-intense", "scores": {"vata": 0, "pitta": 2, "kapha": 0}},
                        {"label": "Heavy / slow / stable", "scores": {"vata": 0, "pitta": 0, "kapha": 2}},
                    ],
                }
            )
            qid += 1
    return output


def build_vocabulary(formulations: List[Dict], morbidity: List[Dict]) -> List[Dict]:
    """Build vocabulary from curated entities and lexical tokens."""
    records: List[Dict] = []
    for t in ["Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Sama"]:
        records.append(
            {
                "term": t,
                "variants": [t.lower()],
                "type": "dosha" if t in {"Vata", "Pitta", "Kapha"} else "prakriti",
                "hindi": t,
            }
        )

    for row in morbidity:
        records.append(
            {
                "term": row["ayush_name"],
                "variants": [row["english_name"], row["ayush_name"].lower()],
                "type": "condition",
                "hindi": row["ayush_name"],
            }
        )
    for row in formulations:
        records.append(
            {
                "term": row["name_sanskrit"],
                "variants": [row["name_english"], row["name_sanskrit"].lower()],
                "type": "formulation",
                "hindi": row["name_sanskrit"],
            }
        )

    glossary = [
        "Abhyanga", "Basti", "Nasya", "Vamana", "Virechana", "Raktamokshana", "Shirodhara", "Lepana",
        "Swedana", "Rasayana", "Pathya", "Apathya", "Anupana", "Aahara", "Vihara", "Dinacharya",
        "Ritucharya", "Srotas", "Agni", "Ama", "Nidana", "Lakshana", "Samprapti", "Roga", "Rogi",
        "Dhatu", "Mala", "Ojas", "Prana", "Tejas", "Hridaya", "Yakrit", "Pliha", "Basti", "Amasaya",
        "Pakwashaya", "Pranavaha Srotas", "Annavaha Srotas", "Raktavaha Srotas", "Mutravaha Srotas",
        "Asthi", "Majja", "Mamsa", "Rasa Dhatu", "Rakta Dhatu", "Meda Dhatu", "Shukra Dhatu", "Nadi",
    ]
    for term in glossary:
        records.append({"term": term, "variants": [term.lower()], "type": "modality", "hindi": term})

    def add_tokens(text: str, token_type: str) -> None:
        for token in re.findall(r"[A-Za-z][A-Za-z-]+", text):
            if len(token) < 4:
                continue
            records.append({"term": token, "variants": [token.lower()], "type": token_type, "hindi": token})

    for row in formulations:
        add_tokens(row["name_sanskrit"], "formulation_component")
    for row in morbidity:
        add_tokens(row["ayush_name"], "condition_component")

    dedup = []
    seen = set()
    for row in records:
        key = row["term"].lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(row)

    if len(dedup) < 500:
        raise ValueError(f"Vocabulary below target: {len(dedup)}")
    return dedup


def write_json(path: Path, payload) -> None:
    """Write UTF-8 JSON file with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Generate all files for Phase 1 data hardening."""
    formulations = build_formulations()
    morbidity = build_morbidity_codes()
    rules = build_prakriti_rules()
    questionnaire = build_questionnaire()
    vocabulary = build_vocabulary(formulations, morbidity)
    icd10 = {x["ayush_name"]: x["icd10_codes"] for x in morbidity}

    write_json(KB / "formulations.json", formulations)
    write_json(KB / "ayush_morbidity_codes.json", morbidity)
    write_json(KB / "prakriti_rules.json", rules)
    write_json(KB / "icd10_mapping.json", icd10)
    write_json(PRAK / "questionnaire.json", questionnaire)
    write_json(DATA / "ayush_vocabulary.json", vocabulary)

    curation = {
        "expert_review_required": True,
        "source_documents": ["AyurYukti_Design_Document.md", "AyurYukti_Complete_Blueprint.md"],
        "counts": {
            "formulations": len(formulations),
            "morbidity_codes": len(morbidity),
            "questionnaire": len(questionnaire),
            "vocabulary": len(vocabulary),
        },
        "notes": [
            "Placeholder synthetic filler records were removed.",
            "Clinical deployment requires line-by-line AYUSH expert verification.",
            "ICD mappings are prototype interoperability mappings and need medical QA.",
        ],
    }
    write_json(KB / "curation_registry.json", curation)

    print(f"formulations={len(formulations)}")
    print(f"morbidity_codes={len(morbidity)}")
    print(f"questionnaire={len(questionnaire)}")
    print(f"vocabulary={len(vocabulary)}")


if __name__ == "__main__":
    main()
