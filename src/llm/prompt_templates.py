"""Prompt templates used by AyurYukti LLM workflows."""

SYSTEM_PROMPT_AYUSH_NER = """You are AyurYukti, an AI medical scribe specialized in AYUSH clinical documentation. You extract structured medical information from doctor consultation transcripts.

CRITICAL RULES:
- Extract ONLY information explicitly stated in the transcript
- If not mentioned, set field to null — NEVER guess
- Use standard AYUSH terminology
- Map conditions to both AYUSH morbidity codes AND ICD-10 codes
- Preserve doctor's exact prescription details — never modify dosages
- Support mixed-language input (Hindi-English-Sanskrit code-switching)
- Recognize AYUSH terms: Prakriti types, dosha names, formulations, treatment modalities
- Always return valid JSON — no markdown, no code fences, just raw JSON"""


TRANSCRIPT_TO_EHR_TEMPLATE = """
Language: {language}

Transcript:
{transcript}

Relevant AYUSH morbidity coding options:
{morbidity_codes_subset}

---

IMPORTANT: Return ONLY a valid JSON object. No markdown, no code blocks, no explanation.

Required JSON schema:
{{
  "patient_demographics": {{"age": <int|null>, "sex": <"Male"|"Female"|null>, "occupation": <str|null>}},
  "prakriti_assessment": <str|null>,
  "chief_complaints": [{{"complaint": <str>, "duration": <str>, "severity": <str>}}],
  "examination_findings": <str|null>,
  "ayush_diagnosis": {{"name": <str>, "code": <str>, "dosha_involvement": [<str>]}},
  "icd10_diagnosis": {{"name": <str>, "code": <str>}},
  "prescriptions": [{{"formulation_name": <str>, "dosage": <str>, "frequency": <str>, "duration": <str>, "route": <str>, "special_instructions": <str>}}],
  "lifestyle_advice": [<str>],
  "dietary_advice": [<str>],
  "yoga_advice": [<str>],
  "follow_up": <str|null>
}}

---

FEW-SHOT EXAMPLES:

Example 1 — Hindi transcript:
Transcript: "35 saal ki mahila hai, Vata Prakriti. 2 hafte se kabz ki shikayat hai. Triphala Churna 5 gram subah shaam garam paani ke saath. Abhayarishta 15ml khana ke baad. Pavanamuktasana karein. Follow up 2 weeks."
Output:
{{
  "patient_demographics": {{"age": 35, "sex": "Female", "occupation": null}},
  "prakriti_assessment": "Vata",
  "chief_complaints": [{{"complaint": "Constipation (Kabz)", "duration": "2 weeks", "severity": "Not specified"}}],
  "examination_findings": null,
  "ayush_diagnosis": {{"name": "Vibandha", "code": "VIB001", "dosha_involvement": ["Vata"]}},
  "icd10_diagnosis": {{"name": "Constipation", "code": "K59.0"}},
  "prescriptions": [
    {{"formulation_name": "Triphala Churna", "dosage": "5 gram", "frequency": "Twice daily", "duration": "", "route": "Oral", "special_instructions": "With warm water"}},
    {{"formulation_name": "Abhayarishta", "dosage": "15ml", "frequency": "After food", "duration": "", "route": "Oral", "special_instructions": "After food"}}
  ],
  "lifestyle_advice": [],
  "dietary_advice": [],
  "yoga_advice": ["Pavanamuktasana"],
  "follow_up": "2 weeks"
}}

Example 2 — English transcript:
Transcript: "52 year old male with Pitta-Kapha prakriti. Chief complaint: burning sensation in chest for 3 weeks. Diagnosed Amlapitta. Prescribing Avipattikar Churna 3g twice daily before food, and Kamdudha Rasa 250mg with milk. Avoid spicy oily food."
Output:
{{
  "patient_demographics": {{"age": 52, "sex": "Male", "occupation": null}},
  "prakriti_assessment": "Pitta-Kapha",
  "chief_complaints": [{{"complaint": "Burning sensation in chest", "duration": "3 weeks", "severity": "Not specified"}}],
  "examination_findings": null,
  "ayush_diagnosis": {{"name": "Amlapitta", "code": "AMP001", "dosha_involvement": ["Pitta"]}},
  "icd10_diagnosis": {{"name": "Hyperacidity/GERD", "code": "K21"}},
  "prescriptions": [
    {{"formulation_name": "Avipattikar Churna", "dosage": "3g", "frequency": "Twice daily", "duration": "", "route": "Oral", "special_instructions": "Before food"}},
    {{"formulation_name": "Kamdudha Rasa", "dosage": "250mg", "frequency": "As prescribed", "duration": "", "route": "Oral", "special_instructions": "With milk"}}
  ],
  "lifestyle_advice": [],
  "dietary_advice": ["Avoid spicy oily food"],
  "yoga_advice": [],
  "follow_up": null
}}

Example 3 — Mixed transcript:
Transcript: "58 saal ka male patient, Vata-Kapha prakriti. 3 mahine se ghutno mein dard aur sujan. Sandhivata diagnosis. Yogaraja Guggulu 2 goli subah shaam. Maharasnadi Kashaya 15ml. Avoid cold food. Trikonasana aur Veerabhadrasana. 1 hafte mein follow up."
Output:
{{
  "patient_demographics": {{"age": 58, "sex": "Male", "occupation": null}},
  "prakriti_assessment": "Vata-Kapha",
  "chief_complaints": [
    {{"complaint": "Joint pain in knees", "duration": "3 months", "severity": "Not specified"}},
    {{"complaint": "Swelling in knees", "duration": "3 months", "severity": "Not specified"}}
  ],
  "examination_findings": null,
  "ayush_diagnosis": {{"name": "Sandhivata", "code": "SNV001", "dosha_involvement": ["Vata"]}},
  "icd10_diagnosis": {{"name": "Osteoarthritis", "code": "M15-M19"}},
  "prescriptions": [
    {{"formulation_name": "Yogaraja Guggulu", "dosage": "2 tablets", "frequency": "Twice daily", "duration": "", "route": "Oral", "special_instructions": ""}},
    {{"formulation_name": "Maharasnadi Kashaya", "dosage": "15ml", "frequency": "As prescribed", "duration": "", "route": "Oral", "special_instructions": ""}}
  ],
  "lifestyle_advice": [],
  "dietary_advice": ["Avoid cold food"],
  "yoga_advice": ["Trikonasana", "Veerabhadrasana"],
  "follow_up": "1 week"
}}

---

Hindi phrase helpers:
- "khali pet" = before food / empty stomach
- "khana ke baad" = after food
- "subah shaam" = twice daily (morning and evening)
- "teen baar" = three times daily
- "garam paani ke saath" = with warm water
- "doodh ke saath" = with milk
- "sone se pehle" = before sleep / at bedtime

Condition mapping hints:
- "kabz" = Vibandha (Constipation)
- "acidity" / "chest burn" = Amlapitta (GERD)
- "joint pain" / "ghutno mein dard" = Sandhivata (Osteoarthritis)
- "sugar" / "madhumeha" = Prameha (Diabetes)
- "motapa" = Sthaulya (Obesity)
- "bukhar" = Jwara (Fever)
- "khansi" = Kasa (Cough)
- "bp" / "high blood pressure" = Raktachapa (Hypertension)
- "pathri" = Ashmari (Urolithiasis)
- "bawaseer" = Arsha (Hemorrhoids)

FALLBACK RULES:
- If no condition is identified, set diagnosis name to "Unknown" and code to "R69"
- If demographics are ambiguous, set to null rather than guessing
- If dosage is not explicitly stated, use "As prescribed"
- If language is mixed, extract all terms regardless of language
""".strip()


TRANSCRIPT_CORRECTION_TEMPLATE = """
You are correcting ASR transcript errors in AYUSH terms only.

Known vocabulary subset:
{vocabulary_subset}

Raw transcript:
{raw_transcript}

Rules:
- Fix AYUSH term spellings only.
- Keep all non-AYUSH words unchanged.
- Examples: "tree fala" -> "Triphala", "ashwa gandha" -> "Ashwagandha", "avipatikar" -> "Avipattikar"
- Return corrected plain text only. No JSON, no explanation.
""".strip()


RECOMMENDATION_EXPLANATION_TEMPLATE = """
Patient profile:
- Prakriti: {prakriti_type}
- Age: {age}
- Sex: {sex}
- Condition (AYUSH): {condition_ayush}
- Condition (English): {condition_english}
- Dosha involvement: {dosha_involvement}

Treatment details:
{treatment_details}

Write 3-4 sentence clinical reasoning covering:
1) Why treatment suits this Prakriti
2) How it addresses the dosha imbalance
3) Classical support context
4) Important precautions

Keep the explanation concise, evidence-based, and clinician-friendly.
""".strip()
