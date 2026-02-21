"""Centralized configuration values for AyurYukti."""

from __future__ import annotations

import os

PROJECT_NAME = "AyurYukti"
PROJECT_SANSKRIT = "आयुर्युक्ति"
TAGLINE = "From Voice to Verdict — Intelligent AYUSH Healthcare"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ayuryukti2026")

POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://ayuryukti:ayuryukti2026@localhost:5432/ayuryukti",
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

BHASHINI_API_URL = os.getenv("BHASHINI_API_URL", "https://meity-auth.ulcacontrib.org")
BHASHINI_INFERENCE_URL = os.getenv("BHASHINI_INFERENCE_URL", "https://dhruva-api.bhashini.gov.in")

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "bn": "Bengali",
    "mr": "Marathi",
    "te": "Telugu",
    "kn": "Kannada",
    "gu": "Gujarati",
}

PRAKRITI_TYPES = [
    "Vata",
    "Pitta",
    "Kapha",
    "Vata-Pitta",
    "Pitta-Kapha",
    "Vata-Kapha",
    "Sama",
]

AYUSH_SYSTEMS = ["Ayurveda", "Yoga", "Unani", "Siddha", "Homeopathy"]

PRIORITY_CONDITIONS = {
    "Vibandha": "K59.0",
    "Amlapitta": "K21",
    "Sandhivata": "M15-M19",
    "Prameha": "E11",
    "Sthaulya": "E66",
    "Raktachapa": "I10-I15",
    "Jwara": "R50",
    "Ashmari": "N20",
}

ALERT_LEVELS = {"WATCH": 2.0, "WARNING": 2.5, "ALERT": 3.5}
