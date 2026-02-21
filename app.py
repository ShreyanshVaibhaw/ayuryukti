"""AyurYukti Streamlit dashboard with integrated module workflows."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    BHASHINI_API_URL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PROJECT_NAME,
    SUPPORTED_LANGUAGES,
)
from src.common.database import get_postgres_engine
from src.common.logger import get_performance_logger, setup_logger
from src.llm.ollama_client import LLMClient
from src.prakritimitra.knowledge_graph import AyushKnowledgeGraph
from src.prakritimitra.lifestyle_advisor import LifestyleAdvisor
from src.prakritimitra.prakriti_classifier import PrakritiClassifier
from src.prakritimitra.recommendation_engine import RecommendationEngine
from src.reporting.analytics_report import export_analytics_excel, generate_analytics_pdf
from src.reporting.ehr_report import export_ehr_json, generate_ehr_pdf
from src.reporting.recommendation_report import generate_recommendation_pdf
from src.reporting.surveillance_report import generate_surveillance_pdf
from src.rogaradar.alert_generator import AlertGenerator
from src.rogaradar.anomaly_detector import AnomalyDetector
from src.rogaradar.baseline_model import BaselineModel
from src.rogaradar.data_ingestion import DataIngestion
from src.rogaradar.geo_cluster import GeoCluster
from src.rogaradar.surveillance_dashboard import SurveillanceDashboard
from src.vaksetu.code_mapper import CodeMapper
from src.vaksetu.ehr_generator import EHRGenerator
from src.vaksetu.medical_ner import MedicalNEREngine
from src.vaksetu.speech_engine import SAMPLE_TRANSCRIPTS, SpeechEngine
from src.yuktishaala.analytics import TreatmentAnalytics
from src.yuktishaala.contextual_bandit import ThompsonSamplingBandit
from src.yuktishaala.outcome_tracker import OutcomeTracker

try:
    from streamlit_folium import st_folium
except Exception:
    st_folium = None

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except Exception:
    WebRtcMode = None
    webrtc_streamer = None


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
KB = DATA / "knowledge_base"
OUTPUTS = ROOT / "outputs"
OUT_REPORTS = OUTPUTS / "reports"
OUT_EHR = OUTPUTS / "ehr"
OUT_LOGS = OUTPUTS / "logs"
OUT_MODELS = OUTPUTS / "models"
SYNTHETIC_VISITS = DATA / "synthetic" / "patient_visits.csv"
SYNTHETIC_SCENARIOS = DATA / "synthetic" / "outbreak_scenarios.csv"
VOCAB_PATH = DATA / "ayush_vocabulary.json"
Q_PATH = DATA / "prakriti" / "questionnaire.json"
TRAIN_PATH = DATA / "prakriti" / "training_data.csv"

LOGGER = setup_logger("AyurYuktiApp")
PERF_LOGGER = get_performance_logger()


st.set_page_config(page_title="AyurYukti — आयुर्युक्ति", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------------------
# UI Strings — Bilingual (English / Hindi)
# ---------------------------------------------------------------------------
UI_STRINGS = {
    "en": {
        "app_title": "AyurYukti",
        "app_sanskrit": "आयुर्युक्ति",
        "app_subtitle": "Rational Intelligence for Ayurveda",
        "tagline": "From Voice to Verdict — Intelligent AYUSH Healthcare",
        "nav_home": "Home",
        "nav_vaksetu": "VakSetu",
        "nav_rogaradar": "RogaRadar",
        "nav_prakriti": "Prakriti",
        "system_health": "System Health",
        "hero_title": "AyurYukti",
        "hero_subtitle": "From Voice to Verdict",
        "hero_desc": "AI-powered clinical decision support for AYUSH practitioners — bridging traditional wisdom with modern intelligence.",
        "hero_badge": "Ministry of AYUSH  |  IndiaAI Innovation Challenge 2026",
        "stat_formulations": "Formulations",
        "stat_morbidity": "Morbidity Codes",
        "stat_patients": "Patient Records",
        "stat_conditions": "Conditions",
        "mod_vaksetu": "VakSetu",
        "mod_vaksetu_hi": "वाक्सेतु",
        "mod_vaksetu_desc": "Voice-to-EHR pipeline. Speak naturally and get structured electronic health records with AYUSH diagnosis codes.",
        "mod_prakriti": "PrakritiMitra",
        "mod_prakriti_hi": "प्रकृतिमित्र",
        "mod_prakriti_desc": "Prakriti-aware treatment recommendations. Personalized formulations based on your constitutional type.",
        "mod_rogaradar": "RogaRadar",
        "mod_rogaradar_hi": "रोगरडार",
        "mod_rogaradar_desc": "Disease outbreak early warning system. Real-time surveillance with anomaly detection across districts.",
        "mod_yukti": "YuktiShaala",
        "mod_yukti_hi": "युक्तिशाला",
        "mod_yukti_desc": "Adaptive learning engine. Gets smarter with every patient outcome using contextual bandits.",
        "btn_open": "Open",
        "how_title": "How It Works",
        "how_step1": "Doctor Speaks",
        "how_step1_desc": "Natural voice input in any supported Indian language",
        "how_step2": "AI Extracts EHR",
        "how_step2_desc": "NER + LLM pipeline structures clinical data",
        "how_step3": "Prakriti-Aware Rx",
        "how_step3_desc": "Personalized treatment based on constitution",
        "how_step4": "System Learns",
        "how_step4_desc": "Outcomes improve future recommendations",
        "vaksetu_title": "VakSetu — Voice to EHR",
        "vaksetu_step1": "Input",
        "vaksetu_step2": "Process",
        "vaksetu_step3": "Results",
        "vaksetu_step4": "Actions",
        "tab_voice": "Voice",
        "tab_type": "Type",
        "tab_demo": "Demo",
        "btn_generate_ehr": "Generate EHR",
        "btn_get_recommendations": "Get Treatment Plan",
        "btn_export_json": "Export AHMIS JSON",
        "btn_export_pdf": "Download EHR PDF",
        "btn_new_consultation": "New Consultation",
        "ehr_title": "Structured EHR",
        "ehr_confidence": "Extraction Confidence",
        "ehr_patient": "Patient Info",
        "ehr_complaints": "Chief Complaints",
        "ehr_diagnosis": "Diagnosis",
        "ehr_prescriptions": "Prescriptions",
        "tab_lifestyle": "Lifestyle",
        "tab_diet": "Diet",
        "tab_yoga": "Yoga",
        "ehr_followup": "Follow-up",
        "ehr_disclaimer": "AI-assisted extraction. The treating physician makes the final clinical decision.",
        "rec_title": "PrakritiMitra Recommendations",
        "rec_disclaimer": "AI-assisted recommendations for decision support only. The treating physician makes the final clinical decision.",
        "rec_outcome": "improvement in similar",
        "rec_patients": "patients",
        "rec_no_history": "No prior outcome history for this arm",
        "rec_record": "Record Outcome",
        "rec_followup": "Follow-up Result",
        "rec_submit": "Submit Outcome",
        "rr_title": "RogaRadar — Surveillance Dashboard",
        "rr_districts": "Districts Monitored",
        "rr_alerts": "Active Alerts",
        "rr_conditions": "Conditions Watched",
        "rr_datapoints": "Data Points",
        "rr_map": "District Alert Map",
        "rr_alert_table": "Active Alerts",
        "rr_inspect": "Inspect Alert",
        "rr_timeseries": "Time Series",
        "rr_heatmap": "Heatmap",
        "rr_explorer": "Time Series Explorer",
        "prakriti_title": "Prakriti Assessment",
        "prakriti_subtitle": "Discover your Ayurvedic constitution through a guided 30-question assessment",
        "prakriti_progress": "completed",
        "prakriti_of": "of",
        "prakriti_questions": "questions",
        "prakriti_category": "Category",
        "btn_previous": "Previous",
        "btn_next": "Next",
        "btn_calculate": "Calculate Prakriti",
        "prakriti_result": "Your Prakriti",
        "prakriti_distribution": "Dosha Distribution",
        "btn_use_prakriti": "Use for Recommendations",
        "demo_banner": "Running in demo mode — some features use cached data",
        "demo_details": "Details",
        "lang_toggle": "हिन्दी",
        "voice_record": "Record your consultation",
        "voice_record_hint": "Select language, then click the microphone to record",
        "voice_transcribing": "Transcribing your voice...",
        "voice_transcribed": "Transcription complete",
        "voice_play": "Play recorded audio",
        "voice_result": "Transcribed text (edit if needed):",
        "voice_method": "Method",
    },
    "hi": {
        "app_title": "आयुर्युक्ति",
        "app_sanskrit": "AyurYukti",
        "app_subtitle": "आयुर्वेद के लिए बुद्धिमान प्रणाली",
        "tagline": "आवाज़ से निदान तक — बुद्धिमान आयुष स्वास्थ्य सेवा",
        "nav_home": "मुख्य पृष्ठ",
        "nav_vaksetu": "वाक्सेतु",
        "nav_rogaradar": "रोगरडार",
        "nav_prakriti": "प्रकृति",
        "system_health": "सिस्टम स्थिति",
        "hero_title": "आयुर्युक्ति",
        "hero_subtitle": "आवाज़ से निदान तक",
        "hero_desc": "आयुष चिकित्सकों के लिए AI-संचालित नैदानिक निर्णय सहायता — पारंपरिक ज्ञान को आधुनिक बुद्धिमत्ता से जोड़ना।",
        "hero_badge": "आयुष मंत्रालय  |  IndiaAI Innovation Challenge 2026",
        "stat_formulations": "फॉर्मूलेशन",
        "stat_morbidity": "रुग्णता कोड",
        "stat_patients": "रोगी रिकॉर्ड",
        "stat_conditions": "रोग स्थितियाँ",
        "mod_vaksetu": "वाक्सेतु",
        "mod_vaksetu_hi": "VakSetu",
        "mod_vaksetu_desc": "वॉइस-टू-EHR पाइपलाइन। स्वाभाविक रूप से बोलें और AYUSH निदान कोड के साथ संरचित EHR प्राप्त करें।",
        "mod_prakriti": "प्रकृतिमित्र",
        "mod_prakriti_hi": "PrakritiMitra",
        "mod_prakriti_desc": "प्रकृति-जागरूक उपचार सिफारिशें। आपके संवैधानिक प्रकार के आधार पर व्यक्तिगत फॉर्मूलेशन।",
        "mod_rogaradar": "रोगरडार",
        "mod_rogaradar_hi": "RogaRadar",
        "mod_rogaradar_desc": "रोग प्रकोप पूर्व चेतावनी प्रणाली। जिलों में विसंगति पहचान के साथ वास्तविक समय निगरानी।",
        "mod_yukti": "युक्तिशाला",
        "mod_yukti_hi": "YuktiShaala",
        "mod_yukti_desc": "अनुकूली शिक्षण इंजन। प्रासंगिक बैंडिट्स का उपयोग करके हर रोगी परिणाम के साथ स्मार्ट होता है।",
        "btn_open": "खोलें",
        "how_title": "यह कैसे काम करता है",
        "how_step1": "डॉक्टर बोलते हैं",
        "how_step1_desc": "किसी भी समर्थित भारतीय भाषा में प्राकृतिक वॉइस इनपुट",
        "how_step2": "AI EHR निकालता है",
        "how_step2_desc": "NER + LLM पाइपलाइन नैदानिक डेटा को संरचित करती है",
        "how_step3": "प्रकृति-आधारित Rx",
        "how_step3_desc": "संविधान के आधार पर व्यक्तिगत उपचार",
        "how_step4": "सिस्टम सीखता है",
        "how_step4_desc": "परिणाम भविष्य की सिफारिशों में सुधार करते हैं",
        "vaksetu_title": "वाक्सेतु — वॉइस से EHR",
        "vaksetu_step1": "इनपुट",
        "vaksetu_step2": "प्रक्रिया",
        "vaksetu_step3": "परिणाम",
        "vaksetu_step4": "कार्रवाई",
        "tab_voice": "आवाज़",
        "tab_type": "टाइप",
        "tab_demo": "डेमो",
        "btn_generate_ehr": "EHR बनाएं",
        "btn_get_recommendations": "उपचार योजना प्राप्त करें",
        "btn_export_json": "AHMIS JSON निर्यात",
        "btn_export_pdf": "EHR PDF डाउनलोड",
        "btn_new_consultation": "नया परामर्श",
        "ehr_title": "संरचित EHR",
        "ehr_confidence": "निष्कर्षण विश्वास",
        "ehr_patient": "रोगी जानकारी",
        "ehr_complaints": "मुख्य शिकायतें",
        "ehr_diagnosis": "निदान",
        "ehr_prescriptions": "नुस्खे",
        "tab_lifestyle": "जीवनशैली",
        "tab_diet": "आहार",
        "tab_yoga": "योग",
        "ehr_followup": "अनुवर्ती",
        "ehr_disclaimer": "AI-सहायित निष्कर्षण। उपचार चिकित्सक अंतिम नैदानिक निर्णय लेता है।",
        "rec_title": "प्रकृतिमित्र सिफारिशें",
        "rec_disclaimer": "केवल निर्णय समर्थन के लिए AI-सहायित सिफारिशें। उपचार चिकित्सक अंतिम नैदानिक निर्णय लेता है।",
        "rec_outcome": "समान में सुधार",
        "rec_patients": "रोगी",
        "rec_no_history": "इस भुजा के लिए कोई पूर्व परिणाम इतिहास नहीं",
        "rec_record": "परिणाम दर्ज करें",
        "rec_followup": "अनुवर्ती परिणाम",
        "rec_submit": "परिणाम जमा करें",
        "rr_title": "रोगरडार — निगरानी डैशबोर्ड",
        "rr_districts": "निगरानी वाले जिले",
        "rr_alerts": "सक्रिय अलर्ट",
        "rr_conditions": "निगरानी में रोग",
        "rr_datapoints": "डेटा बिंदु",
        "rr_map": "जिला अलर्ट मानचित्र",
        "rr_alert_table": "सक्रिय अलर्ट",
        "rr_inspect": "अलर्ट का निरीक्षण",
        "rr_timeseries": "समय श्रृंखला",
        "rr_heatmap": "हीटमैप",
        "rr_explorer": "समय श्रृंखला एक्सप्लोरर",
        "prakriti_title": "प्रकृति मूल्यांकन",
        "prakriti_subtitle": "30-प्रश्न निर्देशित मूल्यांकन के माध्यम से अपनी आयुर्वेदिक प्रकृति खोजें",
        "prakriti_progress": "पूर्ण",
        "prakriti_of": "में से",
        "prakriti_questions": "प्रश्न",
        "prakriti_category": "श्रेणी",
        "btn_previous": "पिछला",
        "btn_next": "अगला",
        "btn_calculate": "प्रकृति की गणना करें",
        "prakriti_result": "आपकी प्रकृति",
        "prakriti_distribution": "दोष वितरण",
        "btn_use_prakriti": "सिफारिशों के लिए उपयोग करें",
        "demo_banner": "डेमो मोड में चल रहा है — कुछ सुविधाएं कैश्ड डेटा का उपयोग करती हैं",
        "demo_details": "विवरण",
        "lang_toggle": "English",
        "voice_record": "अपना परामर्श रिकॉर्ड करें",
        "voice_record_hint": "भाषा चुनें, फिर रिकॉर्ड करने के लिए माइक्रोफ़ोन पर क्लिक करें",
        "voice_transcribing": "आपकी आवाज़ का लिप्यंतरण हो रहा है...",
        "voice_transcribed": "लिप्यंतरण पूर्ण",
        "voice_play": "रिकॉर्ड किया गया ऑडियो चलाएं",
        "voice_result": "लिप्यंतरित पाठ (आवश्यकतानुसार संपादित करें):",
        "voice_method": "विधि",
    },
}


def T(key: str) -> str:
    """Return UI string for current language."""
    lang = "hi" if st.session_state.get("ui_language") == "Hindi" else "en"
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(key, UI_STRINGS["en"].get(key, key))


# ---------------------------------------------------------------------------
# Custom CSS — Modern Design System
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --primary: #0F4C5C;
    --accent: #E8913A;
    --accent-light: #FFF3E6;
    --success: #2D6A4F;
    --danger: #C1292E;
    --surface: #FFFFFF;
    --bg: #FAFAF8;
    --text: #1A1A2E;
    --text-muted: #6B7280;
    --border: #E5E7EB;
    --shadow: 0 4px 24px rgba(0,0,0,0.06);
    --radius: 16px;
}

/* Hide default hamburger menu for cleaner look */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* Base typography */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1200px;
}

html, body, [class*="css"] {
    font-size: 16px;
    color: var(--text);
}

/* Hindi text rendering */
[lang="hi"], .hindi-text {
    line-height: 1.8;
}

/* ---- App Header / Hero ---- */
.app-hero {
    background: linear-gradient(135deg, #0F4C5C 0%, #1a6b7a 40%, #E8913A 100%);
    border-radius: var(--radius);
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.8rem;
    color: #fff;
    position: relative;
    overflow: hidden;
}
.app-hero::before {
    content: "";
    position: absolute;
    top: -40%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.app-hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.app-hero .hero-sanskrit {
    font-size: 1.5rem;
    opacity: 0.85;
    margin: 0.2rem 0 0;
    font-weight: 400;
}
.app-hero .hero-tagline {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0.8rem 0 0;
    font-weight: 300;
    max-width: 600px;
}
.app-hero .hero-badge {
    font-size: 0.82rem;
    opacity: 0.7;
    margin-top: 1.2rem;
    letter-spacing: 0.5px;
}

/* ---- Navigation Pills ---- */
.nav-pill-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin: 0.5rem 0 1rem;
}
.nav-pill {
    display: block;
    padding: 0.65rem 1rem;
    border-radius: 10px;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text);
    background: transparent;
    text-decoration: none;
    transition: all 0.2s ease;
    cursor: pointer;
    border: none;
    text-align: left;
    width: 100%;
}
.nav-pill:hover {
    background: var(--accent-light);
}
.nav-pill.active {
    background: var(--accent);
    color: #fff;
    font-weight: 600;
}

/* ---- Cards ---- */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.09);
}
.card-accent {
    border-left: 4px solid var(--accent);
}
.card-teal {
    border-left: 4px solid var(--primary);
}
.card-green {
    border-left: 4px solid var(--success);
}
.card-warm {
    border-left: 4px solid #D97706;
}

/* ---- Stat Cards ---- */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    box-shadow: var(--shadow);
    text-align: center;
}
.stat-card .stat-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1.1;
}
.stat-card .stat-label {
    font-size: 0.88rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    font-weight: 500;
}

/* ---- Module Cards ---- */
.module-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.module-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
.module-card .mod-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
}
.module-card .mod-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.module-card .mod-name-hi {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin: 0.1rem 0 0.6rem;
}
.module-card .mod-desc {
    font-size: 0.9rem;
    color: var(--text-muted);
    line-height: 1.5;
    flex: 1;
}

/* ---- Step Indicator ---- */
.step-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 1rem 0 2rem;
    padding: 0;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0;
}
.step-circle {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.95rem;
    flex-shrink: 0;
}
.step-circle.active {
    background: var(--accent);
    color: #fff;
}
.step-circle.completed {
    background: var(--success);
    color: #fff;
}
.step-circle.pending {
    background: var(--border);
    color: var(--text-muted);
}
.step-label {
    font-size: 0.82rem;
    margin-left: 0.4rem;
    font-weight: 500;
    white-space: nowrap;
}
.step-label.active {
    color: var(--accent);
    font-weight: 700;
}
.step-label.completed {
    color: var(--success);
}
.step-label.pending {
    color: var(--text-muted);
}
.step-line {
    width: 60px;
    height: 2px;
    margin: 0 0.5rem;
    flex-shrink: 0;
}
.step-line.completed {
    background: var(--success);
}
.step-line.pending {
    background: var(--border);
}

/* ---- Badges ---- */
.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-good {
    background: rgba(45,106,79,0.12);
    color: #2D6A4F;
}
.badge-warn {
    background: rgba(232,145,58,0.16);
    color: #B45309;
}
.badge-alert {
    background: rgba(193,41,46,0.12);
    color: #C1292E;
}
.badge-vata {
    background: rgba(59,130,246,0.14);
    color: #2563EB;
}
.badge-pitta {
    background: rgba(239,68,68,0.14);
    color: #DC2626;
}
.badge-kapha {
    background: rgba(34,197,94,0.14);
    color: #16A34A;
}
.badge-mixed {
    background: linear-gradient(90deg, rgba(59,130,246,0.12), rgba(239,68,68,0.12), rgba(34,197,94,0.12));
    color: var(--text);
}

/* ---- Section Header ---- */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    border-left: 4px solid var(--accent);
    padding-left: 0.8rem;
    margin: 1.8rem 0 1rem;
}

/* ---- EHR Field ---- */
.ehr-field {
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
}
.ehr-field:last-child {
    border-bottom: none;
}
.ehr-field .ehr-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.ehr-field .ehr-value {
    font-size: 1rem;
    color: var(--text);
    font-weight: 500;
    margin-top: 0.15rem;
}

/* ---- Questionnaire Card ---- */
.questionnaire-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}
.questionnaire-card .q-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.8rem;
}
.questionnaire-card .q-text {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.5;
}
.questionnaire-card .q-text-hi {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
    line-height: 1.7;
}

/* ---- Progress Modern ---- */
.progress-modern {
    background: var(--border);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.progress-modern .progress-fill {
    background: linear-gradient(90deg, var(--accent), #D97706);
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}

/* ---- Circular Progress ---- */
.circular-progress {
    text-align: center;
    padding: 1rem;
}
.circular-progress .cp-number {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--accent);
}
.circular-progress .cp-label {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* ---- Alert Banner ---- */
.alert-banner {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    font-size: 0.9rem;
    color: #92400E;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.alert-banner .alert-icon {
    font-size: 1.1rem;
}

/* ---- How It Works ---- */
.how-step {
    text-align: center;
    padding: 1rem 0.5rem;
}
.how-step .how-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.how-step .how-label {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.3rem;
}
.how-step .how-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.4;
}
.how-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--accent);
    font-size: 1.5rem;
    padding-top: 1rem;
}

/* ---- Sidebar Styles ---- */
.sidebar-brand {
    background: linear-gradient(135deg, var(--primary), #1a6b7a);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    color: #fff;
    text-align: center;
}
.sidebar-brand h3 {
    margin: 0;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.3px;
}
.sidebar-brand .brand-sub {
    font-size: 0.92rem;
    opacity: 0.8;
    margin-top: 0.2rem;
}

/* ---- Footer ---- */
.footer-minimal {
    text-align: center;
    font-size: 0.78rem;
    color: var(--text-muted);
    padding: 1rem 0 0.5rem;
    border-top: 1px solid var(--border);
    margin-top: 1rem;
}

/* ---- Severity Dot ---- */
.severity-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.severity-dot.alert { background: var(--danger); }
.severity-dot.warning { background: var(--accent); }
.severity-dot.watch { background: var(--success); }

/* ---- Diagnosis Bridge ---- */
.diagnosis-bridge {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 0;
}
.diagnosis-bridge .diag-box {
    flex: 1;
    padding: 0.8rem;
    border-radius: 10px;
    text-align: center;
}
.diagnosis-bridge .diag-box.ayush {
    background: rgba(15,76,92,0.06);
    border: 1px solid rgba(15,76,92,0.15);
}
.diagnosis-bridge .diag-box.icd {
    background: rgba(232,145,58,0.06);
    border: 1px solid rgba(232,145,58,0.15);
}
.diagnosis-bridge .diag-arrow {
    font-size: 1.3rem;
    color: var(--text-muted);
}
.diagnosis-bridge .diag-code {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
}
.diagnosis-bridge .diag-name {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}
.diagnosis-bridge .diag-system {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.3rem;
}

/* Streamlit overrides */
.stButton > button[kind="primary"] {
    background: var(--accent);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    font-size: 0.95rem;
}
.stButton > button[kind="primary"]:hover {
    background: #D97706;
}
.stButton > button[kind="secondary"] {
    border-radius: 10px;
    font-weight: 500;
}

/* Confidence bar */
.confidence-bar-container {
    margin: 0.3rem 0;
}
.confidence-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}
.confidence-bar {
    background: var(--border);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.confidence-bar .cb-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}
.confidence-bar .cb-fill.high { background: var(--success); }
.confidence-bar .cb-fill.medium { background: var(--accent); }
.confidence-bar .cb-fill.low { background: var(--danger); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _init_session_state() -> None:
    defaults = {
        "nav_page": "Home / Overview",
        "current_ehr": None,
        "current_ahmis_json": None,
        "current_transcript": "",
        "current_recommendations": None,
        "prakriti_result": None,
        "loaded_data": None,
        "alert_results": None,
        "current_encounter_id": None,
        "prakriti_answers": {},
        "prakriti_q_index": 0,
        "ui_language": "English",
        "last_ehr_seconds": None,
        "last_recommendation_seconds": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_output_dirs() -> None:
    for path in [OUT_REPORTS, OUT_EHR, OUT_LOGS, OUT_MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def _read_file_bytes(path: Path) -> bytes:
    return path.read_bytes()


@st.cache_resource(show_spinner=False)
def get_llm_client() -> LLMClient:
    return LLMClient()


@st.cache_resource(show_spinner=False)
def get_knowledge_graph() -> AyushKnowledgeGraph:
    kg = AyushKnowledgeGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    try:
        kg.setup_schema()
        kg.seed_from_json(
            formulations_path=str(KB / "formulations.json"),
            morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
            prakriti_rules_path=str(KB / "prakriti_rules.json"),
        )
    except Exception:
        LOGGER.exception("Knowledge graph initialization failed; using fallback store.")
    return kg


@st.cache_resource(show_spinner=False)
def get_vaksetu_stack() -> Tuple[SpeechEngine, EHRGenerator]:
    speech = SpeechEngine()
    mapper = CodeMapper(
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
        icd10_mapping_path=str(KB / "icd10_mapping.json"),
    )
    ner = MedicalNEREngine(
        llm_client=get_llm_client(),
        vocabulary_path=str(VOCAB_PATH),
        morbidity_codes_path=str(KB / "ayush_morbidity_codes.json"),
    )
    generator = EHRGenerator(ner_engine=ner, code_mapper=mapper)
    return speech, generator


@st.cache_resource(show_spinner=False)
def get_learning_components() -> Tuple[OutcomeTracker, ThompsonSamplingBandit, TreatmentAnalytics]:
    tracker = OutcomeTracker()
    bandit = ThompsonSamplingBandit(exploration_rate=0.15)

    if SYNTHETIC_VISITS.exists():
        tracker.seed_from_synthetic(str(SYNTHETIC_VISITS))
        for row in tracker.list_all():
            reward = 1.0 if row["outcome"] == "Improved" else 0.5 if row["outcome"] == "No Change" else 0.0
            bandit.update(
                prakriti=row["patient_prakriti"],
                condition=row["condition_code"],
                formulation=row["formulation_name"],
                reward=reward,
            )

    analytics = TreatmentAnalytics(tracker, bandit)
    return tracker, bandit, analytics


@st.cache_resource(show_spinner=False)
def get_recommendation_engine() -> RecommendationEngine:
    tracker, bandit, _ = get_learning_components()
    return RecommendationEngine(
        knowledge_graph=get_knowledge_graph(),
        llm_client=get_llm_client(),
        outcome_tracker=tracker,
        bandit=bandit,
    )


@st.cache_resource(show_spinner=False)
def get_prakriti_tools() -> Tuple[PrakritiClassifier, LifestyleAdvisor, List[Dict]]:
    classifier = PrakritiClassifier(questionnaire_path=str(Q_PATH), training_data_path=str(TRAIN_PATH))
    advisor = LifestyleAdvisor(prakriti_rules_path=KB / "prakriti_rules.json")
    questionnaire = _load_json(Q_PATH)
    return classifier, advisor, questionnaire


@st.cache_data(show_spinner=False)
def get_quick_stats() -> Dict[str, int]:
    formulations = len(_load_json(KB / "formulations.json")) if (KB / "formulations.json").exists() else 0
    morbidity_codes = len(_load_json(KB / "ayush_morbidity_codes.json")) if (KB / "ayush_morbidity_codes.json").exists() else 0
    patients = len(pd.read_csv(SYNTHETIC_VISITS)) if SYNTHETIC_VISITS.exists() else 0
    conditions = pd.read_csv(SYNTHETIC_VISITS)["ayush_diagnosis_name"].nunique() if SYNTHETIC_VISITS.exists() else 0
    return {
        "formulations": int(formulations),
        "morbidity_codes": int(morbidity_codes),
        "synthetic_patients": int(patients),
        "conditions": int(conditions),
    }


@st.cache_resource(show_spinner=False)
def load_surveillance_data() -> Dict[str, object]:
    try:
        ingestion = DataIngestion(str(SYNTHETIC_VISITS))
        visits = ingestion.load_visit_data()
        agg = ingestion.aggregate_by_district_condition_week(visits)

        baseline = BaselineModel()
        if not agg.empty:
            baseline.fit_all(agg)

        detector = AnomalyDetector(baseline_model=baseline)
        anomalies = detector.run_all_detectors(agg) if not agg.empty else []
        district_meta = ingestion.get_district_metadata()
        geo = GeoCluster(district_metadata=district_meta)
        clusters = geo.cluster_anomalies(anomalies)
        alerts = AlertGenerator({"WATCH": 1.5, "WARNING": 2.0, "ALERT": 3.0}).generate_alerts(anomalies, clusters)

        dashboard = SurveillanceDashboard()
        fmap = dashboard.create_district_map(alerts, district_meta)
        alert_table = dashboard.create_alert_summary_table(alerts)
        heatmap = dashboard.create_condition_heatmap(agg)

        return {
            "visits": visits,
            "aggregated": agg,
            "anomalies": anomalies,
            "clusters": clusters,
            "alerts": alerts,
            "district_meta": district_meta,
            "map": fmap,
            "alert_table": alert_table,
            "heatmap": heatmap,
            "dashboard": dashboard,
        }
    except Exception:
        LOGGER.exception("Surveillance pipeline failed; returning empty fallback payload.")
        dashboard = SurveillanceDashboard()
        empty = pd.DataFrame()
        return {
            "visits": empty,
            "aggregated": empty,
            "anomalies": [],
            "clusters": [],
            "alerts": [],
            "district_meta": pd.DataFrame(columns=["district", "state", "lat", "lon"]),
            "map": dashboard.create_district_map([], pd.DataFrame(columns=["district", "state", "lat", "lon"])),
            "alert_table": empty,
            "heatmap": dashboard.create_condition_heatmap(empty),
            "dashboard": dashboard,
        }


# ---------------------------------------------------------------------------
# Badge / Helper Functions
# ---------------------------------------------------------------------------

def _status_badge(label: str, ok: bool) -> str:
    dot = "🟢" if ok else "🔴"
    return f"{dot} {label}"


def _confidence_badge(score: float) -> str:
    if score >= 0.75:
        return '<span class="badge badge-good">HIGH</span>'
    if score >= 0.5:
        return '<span class="badge badge-warn">MEDIUM</span>'
    return '<span class="badge badge-alert">LOW</span>'


def _confidence_bar(label: str, score: float) -> str:
    pct = int(score * 100)
    level = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    return f"""
    <div class="confidence-bar-container">
        <div class="confidence-bar-label"><span>{label}</span><span>{pct}%</span></div>
        <div class="confidence-bar"><div class="cb-fill {level}" style="width:{pct}%"></div></div>
    </div>"""


def _prakriti_badge(prakriti: str) -> str:
    if prakriti in {"Vata"}:
        cls = "badge-vata"
    elif prakriti in {"Pitta"}:
        cls = "badge-pitta"
    elif prakriti in {"Kapha"}:
        cls = "badge-kapha"
    else:
        cls = "badge-mixed"
    return f"<span class='badge {cls}'>{prakriti}</span>"


def _severity_badge(level: str) -> str:
    if level == "ALERT":
        return '<span class="badge badge-alert">ALERT</span>'
    if level == "WARNING":
        return '<span class="badge badge-warn">WARNING</span>'
    return '<span class="badge badge-good">WATCH</span>'


def _step_indicator_html(active: int, labels: list) -> str:
    """Build HTML step indicator. active is 1-indexed."""
    parts = []
    for i, label in enumerate(labels, 1):
        if i < active:
            state = "completed"
            num = "&#10003;"
        elif i == active:
            state = "active"
            num = str(i)
        else:
            state = "pending"
            num = str(i)
        parts.append(f'<div class="step-item"><div class="step-circle {state}">{num}</div><span class="step-label {state}">{label}</span></div>')
        if i < len(labels):
            line_state = "completed" if i < active else "pending"
            parts.append(f'<div class="step-line {line_state}"></div>')
    return f'<div class="step-indicator">{"".join(parts)}</div>'


def _check_postgres() -> bool:
    try:
        return get_postgres_engine() is not None
    except Exception:
        LOGGER.exception("PostgreSQL status check failed.")
        return False


def _check_bhashini() -> bool:
    return bool(BHASHINI_API_URL and os.getenv("BHASHINI_API_KEY"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> Tuple[bool, bool, bool, bool]:
    llm = get_llm_client()
    kg = get_knowledge_graph()

    ollama_ok = bool(llm.health_check().get("ok"))
    neo4j_ok = bool(kg.driver)
    postgres_ok = _check_postgres()
    bhashini_ok = _check_bhashini()

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
              <h3>{T("app_title")}</h3>
              <div class="brand-sub">{T("app_sanskrit")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        lang_options = ["English", "Hindi"]
        current_lang = st.session_state.get("ui_language", "English")
        st.radio(
            T("lang_toggle"),
            lang_options,
            index=lang_options.index(current_lang),
            key="ui_language",
            horizontal=True,
        )

        st.markdown("---")

        nav_items = {
            "Home / Overview": ("nav_home", "🏠"),
            "VakSetu": ("nav_vaksetu", "🎙️"),
            "RogaRadar": ("nav_rogaradar", "🗺️"),
            "Prakriti Assessment": ("nav_prakriti", "🧬"),
        }
        current = st.session_state.get("nav_page", "Home / Overview")

        for page_key, (label_key, icon) in nav_items.items():
            is_active = current == page_key
            btn_label = f"{icon} {T(label_key)}"
            if st.button(
                btn_label,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["nav_page"] = page_key
                st.rerun()

        st.markdown("---")

        with st.expander(T("system_health"), expanded=False):
            st.caption(_status_badge("Ollama LLM", ollama_ok))
            st.caption(_status_badge("Neo4j Graph", neo4j_ok))
            st.caption(_status_badge("PostgreSQL", postgres_ok))
            st.caption(_status_badge("Bhashini NLP", bhashini_ok))

        st.markdown('<div class="footer-minimal">v1.0 | IndiaAI 2026</div>', unsafe_allow_html=True)

    return ollama_ok, neo4j_ok, postgres_ok, bhashini_ok


# ---------------------------------------------------------------------------
# Home Page
# ---------------------------------------------------------------------------

def render_home() -> None:
    stats = get_quick_stats()

    # Hero
    st.markdown(
        f"""
        <div class="app-hero">
          <h1>{T("hero_title")}</h1>
          <p class="hero-sanskrit">{T("app_sanskrit")}</p>
          <p class="hero-tagline">{T("hero_desc")}</p>
          <p class="hero-badge">{T("hero_badge")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Stats Row
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["formulations"]}</div><div class="stat-label">{T("stat_formulations")}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["morbidity_codes"]}</div><div class="stat-label">{T("stat_morbidity")}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["synthetic_patients"]}</div><div class="stat-label">{T("stat_patients")}</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["conditions"]}</div><div class="stat-label">{T("stat_conditions")}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Module Cards (2x2)
    modules = [
        {
            "icon": "🎙️", "name_key": "mod_vaksetu", "hi_key": "mod_vaksetu_hi",
            "desc_key": "mod_vaksetu_desc", "border": "card-accent", "page": "VakSetu",
        },
        {
            "icon": "🧬", "name_key": "mod_prakriti", "hi_key": "mod_prakriti_hi",
            "desc_key": "mod_prakriti_desc", "border": "card-green", "page": "Prakriti Assessment",
        },
        {
            "icon": "🗺️", "name_key": "mod_rogaradar", "hi_key": "mod_rogaradar_hi",
            "desc_key": "mod_rogaradar_desc", "border": "card-teal", "page": "RogaRadar",
        },
        {
            "icon": "📈", "name_key": "mod_yukti", "hi_key": "mod_yukti_hi",
            "desc_key": "mod_yukti_desc", "border": "card-warm", "page": "VakSetu",
        },
    ]

    row1_c1, row1_c2 = st.columns(2)
    row2_c1, row2_c2 = st.columns(2)
    cols = [row1_c1, row1_c2, row2_c1, row2_c2]

    for col, mod in zip(cols, modules):
        with col:
            st.markdown(
                f"""
                <div class="module-card {mod['border']}">
                  <div class="mod-icon">{mod['icon']}</div>
                  <div class="mod-name">{T(mod['name_key'])}</div>
                  <div class="mod-name-hi">{T(mod['hi_key'])}</div>
                  <div class="mod-desc">{T(mod['desc_key'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"{T('btn_open')} →", key=f"home_{mod['name_key']}", use_container_width=True):
                st.session_state["nav_page"] = mod["page"]
                st.rerun()

    # How It Works
    st.markdown(f'<div class="section-header">{T("how_title")}</div>', unsafe_allow_html=True)

    h1, ha1, h2, ha2, h3, ha3, h4 = st.columns([2, 1, 2, 1, 2, 1, 2])
    with h1:
        st.markdown(f'<div class="how-step"><div class="how-icon">🎙️</div><div class="how-label">{T("how_step1")}</div><div class="how-desc">{T("how_step1_desc")}</div></div>', unsafe_allow_html=True)
    with ha1:
        st.markdown('<div class="how-arrow">→</div>', unsafe_allow_html=True)
    with h2:
        st.markdown(f'<div class="how-step"><div class="how-icon">📋</div><div class="how-label">{T("how_step2")}</div><div class="how-desc">{T("how_step2_desc")}</div></div>', unsafe_allow_html=True)
    with ha2:
        st.markdown('<div class="how-arrow">→</div>', unsafe_allow_html=True)
    with h3:
        st.markdown(f'<div class="how-step"><div class="how-icon">🧬</div><div class="how-label">{T("how_step3")}</div><div class="how-desc">{T("how_step3_desc")}</div></div>', unsafe_allow_html=True)
    with ha3:
        st.markdown('<div class="how-arrow">→</div>', unsafe_allow_html=True)
    with h4:
        st.markdown(f'<div class="how-step"><div class="how-icon">📈</div><div class="how-label">{T("how_step4")}</div><div class="how-desc">{T("how_step4_desc")}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# VakSetu Page
# ---------------------------------------------------------------------------

def _render_ehr(ehr) -> None:
    if not ehr:
        return

    st.markdown(f'<div class="section-header">{T("ehr_title")}</div>', unsafe_allow_html=True)

    # Confidence scores — horizontal bar chart
    conf = getattr(ehr, "confidence_scores", {}) or {}
    if conf:
        overall = conf.get("overall", 0)
        st.markdown(
            f"**{T('ehr_confidence')}:** {_confidence_badge(overall)} ({overall:.0%})",
            unsafe_allow_html=True,
        )
        field_labels = [
            ("age", "Age"), ("sex", "Sex"), ("prakriti", "Prakriti"),
            ("diagnosis", "Diagnosis"), ("prescriptions", "Rx"), ("chief_complaints", "Complaints"),
        ]
        bars_html = "".join(_confidence_bar(label, conf.get(key, 0)) for key, label in field_labels)
        st.markdown(f'<div class="card" style="padding:1rem;margin-bottom:1rem;">{bars_html}</div>', unsafe_allow_html=True)

    # Patient + Diagnosis row
    c1, c2 = st.columns([1, 1])
    with c1:
        age = ehr.patient_demographics.get("age")
        sex = ehr.patient_demographics.get("sex", "NA")
        prak = ehr.prakriti_assessment or "Not Captured"
        prak_html = _prakriti_badge(prak) if prak != "Not Captured" else f'<span class="badge badge-warn">{prak}</span>'
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:1rem;">
                <div style="font-weight:700;margin-bottom:0.8rem;">{T("ehr_patient")}</div>
                <div class="ehr-field"><div class="ehr-label">Age</div><div class="ehr-value">{age if age else 'NA'}</div></div>
                <div class="ehr-field"><div class="ehr-label">Sex</div><div class="ehr-value">{sex}</div></div>
                <div class="ehr-field"><div class="ehr-label">Prakriti</div><div class="ehr-value">{prak_html}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        ayush_code = ehr.ayush_diagnosis.get("code", "-")
        ayush_name = ehr.ayush_diagnosis.get("name", "-")
        icd_code = ehr.icd10_diagnosis.get("code", "-")
        icd_name = ehr.icd10_diagnosis.get("name", "-")
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:1rem;">
                <div style="font-weight:700;margin-bottom:0.8rem;">{T("ehr_diagnosis")}</div>
                <div class="diagnosis-bridge">
                    <div class="diag-box ayush">
                        <div class="diag-system">AYUSH</div>
                        <div class="diag-code">{ayush_code}</div>
                        <div class="diag-name">{ayush_name}</div>
                    </div>
                    <div class="diag-arrow">→</div>
                    <div class="diag-box icd">
                        <div class="diag-system">ICD-10</div>
                        <div class="diag-code">{icd_code}</div>
                        <div class="diag-name">{icd_name}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chief Complaints
    complaints = ehr.chief_complaints or []
    if complaints:
        st.markdown(f'<div class="section-header" style="font-size:1.1rem;">{T("ehr_complaints")}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(complaints), use_container_width=True, hide_index=True)

    # Prescriptions
    st.markdown(f'<div class="section-header" style="font-size:1.1rem;">{T("ehr_prescriptions")}</div>', unsafe_allow_html=True)
    if ehr.prescriptions:
        st.dataframe(pd.DataFrame(ehr.prescriptions), use_container_width=True, hide_index=True)
    else:
        st.info("No prescriptions extracted.")

    # Advice as tabs
    tab_lifestyle, tab_diet, tab_yoga = st.tabs([
        f"🧘 {T('tab_lifestyle')}",
        f"🍽️ {T('tab_diet')}",
        f"🧎 {T('tab_yoga')}",
    ])
    with tab_lifestyle:
        for item in ehr.lifestyle_advice or []:
            st.write(f"- {item}")
        if not ehr.lifestyle_advice:
            st.caption("No lifestyle advice extracted.")
    with tab_diet:
        for item in ehr.dietary_advice or []:
            st.write(f"- {item}")
        if not ehr.dietary_advice:
            st.caption("No dietary advice extracted.")
    with tab_yoga:
        for item in ehr.yoga_advice or []:
            st.write(f"- {item}")
        if not ehr.yoga_advice:
            st.caption("No yoga advice extracted.")

    st.caption(f"{T('ehr_followup')}: {ehr.follow_up or 'To be scheduled'}")
    st.caption(T("ehr_disclaimer"))


def _render_recommendations() -> None:
    rec = st.session_state.get("current_recommendations")
    if not rec:
        return

    tracker, _, analytics = get_learning_components()

    st.markdown(f'<div class="section-header">{T("rec_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-banner"><span class="alert-icon">ℹ️</span>{T("rec_disclaimer")}</div>', unsafe_allow_html=True)
    st.write(f"**Prakriti:** {rec.patient_prakriti} | **Condition:** {rec.condition}")

    for idx, row in enumerate(rec.recommended_formulations, start=1):
        stats = tracker.get_outcomes_for_treatment(
            rec.patient_prakriti,
            rec.condition,
            row["formulation_name"],
        )
        outcome_text = (
            f"{stats['success_rate'] * 100:.0f}% {T('rec_outcome')} {rec.patient_prakriti} {T('rec_patients')} (n={stats['total']})"
            if stats["total"] > 0
            else T("rec_no_history")
        )

        st.markdown(
            f"""
            <div class="card card-accent" style="margin-bottom:0.8rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <h4 style="margin:0;font-size:1.05rem;">#{idx} {row['formulation_name']}</h4>
                {_confidence_badge(float(row['score']))}
              </div>
              <p style="margin:0.5rem 0 0.3rem;font-size:0.9rem;"><b>Dosage:</b> {row['dosage']} &nbsp; <b>Reference:</b> {row.get('classical_reference', '-')}</p>
              <p style="margin:0;font-size:0.88rem;color:var(--success);">{outcome_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Suggestion tabs
    tab_ls, tab_dt, tab_yg = st.tabs([
        f"🧘 {T('tab_lifestyle')}",
        f"🍽️ {T('tab_diet')}",
        f"🧎 {T('tab_yoga')}",
    ])
    with tab_ls:
        for item in rec.lifestyle_suggestions:
            st.write(f"- {item}")
    with tab_dt:
        for item in rec.dietary_suggestions:
            st.write(f"- {item}")
    with tab_yg:
        for item in rec.yoga_suggestions:
            st.write(f"- {item}")

    with st.expander(T("rec_record")):
        follow_up = st.selectbox(
            T("rec_followup"),
            ["Improved", "No Change", "Worsened"],
            key="followup_result",
            help="Used by YuktiShaala to improve future recommendation ranking.",
        )
        if st.button(T("rec_submit"), key="submit_followup"):
            start = time.perf_counter()
            ok = get_recommendation_engine().record_feedback(st.session_state.get("current_encounter_id"), follow_up)
            PERF_LOGGER.info("record_outcome completed in %.3fs", time.perf_counter() - start)
            if ok:
                st.toast("Outcome recorded and learning engine updated.")
            else:
                st.error("Could not record feedback. Generate recommendations first.")

    rep_col1, rep_col2, rep_col3 = st.columns(3)
    with rep_col1:
        try:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            rec_pdf = OUT_REPORTS / f"recommendation_{ts}.pdf"
            generate_recommendation_pdf(rec, str(rec_pdf))
            st.download_button(
                "Download Recommendation PDF",
                data=_read_file_bytes(rec_pdf),
                file_name=rec_pdf.name,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            LOGGER.exception("Recommendation PDF generation failed.")
            st.button("Download Recommendation PDF", disabled=True, use_container_width=True)
    with rep_col2:
        try:
            eff = analytics.get_treatment_effectiveness(rec.condition)
            response = analytics.get_prakriti_response_analysis(rec.condition)
            raw = tracker.get_all_outcomes_for_condition(rec.condition)
            analytics_payload = {"effectiveness": eff, "prakriti_response": response, "raw_outcomes": raw}
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            analytics_pdf = OUT_REPORTS / f"analytics_{rec.condition}_{ts}.pdf"
            generate_analytics_pdf(analytics_payload, str(analytics_pdf))
            st.download_button(
                "Download Analytics PDF",
                data=_read_file_bytes(analytics_pdf),
                file_name=analytics_pdf.name,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            LOGGER.exception("Analytics PDF generation failed.")
            st.button("Download Analytics PDF", disabled=True, use_container_width=True)
    with rep_col3:
        try:
            eff = analytics.get_treatment_effectiveness(rec.condition)
            response = analytics.get_prakriti_response_analysis(rec.condition)
            raw = tracker.get_all_outcomes_for_condition(rec.condition)
            analytics_payload = {"effectiveness": eff, "prakriti_response": response, "raw_outcomes": raw}
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            analytics_xlsx = OUT_REPORTS / f"analytics_{rec.condition}_{ts}.xlsx"
            export_analytics_excel(analytics_payload, str(analytics_xlsx))
            st.download_button(
                "Export Analytics Excel",
                data=_read_file_bytes(analytics_xlsx),
                file_name=analytics_xlsx.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception:
            LOGGER.exception("Analytics Excel export failed.")
            st.button("Export Analytics Excel", disabled=True, use_container_width=True)


def render_vaksetu_page() -> None:
    speech, generator = get_vaksetu_stack()

    st.markdown(f'<div class="section-header" style="font-size:1.5rem;border-left-width:5px;">{T("vaksetu_title")}</div>', unsafe_allow_html=True)

    # Determine active step
    has_transcript = bool(st.session_state.get("current_transcript", "").strip())
    has_ehr = st.session_state.get("current_ehr") is not None
    has_rec = st.session_state.get("current_recommendations") is not None
    if has_rec:
        active_step = 4
    elif has_ehr:
        active_step = 3
    elif has_transcript:
        active_step = 2
    else:
        active_step = 1

    step_labels = [T("vaksetu_step1"), T("vaksetu_step2"), T("vaksetu_step3"), T("vaksetu_step4")]
    st.markdown(_step_indicator_html(active_step, step_labels), unsafe_allow_html=True)

    # Language selector — placed above tabs so it applies to Voice recording
    language_label = st.selectbox("Language", list(SUPPORTED_LANGUAGES.values()), help="Clinical input language.")
    language_code = next((k for k, v in SUPPORTED_LANGUAGES.items() if v == language_label), "hi")

    # Step 1: Input — Tabs
    tab_voice, tab_type, tab_demo = st.tabs([
        f"🎙️ {T('tab_voice')}", f"⌨️ {T('tab_type')}", f"📋 {T('tab_demo')}"
    ])

    transcript_input = ""
    input_mode = "Use Demo Sample"

    with tab_voice:
        st.markdown(f'<p style="color:var(--text-muted);font-size:0.9rem;">{T("voice_record_hint")}</p>', unsafe_allow_html=True)
        audio_value = st.audio_input(T("voice_record"))
        if audio_value:
            wav_bytes = audio_value.getvalue()
            st.audio(wav_bytes, format="audio/wav")
            with st.spinner(T("voice_transcribing")):
                try:
                    result = speech.transcribe(audio_bytes=wav_bytes, language=language_code)
                    st.session_state["_voice_transcript"] = result.text
                    st.session_state["_voice_method"] = result.method
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")
                    st.session_state["_voice_transcript"] = ""

        voice_text = st.session_state.get("_voice_transcript", "")
        if voice_text:
            st.success(T("voice_transcribed"))
            method = st.session_state.get("_voice_method", "")
            if method:
                st.caption(f"{T('voice_method')}: {method}")
            voice_edited = st.text_area(
                T("voice_result"),
                value=voice_text,
                height=180,
                key="voice_transcript_edit",
            )
            if voice_edited.strip():
                transcript_input = voice_edited
                input_mode = "Record Voice"

    with tab_type:
        transcript_input_typed = st.text_area(
            "Paste transcript",
            height=200,
            value=st.session_state.get("current_transcript", ""),
            placeholder="Paste your clinical transcript here...",
        )
        if transcript_input_typed.strip():
            transcript_input = transcript_input_typed
            input_mode = "Type Transcript"

    with tab_demo:
        st.markdown("Select a demo clinical transcript:")
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        demo_labels = ["Sample 1", "Sample 2", "Sample 3"]
        for i, (col, label) in enumerate(zip([demo_col1, demo_col2, demo_col3], demo_labels)):
            with col:
                preview = SAMPLE_TRANSCRIPTS[i][:80] + "..." if len(SAMPLE_TRANSCRIPTS[i]) > 80 else SAMPLE_TRANSCRIPTS[i]
                st.markdown(f'<div class="card" style="min-height:80px;font-size:0.85rem;padding:1rem;"><b>{label}</b><br><span style="color:var(--text-muted);">{preview}</span></div>', unsafe_allow_html=True)
                if st.button(f"Use {label}", key=f"demo_{i}", use_container_width=True):
                    transcript_input = SAMPLE_TRANSCRIPTS[i]
                    input_mode = "Use Demo Sample"
                    st.session_state["_demo_selected"] = i

        if st.session_state.get("_demo_selected") is not None:
            idx = st.session_state["_demo_selected"]
            transcript_input = SAMPLE_TRANSCRIPTS[idx]
            input_mode = "Use Demo Sample"
            st.text_area("Selected Transcript", value=transcript_input, height=160, disabled=True)

    # Step 2: Process
    if not transcript_input.strip():
        st.markdown(f'<div class="alert-banner"><span class="alert-icon">⚠️</span>Please enter, record, or select a transcript before generating EHR.</div>', unsafe_allow_html=True)

    if st.button(T("btn_generate_ehr"), type="primary", use_container_width=True):
        if not transcript_input.strip():
            st.error("Transcript is empty. Please provide input.")
            st.stop()
        try:
            start = time.perf_counter()
            transcribed = transcript_input
            with st.spinner("📝 Correcting medical terms..."):
                time.sleep(0.35)
            with st.spinner("🔍 Extracting clinical data..."):
                ehr = generator.generate_from_transcript(
                    transcript=transcribed,
                    language=language_code,
                    centre_id="C001",
                    doctor_id="D001",
                )
            with st.spinner("🏷️ Mapping diagnosis codes..."):
                payload = generator.to_ahmis_json(ehr)
                time.sleep(0.25)

            st.session_state["current_transcript"] = transcribed
            st.session_state["current_ehr"] = ehr
            st.session_state["current_ahmis_json"] = payload
            st.session_state["last_ehr_seconds"] = time.perf_counter() - start
            PERF_LOGGER.info("generate_ehr completed in %.3fs", st.session_state["last_ehr_seconds"])
            st.toast("EHR generated successfully.")
        except Exception:
            LOGGER.exception("EHR generation failed.")
            st.toast("EHR generation failed. Please retry or use demo sample.")
            st.error("EHR generation failed. Please check input and service status.")

    # Step 3: Results
    _render_ehr(st.session_state.get("current_ehr"))

    # Step 4: Actions
    if st.session_state.get("current_ehr"):
        st.markdown(f'<div class="section-header" style="font-size:1.1rem;">{T("vaksetu_step4")}</div>', unsafe_allow_html=True)

        if st.button(T("btn_get_recommendations"), type="primary", use_container_width=True):
            ehr = st.session_state.get("current_ehr")
            if not ehr:
                st.error("Generate EHR first.")
            else:
                prakriti = ehr.prakriti_assessment or st.session_state.get("prakriti_result")
                if not prakriti:
                    st.warning("Prakriti not found in transcript. Complete the questionnaire in Prakriti Assessment page.")
                    if st.button("Go to Prakriti Assessment"):
                        st.session_state["nav_page"] = "Prakriti Assessment"
                        st.rerun()
                else:
                    existing = [x.get("formulation_name") for x in ehr.prescriptions if x.get("formulation_name")]
                    try:
                        start = time.perf_counter()
                        with st.spinner("🧬 PrakritiMitra is analyzing..."):
                            rec = get_recommendation_engine().recommend(
                                patient_prakriti=prakriti,
                                condition_ayush_code=ehr.ayush_diagnosis.get("name", ""),
                                patient_age=int(ehr.patient_demographics.get("age") or 35),
                                patient_sex=ehr.patient_demographics.get("sex") or "Unknown",
                                existing_prescriptions=existing,
                            )
                        st.session_state["current_recommendations"] = rec
                        st.session_state["current_encounter_id"] = rec.encounter_id
                        st.session_state["last_recommendation_seconds"] = time.perf_counter() - start
                        PERF_LOGGER.info(
                            "recommendation completed in %.3fs",
                            st.session_state["last_recommendation_seconds"],
                        )
                        st.toast("Recommendations generated.")
                    except Exception:
                        LOGGER.exception("Recommendation generation failed.")
                        st.error("Recommendation generation failed. Try again in demo mode.")

        action_c1, action_c2, action_c3 = st.columns(3)
        with action_c1:
            payload = st.session_state.get("current_ahmis_json")
            ehr = st.session_state.get("current_ehr")
            if payload and ehr:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                json_path = OUT_EHR / f"ahmis_ehr_{ts}.json"
                export_ehr_json(ehr, str(json_path))
                st.download_button(
                    T("btn_export_json"),
                    data=_read_file_bytes(json_path),
                    file_name=json_path.name,
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.button(T("btn_export_json"), disabled=True, use_container_width=True)

        with action_c2:
            ehr = st.session_state.get("current_ehr")
            if ehr:
                try:
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    pdf_path = OUT_REPORTS / f"ehr_report_{ts}.pdf"
                    generate_ehr_pdf(ehr, str(pdf_path))
                    st.download_button(
                        T("btn_export_pdf"),
                        data=_read_file_bytes(pdf_path),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception:
                    LOGGER.exception("EHR PDF generation failed.")
                    st.button(T("btn_export_pdf"), disabled=True, use_container_width=True)
            else:
                st.button(T("btn_export_pdf"), disabled=True, use_container_width=True)

        with action_c3:
            if st.button(T("btn_new_consultation"), use_container_width=True):
                st.session_state["current_ehr"] = None
                st.session_state["current_ahmis_json"] = None
                st.session_state["current_transcript"] = ""
                st.session_state["current_recommendations"] = None
                st.session_state["current_encounter_id"] = None
                st.session_state.pop("_demo_selected", None)
                st.session_state.pop("_voice_transcript", None)
                st.session_state.pop("_voice_method", None)
                st.toast("Consultation state cleared.")

    _render_recommendations()


# ---------------------------------------------------------------------------
# RogaRadar Page
# ---------------------------------------------------------------------------

def render_rogaradar_page() -> None:
    st.markdown(f'<div class="section-header" style="font-size:1.5rem;border-left-width:5px;">{T("rr_title")}</div>', unsafe_allow_html=True)

    start = time.perf_counter()
    with st.spinner("🗺️ RogaRadar is scanning..."):
        data = load_surveillance_data()
    PERF_LOGGER.info("rogaradar_scan completed in %.3fs", time.perf_counter() - start)
    st.session_state["loaded_data"] = data
    st.session_state["alert_results"] = data.get("alerts", [])

    alerts = data["alerts"]
    agg = data["aggregated"]
    visits = data["visits"]

    # Stat cards
    m1, m2, m3, m4 = st.columns(4)
    n_districts = int(data["district_meta"]["district"].nunique())
    n_alerts = len(alerts)
    n_conditions = int(len({a.condition_ayush for a in alerts}))
    n_datapoints = int(len(agg))

    with m1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{n_districts}</div><div class="stat-label">{T("rr_districts")}</div></div>', unsafe_allow_html=True)
    with m2:
        alert_color = "var(--danger)" if n_alerts > 0 else "var(--accent)"
        st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{alert_color};">{n_alerts}</div><div class="stat-label">{T("rr_alerts")}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{n_conditions}</div><div class="stat-label">{T("rr_conditions")}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{n_datapoints}</div><div class="stat-label">{T("rr_datapoints")}</div></div>', unsafe_allow_html=True)

    # Map section
    st.markdown(f'<div class="section-header">{T("rr_map")}</div>', unsafe_allow_html=True)

    dl_col1, dl_col2 = st.columns([1, 1])
    with dl_col1:
        try:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_path = OUT_REPORTS / f"surveillance_report_{ts}.pdf"
            generate_surveillance_pdf(alerts, agg, str(report_path))
            st.download_button(
                "Download Report",
                data=_read_file_bytes(report_path),
                file_name=report_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            LOGGER.exception("Surveillance PDF generation failed.")
            st.button("Download Report", disabled=True, use_container_width=True)
    with dl_col2:
        if alerts:
            alert_rows = [
                {
                    "District": a.district, "State": a.state, "Condition": a.condition_ayush,
                    "Level": a.alert_level, "Cases": a.current_cases,
                    "Baseline": round(a.baseline_cases, 1), "Ratio": round(a.ratio, 2),
                    "Trend": a.trend, "Action": a.recommended_action,
                }
                for a in alerts
            ]
            csv_data = pd.DataFrame(alert_rows).to_csv(index=False)
            st.download_button(
                "Export Alerts CSV",
                data=csv_data,
                file_name=f"rogaradar_alerts_{datetime.utcnow().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Export Alerts CSV", disabled=True, use_container_width=True)

    if st_folium is not None:
        st_folium(data["map"], width=1100, height=460)
    else:
        st.components.v1.html(data["map"]._repr_html_(), height=460)

    # Alert Table + Charts in tabs
    tab_alerts, tab_ts, tab_heatmap = st.tabs([
        f"🚨 {T('rr_alert_table')}", f"📈 {T('rr_timeseries')}", f"🌡️ {T('rr_heatmap')}"
    ])

    with tab_alerts:
        table = data["alert_table"].copy()
        if table.empty:
            st.info("No active alerts detected in current synthetic run.")
        else:
            table = table.sort_values(by=["Alert Level", "Ratio"], ascending=[False, False]).reset_index(drop=True)
            table["Severity"] = table["Alert Level"].map(
                lambda x: "ALERT" if x == "ALERT" else "WARNING" if x == "WARNING" else "WATCH"
            )
            st.dataframe(
                table[["Severity", "District", "State", "Condition", "Cases", "Baseline", "Ratio", "Trend"]],
                hide_index=True,
                use_container_width=True,
            )

            row_choice = st.selectbox(
                T("rr_inspect"),
                table.index.tolist(),
                format_func=lambda i: f"{table.loc[i, 'District']} — {table.loc[i, 'Condition']} ({table.loc[i, 'Severity']})",
            )
            selected = table.loc[row_choice]
            st.markdown(_severity_badge(selected["Severity"]), unsafe_allow_html=True)

            dashboard: SurveillanceDashboard = data["dashboard"]
            fig = dashboard.create_time_series_chart(agg, selected["District"], selected["Condition"])
            st.plotly_chart(fig, use_container_width=True)

    with tab_ts:
        district_options = sorted(agg["district"].unique().tolist()) if not agg.empty else []
        cond_options = sorted(agg["condition_ayush"].unique().tolist()) if not agg.empty else []

        if district_options and cond_options:
            col1, col2 = st.columns(2)
            district = col1.selectbox("District", district_options, key="rr_district")
            condition = col2.selectbox("Condition", cond_options, key="rr_condition")
            ts_fig = data["dashboard"].create_time_series_chart(agg, district, condition)
            st.plotly_chart(ts_fig, use_container_width=True)
        else:
            st.info("No time series data available.")

    with tab_heatmap:
        st.plotly_chart(data["heatmap"], use_container_width=True)


# ---------------------------------------------------------------------------
# Prakriti Page
# ---------------------------------------------------------------------------

def _prakriti_description(prakriti: str) -> str:
    descriptions = {
        "Vata": "Fast, light, and variable tendencies. Benefits from grounding routines.",
        "Pitta": "Sharp, focused, and heat-dominant profile. Benefits from cooling balance.",
        "Kapha": "Steady, strong, and calm profile. Benefits from stimulation and movement.",
        "Vata-Pitta": "Mobile + intense profile; regular meals and stress moderation are key.",
        "Pitta-Kapha": "Strong metabolism with stable endurance; needs heat and heaviness control.",
        "Vata-Kapha": "Alternates between lightness and heaviness; routine and circulation support help.",
        "Sama": "Balanced tri-dosha profile. Maintain seasonal and daily rhythm.",
    }
    return descriptions.get(prakriti, "Prakriti profile available.")


def render_prakriti_page() -> None:
    classifier, advisor, questionnaire = get_prakriti_tools()

    st.markdown(f'<div class="section-header" style="font-size:1.5rem;border-left-width:5px;">{T("prakriti_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:var(--text-muted);margin-top:-0.5rem;">{T("prakriti_subtitle")}</p>', unsafe_allow_html=True)

    total = len(questionnaire)
    index = int(st.session_state.get("prakriti_q_index", 0))
    answers = st.session_state.get("prakriti_answers", {})
    n_answered = len(answers)

    # Circular progress + progress bar
    prog_col, info_col = st.columns([1, 3])
    with prog_col:
        pct = int(100 * n_answered / max(1, total))
        st.markdown(
            f"""
            <div class="circular-progress">
                <div class="cp-number">{n_answered}/{total}</div>
                <div class="cp-label">{T("prakriti_progress")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col:
        progress = min(1.0, n_answered / max(1, total))
        st.markdown(f'<div class="progress-modern"><div class="progress-fill" style="width:{int(progress * 100)}%"></div></div>', unsafe_allow_html=True)
        if index < total:
            cat = questionnaire[index].get("category", "").replace("_", " ").title()
            st.markdown(f'<span style="font-size:0.9rem;color:var(--text-muted);">{T("prakriti_category")}: <b>{cat}</b></span>', unsafe_allow_html=True)

    if index < total:
        q = questionnaire[index]
        q_id = q.get("id", f"Q{index + 1:02d}")
        text_en = q.get("text_en", "")
        text_hi = q.get("text_hi", "")

        st.markdown(
            f"""
            <div class="questionnaire-card">
                <div class="q-number">{index + 1}</div>
                <div class="q-text">{text_en}</div>
                {"<div class='q-text-hi'>" + text_hi + "</div>" if text_hi else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        options = [opt["label"] for opt in q.get("options", [])]
        key = f"prakriti_option_{index}"
        preselect = answers.get(f"q{index + 1}", 3) - 1
        selected_label = st.radio(
            "Select one option",
            options,
            index=max(0, min(preselect, len(options) - 1)),
            key=key,
            label_visibility="collapsed",
        )
        selected_index = options.index(selected_label) + 1
        answers[f"q{index + 1}"] = selected_index
        st.session_state["prakriti_answers"] = answers

        prev_col, spacer, next_col = st.columns([2, 1, 2])
        with prev_col:
            if st.button(f"← {T('btn_previous')}", disabled=index == 0, use_container_width=True):
                st.session_state["prakriti_q_index"] = max(0, index - 1)
                st.rerun()
        with next_col:
            if st.button(f"{T('btn_next')} →", use_container_width=True):
                st.session_state["prakriti_q_index"] = min(total, index + 1)
                st.rerun()

    if n_answered == total:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(T("btn_calculate"), type="primary", use_container_width=True):
            result = classifier.classify(answers)
            st.session_state["prakriti_assessment_result"] = result

    result = st.session_state.get("prakriti_assessment_result")
    if result:
        st.markdown(f'<div class="section-header">{T("prakriti_result")}</div>', unsafe_allow_html=True)

        dosha_total = result.vata_score + result.pitta_score + result.kapha_score
        vata_pct = 100 * (result.vata_score / dosha_total) if dosha_total else 0
        pitta_pct = 100 * (result.pitta_score / dosha_total) if dosha_total else 0
        kapha_pct = 100 * (result.kapha_score / dosha_total) if dosha_total else 0

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            pie = px.pie(
                names=["Vata", "Pitta", "Kapha"],
                values=[vata_pct, pitta_pct, kapha_pct],
                title=T("prakriti_distribution"),
                color=["Vata", "Pitta", "Kapha"],
                color_discrete_map={"Vata": "#3B82F6", "Pitta": "#EF4444", "Kapha": "#22C55E"},
            )
            pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=14),
            )
            st.plotly_chart(pie, use_container_width=True)

        with res_col2:
            st.markdown(
                f"""
                <div class="card card-accent" style="margin-top:1rem;">
                    <div style="font-size:0.85rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;">Dominant Prakriti</div>
                    <div style="font-size:1.8rem;font-weight:800;color:var(--primary);margin:0.3rem 0;">{result.prakriti_type}</div>
                    <p style="color:var(--text-muted);font-size:0.95rem;line-height:1.6;">{_prakriti_description(result.prakriti_type)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Advice as tabs
        tab_diet, tab_ls, tab_yg = st.tabs([
            f"🍽️ {T('tab_diet')}",
            f"🧘 {T('tab_lifestyle')}",
            f"🧎 {T('tab_yoga')}",
        ])
        with tab_diet:
            for item in advisor.get_dietary_advice(result.prakriti_type, "general")[:8]:
                st.write(f"- {item}")
        with tab_ls:
            for item in advisor.get_lifestyle_advice(result.prakriti_type, "general")[:8]:
                st.write(f"- {item}")
        with tab_yg:
            for item in advisor.get_yoga_advice(result.prakriti_type, "general")[:8]:
                st.write(f"- {item}")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"{T('btn_use_prakriti')} →", type="primary", use_container_width=True):
            st.session_state["prakriti_result"] = result.prakriti_type
            st.session_state["nav_page"] = "VakSetu"
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()
    _ensure_output_dirs()
    ollama_ok, neo4j_ok, postgres_ok, bhashini_ok = render_sidebar()

    # Collapsible demo mode banner — only show if something is down
    issues = []
    if not ollama_ok:
        issues.append("LLM service not available — using cached results")
    if not neo4j_ok:
        issues.append("Using in-memory knowledge base (limited)")
    if not postgres_ok:
        issues.append("PostgreSQL not available — using SQLite fallback")

    if issues:
        with st.expander(f"⚠️ {T('demo_banner')}", expanded=False):
            for issue in issues:
                st.caption(f"• {issue}")

    page = st.session_state.get("nav_page", "Home / Overview")
    if page == "Home / Overview":
        render_home()
    elif page == "VakSetu":
        render_vaksetu_page()
    elif page == "RogaRadar":
        render_rogaradar_page()
    elif page == "Prakriti Assessment":
        render_prakriti_page()


if __name__ == "__main__":
    main()
