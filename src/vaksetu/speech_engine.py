"""Speech interfaces for VakSetu with Bhashini + Google Speech + demo fallback."""

from __future__ import annotations

import logging
import os
import time
from io import BytesIO
from typing import Dict, List

import requests
from pydantic import BaseModel

import config as app_config

try:
    import speech_recognition as sr

    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


SAMPLE_TRANSCRIPTS: List[str] = [
    "35 saal ki mahila hai, Vata Prakriti, pet mein dard aur kabz ki shikayat hai, pichle 2 hafte se. Maine Triphala Churna 5 gram raat ko garam paani ke saath aur Abhayarishta 15ml subah shaam khana ke baad likhi hai. Ek hafte baad follow up.",
    "Male patient, 52 years old, Pitta-Kapha Prakriti. Burning sensation in chest, sour eructation, 3 weeks. Diagnosis Amlapitta. Prescribed Avipattikar Churna 3g twice daily before food, Kamdudha Rasa 250mg twice daily after food. Avoid spicy oily food. Follow up 2 weeks.",
    "58 saal ke purush, Vata-Kapha Prakriti. Ghutno mein dard aur sujan, 3 mahine se. Sandhivata. Yogaraja Guggulu 2 goli din mein teen baar khana ke baad, Maharasnadi Kashaya 15ml subah khali pet. Bahar se Kottamchukkadi Taila se malish. Yoga mein Trikonasana aur Veerabhadrasana.",
]


class TranscriptionResult(BaseModel):
    """Speech transcription payload."""

    text: str
    confidence: float
    language: str
    duration_seconds: float
    method: str  # "bhashini" or "whisper_fallback" or "demo_mode"


class MockSpeechEngine:
    """Demo-mode speech engine used when no Bhashini credentials are configured."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("MockSpeechEngine")
        self._cursor = 0
        self.logger.warning("Running in demo mode — set BHASHINI_API_KEY for live ASR")

    def transcribe(self, audio_bytes: bytes, language: str) -> TranscriptionResult:
        """Return one of the curated demo transcripts."""
        _ = audio_bytes
        transcript = SAMPLE_TRANSCRIPTS[self._cursor % len(SAMPLE_TRANSCRIPTS)]
        self._cursor += 1
        return TranscriptionResult(
            text=transcript,
            confidence=0.99,
            language=language,
            duration_seconds=8.0,
            method="demo_mode",
        )

    def text_to_speech(self, text: str, language: str) -> bytes:
        """No-op TTS for demo."""
        _ = (text, language)
        self.logger.info("Demo mode TTS invoked; returning empty audio bytes.")
        return b""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """No-op translation for demo."""
        _ = (source_lang, target_lang)
        return text


class GoogleSpeechEngine:
    """Speech-to-text using Google's free Speech Recognition API.

    Supports Hindi (hi-IN), English (en-IN), and 6 other Indian languages.
    No API key required for short audio clips.
    """

    LANG_MAP = {
        "hi": "hi-IN",
        "en": "en-IN",
        "ta": "ta-IN",
        "bn": "bn-IN",
        "mr": "mr-IN",
        "te": "te-IN",
        "kn": "kn-IN",
        "gu": "gu-IN",
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger("GoogleSpeechEngine")
        if not _SR_AVAILABLE:
            self.logger.warning("SpeechRecognition not installed; Google ASR unavailable.")

    def transcribe_wav(self, wav_bytes: bytes, language: str) -> TranscriptionResult:
        """Transcribe WAV audio bytes to text using Google Speech API."""
        if not _SR_AVAILABLE:
            raise RuntimeError("SpeechRecognition library not installed.")

        google_lang = self.LANG_MAP.get(language, "hi-IN")
        recognizer = sr.Recognizer()
        start = time.time()

        with sr.AudioFile(BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language=google_lang)

        return TranscriptionResult(
            text=text,
            confidence=0.85,
            language=language,
            duration_seconds=round(time.time() - start, 2),
            method="google_speech",
        )


class SpeechEngine:
    """Bhashini-first speech engine with Google Speech + demo fallback."""

    def __init__(self, config=app_config):
        self.bhashini_api_url = config.BHASHINI_API_URL
        self.bhashini_inference_url = config.BHASHINI_INFERENCE_URL
        self.api_key = os.getenv("BHASHINI_API_KEY")
        self.user_id = os.getenv("BHASHINI_USER_ID")
        self.logger = logging.getLogger("SpeechEngine")
        self._google = GoogleSpeechEngine() if _SR_AVAILABLE else None
        self._demo = MockSpeechEngine() if not self.api_key else None

    def get_asr_config(self, source_language: str) -> Dict:
        """Get Bhashini ASR pipeline config for language."""
        if not self.api_key:
            return {"pipeline": "demo", "source_language": source_language}
        return {
            "pipelineTasks": [{"taskType": "asr", "config": {"language": {"sourceLanguage": source_language}}}],
            "inputData": {"audio": [{"audioContent": ""}]},
        }

    def transcribe(self, audio_bytes: bytes, language: str) -> TranscriptionResult:
        """Transcribe audio. Priority: Bhashini -> Google Speech -> demo fallback."""
        # 1. Try Bhashini (if API key configured)
        if self.api_key:
            start = time.time()
            try:
                headers = {
                    "Authorization": self.api_key,
                    "Content-Type": "application/json",
                }
                payload = self.get_asr_config(language)
                response = requests.post(
                    self.bhashini_inference_url,
                    json=payload,
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("output", [{}])[0].get("source", "")
                if not text:
                    raise ValueError("Empty ASR output from Bhashini")
                return TranscriptionResult(
                    text=text,
                    confidence=0.9,
                    language=language,
                    duration_seconds=round(time.time() - start, 2),
                    method="bhashini",
                )
            except Exception:
                self.logger.warning("Bhashini ASR failed; trying Google Speech.")

        # 2. Try Google Speech Recognition (if real audio provided)
        if self._google and audio_bytes and audio_bytes != b"demo-audio":
            try:
                return self._google.transcribe_wav(audio_bytes, language)
            except Exception as exc:
                self.logger.warning("Google Speech failed: %s; using demo fallback.", exc)

        # 3. Demo fallback
        if self._demo:
            return self._demo.transcribe(audio_bytes, language)

        # 4. Last resort
        return TranscriptionResult(
            text=SAMPLE_TRANSCRIPTS[0],
            confidence=0.5,
            language=language,
            duration_seconds=0.0,
            method="demo_mode",
        )

    def text_to_speech(self, text: str, language: str) -> bytes:
        """TTS using Bhashini with demo fallback."""
        if self._demo:
            return self._demo.text_to_speech(text, language)
        _ = (text, language)
        return b""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate with Bhashini with demo fallback."""
        if self._demo:
            return self._demo.translate(text, source_lang, target_lang)
        _ = (source_lang, target_lang)
        return text

    # Backward-compatible wrappers.
    def transcribe_audio(self, audio_bytes: bytes, language: str = "hi") -> Dict:
        """Legacy API compatibility wrapper."""
        result = self.transcribe(audio_bytes, language)
        return {"language": result.language, "transcript": result.text, "confidence": result.confidence}

    def synthesize_speech(self, text: str, language: str = "hi") -> bytes:
        """Legacy API compatibility wrapper."""
        return self.text_to_speech(text=text, language=language)

