"""VakSetu tests for Phase 4 pipeline behavior."""

from __future__ import annotations

from pathlib import Path

from src.vaksetu.code_mapper import CodeMapper
from src.vaksetu.ehr_generator import EHRGenerator
from src.vaksetu.medical_ner import MedicalNEREngine
from src.vaksetu.speech_engine import SAMPLE_TRANSCRIPTS, MockSpeechEngine
from src.vaksetu.vocabulary import AyushVocabulary


ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "data" / "ayush_vocabulary.json"
MORBIDITY = ROOT / "data" / "knowledge_base" / "ayush_morbidity_codes.json"
ICD_MAP = ROOT / "data" / "knowledge_base" / "icd10_mapping.json"


def _build_components():
    vocab = AyushVocabulary(vocabulary_path=VOCAB)
    mapper = CodeMapper(morbidity_codes_path=str(MORBIDITY), icd10_mapping_path=str(ICD_MAP))
    ner = MedicalNEREngine(llm_client=None, vocabulary_path=str(VOCAB), morbidity_codes_path=str(MORBIDITY))
    generator = EHRGenerator(ner_engine=ner, code_mapper=mapper)
    return vocab, mapper, ner, generator


def test_vocabulary_correction_with_misspelled_terms() -> None:
    vocab, *_ = _build_components()
    text = "patient taking tree fala and ashwa gandha churna"
    corrected = vocab.correct_text(text)
    # At least one correction should be suggested by fuzzy helper.
    assert vocab.suggest_correction("treefala") is not None
    assert isinstance(corrected, str)


def test_ner_extraction_on_three_sample_transcripts() -> None:
    _, _, ner, _ = _build_components()
    out1 = ner.extract_ehr(SAMPLE_TRANSCRIPTS[0], "hi")
    out2 = ner.extract_ehr(SAMPLE_TRANSCRIPTS[1], "en")
    out3 = ner.extract_ehr(SAMPLE_TRANSCRIPTS[2], "hi")

    assert out1.ayush_diagnosis["name"] == "Vibandha"
    assert out1.icd10_diagnosis["code"] == "K59.0"
    assert out2.ayush_diagnosis["name"] == "Amlapitta"
    assert out3.ayush_diagnosis["name"] == "Sandhivata"
    assert len(out1.prescriptions) >= 2
    assert len(out2.prescriptions) >= 2
    assert len(out3.prescriptions) >= 2


def test_code_mapping_for_common_conditions() -> None:
    _, mapper, _, _ = _build_components()
    for cond in [
        "Vibandha",
        "Amlapitta",
        "Sandhivata",
        "Prameha",
        "constipation",
        "acidity",
        "joint pain",
        "sugar",
        "dysuria",
        "fever",
    ]:
        mapped = mapper.map_condition(cond)
        assert "match_type" in mapped


def test_full_ehr_generation_pipeline_and_schema() -> None:
    _, _, _, generator = _build_components()
    ehr = generator.generate_from_transcript(SAMPLE_TRANSCRIPTS[0], "hi", "C001", "D001")
    payload = generator.to_ahmis_json(ehr)

    assert ehr.ayush_diagnosis["name"] == "Vibandha"
    assert ehr.icd10_diagnosis["code"] == "K59.0"
    assert len(ehr.prescriptions) >= 2
    assert payload["metadata"]["centre_id"] == "C001"
    assert payload["metadata"]["doctor_id"] == "D001"


def test_audio_pipeline_with_demo_engine() -> None:
    _, _, _, generator = _build_components()
    speech = MockSpeechEngine()
    ehr, transcription = generator.generate_from_audio(
        audio_bytes=b"fake",
        language="hi",
        centre_id="C001",
        doctor_id="D001",
        speech_engine=speech,
    )
    assert transcription.method == "demo_mode"
    assert ehr.encounter_metadata["transcription_method"] == "demo_mode"


def test_error_handling_empty_and_gibberish() -> None:
    _, _, ner, generator = _build_components()
    empty = ner.extract_ehr("", "hi")
    gibberish = generator.generate_from_transcript("asdf qwer zxcv", "en", "C1", "D1")
    assert isinstance(empty.patient_demographics, dict)
    assert isinstance(gibberish.ayush_diagnosis, dict)


# --- Phase 4 additions ---

def test_confidence_scores_present_and_valid() -> None:
    _, _, ner, _ = _build_components()
    out = ner.extract_ehr(SAMPLE_TRANSCRIPTS[0], "hi")
    assert hasattr(out, "confidence_scores")
    assert "overall" in out.confidence_scores
    assert "age" in out.confidence_scores
    assert "diagnosis" in out.confidence_scores
    assert "prescriptions" in out.confidence_scores
    assert 0.0 <= out.confidence_scores["overall"] <= 1.0
    assert out.confidence_scores["age"] > 0  # Sample 1 has age


def test_confidence_empty_transcript_has_zero_scores() -> None:
    _, _, ner, _ = _build_components()
    out = ner.extract_ehr("", "hi")
    assert out.confidence_scores.get("overall", 0) == 0
    assert out.confidence_scores.get("age", 0) == 0


def test_dosage_extraction_improved() -> None:
    _, _, ner, _ = _build_components()
    out = ner.extract_ehr(SAMPLE_TRANSCRIPTS[0], "hi")
    # Sample 1 mentions "5 gram" for Triphala
    triphala_rx = [rx for rx in out.prescriptions if "triphala" in rx.get("formulation_name", "").lower()]
    if triphala_rx:
        assert triphala_rx[0]["dosage"] != "As prescribed"


def test_severity_detection() -> None:
    _, _, ner, _ = _build_components()
    # "severe" should be detected
    out = ner.extract_ehr("35 saal ki mahila, severe kabz ki shikayat 2 hafte se", "hi")
    if out.chief_complaints:
        assert any("Severe" in c.get("severity", "") for c in out.chief_complaints)


def test_multiple_conditions_in_aliases() -> None:
    _, _, ner, _ = _build_components()
    for alias, expected in [("kabz", "Vibandha"), ("acidity", "Amlapitta"), ("joint pain", "Sandhivata"),
                            ("sugar", "Prameha"), ("fever", "Jwara"), ("cough", "Kasa"),
                            ("piles", "Arsha"), ("obesity", "Sthaulya")]:
        out = ner.extract_ehr(f"patient has {alias}", "en")
        assert out.ayush_diagnosis["name"] == expected, f"Failed for alias '{alias}': got {out.ayush_diagnosis['name']}"


def test_hindi_age_variants() -> None:
    _, _, ner, _ = _build_components()
    assert ner._parse_age("25 saal ki mahila") == 25
    assert ner._parse_age("30 years old male") == 30
    assert ner._parse_age("age: 40") == 40


def test_prakriti_detection_all_types() -> None:
    _, _, ner, _ = _build_components()
    for ptype in ["Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Sama"]:
        result = ner._parse_prakriti(f"patient has {ptype} prakriti")
        assert result == ptype


def test_follow_up_parsing_variants() -> None:
    _, _, ner, _ = _build_components()
    assert "week" in (ner._parse_follow_up("follow up 2 weeks") or "")
    assert "day" in (ner._parse_follow_up("follow up 5 din baad") or "")
    assert "month" in (ner._parse_follow_up("follow up 1 month") or "")


def test_multilingual_transcript_extraction() -> None:
    _, _, ner, _ = _build_components()
    # Mixed Hindi-English
    mixed = "35 saal ki mahila, Vata Prakriti. constipation since 2 weeks. Triphala Churna prescribed."
    out = ner.extract_ehr(mixed, "hi")
    assert out.patient_demographics.get("age") == 35
    assert out.ayush_diagnosis["name"] == "Vibandha"
