"""Seed AYUSH knowledge base into Neo4j with graceful in-memory export fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from src.prakritimitra.knowledge_graph import AyushKnowledgeGraph


FORM_PATH = ROOT / "data" / "knowledge_base" / "formulations.json"
CODE_PATH = ROOT / "data" / "knowledge_base" / "ayush_morbidity_codes.json"
RULES_PATH = ROOT / "data" / "knowledge_base" / "prakriti_rules.json"
EXPORT_PATH = ROOT / "data" / "knowledge_base" / "inmemory_seed_export.json"

CLASSICAL_TEXTS = [
    "Charaka Samhita",
    "Sushruta Samhita",
    "Ashtanga Hridaya",
    "Bhaishajya Ratnavali",
    "Sharangadhara Samhita",
    "Rasatarangini",
    "Bhavaprakasha",
    "Sahasrayogam",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _text_hits(reference: str) -> List[str]:
    lower = (reference or "").lower()
    hits = []
    for text in CLASSICAL_TEXTS:
        if text.lower().replace(" samhita", "") in lower or text.lower() in lower:
            hits.append(text)
    return hits


def _relationship_estimate(formulations: List[Dict], codes: List[Dict]) -> int:
    count = 0
    for row in formulations:
        count += len(row.get("ingredients", []))
        count += len(row.get("indicated_conditions", []))
        count += len(row.get("indicated_prakriti", []))
        count += len(row.get("contraindicated_prakriti", []))
        count += len([k for k, v in (row.get("dosha_action") or {}).items() if str(v).lower() != "neutral"])
        count += len(_text_hits(row.get("classical_reference", "")))
    for code in codes:
        count += len(code.get("dosha_involvement", []))
    return count


def _seed_neo4j(kg: AyushKnowledgeGraph, formulations: List[Dict], codes: List[Dict], rules: Dict) -> int:
    relationships_created = 0
    with kg.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        session.run("UNWIND $prakritis AS p MERGE (:Prakriti {name:p})", prakritis=list(rules.keys()))
        session.run("UNWIND $doshas AS d MERGE (:Dosha {name:d})", doshas=["Vata", "Pitta", "Kapha"])
        session.run("UNWIND $texts AS t MERGE (:ClassicalText {name:t})", texts=CLASSICAL_TEXTS)

        for code in codes:
            session.run(
                "MERGE (c:Condition {code_id:$code_id}) "
                "SET c.name=$name, c.english_name=$english, c.system=$system, c.category=$category, c.icd10_codes=$icd10",
                code_id=code["code_id"],
                name=code["ayush_name"],
                english=code.get("english_name", ""),
                system=code.get("system", "Ayurveda"),
                category=code.get("category", ""),
                icd10=code.get("icd10_codes", []),
            )
            for dosha in code.get("dosha_involvement", []):
                session.run(
                    "MATCH (c:Condition {code_id:$code_id}), (d:Dosha {name:$dosha}) "
                    "MERGE (c)-[:PACIFIES]->(d)",
                    code_id=code["code_id"],
                    dosha=dosha,
                )
                relationships_created += 1

        for prakriti, payload in rules.items():
            session.run(
                "MATCH (p:Prakriti {name:$name}) "
                "SET p.general_principles=$gp, p.preferred_dosage_forms=$forms",
                name=prakriti,
                gp=payload.get("general_principles", ""),
                forms=payload.get("preferred_dosage_forms", []),
            )

        for row in formulations:
            session.run(
                "MERGE (f:Formulation {formulation_id:$id}) "
                "SET f.name_sanskrit=$name_sanskrit, f.name_english=$name_english, "
                "f.formulation_type=$ftype, f.system=$system, f.dosage_range=$dose, "
                "f.route=$route, f.classical_reference=$cref, f.chapter_reference=$chapter, "
                "f.safety_notes=$safety",
                id=row["formulation_id"],
                name_sanskrit=row.get("name_sanskrit", ""),
                name_english=row.get("name_english", ""),
                ftype=row.get("formulation_type", ""),
                system=row.get("system", "Ayurveda"),
                dose=row.get("dosage_range", ""),
                route=row.get("route", ""),
                cref=row.get("classical_reference", ""),
                chapter=row.get("chapter_reference", ""),
                safety=row.get("safety_notes", ""),
            )

            for ingredient in row.get("ingredients", []):
                name = ingredient.get("name", "Unknown")
                session.run(
                    "MATCH (f:Formulation {formulation_id:$id}) "
                    "MERGE (i:Ingredient {name:$name}) "
                    "SET i.part=$part "
                    "MERGE (f)-[:CONTAINS {proportion:$prop}]->(i)",
                    id=row["formulation_id"],
                    name=name,
                    part=ingredient.get("part", ""),
                    prop=ingredient.get("proportion", ""),
                )
                relationships_created += 1

            for condition_name in row.get("indicated_conditions", []):
                session.run(
                    "MATCH (f:Formulation {formulation_id:$id}), (c:Condition {name:$condition}) "
                    "MERGE (f)-[:INDICATED_FOR]->(c) "
                    "MERGE (f)-[:TREATS]->(c)",
                    id=row["formulation_id"],
                    condition=condition_name,
                )
                relationships_created += 2

            for prakriti in row.get("indicated_prakriti", []):
                session.run(
                    "MATCH (f:Formulation {formulation_id:$id}), (p:Prakriti {name:$prakriti}) "
                    "MERGE (f)-[:SUITABLE_FOR]->(p)",
                    id=row["formulation_id"],
                    prakriti=prakriti,
                )
                relationships_created += 1

            for prakriti in row.get("contraindicated_prakriti", []):
                session.run(
                    "MATCH (f:Formulation {formulation_id:$id}), (p:Prakriti {name:$prakriti}) "
                    "MERGE (f)-[:CONTRAINDICATED_FOR]->(p)",
                    id=row["formulation_id"],
                    prakriti=prakriti,
                )
                relationships_created += 1

            for dosha, action in (row.get("dosha_action") or {}).items():
                action_l = str(action).lower()
                if action_l == "neutral":
                    continue
                relation = "PACIFIES" if action_l == "pacifies" else "AGGRAVATES"
                session.run(
                    f"MATCH (f:Formulation {{formulation_id:$id}}), (d:Dosha {{name:$dosha}}) MERGE (f)-[:{relation}]->(d)",
                    id=row["formulation_id"],
                    dosha=dosha,
                )
                relationships_created += 1

            for text_name in _text_hits(row.get("classical_reference", "")):
                session.run(
                    "MATCH (f:Formulation {formulation_id:$id}), (t:ClassicalText {name:$text}) "
                    "MERGE (f)-[:REFERENCED_IN]->(t)",
                    id=row["formulation_id"],
                    text=text_name,
                )
                relationships_created += 1

    return relationships_created


def _seed_qdrant(formulations_path: str) -> int:
    """Index formulations into Qdrant vector store."""
    try:
        from src.llm.vector_store import VectorStore
        vs = VectorStore()
        if vs.is_ready:
            count = vs.index_formulations(formulations_path)
            return count
        print("qdrant_status=not_available")
        return 0
    except Exception as e:
        print(f"qdrant_status=error ({e})")
        return 0


def seed_all() -> None:
    """Seed knowledge graph entities and relationships with fallback export mode."""
    kg = AyushKnowledgeGraph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    formulations = _load_json(FORM_PATH)
    codes = _load_json(CODE_PATH)
    rules = _load_json(RULES_PATH)

    if kg.driver:
        relationships_created = _seed_neo4j(kg, formulations, codes, rules)
        mode = "neo4j"
    else:
        # Keep fallback artifact loadable by InMemoryKnowledgeGraph.
        payload = {
            "formulations": formulations,
            "morbidity_codes": codes,
            "prakriti_rules": rules,
        }
        EXPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        relationships_created = _relationship_estimate(formulations, codes)
        mode = "in_memory_export"

    # Index into Qdrant
    qdrant_count = _seed_qdrant(str(FORM_PATH))

    print(f"seed_mode={mode}")
    print(f"formulations={len(formulations)}")
    print(f"conditions={len(codes)}")
    print(f"prakriti_nodes={len(rules)}")
    print("dosha_nodes=3")
    print(f"classical_text_nodes={len(CLASSICAL_TEXTS)}")
    print(f"relationships_created={relationships_created}")
    print(f"qdrant_indexed={qdrant_count}")
    if mode == "in_memory_export":
        print(f"fallback_export={EXPORT_PATH}")


if __name__ == "__main__":
    seed_all()
