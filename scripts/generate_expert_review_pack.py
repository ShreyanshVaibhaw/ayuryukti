"""Generate an AYUSH expert-review pack from Phase 1 curated datasets.

Outputs:
- data/review_pack/top50_formulations_review.csv
- data/review_pack/priority_morbidity_review.csv
- data/review_pack/condition_mapping_matrix.csv
- data/review_pack/expert_review_pack.xlsx
- data/review_pack/REVIEW_GUIDE.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KB = DATA / "knowledge_base"
OUT = DATA / "review_pack"
PROMPT_DOC = ROOT.parent / "AyurYukti_Design_Document.md"


PRIORITY_CONDITIONS = {
    "Vibandha": "K59.0",
    "Amlapitta": "K21",
    "Sandhivata": "M15-M19",
    "Prameha": "E11",
    "Sthaulya": "E66",
    "Raktachapa": "I10-I15",
    "Ashmari": "N20",
    "Mutraashmari": "N20-N23",
    "Jwara": "R50",
}


CONDITION_CATEGORIES = {
    "Vibandha": ["Digestive"],
    "Amlapitta": ["Digestive"],
    "Sandhivata": ["Musculoskeletal"],
    "Prameha": ["Metabolic", "Urinary"],
    "Sthaulya": ["Metabolic"],
    "Raktachapa": ["Metabolic"],
    "Ashmari": ["Urinary"],
    "Mutraashmari": ["Urinary"],
    "Jwara": ["Fever/Infection", "Respiratory"],
}


CONDITION_TARGETS = {
    "Vibandha": 7,
    "Amlapitta": 6,
    "Sandhivata": 6,
    "Prameha": 8,
    "Sthaulya": 5,
    "Raktachapa": 4,
    "Ashmari": 7,
    "Mutraashmari": 3,
    "Jwara": 4,
}


def load_json(path: Path):
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_category(raw: str) -> str:
    """Normalize category names extracted from prompt text."""
    cat = raw.strip().title()
    cat = cat.replace("Women'S", "Women's")
    return cat


def extract_prompt_category_map() -> Dict[str, str]:
    """Map formulation name -> prompt category from Prompt 1 definitions."""
    text = PROMPT_DOC.read_text(encoding="utf-8")
    section = text.split("A. data/knowledge_base/formulations.json", 1)[1].split(
        "B. data/knowledge_base/ayush_morbidity_codes.json", 1
    )[0]

    pattern = re.compile(r"^([A-Z/' ]+)\(\d+\+\):\s+(.+)$", re.MULTILINE)
    mapping: Dict[str, str] = {}

    for m in pattern.finditer(section):
        category = _normalize_category(m.group(1))
        for part in m.group(2).split(","):
            name = re.sub(r"\s+for\s+.+$", "", part.strip(), flags=re.IGNORECASE).strip()
            if name:
                mapping[name.lower()] = category
    return mapping


def build_top50_formulation_sheet(formulations: List[Dict], category_map: Dict[str, str]) -> pd.DataFrame:
    """Build a prioritized 50-formulation review sheet."""
    indexed: List[Tuple[int, Dict]] = []
    for idx, row in enumerate(formulations):
        cat = category_map.get(row["name_sanskrit"].lower(), "Uncategorized")
        clone = dict(row)
        clone["prompt_category"] = cat
        indexed.append((idx, clone))

    selected: List[Dict] = []
    selected_ids = set()

    for condition, target in CONDITION_TARGETS.items():
        categories = CONDITION_CATEGORIES[condition]
        pool = [r for _, r in indexed if r["prompt_category"] in categories and r["formulation_id"] not in selected_ids]
        pool = sorted(pool, key=lambda r: int(r["formulation_id"].split("_")[1]))

        for rank, row in enumerate(pool[:target], start=1):
            selected_ids.add(row["formulation_id"])
            selected.append(
                {
                    "review_row_id": f"FR-{len(selected) + 1:03d}",
                    "priority_condition": condition,
                    "priority_icd10": PRIORITY_CONDITIONS[condition],
                    "priority_rank_within_condition": rank,
                    "formulation_id": row["formulation_id"],
                    "name_sanskrit": row["name_sanskrit"],
                    "name_english": row["name_english"],
                    "prompt_category": row["prompt_category"],
                    "formulation_type": row["formulation_type"],
                    "route": row["route"],
                    "dosage_range": row["dosage_range"],
                    "classical_reference": row["classical_reference"],
                    "chapter_reference": row["chapter_reference"],
                    "safety_notes": row["safety_notes"],
                    "expert_status": "Pending",
                    "expert_comments": "",
                }
            )

    if len(selected) < 50:
        fill_pool = [r for _, r in indexed if r["formulation_id"] not in selected_ids]
        fill_pool = sorted(fill_pool, key=lambda r: int(r["formulation_id"].split("_")[1]))
        for row in fill_pool:
            if len(selected) >= 50:
                break
            selected_ids.add(row["formulation_id"])
            selected.append(
                {
                    "review_row_id": f"FR-{len(selected) + 1:03d}",
                    "priority_condition": "General Review",
                    "priority_icd10": "",
                    "priority_rank_within_condition": "",
                    "formulation_id": row["formulation_id"],
                    "name_sanskrit": row["name_sanskrit"],
                    "name_english": row["name_english"],
                    "prompt_category": row["prompt_category"],
                    "formulation_type": row["formulation_type"],
                    "route": row["route"],
                    "dosage_range": row["dosage_range"],
                    "classical_reference": row["classical_reference"],
                    "chapter_reference": row["chapter_reference"],
                    "safety_notes": row["safety_notes"],
                    "expert_status": "Pending",
                    "expert_comments": "",
                }
            )

    return pd.DataFrame(selected[:50])


def build_priority_morbidity_sheet(morbidity: List[Dict]) -> pd.DataFrame:
    """Build morbidity review rows for priority conditions and related entries."""
    rows: List[Dict] = []
    idx = 1

    for record in morbidity:
        name_lower = record["ayush_name"].lower()
        condition_hit = any(cond.lower() in name_lower for cond in PRIORITY_CONDITIONS.keys())
        category_hit = record["category"] in {"Digestive", "Musculoskeletal", "Metabolic", "Urinary", "Fever/Infection"}
        if not (condition_hit or category_hit):
            continue

        rows.append(
            {
                "review_row_id": f"MR-{idx:03d}",
                "code_id": record["code_id"],
                "ayush_name": record["ayush_name"],
                "english_name": record["english_name"],
                "category": record["category"],
                "icd10_codes": ", ".join(record["icd10_codes"]),
                "common_symptoms": "; ".join(record["common_symptoms"]),
                "dosha_involvement": ", ".join(record["dosha_involvement"]),
                "mapping_confidence": "Prototype",
                "expert_status": "Pending",
                "expert_comments": "",
            }
        )
        idx += 1

    return pd.DataFrame(rows)


def build_condition_matrix(formulation_df: pd.DataFrame, morbidity_df: pd.DataFrame) -> pd.DataFrame:
    """Build condition-level matrix for faster expert sign-off."""
    matrix_rows = []
    for condition, icd10 in PRIORITY_CONDITIONS.items():
        f_rows = formulation_df[formulation_df["priority_condition"] == condition]
        m_rows = morbidity_df[morbidity_df["ayush_name"].str.contains(condition, case=False, na=False)]

        matrix_rows.append(
            {
                "priority_condition": condition,
                "priority_icd10": icd10,
                "top_formulations": "; ".join(f_rows["name_sanskrit"].head(10).tolist()),
                "mapped_morbidity_codes": "; ".join(m_rows["code_id"].tolist()),
                "expert_signoff": "Pending",
                "signoff_notes": "",
            }
        )
    return pd.DataFrame(matrix_rows)


def write_review_guide(path: Path) -> None:
    """Write short review guide markdown."""
    path.write_text(
        "\n".join(
            [
                "# AyurYukti Expert Review Pack",
                "",
                "## Review Scope",
                "- Validate formulation names, dosage ranges, and route suitability.",
                "- Validate condition-to-formulation prioritization for top 50 entries.",
                "- Validate AYUSH morbidity to ICD-10 mappings for priority use-cases.",
                "",
                "## Status Labels",
                "- `Approved`: Expert confirms entry.",
                "- `Needs Update`: Entry is conceptually valid but needs edits.",
                "- `Reject`: Entry should be removed from Phase 1 pack.",
                "",
                "## Output Files",
                "- `top50_formulations_review.csv`",
                "- `priority_morbidity_review.csv`",
                "- `condition_mapping_matrix.csv`",
                "- `expert_review_pack.xlsx`",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Generate review pack files from current curated assets."""
    OUT.mkdir(parents=True, exist_ok=True)

    formulations = load_json(KB / "formulations.json")
    morbidity = load_json(KB / "ayush_morbidity_codes.json")
    category_map = extract_prompt_category_map()

    top50_df = build_top50_formulation_sheet(formulations, category_map)
    morbidity_df = build_priority_morbidity_sheet(morbidity)
    matrix_df = build_condition_matrix(top50_df, morbidity_df)

    top50_csv = OUT / "top50_formulations_review.csv"
    morbidity_csv = OUT / "priority_morbidity_review.csv"
    matrix_csv = OUT / "condition_mapping_matrix.csv"
    xlsx_path = OUT / "expert_review_pack.xlsx"
    guide = OUT / "REVIEW_GUIDE.md"

    top50_df.to_csv(top50_csv, index=False, encoding="utf-8")
    morbidity_df.to_csv(morbidity_csv, index=False, encoding="utf-8")
    matrix_df.to_csv(matrix_csv, index=False, encoding="utf-8")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        top50_df.to_excel(writer, sheet_name="top50_formulations", index=False)
        morbidity_df.to_excel(writer, sheet_name="priority_morbidity", index=False)
        matrix_df.to_excel(writer, sheet_name="condition_matrix", index=False)

    write_review_guide(guide)

    print(f"top50={len(top50_df)}")
    print(f"morbidity_review_rows={len(morbidity_df)}")
    print(f"condition_rows={len(matrix_df)}")
    print(f"output_dir={OUT}")


if __name__ == "__main__":
    main()
