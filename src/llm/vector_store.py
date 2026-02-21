"""Qdrant vector store integration for semantic search over AYUSH formulations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("AyurYukti.VectorStore")


class VectorStore:
    """Qdrant-backed semantic search for AYUSH knowledge base."""

    COLLECTION_NAME = "ayush_formulations"
    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

    def __init__(self) -> None:
        self.client = None
        self.model = None
        self._ready = False
        self._init_clients()

    def _init_clients(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
                timeout=5,
            )
            # Test connection
            self.client.get_collections()
            logger.info("Qdrant connected at %s:%s", config.QDRANT_HOST, config.QDRANT_PORT)
        except Exception:
            logger.warning("Qdrant not available. Vector search disabled.")
            self.client = None
            return

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded: %s", config.EMBEDDING_MODEL)
        except Exception:
            logger.warning("Sentence transformer not available. Vector search disabled.")
            self.model = None
            return

        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready and self.client is not None and self.model is not None

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        if not self.is_ready:
            return
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self.COLLECTION_NAME)

    def index_formulations(self, formulations_path: str) -> int:
        """Index formulations into Qdrant for semantic search."""
        if not self.is_ready:
            logger.warning("Vector store not ready; skipping indexing.")
            return 0

        from qdrant_client.models import PointStruct

        self._ensure_collection()

        formulations = json.loads(Path(formulations_path).read_text(encoding="utf-8"))
        points = []
        texts = []
        for i, f in enumerate(formulations):
            text = self._formulation_to_text(f)
            texts.append(text)

        if not texts:
            return 0

        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=64)

        for i, (f, embedding) in enumerate(zip(formulations, embeddings)):
            points.append(
                PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={
                        "formulation_id": f.get("formulation_id", ""),
                        "name_sanskrit": f.get("name_sanskrit", ""),
                        "name_english": f.get("name_english", ""),
                        "formulation_type": f.get("formulation_type", ""),
                        "system": f.get("system", ""),
                        "indicated_conditions": f.get("indicated_conditions", []),
                        "indicated_prakriti": f.get("indicated_prakriti", []),
                        "contraindicated_prakriti": f.get("contraindicated_prakriti", []),
                        "dosage_range": f.get("dosage_range", ""),
                        "classical_reference": f.get("classical_reference", ""),
                        "safety_notes": f.get("safety_notes", ""),
                    },
                )
            )

        # Batch upsert
        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=batch)

        logger.info("Indexed %d formulations into Qdrant.", len(points))
        return len(points)

    def _formulation_to_text(self, f: Dict[str, Any]) -> str:
        """Convert formulation record to searchable text."""
        parts = [
            f.get("name_sanskrit", ""),
            f.get("name_english", ""),
            f.get("formulation_type", ""),
            f"System: {f.get('system', '')}",
            f"Conditions: {', '.join(f.get('indicated_conditions', []))}",
            f"Prakriti: {', '.join(f.get('indicated_prakriti', []))}",
            f"Dosha action: {json.dumps(f.get('dosha_action', {}))}",
            f"Ingredients: {', '.join(i.get('name', '') for i in f.get('ingredients', []))}",
            f"Reference: {f.get('classical_reference', '')}",
        ]
        return " | ".join(p for p in parts if p.strip())

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        prakriti_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search for formulations similar to query."""
        if not self.is_ready:
            return []

        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_vector = self.model.encode(query).tolist()

        search_filter = None
        if prakriti_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="indicated_prakriti",
                        match=MatchAny(any=[prakriti_filter]),
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
        )

        return [
            {
                "formulation_id": hit.payload.get("formulation_id", ""),
                "name": hit.payload.get("name_sanskrit", ""),
                "name_english": hit.payload.get("name_english", ""),
                "score": round(hit.score, 3),
                "conditions": hit.payload.get("indicated_conditions", []),
                "classical_reference": hit.payload.get("classical_reference", ""),
                "safety_notes": hit.payload.get("safety_notes", ""),
            }
            for hit in results
        ]

    def find_similar_formulations(
        self,
        formulation_name: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find formulations similar to a given one."""
        return self.search_similar(f"formulation similar to {formulation_name}", top_k=top_k)

    def search_classical_evidence(
        self,
        condition: str,
        prakriti: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find classical text evidence for a condition-prakriti pair."""
        query = f"classical Ayurvedic treatment for {condition}"
        if prakriti:
            query += f" in {prakriti} prakriti patient"
        return self.search_similar(query, top_k=5, prakriti_filter=prakriti)
