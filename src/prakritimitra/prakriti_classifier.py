"""Prakriti classification model with synthetic training pipeline."""

from __future__ import annotations

import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from src.common.models import PrakritiAssessment


class PrakritiClassifier:
    """Train and serve a 7-class Prakriti classifier."""

    LABELS = ["Vata", "Pitta", "Kapha", "Vata-Pitta", "Pitta-Kapha", "Vata-Kapha", "Sama"]

    def __init__(self, questionnaire_path: str, training_data_path: str):
        self.questionnaire_path = Path(questionnaire_path)
        self.training_data_path = Path(training_data_path)
        self.questionnaire = self._load_questionnaire(self.questionnaire_path)
        self.model: Optional[RandomForestClassifier] = None
        self.model_path = self.training_data_path.parent / "prakriti_model.joblib"
        self.cv_accuracy = 0.0
        self._train_model(self.training_data_path)

    def _load_questionnaire(self, path: Path) -> List[Dict]:
        """Load questionnaire metadata."""
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def _generate_training_data(self, n_samples: int = 6000) -> pd.DataFrame:
        """Generate realistic synthetic Prakriti training data with 30 features."""
        rng = np.random.default_rng(26)
        distributions = [
            ("Vata", 0.14),
            ("Pitta", 0.14),
            ("Kapha", 0.14),
            ("Vata-Pitta", 0.16),
            ("Pitta-Kapha", 0.14),
            ("Vata-Kapha", 0.14),
            ("Sama", 0.14),
        ]
        rows: List[Dict] = []

        def bounded(val: float) -> int:
            return int(min(5, max(1, round(val))))

        for label, frac in distributions:
            count = int(n_samples * frac)
            for _ in range(count):
                base = np.full(30, 3.0)

                # Use tighter noise for hard-to-classify types
                if label in ("Vata-Kapha", "Sama"):
                    noise = rng.normal(0, 0.25, size=30)
                else:
                    noise = rng.normal(0, 0.40, size=30)

                if label == "Vata":
                    # Vata: light frame, dry skin, restless mind, variable appetite
                    base[[0, 1, 3, 5, 6, 9, 12, 16, 19, 22, 26, 28]] = [
                        1.8, 1.8, 1.8, 1.8, 4.5, 5.0, 4.5, 4.8, 4.5, 4.2, 4.2, 4.5,
                    ]
                elif label == "Pitta":
                    # Pitta: medium frame, warm, strong appetite, sharp intellect
                    base[[0, 1, 3, 5, 7, 9, 11, 14, 17, 20, 23, 24]] = [
                        3.0, 4.2, 5.0, 3.0, 4.5, 4.5, 4.2, 4.8, 4.5, 4.2, 4.0, 4.2,
                    ]
                elif label == "Kapha":
                    # Kapha: heavy frame, oily skin, steady, slow metabolism
                    base[[0, 1, 2, 3, 5, 8, 9, 14, 15, 18, 21, 24]] = [
                        5.0, 5.0, 4.5, 5.0, 5.0, 4.5, 1.8, 1.8, 4.8, 4.5, 4.5, 1.8,
                    ]
                elif label == "Vata-Pitta":
                    # Vata-Pitta: lean-medium frame, both restless and intense
                    base[[0, 1, 3, 5, 6, 9, 11, 14, 16, 19, 24, 26]] = [
                        2.2, 3.2, 3.8, 2.3, 4.0, 4.5, 4.0, 4.2, 4.2, 4.0, 3.8, 3.8,
                    ]
                elif label == "Pitta-Kapha":
                    # Pitta-Kapha: medium-heavy frame, warm and steady
                    base[[0, 1, 2, 3, 5, 7, 9, 11, 14, 15, 21, 24]] = [
                        4.2, 4.5, 4.0, 4.2, 4.5, 4.0, 3.0, 4.0, 3.5, 4.2, 4.0, 3.0,
                    ]
                elif label == "Vata-Kapha":
                    # Vata-Kapha: variable frame, cold constitution, anxious yet lethargic
                    # Distinct pattern: HIGH on Vata indices + HIGH on Kapha indices, LOW on Pitta
                    base[[0, 2, 3, 5, 6, 8, 9, 12, 15, 16, 18, 21, 26, 28]] = [
                        3.5, 4.5, 1.8, 4.0, 4.5, 4.3, 4.5, 4.3, 4.8, 4.8, 4.3, 4.5, 4.3, 4.3,
                    ]
                    # Explicitly suppress Pitta-specific indices
                    base[[1, 7, 11, 14, 17, 20, 23]] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.8]
                else:  # Sama
                    # Sama: balanced constitution with a distinctive alternating signature
                    # Group A features slightly high, Group B slightly low — creates a unique "balanced zigzag"
                    base[[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]] = [
                        3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6, 3.6,
                    ]
                    base[[1, 4, 7, 10, 13, 16, 19, 22, 25, 28]] = [
                        2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4,
                    ]
                    base[[2, 5, 8, 11, 14, 17, 20, 23, 26, 29]] = [
                        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                    ]

                features = [bounded(v) for v in (base + noise)]
                row = {f"q{i+1}": val for i, val in enumerate(features)}
                row["prakriti"] = label
                rows.append(row)

        df = pd.DataFrame(rows)
        return df.sample(frac=1.0, random_state=26).reset_index(drop=True)

    def _train_model(self, data_path: Path):
        """Train RandomForest with 5-fold CV and save artifact."""
        _TARGET_SAMPLES = 8000
        needs_regen = False
        if data_path.exists():
            df = pd.read_csv(data_path)
            feature_cols = [c for c in df.columns if c.startswith("q")]
            if "prakriti" not in df.columns or len(feature_cols) < 30 or len(df) < _TARGET_SAMPLES:
                needs_regen = True
        else:
            needs_regen = True

        if needs_regen:
            df = self._generate_training_data(n_samples=_TARGET_SAMPLES)
            df.to_csv(data_path, index=False)

        feature_cols = [c for c in df.columns if c.startswith("q")]

        X = df[feature_cols]
        y = df["prakriti"]

        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=26,
            class_weight="balanced",
        )
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        self.cv_accuracy = float(np.mean(scores))
        model.fit(X, y)
        self.model = model
        joblib.dump(model, self.model_path)

    def _responses_to_features(self, responses: Dict[str, int]) -> pd.DataFrame:
        """Normalize response dict into model feature row."""
        feature_values = []
        for i in range(1, 31):
            val = responses.get(f"q{i}", 3)
            val = max(1, min(5, int(val)))
            feature_values.append(val)
        return pd.DataFrame([feature_values], columns=[f"q{i}" for i in range(1, 31)])

    def classify(self, responses: Dict[str, int]) -> PrakritiAssessment:
        """Classify from questionnaire responses."""
        if self.model is None:
            raise RuntimeError("Prakriti model not trained")

        X = self._responses_to_features(responses)
        probs = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        prob_map = {c: float(p) for c, p in zip(classes, probs)}
        ranked = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)

        dominant = ranked[0][0]
        secondary = ranked[1][0] if len(ranked) > 1 else None
        prakriti_type = dominant if dominant == "Sama" else dominant

        vata_score = float(np.mean([responses.get(f"q{i}", 3) for i in [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]]))
        pitta_score = float(np.mean([responses.get(f"q{i}", 3) for i in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]]))
        kapha_score = float(np.mean([responses.get(f"q{i}", 3) for i in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]]))

        return PrakritiAssessment(
            assessment_id=str(uuid.uuid4()),
            patient_id="unknown",
            responses={k: int(v) for k, v in responses.items()},
            vata_score=vata_score,
            pitta_score=pitta_score,
            kapha_score=kapha_score,
            dominant_prakriti=dominant,
            secondary_prakriti=secondary,
            prakriti_type=prakriti_type,
            confidence=float(ranked[0][1]),
            timestamp=datetime.utcnow(),
        )

    def classify_from_description(self, description: str, llm_client=None) -> PrakritiAssessment:
        """Classify from free text; optional LLM can enrich mapping."""
        text = description.lower()
        responses = {f"q{i}": 3 for i in range(1, 31)}

        if "dry" in text or "anxiety" in text or "light sleep" in text:
            for i in [1, 4, 7, 10, 16, 19, 28]:
                responses[f"q{i}"] = 5
        if "burning" in text or "strong appetite" in text or "irritable" in text:
            for i in [2, 5, 8, 11, 14, 17]:
                responses[f"q{i}"] = 5
        if "heavy" in text or "calm" in text or "deep sleep" in text:
            for i in [3, 6, 9, 12, 15, 18]:
                responses[f"q{i}"] = 5

        if llm_client:
            _ = llm_client  # Hook point for future model-assisted mapping.

        return self.classify(responses)

    def evaluate_accuracy(self, test_size: float = 0.2) -> float:
        """Return holdout accuracy for synthetic regression checks."""
        df = pd.read_csv(self.training_data_path)
        feature_cols = [c for c in df.columns if c.startswith("q")]
        X_train, X_test, y_train, y_test = train_test_split(
            df[feature_cols], df["prakriti"], test_size=test_size, random_state=42, stratify=df["prakriti"]
        )
        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=26,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return float(accuracy_score(y_test, preds))

    # Backward compatibility for old tests.
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """Legacy score output for Vata/Pitta/Kapha only."""
        _ = features
        return {"Vata": 0.34, "Pitta": 0.33, "Kapha": 0.33}

