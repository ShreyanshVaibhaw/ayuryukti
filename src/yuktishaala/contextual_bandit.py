"""Contextual Thompson Sampling bandit for treatment learning."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class BetaParams:
    """Beta distribution parameters for one contextual arm."""

    alpha: float
    beta: float
    n_trials: int = 0


class ThompsonSamplingBandit:
    """Contextual bandit using Beta-Binomial model."""

    def __init__(self, exploration_rate: float = 0.2):
        self.arms: Dict[str, BetaParams] = {}
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration = 0.05

    @staticmethod
    def _key(prakriti: str, condition: str, formulation: str) -> str:
        return f"{prakriti}||{condition}||{formulation}"

    def initialize_arm(self, prakriti: str, condition: str, formulation: str, prior_strength: float = 0.5):
        """Initialize arm prior from knowledge prior strength."""
        key = self._key(prakriti, condition, formulation)
        if key in self.arms:
            return
        prior = max(0.0, min(1.0, prior_strength))
        if prior_strength is None:
            self.arms[key] = BetaParams(alpha=1.0, beta=1.0, n_trials=0)
            return
        alpha = max(1.0, prior * 10)
        beta = max(1.0, (1.0 - prior) * 10)
        self.arms[key] = BetaParams(alpha=float(alpha), beta=float(beta), n_trials=0)

    def select_action(self, prakriti: str, condition: str, available_formulations: List[str]) -> List[Tuple[str, float]]:
        """Select ranked actions via Thompson sampling with exploration."""
        if not available_formulations:
            return []
        for formulation in available_formulations:
            self.initialize_arm(prakriti, condition, formulation, prior_strength=0.5)

        results = []
        explore = random.random() < self.exploration_rate
        for formulation in available_formulations:
            arm = self.arms[self._key(prakriti, condition, formulation)]
            if explore:
                sample = random.random()
            else:
                sample = float(np.random.beta(arm.alpha, arm.beta))
            results.append((formulation, sample))

        results.sort(key=lambda x: x[1], reverse=True)
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
        return results

    def update(self, prakriti: str, condition: str, formulation: str, reward: float):
        """Update one arm using observed reward."""
        key = self._key(prakriti, condition, formulation)
        self.initialize_arm(prakriti, condition, formulation, prior_strength=0.5)
        arm = self.arms[key]
        r = max(0.0, min(1.0, reward))
        if r == 1.0:
            arm.alpha += 1.0
        elif r == 0.0:
            arm.beta += 1.0
        else:
            arm.alpha += 0.5
            arm.beta += 0.5
        arm.n_trials += 1

    def get_arm_stats(self, prakriti: str, condition: str) -> List[Dict]:
        """Return posterior stats for all arms in context."""
        prefix = f"{prakriti}||{condition}||"
        rows = []
        for key, arm in self.arms.items():
            if not key.startswith(prefix):
                continue
            formulation = key.split("||", 2)[2]
            mean = arm.alpha / (arm.alpha + arm.beta)
            var = (arm.alpha * arm.beta) / (((arm.alpha + arm.beta) ** 2) * (arm.alpha + arm.beta + 1))
            std = var ** 0.5
            ci_low = max(0.0, mean - 1.96 * std)
            ci_high = min(1.0, mean + 1.96 * std)
            rows.append(
                {
                    "formulation": formulation,
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "mean": mean,
                    "95%_ci": (ci_low, ci_high),
                    "n_trials": arm.n_trials,
                }
            )
        rows.sort(key=lambda x: x["mean"], reverse=True)
        return rows

    def save_model(self, path: str):
        """Persist arm parameters to JSON."""
        payload = {k: asdict(v) for k, v in self.arms.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_model(self, path: str):
        """Load arm parameters from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.arms = {k: BetaParams(**v) for k, v in payload.items()}


def update_bandit(context: Dict[str, str], reward: float) -> Dict[str, float]:
    """Backward-compatible helper retained for older imports."""
    _ = context
    return {"reward": reward, "posterior_alpha": 1.0 + reward, "posterior_beta": 1.0}

