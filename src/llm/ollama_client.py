"""LLM integration for local Ollama with resilient fallbacks."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import requests

from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_PORT


class LLMClient:
    """Client for Ollama generate/chat endpoints with safe fallback behavior."""

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        port: int = OLLAMA_PORT,
        model: str = OLLAMA_MODEL,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        """Initialize client configuration."""
        self.host = host
        self.port = port
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_url = f"http://{host}:{port}"
        self._last_health_ok: Optional[bool] = None

    def _request_with_retry(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send POST request with exponential-backoff retry policy."""
        backoff = 1.0
        last_error: Optional[Exception] = None

        for _attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"Ollama request failed after retries: {last_error}")

    def _mock_generate(self, prompt: str) -> str:
        """Return conservative deterministic fallback output when Ollama is down."""
        # Minimal fallback that never fabricates detailed clinical claims.
        if "json" in prompt.lower():
            return "{}"
        return ""

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.1) -> str:
        """Generate completion text from Ollama; return graceful fallback if unavailable."""
        health = self.health_check()
        if not health.get("ok", False) or not health.get("model_available", False):
            return self._mock_generate(prompt)

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            data = self._request_with_retry("/api/generate", payload)
            return str(data.get("response", "")).strip()
        except Exception:
            return self._mock_generate(prompt)

    def generate_json(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """Generate JSON response with wrapper stripping and retries."""
        for _ in range(self.max_retries):
            text = self.generate(prompt=prompt, system=system, temperature=0.1)
            cleaned = text.strip()
            cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            if not cleaned:
                continue
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return {}

    def health_check(self) -> Dict[str, Any]:
        """Check whether Ollama service and configured model are available."""
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=2)
            tags.raise_for_status()
            model_list = tags.json().get("models", [])
            available = any(self.model in m.get("name", "") for m in model_list)
            self._last_health_ok = True
            return {"ok": True, "model_available": available, "model": self.model}
        except Exception:
            self._last_health_ok = False
            return {"ok": False, "model_available": False, "model": self.model}


class OllamaClient(LLMClient):
    """Backward-compatible alias for earlier imports."""
