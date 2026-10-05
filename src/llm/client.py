"""
LLM client abstraction.

Supports any OpenAI-compatible chat-completions API, configured entirely
via environment variables (never hardcoded):
    LLM_API_KEY, LLM_MODEL, LLM_BASE_URL

If no API key is configured (or a live call fails), the client transparently
falls back to a deterministic MOCK mode that clearly labels its output as
such rather than pretending to have produced a real LLM analysis. This
keeps the rest of the application fully functional without any API key.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests

from src.config import settings, get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self):
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")
        self.temperature = settings.get("llm.temperature", 0.2)
        self.max_tokens = settings.get("llm.max_tokens", 900)
        self.mode = "live" if settings.has_llm_key else "mock"
        if self.mode == "mock":
            logger.info("LLMClient running in MOCK mode (no LLM_API_KEY configured). "
                        "All LLM-dependent outputs will be clearly labeled as mock/deterministic.")
        else:
            logger.info("LLMClient configured for live calls to %s (model=%s)", self.base_url, self.model)

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    # ------------------------------------------------------------ raw generate
    def generate(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Returns raw text response, or None if the call failed (caller must
        handle fallback - this method never silently fabricates content)."""
        if self.mode != "live":
            return None
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("Live LLM call failed (%s); caller should use mock fallback", e)
            return None

    # ------------------------------------------------------------ structured generate
    def generate_structured(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Requests a JSON object response and parses it. Returns None on any
        failure (network, non-JSON response, etc.) so the caller can fall
        back to the deterministic mock generator."""
        raw = self.generate(
            system_prompt + "\n\nRespond ONLY with a single valid JSON object. "
                             "No markdown fences, no preamble, no explanation outside the JSON.",
            user_prompt,
        )
        if raw is None:
            return None
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM JSON response (%s). Raw (truncated): %.300s", e, cleaned)
            return None


_client_singleton: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton
