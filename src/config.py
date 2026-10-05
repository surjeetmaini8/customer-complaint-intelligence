from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings:
    def __init__(self):
        self._raw = _load_yaml(PROJECT_ROOT / "config.yaml")
        self.root = PROJECT_ROOT

        # ---- env-backed settings (never hardcode secrets) ----
        self.llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        self.database_url = os.getenv("DATABASE_URL", f"sqlite:///{self.path('paths.database')}")
        self.api_host = os.getenv("API_HOST", self.get("api.host", "0.0.0.0"))
        self.api_port = int(os.getenv("API_PORT", self.get("api.port", 8000)))
        self.log_level = os.getenv("LOG_LEVEL", self.get("logging.level", "INFO"))
        self.api_require_key = os.getenv("API_REQUIRE_KEY", str(self.get("api.require_api_key", False))).lower() in {"1", "true", "yes", "on"}
        self.api_key = os.getenv("API_KEY", "").strip()
        self.retain_original_text = os.getenv("RETAIN_ORIGINAL_TEXT", str(self.get("privacy.retain_original_text", True))).lower() in {"1", "true", "yes", "on"}

        self._setup_logging()

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node = self._raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted_key: str) -> Path:
        rel = self.get(dotted_key)
        if rel is None:
            raise KeyError(f"No path configured for {dotted_key}")
        return self.root / rel

    def _setup_logging(self):
        log_path = self.root / self.get("logging.log_file", "storage/app.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
            force=True,
        )

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
