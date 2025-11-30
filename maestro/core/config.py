"""Application configuration utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """Configuration values for Maestro.

    Values are loaded from environment variables with sensible defaults so the
    application can run locally without additional configuration. Paths are
    resolved relative to the project root when possible.
    """

    database_url: str = os.getenv("MAESTRO_DATABASE_URL", "sqlite:///./maestro.db")
    faiss_index_path: Path = Path(os.getenv("MAESTRO_FAISS_INDEX", "./data/faiss.index"))
    embedding_model_name: str = os.getenv("MAESTRO_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    summarizer_model_name: str = os.getenv("MAESTRO_SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
    llm_model_name: str = os.getenv("MAESTRO_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    gmail_credentials_path: Path = Path(os.getenv("MAESTRO_GMAIL_CREDENTIALS", "./config/credentials.json"))
    gmail_token_path: Path = Path(os.getenv("MAESTRO_GMAIL_TOKEN", "./config/token.json"))
    device: str = "cuda" if os.getenv("MAESTRO_DEVICE", "cuda") == "cuda" else "cpu"


settings = Settings()

