"""Central configuration for the luxury transport RAG pipeline.

All tunable parameters live here so ingest.py, chain.py, and (later) app.py /
eval scripts share a single source of truth. Secrets are loaded from a local
.env file via python-dotenv (see .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL: str = "text-embedding-3-small"

# Use the cheap model for development. Switch to "gpt-4o" for the final
# RAGAS evaluation run as required by the project brief.
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.2

# Chunking is measured in tokens (via tiktoken) so we stay within the
# 500-1000 token / 100 overlap spec from the project brief.
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100

TOP_K: int = 3

CHROMA_DIR: str = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME: str = "luxury_transport"
DISTANCE_METRIC: str = "cosine"

DOCS_DIR: str = str(PROJECT_ROOT / "docs")
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".pdf")


def require_api_key() -> str:
    """Return the OpenAI API key, raising a clear error if it's missing.

    Centralised so every entry point gives the same actionable message
    instead of a deep openai SDK traceback.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export OPENAI_API_KEY in your shell."
        )
    return OPENAI_API_KEY
