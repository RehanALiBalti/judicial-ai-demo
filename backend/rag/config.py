"""RAG configuration."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "jams_chunks")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


def ollama_base_url() -> str:
    url = OLLAMA_URL.rstrip("/")
    if url.endswith("/api/generate"):
        return url[: -len("/api/generate")]
    return url.replace("/api/chat", "")
