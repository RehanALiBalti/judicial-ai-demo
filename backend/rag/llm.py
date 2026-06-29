"""LangChain Ollama LLM wrapper."""

from __future__ import annotations

from typing import Any, Dict

import requests
from langchain_ollama import ChatOllama

from backend.rag.config import OLLAMA_MODEL, ollama_base_url


def get_chat_llm(max_tokens: int = 500) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=ollama_base_url(),
        temperature=0.1,
        num_predict=max_tokens,
    )


def generate_text(prompt: str, max_new_tokens: int = 500) -> str:
    try:
        llm = get_chat_llm(max_tokens=max_new_tokens)
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        return (content or "").strip()
    except Exception as exc:
        return f"Error calling Ollama via LangChain: {exc}"


def check_ollama() -> Dict[str, Any]:
    try:
        base = ollama_base_url()
        res = requests.get(f"{base}/api/tags", timeout=5)
        res.raise_for_status()
        models = [m.get("name", "") for m in res.json().get("models", [])]
        has_model = any(OLLAMA_MODEL in m or m.startswith(OLLAMA_MODEL) for m in models)
        return {
            "ok": True,
            "model": OLLAMA_MODEL,
            "model_available": has_model,
            "models": models,
            "via": "langchain-ollama",
        }
    except requests.RequestException as exc:
        return {"ok": False, "model": OLLAMA_MODEL, "error": str(exc)}
