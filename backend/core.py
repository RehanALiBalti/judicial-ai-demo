"""
JAMS core business logic — PDF indexing, search, and local LLM chat.
No UI dependencies; used by the FastAPI backend.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from pypdf import PdfReader

from backend.persistence import load_store, manifest_item_key, save_store
from backend.rag import llm as rag_llm
from backend.rag.vectorstore import (
    add_document_chunks,
    build_document_search_text,
    deduplicate_results as _dedupe_results,
    diversify_by_case,
    search_documents,
    search_temp_documents,
    sync_vectorstore,
)

# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

cases: List[Dict[str, Any]] = []
documents: List[Dict[str, Any]] = []
vector_index_ready = False
vector_index_error: Optional[str] = None

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")


def _configure_runtime_cache() -> None:
    """Use writable cache under the app dir (www-data cannot write /var/www/.cache)."""
    app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_root = os.getenv("JAMS_CACHE_DIR", os.path.join(app_root, ".cache"))
    chroma_dir = os.getenv("CHROMA_DIR", os.path.join(app_root, "data", "chroma"))
    home = os.getenv("HOME", os.path.join(app_root, ".home"))
    os.environ.setdefault("HOME", home)
    os.environ.setdefault("CHROMA_DIR", chroma_dir)
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("HF_HOME", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_root, "torch"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(cache_root, "sentence_transformers"))
    for path in {
        home,
        cache_root,
        chroma_dir,
        os.environ["HF_HOME"],
        os.environ["TORCH_HOME"],
        os.environ["SENTENCE_TRANSFORMERS_HOME"],
    }:
        os.makedirs(path, exist_ok=True)


_configure_runtime_cache()
print(f"JAMS ready — LangChain RAG + Ollama: {OLLAMA_MODEL}")


# ---------------------------------------------------------------------------
# Text + PDF utilities
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9\s\-:/]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_filename_title(file_name: str) -> str:
    name = os.path.splitext(file_name or "")[0]
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"^case\s*\d+\s*", "", name, flags=re.IGNORECASE).strip()
    return name.title()


def extract_pdf_text(file_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(file_path)
    pages_text: List[Dict[str, Any]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append({"page": page_no, "text": text.strip()})
    return pages_text


def extract_first_pages_text(file_path: str, max_pages: int = 2) -> str:
    try:
        reader = PdfReader(file_path)
        text_parts: List[str] = []
        total_pages = min(len(reader.pages), max_pages)
        for page_index in range(total_pages):
            text = reader.pages[page_index].extract_text() or ""
            if text.strip():
                text_parts.append(text.strip())
        return "\n".join(text_parts).strip()
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    clean_text = " ".join(str(text or "").split())
    chunks: List[str] = []
    start = 0
    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def guess_case_title(text: str, file_name: str = "") -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title_patterns = [
        r"([A-Z][A-Za-z\s\.\-]+)\s+(?:vs\.?|v\.?|versus)\s+([A-Z][A-Za-z\s\.\-]+)",
        r"case\s*title\s*[:\-]\s*(.+)",
        r"title\s*[:\-]\s*(.+)",
        r"matter\s*of\s*[:\-]\s*(.+)",
    ]
    for line in lines[:50]:
        for pattern in title_patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    return f"{match.group(1).strip()} Vs {match.group(2).strip()}".title()
                return match.group(1).strip().title()
    if file_name:
        return clean_filename_title(file_name)
    return ""


def guess_court_name(text: str) -> str:
    court_patterns = [
        r"(Supreme Court(?:\s+of\s+[A-Za-z\s]+)?)",
        r"([A-Za-z\s]+High Court)",
        r"([A-Za-z\s]+Sessions Court)",
        r"(Sessions Court)",
        r"([A-Za-z\s]+Civil Court)",
        r"(Civil Court)",
        r"([A-Za-z\s]+Family Court)",
        r"(Family Court)",
        r"([A-Za-z\s]+Service Tribunal)",
        r"(Service Tribunal)",
        r"([A-Za-z\s]+Tribunal)",
        r"(Tribunal)",
        r"court\s*name\s*[:\-]\s*(.+)",
        r"court\s*[:\-]\s*(.+)",
    ]
    for pattern in court_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            court = re.sub(r"\s+", " ", match.group(1).strip())
            if 3 <= len(court) <= 80:
                return court.title()
    return ""


def normalize_date_value(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    date_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return ""


def guess_decision_date(text: str, file_name: str = "") -> str:
    date_patterns = [
        r"decision\s*date\s*[:\-]\s*([A-Za-z0-9,\s/\-]+)",
        r"date\s*of\s*decision\s*[:\-]\s*([A-Za-z0-9,\s/\-]+)",
        r"decided\s*on\s*[:\-]?\s*([A-Za-z0-9,\s/\-]+)",
        r"order\s*date\s*[:\-]\s*([A-Za-z0-9,\s/\-]+)",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b(\d{2}-\d{2}-\d{4})\b",
        r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
        r"\b([A-Za-z]+\s+\d{1,2},\s*\d{4})\b",
    ]
    combined_text = f"{text}\n{file_name}"
    for pattern in date_patterns:
        match = re.search(pattern, combined_text, flags=re.IGNORECASE)
        if match:
            raw_date = match.group(1).strip().split("\n")[0].strip()
            normalized = normalize_date_value(raw_date)
            if normalized:
                return normalized
    return ""


def auto_fill_metadata_from_path(file_path: str, file_name: str) -> Dict[str, Any]:
    first_text = extract_first_pages_text(file_path, max_pages=2)
    if not first_text:
        return {
            "case_title": clean_filename_title(file_name),
            "court_name": "",
            "decision_date": "",
            "status": "warning",
            "message": "No extractable text found. Title guessed from file name. OCR may be required.",
        }
    case_title = guess_case_title(first_text, file_name)
    court_name = guess_court_name(first_text)
    decision_date = guess_decision_date(first_text, file_name)
    missing = []
    if not case_title:
        missing.append("case title")
    if not court_name:
        missing.append("court name")
    if not decision_date:
        missing.append("decision date")
    if missing:
        return {
            "case_title": case_title,
            "court_name": court_name,
            "decision_date": decision_date,
            "status": "warning",
            "message": "Auto-fill partially completed. Missing: " + ", ".join(missing),
        }
    return {
        "case_title": case_title,
        "court_name": court_name,
        "decision_date": decision_date,
        "status": "success",
        "message": "Case metadata detected from the PDF.",
    }


# ---------------------------------------------------------------------------
# Indexing + search (LangChain Chroma + MMR)
# ---------------------------------------------------------------------------


def rebuild_faiss_index() -> None:
    """Rebuild persisted vector index (Chroma). Kept name for compatibility."""
    sync_vectorstore(documents)


def start_vector_index_sync() -> None:
    """Build Chroma index in background so API can start immediately."""
    import threading

    global vector_index_ready, vector_index_error

    def _run() -> None:
        global vector_index_ready, vector_index_error
        try:
            if not documents:
                vector_index_ready = True
                return
            print(f"Building Chroma index ({len(documents)} chunks)…")
            sync_vectorstore(documents)
            vector_index_ready = True
            print(f"Chroma index ready ({len(documents)} chunks)")
        except Exception as exc:
            vector_index_error = str(exc)
            print(f"Chroma sync failed: {exc}")

    threading.Thread(target=_run, daemon=True, name="chroma-sync").start()


def get_vector_index_status() -> Dict[str, Any]:
    return {
        "ready": vector_index_ready,
        "error": vector_index_error,
        "chunks": len(documents),
    }


def validate_case_upload(file_path: Optional[str], case_title: str, court_name: str, decision_date: str) -> List[str]:
    errors: List[str] = []
    if not file_path:
        errors.append("PDF file is required.")
    if not case_title or not case_title.strip():
        errors.append("Case title is required.")
    if not court_name or not court_name.strip():
        errors.append("Court name is required.")
    if not decision_date or not decision_date.strip():
        errors.append("Decision date is required.")
    elif not normalize_date_value(decision_date.strip()):
        errors.append("Decision date must be valid. Recommended format: YYYY-MM-DD.")
    return errors


def upload_case_from_path(
    file_path: str,
    file_name: str,
    case_title: str,
    court_name: str,
    decision_date: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extra_meta = extra_meta or {}

    if extra_meta.get("source") == "fccp":
        stable_key = extra_meta.get("source_id") or manifest_item_key({
            "case_title": case_title,
            "upload_date": extra_meta.get("upload_date") or decision_date,
        })
        for existing in cases:
            if existing.get("source") != "fccp":
                continue
            existing_key = existing.get("source_id") or manifest_item_key({
                "case_title": existing.get("title", ""),
                "upload_date": existing.get("upload_date") or existing.get("decision_date", ""),
            })
            if existing_key == stable_key:
                return {
                    "success": True,
                    "message": f"Already indexed: {existing.get('title')}",
                    "case_id": existing.get("case_id"),
                    "pages": existing.get("pages"),
                    "chunks": 0,
                }
        extra_meta["source_id"] = stable_key

    if extra_meta.get("source") == "lhc":
        from backend.persistence import lhc_manifest_item_key

        stable_key = extra_meta.get("source_id") or lhc_manifest_item_key({
            "pdf_url": extra_meta.get("pdf_url"),
            "lhc_citation": extra_meta.get("lhc_citation"),
            "case_number": case_title,
        })
        for existing in cases:
            if existing.get("source") != "lhc":
                continue
            if existing.get("source_id") == stable_key:
                return {
                    "success": True,
                    "message": f"Already indexed: {existing.get('title')}",
                    "case_id": existing.get("case_id"),
                    "pages": existing.get("pages"),
                    "chunks": 0,
                }
        extra_meta["source_id"] = stable_key

    errors = validate_case_upload(file_path, case_title, court_name, decision_date)
    if errors:
        return {"success": False, "message": " ".join(errors)}

    pages = extract_pdf_text(file_path)
    if not pages:
        return {
            "success": False,
            "message": "No text found in this PDF. This may be a scanned PDF and OCR is required.",
        }

    normalized_date = normalize_date_value(decision_date.strip()) or decision_date.strip()
    case_id = f"CASE-{len(cases) + 1:03d}"
    case_record = {
        "case_id": case_id,
        "title": case_title.strip(),
        "court": court_name.strip(),
        "decision_date": normalized_date,
        "file_name": file_name,
        "pages": len(pages),
        **{k: v for k, v in extra_meta.items() if v is not None},
    }
    cases.append(case_record)

    added_chunks = 0
    new_chunks: List[Dict[str, Any]] = []
    for page in pages:
        for chunk_index, chunk in enumerate(chunk_text(page["text"])):
            chunk_doc = {
                "case_id": case_id,
                "title": case_title.strip(),
                "court": court_name.strip(),
                "decision_date": normalized_date,
                "page": page["page"],
                "chunk_index": chunk_index,
                "text": chunk,
                "source_type": "indexed_case",
                "author_judge": extra_meta.get("author_judge"),
            }
            documents.append(chunk_doc)
            new_chunks.append(chunk_doc)
            added_chunks += 1

    add_document_chunks(new_chunks)
    persist_cases()
    return {
        "success": True,
        "message": f"Uploaded and indexed: {case_title} | Pages: {len(pages)} | Chunks: {added_chunks}",
        "case_id": case_id,
        "pages": len(pages),
        "chunks": added_chunks,
    }


def persist_cases() -> None:
    save_store(cases, documents)


def load_persisted_cases() -> int:
    """Load cases and chunks from disk on startup."""
    global cases, documents
    store = load_store()
    cases = store.get("cases", [])
    documents = store.get("documents", [])
    return len(cases)


def index_fccp_judgment(pdf_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Index a scraped FCCP judgment PDF into the JAMS dataset."""
    return upload_case_from_path(
        file_path=pdf_path,
        file_name=meta.get("file_name") or os.path.basename(pdf_path),
        case_title=meta.get("case_title", "FCCP Judgment"),
        court_name=meta.get("court", "Federal Constitutional Court of Pakistan"),
        decision_date=meta.get("upload_date", "2026-01-01"),
        extra_meta={
            "source": "fccp",
            "source_id": meta.get("source_id"),
            "author_judge": meta.get("author_judge"),
            "download_url": meta.get("download_url"),
            "upload_date": meta.get("upload_date"),
            "pdf_path": pdf_path,
        },
    )


def index_lhc_judgment(pdf_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Index a scraped LHC judgment PDF into the JAMS dataset."""
    return upload_case_from_path(
        file_path=pdf_path,
        file_name=meta.get("file_name") or os.path.basename(pdf_path),
        case_title=meta.get("case_title", "LHC Judgment"),
        court_name=meta.get("court", "Lahore High Court, Lahore"),
        decision_date=meta.get("decision_date", "2026-01-01"),
        extra_meta={
            "source": "lhc",
            "source_id": meta.get("source_id"),
            "author_judge": meta.get("author_judge"),
            "pdf_url": meta.get("pdf_url"),
            "lhc_citation": meta.get("lhc_citation"),
            "case_number": meta.get("case_number"),
            "tag_line": meta.get("tag_line"),
            "pdf_path": pdf_path,
        },
    )


def get_query_tokens(query: str) -> List[str]:
    stopwords = {
        "court", "name", "case", "title", "date", "decision", "the", "a", "an",
        "of", "in", "on", "for", "to", "vs", "versus", "and",
    }
    normalized = normalize_text(query)
    return [token for token in normalized.split() if token and token not in stopwords and len(token) > 1]


def search_cases_by_metadata(query: str) -> List[Dict[str, Any]]:
    query_normalized = normalize_text(query)
    if not query_normalized:
        return []
    query_tokens = get_query_tokens(query)
    matched_cases: List[Dict[str, Any]] = []
    for case in cases:
        metadata_text = normalize_text(
            f"{case.get('case_id', '')} {case.get('title', '')} {case.get('court', '')} "
            f"{case.get('decision_date', '')} {case.get('upload_date', '')} {case.get('author_judge', '')}"
        )
        direct_match = query_normalized in metadata_text
        token_match_count = sum(1 for token in query_tokens if token in metadata_text)
        token_match = len(query_tokens) >= 2 and token_match_count >= 2
        if direct_match or token_match:
            matched_cases.append(case)
    return matched_cases


def is_topic_case_request(query: str) -> bool:
    """User wants cases on a legal topic (not a single-case deep dive)."""
    q = normalize_text(query)
    phrases = (
        "cases regarding", "cases about", "cases on", "cases related",
        "cases involving", "give me cases", "show me cases", "find cases",
        "human rights", "fundamental rights", "regarding human",
        "any cases on", "search cases",
    )
    return any(p in q for p in phrases)


def search_documents_by_keyword(
    query: str,
    top_k: int = 8,
    diverse_cases: bool = True,
) -> List[Dict[str, Any]]:
    """Fallback when vector search returns nothing — scan indexed chunk text."""
    tokens = get_query_tokens(query)
    if not tokens or not documents:
        return []
    scored: List[tuple] = []
    for doc in documents:
        blob = normalize_text(
            f"{doc.get('title', '')} {doc.get('text', '')} {doc.get('author_judge', '')}"
        )
        score = sum(2 if len(t) > 5 else 1 for t in tokens if t in blob)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: (-x[0], str(x[1].get("case_id", ""))))
    raw: List[Dict[str, Any]] = []
    for rank, (_, doc) in enumerate(scored[: top_k * 4], start=1):
        raw.append({
            "rank": rank,
            "case_id": doc.get("case_id"),
            "title": doc.get("title"),
            "court": doc.get("court"),
            "decision_date": doc.get("decision_date"),
            "page": doc.get("page"),
            "chunk_index": doc.get("chunk_index"),
            "source_type": doc.get("source_type", "indexed_case"),
            "author_judge": doc.get("author_judge"),
            "text": doc.get("text", ""),
        })
    if diverse_cases:
        raw = diversify_by_case(raw, max_results=top_k)
    return raw[:top_k]


def reply_topic_search_unavailable(user_question: str) -> str:
    stats = get_dashboard_stats()
    try:
        from backend.scraper.lhc import get_lhc_status

        lhc = get_lhc_status()
        downloaded = lhc.get("downloaded") or 0
        indexed_lhc = lhc.get("indexed") or 0
        if downloaded > indexed_lhc or (downloaded > 0 and stats["cases"] < 100):
            return (
                f"I couldn't search **{user_question.strip()}** in indexed text yet.\n\n"
                f"**On disk:** {downloaded} LHC PDFs  \n"
                f"**Indexed for AI chat:** {stats['cases']} case(s) "
                f"({indexed_lhc} LHC marked in manifest)\n\n"
                "PDFs alone are not searchable — run **indexing** (`--index-only`), "
                "then push `data/jams_store.json` and `data/chroma/` to the server."
            )
    except Exception:
        pass
    if stats["cases"] == 0:
        return (
            "No cases are indexed for AI search yet.\n\n"
            "• Use **FCCP/LHC Judgments** to index PDFs, or\n"
            "• **Upload Case** for manual uploads."
        )
    return (
        f"No indexed passages matched **{user_question.strip()}** "
        f"in the current **{stats['cases']}** case(s).\n\n"
        "Try a narrower term (e.g. *Article 9*, *bail*, *constitutional petition*), "
        "or index more LHC judgments for broader coverage."
    )


def extract_judge_name_from_query(query: str) -> str:
    patterns = [
        r"(?:decision|decession|judgment|judgement)\s+of\s+(?:mr\.?\s*)?justice\s+([a-z][a-z\-\.\s]{2,60}?)\s*$",
        r"(?:mr\.?\s*)?justice\s+([a-z][a-z\-\.\s]{2,60}?)(?:\s*(?:latest|last|recent|decision|judgment|judgement|case|order)|\s*$)",
        r"(?:honourable|honorable|hon)\.?\s*(?:mr\.?\s*)?justice\s+([a-z][a-z\-\.\s]{2,60}?)(?:\s|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            name = re.sub(r"\s+", " ", match.group(1).strip(" .,-"))
            return name.title()
    return ""


def search_cases_by_judge(query: str) -> List[Dict[str, Any]]:
    judge_hint = extract_judge_name_from_query(query)
    hint_norm = normalize_text(judge_hint) if judge_hint else ""
    query_tokens = [t for t in get_query_tokens(query) if t not in {"justice", "mr", "latest", "decision", "give", "me"}]
    matched: List[Dict[str, Any]] = []

    for case in cases:
        judge_field = normalize_text(case.get("author_judge", ""))
        if not judge_field:
            continue
        if hint_norm and hint_norm in judge_field:
            matched.append(case)
            continue
        if len(query_tokens) >= 2:
            token_hits = sum(1 for token in query_tokens if token in judge_field)
            if token_hits >= 2:
                matched.append(case)
    return matched


def case_sort_date(case: Dict[str, Any]) -> str:
    return case.get("upload_date") or case.get("decision_date") or "0000-00-00"


def sort_cases_by_date(case_list: List[Dict[str, Any]], reverse: bool = True) -> List[Dict[str, Any]]:
    return sorted(case_list, key=case_sort_date, reverse=reverse)


def format_judge_cases_summary(judge_cases: List[Dict[str, Any]], judge_label: str) -> str:
    lines = [f"### Cases by **{judge_label}** (newest first)\n"]
    for case in judge_cases[:8]:
        lines.append(
            f"- **{case.get('title')}**  \n"
            f"  Date: {case.get('upload_date') or case.get('decision_date') or 'N/A'} | "
            f"`{case.get('case_id')}`"
        )
    return "\n".join(lines)


def reply_judge_latest_decision(
    latest_case: Dict[str, Any],
    indexed_results: List[Dict[str, Any]],
    judge_label: str,
) -> str:
    date = latest_case.get("upload_date") or latest_case.get("decision_date") or "N/A"
    lines = [
        f"## Latest decision by {judge_label}\n",
        f"**Case:** {latest_case.get('title')}",
        f"**Case ID:** `{latest_case.get('case_id')}`",
        f"**Court:** {latest_case.get('court') or 'N/A'}",
        f"**Date:** {date}",
        f"**Pages:** {latest_case.get('pages') or 'N/A'}",
    ]
    if indexed_results:
        top = indexed_results[0]
        excerpt = (top.get("text") or "").strip()
        if len(excerpt) > 500:
            excerpt = excerpt[:500].rstrip() + "..."
        if excerpt:
            lines.append(f"\n**Excerpt (page {top.get('page', '?')}):**\n{excerpt}")
    lines.append("\n---\n_Ask for a full summary of this case for more detail._")
    return "\n".join(lines)


def is_latest_query(query: str) -> bool:
    q = normalize_text(query)
    return any(word in q for word in ("latest", "last", "recent", "newest", "most recent"))


def is_judge_query(query: str) -> bool:
    q = normalize_text(query)
    return bool(extract_judge_name_from_query(query)) or "justice" in q or "judge" in q


def lookup_case(case_id: str) -> Optional[Dict[str, Any]]:
    for case in cases:
        if case.get("case_id") == case_id:
            return case
    return None


def deduplicate_results(results: List[Dict[str, Any]], max_results: int = 4) -> List[Dict[str, Any]]:
    return _dedupe_results(results, max_results=max_results)


def search_indexed_docs(
    query: str,
    top_k: int = 3,
    case_ids: Optional[List[str]] = None,
    diverse_cases: bool = False,
) -> List[Dict[str, Any]]:
    return search_documents(
        query,
        top_k=top_k,
        case_ids=case_ids,
        diverse_cases=diverse_cases,
    )


def search_indexed_docs_global(
    query: str,
    top_k: int = 3,
    diverse_cases: bool = False,
) -> List[Dict[str, Any]]:
    return search_documents(query, top_k=top_k, diverse_cases=diverse_cases)


def search_temp_docs(query: str, temp_docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    return search_temp_documents(query, temp_docs, top_k=top_k)


def process_chat_attachment_from_path(file_path: str, file_name: str) -> Tuple[List[Dict[str, Any]], str]:
    title = clean_filename_title(file_name) or "Temporary Chat PDF"
    pages = extract_pdf_text(file_path)
    if not pages:
        return [], "No extractable text found in attached PDF. OCR may be required."
    temp_docs: List[Dict[str, Any]] = []
    added_chunks = 0
    for page in pages:
        for chunk_index, chunk in enumerate(chunk_text(page["text"])):
            temp_docs.append({
                "case_id": "CHAT-PDF",
                "title": title,
                "court": "Temporary Chat PDF",
                "decision_date": "N/A",
                "page": page["page"],
                "chunk_index": chunk_index,
                "text": chunk,
                "source_type": "chat_temp_pdf",
            })
            added_chunks += 1
    status = f"Attached PDF processed for this chat only: {title} | Pages: {len(pages)} | Chunks: {added_chunks}."
    return temp_docs, status


def build_sources_text(results: List[Dict[str, Any]], max_chars_per_source: int = 650) -> str:
    sources_text = ""
    for idx, item in enumerate(results, start=1):
        source_label = "Temporary Chat PDF" if item.get("source_type") == "chat_temp_pdf" else "Indexed Case"
        sources_text += f"""
Source {idx}
Source Type: {source_label}
Case ID: {item.get('case_id')}
Case Title: {item.get('title')}
Court: {item.get('court') or 'N/A'}
Decision Date: {item.get('decision_date') or 'N/A'}
Page: {item.get('page')}
Text: {item.get('text', '')[:max_chars_per_source]}
"""
    return sources_text.strip()


def build_chat_history_text(history: List[Dict[str, str]], max_turns: int = 2) -> str:
    if not history:
        return ""
    recent = history[-max_turns * 2:]
    lines = []
    for item in recent:
        role = item.get("role", "")
        content = item.get("content", "")
        lines.append(f"{role.title()}: {content[:400]}")
    return "\n".join(lines)


def call_ollama(prompt: str, max_new_tokens: int = 350) -> str:
    return rag_llm.generate_text(prompt, max_new_tokens=max_new_tokens)


def check_ollama() -> Dict[str, Any]:
    return rag_llm.check_ollama()


def generate_from_model(prompt: str, max_new_tokens: int = 350) -> str:
    return rag_llm.generate_text(prompt, max_new_tokens=max_new_tokens)


def get_dashboard_stats() -> Dict[str, int]:
    total_pages = sum(c.get("pages", 0) for c in cases)
    return {
        "cases": len(cases),
        "chunks": len(documents),
        "pages": total_pages,
    }


def list_cases() -> List[Dict[str, Any]]:
    return list(cases)


def is_conversational_query(query: str) -> bool:
    """Greetings and general chat — no case sources required."""
    q = normalize_text(query).strip()
    if not q:
        return False
    word_count = len(q.split())
    if word_count > 12:
        return False
    patterns = [
        r"^(hi|hello|hey|hola|salam|assalam|aoa|aslam|good\s+(morning|afternoon|evening|night))[!.?\s]*$",
        r"^(thanks|thank\s+you|shukriya|ok|okay|bye|goodbye)[!.?\s]*$",
        r"^(help|help me)[?.!\s]*$",
        r"^who are you[?.!\s]*$",
        r"^what can you do[?.!\s]*$",
        r"^how (does|do) (this|jams) work[?.!\s]*$",
    ]
    return any(re.match(p, q, re.IGNORECASE) for p in patterns)


def is_case_inventory_query(query: str) -> bool:
    """User asking what cases exist — answer from metadata, not vector search."""
    q = normalize_text(query)
    phrases = (
        "what cases", "which cases", "list cases", "indexed cases",
        "how many cases", "how much cases", "any cases", "show cases",
        "cases indexed", "cases are indexed", "cases do you have",
        "cases of record", "case records", "case record",
        "total cases", "number of cases", "count cases",
        "how many record", "how much record", "records do you have",
        "cases in database", "cases in system", "in the dataset",
    )
    if any(p in q for p in phrases):
        return True
    if re.search(r"how\s+(many|much)\s+.*\b(case|record)", q):
        return True
    return False


def reply_case_inventory() -> str:
    stats = get_dashboard_stats()
    lines = [
        f"### Case records in JAMS\n",
        f"**AI chat (indexed):** {stats['cases']} case(s), "
        f"{stats['chunks']} text chunks, {stats['pages']} pages\n",
    ]

    try:
        from backend.scraper.fccp import get_fccp_status
        from backend.scraper.lhc import get_lhc_status

        fccp = get_fccp_status()
        lhc = get_lhc_status()
        lines.append(
            f"**FCCP judgments:** {fccp.get('total_items', 0)} in manifest, "
            f"{fccp.get('downloaded', 0)} PDFs, {fccp.get('indexed', 0)} indexed for chat"
        )
        lines.append(
            f"**LHC judgments:** {lhc.get('total_items', 0)} in manifest, "
            f"{lhc.get('downloaded', 0)} PDFs, {lhc.get('indexed', 0)} indexed for chat"
        )
        total_records = (fccp.get("total_items") or 0) + (lhc.get("total_items") or 0)
        lines.insert(1, f"**Total records (FCCP + LHC manifests):** {total_records}\n")
    except Exception:
        pass

    if not cases:
        lines.append(
            "\nNo cases are indexed for AI chat yet. "
            "Use **FCCP/LHC Judgments** tabs to index PDFs, or **Upload Case**."
        )
        return "\n".join(lines)

    lines.append("\n**Indexed cases (sample):**")
    for case in cases[:15]:
        lines.append(
            f"- **{case.get('title')}** (`{case.get('case_id')}`) — "
            f"{case.get('court') or 'N/A'}, {case.get('decision_date') or 'N/A'}"
        )
    if len(cases) > 15:
        lines.append(f"\n_…and {len(cases) - 15} more indexed case(s)._")
    return "\n".join(lines)


def wants_case_content_answer(query: str) -> bool:
    """User wants substance from case PDF text, not just a metadata card."""
    if is_topic_case_request(query):
        return False
    q = normalize_text(query)
    content_phrases = (
        "detail", "details", "about", "summar", "explain", "tell me",
        "what happened", "decision", "judgment", "judgement", "verdict",
        "outcome", "ruling", "holding", "facts", "issue", "reasoning",
        "analysis", "share", "describe", "overview", "key point",
        "legal point", "bail", "sentence", "conviction", "appeal",
        "order", "disposed", "dismissed", "allowed", "petition",
    )
    return any(phrase in q for phrase in content_phrases) or len(q.split()) > 7


def is_broad_legal_topic_query(query: str) -> bool:
    """Cross-case legal topics — retrieve from multiple cases."""
    if is_case_record_lookup(query) or is_judge_query(query):
        return False
    q = normalize_text(query)
    terms = (
        "right", "rights", "human rights", "fundamental rights",
        "article", "constitutional", "fundamental",
        "section", "precedent", "jurisprudence", "cases on", "law on",
        "across cases", "multiple cases", "various cases",
    )
    return any(term in q for term in terms) or is_topic_case_request(query)


def is_case_record_lookup(query: str) -> bool:
    """Short lookup — show case record card only (no content analysis)."""
    if wants_case_content_answer(query):
        return False
    q = normalize_text(query)
    lookup_phrases = (
        "find case", "search case", "lookup", "case record",
        "case id", "matching case", "show case",
    )
    return any(p in q for p in lookup_phrases)


def reply_conversational(user_question: str) -> str:
    stats = get_dashboard_stats()
    prompt = f"""You are JAMS, a friendly judicial AI management assistant.

Session: {stats['cases']} indexed cases, {stats['chunks']} text chunks.

The user sent a casual message (greeting, thanks, or general question).
Reply warmly in 2-4 short sentences.
Explain you help with judicial case PDFs — upload, search, and answers from indexed sources.
If no cases indexed yet, suggest Upload Case tab or attach a PDF in chat.
Do NOT invent case names or legal facts. Do NOT refuse simple greetings.

User: {user_question}

Reply:"""
    answer = call_ollama(prompt, max_new_tokens=180)
    if answer and not answer.startswith("Error"):
        return answer
    if stats["cases"] == 0:
        return (
            "Hello! I'm **JAMS**, your judicial AI assistant.\n\n"
            "No cases are indexed yet. Use **Upload Case** to add PDFs, "
            "or attach a PDF here in chat for session-only Q&A."
        )
    return (
        f"Hello! I'm **JAMS**. You have **{stats['cases']}** indexed case(s) ready. "
        "Ask me about case content, or attach a PDF for quick analysis."
    )


def chat(
    message: str,
    history: List[Dict[str, str]],
    temp_docs: List[Dict[str, Any]],
    pdf_path: Optional[str] = None,
    pdf_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a chat turn. Returns updated history, temp_docs, and status."""
    if history is None:
        history = []
    if temp_docs is None:
        temp_docs = []

    has_pdf = pdf_path is not None
    user_question = message.strip() if message and message.strip() else ""
    if not user_question and has_pdf:
        user_question = "Summarize this PDF."
    if not user_question:
        return {
            "history": history,
            "temp_docs": temp_docs,
            "status": "warning",
            "message": "Type a message or attach a PDF.",
        }

    attachment_notice = ""
    if has_pdf and pdf_path and pdf_filename:
        new_docs, attachment_notice = process_chat_attachment_from_path(pdf_path, pdf_filename)
        if new_docs:
            temp_docs = new_docs
        elif attachment_notice:
            history = history + [
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": attachment_notice},
            ]
            return {
                "history": history,
                "temp_docs": temp_docs,
                "status": "warning",
                "message": attachment_notice,
            }

    display_user_message = user_question
    if has_pdf and pdf_filename:
        display_user_message = f"📎 {pdf_filename}\n\n{user_question}"

    history = history + [{"role": "user", "content": display_user_message}]

    if is_conversational_query(user_question):
        response = reply_conversational(user_question)
        history = history + [{"role": "assistant", "content": response}]
        return {
            "history": history,
            "temp_docs": temp_docs,
            "status": "success",
            "message": "Reply sent.",
        }

    if is_case_inventory_query(user_question) and not temp_docs:
        response = reply_case_inventory()
        history = history + [{"role": "assistant", "content": response}]
        return {
            "history": history,
            "temp_docs": temp_docs,
            "status": "success",
            "message": "Case list sent.",
        }

    judge_hint = extract_judge_name_from_query(user_question)
    if is_judge_query(user_question) and not temp_docs:
        judge_cases = search_cases_by_judge(user_question)
        if judge_cases:
            sorted_judge_cases = sort_cases_by_date(judge_cases)
            judge_label = sorted_judge_cases[0].get("author_judge") or judge_hint or "the judge"

            if is_latest_query(user_question):
                latest_case = sorted_judge_cases[0]
                indexed_results = search_indexed_docs(
                    user_question, top_k=3, case_ids=[latest_case["case_id"]]
                )
                assistant_answer = reply_judge_latest_decision(
                    latest_case, indexed_results, judge_label
                )
                history = history + [{"role": "assistant", "content": assistant_answer}]
                return {
                    "history": history,
                    "temp_docs": temp_docs,
                    "status": "success",
                    "message": "Latest judge decision returned.",
                }

            focus_cases = sorted_judge_cases[:5]
            focus_ids = [c["case_id"] for c in focus_cases]
            judge_summary = format_judge_cases_summary(sorted_judge_cases, judge_label)
            indexed_results = search_indexed_docs(user_question, top_k=6, case_ids=focus_ids)
            if not indexed_results:
                indexed_results = search_indexed_docs_global(user_question, top_k=4)

            sources_text = judge_summary + "\n\n" + build_sources_text(indexed_results or [])
            previous_chat = build_chat_history_text(history[:-1])
            prompt = f"""
You are an AI judicial research chat assistant.

CRITICAL:
- Answer ONLY the CURRENT user question below.
- Do NOT repeat or copy any previous assistant answer.
- Previous conversation is background context only.

Strict rules:
- Answer only from the provided sources and judge case list.
- Do not invent case names, citations, laws, facts, or decisions.
- If not supported by sources, say: "No supported source found."

Previous conversation:
{previous_chat if previous_chat else "No previous conversation."}

CURRENT user question (answer this only):
{user_question}

Available sources:
{sources_text}

Answer format:
1. Answer
2. Relevant Source
3. Reasoning
4. Source References
"""
            try:
                assistant_answer = generate_from_model(prompt, max_new_tokens=500)
            except Exception as error:
                assistant_answer = f"AI generation failed: {str(error)}"

            if assistant_answer.startswith("Error"):
                assistant_answer = judge_summary

            history = history + [{"role": "assistant", "content": assistant_answer}]
            return {
                "history": history,
                "temp_docs": temp_docs,
                "status": "success",
                "message": "Judge query answered.",
            }

    metadata_matches = search_cases_by_metadata(user_question)
    matched_case_ids = [c["case_id"] for c in metadata_matches]

    # Metadata card only for short lookups — not for "details about Ali Khan case"
    if metadata_matches and not temp_docs and is_case_record_lookup(user_question):
        response = "### Matching Case Records\n\n"
        for case in metadata_matches:
            response += (
                f"**Case ID:** {case.get('case_id')}  \n"
                f"**Title:** {case.get('title')}  \n"
                f"**Court:** {case.get('court') or 'N/A'}  \n"
                f"**Decision Date:** {case.get('decision_date') or 'N/A'}  \n"
                f"**Pages:** {case.get('pages') or 'N/A'}\n\n---\n"
            )
        history = history + [{"role": "assistant", "content": response}]
        return {
            "history": history,
            "temp_docs": temp_docs,
            "status": "success",
            "message": "Matching cases found.",
        }

    temp_results = search_temp_docs(user_question, temp_docs, top_k=3) if temp_docs else []

    if matched_case_ids and wants_case_content_answer(user_question):
        indexed_results = search_indexed_docs(
            user_question, top_k=5, case_ids=matched_case_ids,
        )
        if not indexed_results:
            indexed_results = search_indexed_docs_global(user_question, top_k=4)
    else:
        broad = is_broad_legal_topic_query(user_question)
        indexed_results = search_indexed_docs_global(
            user_question, top_k=8 if broad else 4, diverse_cases=broad,
        )
    if temp_results:
        results = temp_results + indexed_results[:1]
    else:
        results = indexed_results
    results = deduplicate_results(results, max_results=6)

    if not results and (
        is_broad_legal_topic_query(user_question) or is_topic_case_request(user_question)
    ):
        if documents:
            results = search_documents_by_keyword(user_question, top_k=8)
            results = deduplicate_results(results, max_results=6)

    if not results and metadata_matches and not temp_docs:
        response = "### Matching Case Records\n\n"
        for case in metadata_matches:
            response += (
                f"**Case ID:** {case.get('case_id')}  \n"
                f"**Title:** {case.get('title')}  \n"
                f"**Court:** {case.get('court') or 'N/A'}  \n"
                f"**Decision Date:** {case.get('decision_date') or 'N/A'}  \n"
                f"**Pages:** {case.get('pages') or 'N/A'}\n\n"
                f"_Ask a specific question (e.g. summarize the decision) for a detailed AI answer._\n\n---\n"
            )
        history = history + [{"role": "assistant", "content": response}]
        return {
            "history": history,
            "temp_docs": temp_docs,
            "status": "success",
            "message": "Matching cases found.",
        }

    if not results:
        if is_conversational_query(user_question):
            response = reply_conversational(user_question)
        elif is_case_inventory_query(user_question):
            response = reply_case_inventory()
        elif is_broad_legal_topic_query(user_question) or is_topic_case_request(user_question):
            response = reply_topic_search_unavailable(user_question)
        else:
            response = (
                "I couldn't find relevant content in your indexed cases or attached PDF "
                "for this question.\n\n"
                "• Upload a case in **Upload Case**, or\n"
                "• Attach a PDF in this chat, then ask again."
            )
        history = history + [{"role": "assistant", "content": response}]
        return {
            "history": history,
            "temp_docs": temp_docs,
            "status": "info",
            "message": "No matching source found.",
        }

    sources_text = build_sources_text(results)
    previous_chat = build_chat_history_text(history[:-1])
    multi_case_note = ""
    if is_broad_legal_topic_query(user_question):
        multi_case_note = (
            "- This is a cross-case legal topic: cite ALL relevant cases in the sources "
            "(at least 3 different case IDs if available).\n"
        )
    prompt = f"""
You are an AI judicial research chat assistant.

CRITICAL:
- Answer ONLY the CURRENT user question below.
- Do NOT repeat or copy any previous assistant answer.
- Previous conversation is background context only.

Strict rules:
- Answer only from the provided sources.
- Sources may include indexed cases and/or a temporary PDF attached in chat.
- A temporary chat PDF is only chat context and must not be treated as stored case database.
- Do not invent case names, citations, laws, facts, or decisions.
- If not supported by sources, say: "No supported source found."
- Always mention source type, title, case id, and page number.
- Keep answer clear, detailed, and structured.
- When the user asks for details about a named case, summarize facts, issues, and outcome from the sources.
{multi_case_note}
Previous conversation:
{previous_chat if previous_chat else "No previous conversation."}

CURRENT user question (answer this only):
{user_question}

Available sources:
{sources_text}

Answer format:
1. Answer
2. Relevant Source
3. Reasoning
4. Source References
"""
    try:
        assistant_answer = generate_from_model(prompt, max_new_tokens=500)
    except Exception as error:
        assistant_answer = f"AI generation failed: {str(error)}"

    if attachment_notice:
        assistant_answer = f"**Attachment processed:** {attachment_notice}\n\n{assistant_answer}"

    history = history + [{"role": "assistant", "content": assistant_answer}]
    return {
        "history": history,
        "temp_docs": temp_docs,
        "status": "success",
        "message": "Answer generated.",
    }


_loaded = load_persisted_cases()
if _loaded:
    print(f"Loaded {_loaded} persisted case(s) from disk")
