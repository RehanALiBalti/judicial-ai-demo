"""Chroma vector store + MMR / multi-case retrieval."""

from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from backend.rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def build_document_search_text(doc: Dict[str, Any]) -> str:
    return f"""
Case ID: {doc.get('case_id', '')}
Case Title: {doc.get('title', '')}
Court: {doc.get('court', '')}
Author Judge: {doc.get('author_judge', '')}
Decision Date: {doc.get('decision_date', '')}
Page: {doc.get('page', '')}
Text: {doc.get('text', '')}
""".strip()


def chunk_id(doc: Dict[str, Any]) -> str:
    return f"{doc.get('case_id')}|{doc.get('page')}|{doc.get('chunk_index')}"


def doc_to_langchain(doc: Dict[str, Any]) -> Document:
    metadata = {
        "case_id": str(doc.get("case_id", "")),
        "title": str(doc.get("title", ""))[:500],
        "court": str(doc.get("court", ""))[:300],
        "decision_date": str(doc.get("decision_date", "")),
        "page": int(doc.get("page") or 0),
        "chunk_index": int(doc.get("chunk_index") or 0),
        "source_type": str(doc.get("source_type", "indexed_case")),
        "author_judge": str(doc.get("author_judge", ""))[:200],
    }
    return Document(page_content=build_document_search_text(doc), metadata=metadata)


def langchain_to_result(doc: Document, rank: int) -> Dict[str, Any]:
    meta = doc.metadata or {}
    text = doc.page_content
    body = ""
    if "Text:" in text:
        body = text.split("Text:", 1)[-1].strip()
    else:
        body = text
    return {
        "rank": rank,
        "case_id": meta.get("case_id"),
        "title": meta.get("title"),
        "court": meta.get("court"),
        "decision_date": meta.get("decision_date"),
        "page": meta.get("page"),
        "chunk_index": meta.get("chunk_index"),
        "source_type": meta.get("source_type", "indexed_case"),
        "author_judge": meta.get("author_judge"),
        "text": body,
    }


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return _vectorstore


def reset_vectorstore() -> None:
    global _vectorstore
    _vectorstore = None
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def _add_batches(docs: List[Dict[str, Any]], batch_size: int = 400) -> None:
    if not docs:
        return
    vs = get_vectorstore()
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        vs.add_documents(
            [doc_to_langchain(d) for d in batch],
            ids=[chunk_id(d) for d in batch],
        )


def rebuild_from_documents(documents: List[Dict[str, Any]]) -> None:
    reset_vectorstore()
    _add_batches(documents)


def sync_vectorstore(documents: List[Dict[str, Any]]) -> None:
    """Rebuild Chroma if empty or out of sync with in-memory document list."""
    if not documents:
        reset_vectorstore()
        return
    vs = get_vectorstore()
    try:
        count = vs._collection.count()
    except Exception:
        count = 0
    if count == len(documents):
        return
    rebuild_from_documents(documents)


def add_document_chunks(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return
    _add_batches(chunks)


def diversify_by_case(results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
    """Spread results across multiple cases (round-robin)."""
    if not results:
        return []
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        cid = str(item.get("case_id") or "unknown")
        buckets.setdefault(cid, []).append(item)

    diverse: List[Dict[str, Any]] = []
    case_ids = list(buckets.keys())
    guard = 0
    while len(diverse) < max_results and case_ids and guard < max_results * len(case_ids) * 3:
        guard += 1
        for cid in list(case_ids):
            if len(diverse) >= max_results:
                break
            if buckets.get(cid):
                diverse.append(buckets[cid].pop(0))
            if not buckets.get(cid):
                case_ids.remove(cid)
    return diverse[:max_results]


def deduplicate_results(results: List[Dict[str, Any]], max_results: int = 6) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()
    for item in results:
        key = (
            item.get("source_type"),
            item.get("case_id"),
            item.get("page"),
            item.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= max_results:
            break
    return unique


def search_documents(
    query: str,
    top_k: int = 6,
    case_ids: Optional[List[str]] = None,
    diverse_cases: bool = False,
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    vs = get_vectorstore()
    try:
        if vs._collection.count() == 0:
            return []
    except Exception:
        return []

    fetch_k = min(max(top_k * 5, 20), 80)

    if case_ids:
        # Filtered similarity search per case subset
        filter_expr: Dict[str, Any]
        if len(case_ids) == 1:
            filter_expr = {"case_id": case_ids[0]}
        else:
            filter_expr = {"case_id": {"$in": case_ids}}
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": fetch_k, "filter": filter_expr},
        )
    else:
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": fetch_k,
                "fetch_k": min(fetch_k * 4, 120),
                "lambda_mult": 0.55,
            },
        )

    lc_docs = retriever.invoke(query)
    results = [langchain_to_result(d, rank) for rank, d in enumerate(lc_docs, start=1)]

    if diverse_cases and not case_ids:
        results = diversify_by_case(results, top_k)
    else:
        results = deduplicate_results(results, max_results=top_k)

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results


def search_temp_documents(query: str, temp_docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    if not temp_docs:
        return []
    from langchain_community.vectorstores import FAISS

    lc_docs = [doc_to_langchain(d) for d in temp_docs]
    temp_store = FAISS.from_documents(lc_docs, get_embeddings())
    found = temp_store.similarity_search(query, k=min(top_k, len(temp_docs)))
    return [langchain_to_result(d, rank) for rank, d in enumerate(found, start=1)]
